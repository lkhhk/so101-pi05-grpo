"""Loopback-only frozen flow-SDE sampler and portable replay-record transport."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import struct
import threading
import time
import urllib.request
from urllib.parse import urlsplit
import uuid

import numpy as np
from PIL import Image, features
import torch

from pi05_rl import FlowRLConfig, LeRobotFlowRL, observation_digest
from pi05_rl_rtc import MODE as RTC_MODE, PREFIX, WINDOW
import pi05_rl_sparse_rgb as sparse_rgb
from pi05_rl_episodes import pack_record, encode_record, decode_record, decode_transport_record, encode_transport_record
from pi05_rl_runtime import load_ira_lora, make_processors, sha256

PROTOCOL = "pi05-flow-sde-http-v1"
TASK = "Pick up the block and place it inside the box."
LOSSLESS_TRANSPORT = "png-v1"
SPLIT_TRANSPORT = "actions-first-v1"
DEFERRED_TRANSPORT = "deferred-trace-v1"
WEBP_UPLOAD = "webp-lossless-v1"
JPEG_DIAGNOSTIC_UPLOAD = "jpeg70-diagnostic-v1"
MAX_RECORD = 32 * 1024 * 1024


class DeferredTraces:
    """Bounded in-memory cache for short probes; reads can be retried until expiry."""

    def __init__(self, *, capacity=24, ttl=180, max_bytes=96*1024*1024):
        self.capacity, self.ttl, self.max_bytes = capacity, ttl, max_bytes
        self.items = {}
        self.lock = threading.Lock()

    def _prune(self):
        now = time.monotonic()
        self.items = {key: item for key, item in self.items.items() if item[1] > now}

    def reserve(self, request_id):
        with self.lock:
            self._prune()
            if request_id in self.items or len(self.items) >= self.capacity:
                raise ValueError("Deferred trace cache full or duplicate request")
            token = object()
            self.items[request_id] = (token, time.monotonic()+self.ttl, 202, b"")
            return token

    def finish(self, request_id, token, payload=None):
        with self.lock:
            self._prune()
            entry = self.items.get(request_id)
            if entry is None or entry[0] is not token:
                return
            room = self.max_bytes - sum(len(item[3]) for item in self.items.values())
            if payload is not None and len(payload) <= room:
                status, body = 200, payload
            else:
                status, body = 422, b"Deferred trace failed or cache byte limit reached"
            self.items[request_id] = (token, time.monotonic()+self.ttl, status, body)

    def get(self, request_id):
        with self.lock:
            self._prune()
            entry = self.items.get(request_id)
            return (404, b"Unknown or expired trace") if entry is None else entry[2:]


def read_frame(stream, limit):
    prefix = stream.read(4)
    if len(prefix) != 4:
        raise ValueError("Missing sampler frame")
    size, = struct.unpack("!I", prefix)
    if not 0 < size <= limit:
        raise ValueError("Invalid sampler frame size")
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("Truncated sampler frame")
    return payload


def encode_rgb(image, *, webp=False, diagnostic_jpeg=False, webp_method=2):
    stream = io.BytesIO()
    if diagnostic_jpeg:
        Image.fromarray(image).save(stream, format="JPEG", quality=70, subsampling=2)
    elif webp:
        Image.fromarray(image).save(stream, format="WEBP", lossless=True, method=webp_method)
    else:
        Image.fromarray(image).save(stream, format="PNG", compress_level=1)
    return stream.getvalue()


def decode_rgb(payload, *, webp=False, diagnostic_jpeg=False):
    if webp or diagnostic_jpeg:
        with Image.open(io.BytesIO(payload), formats=["JPEG" if diagnostic_jpeg else "WEBP"]) as image:
            if image.mode != "RGB" or not all(0 < side <= 1200 for side in image.size):
                raise ValueError("Invalid compressed RGB image")
            return np.array(image)
    from pi05_rl_episodes import open_rgb_png
    return np.array(open_rgb_png(payload))


def pipeline_identity(adapter, provenance, cfg):
    files = {p.name: sha256(p) for p in sorted(Path(adapter).glob("policy_*")) if p.is_file()}
    data = {"policy_version": provenance["policy_version"], "config": cfg.serialized(),
            "processors": files, "task": TASK, "units": "body_degrees_gripper_percent"}
    if "placement" in provenance:
        data["placement"] = provenance["placement"]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), data


class SamplerEngine:
    def __init__(self, base, checkpoint, tokenizer, device, cfg, *, verify_replay=False, profile=False,
                 diagnostic_jpeg=False, offload_text_tables=False, reuse_replay_prefix=False, sparse_upload=False):
        self.policy, provenance = load_ira_lora(base, checkpoint, device=device,
                                                offload_text_tables=offload_text_tables)
        self.adapter = LeRobotFlowRL(self.policy, cfg, policy_version=provenance["policy_version"])
        self.pre, self.post = make_processors(self.policy, checkpoint, tokenizer, device)
        if sparse_upload and (not cfg.rtc or tuple(self.policy.config.image_resolution) != sparse_rgb.TARGET):
            raise ValueError("Sparse RGB requires guided PI05 with 224×224 vision input")
        self.verify_replay = verify_replay
        self.reuse_replay_prefix = reuse_replay_prefix
        self.profile, self.device = profile, device
        pid, processing = pipeline_identity(checkpoint, provenance, cfg)
        self.health = {"ready": True, "protocol": PROTOCOL, "pipeline_id": pid,
                       "policy_version": provenance["policy_version"], "config": cfg.serialized(),
                       "provenance": provenance, "processing": processing,
                       "rtc_enabled": bool(cfg.rtc), "rtc_mode": cfg.rtc, "units": "body_degrees_gripper_percent", "fps": 30,
                       "verify_replay": verify_replay,
                       "replay_prefix": "same_request" if reuse_replay_prefix else "independent",
                       "lossless_transport": LOSSLESS_TRANSPORT,
                       "split_transport": SPLIT_TRANSPORT, "deferred_transport": DEFERRED_TRANSPORT,
                       "webp_upload": WEBP_UPLOAD if features.check("webp") else None,
                       "sparse_rgb": sparse_rgb.CODEC if sparse_upload and features.check("webp") else None,
                       "diagnostic_jpeg_upload": JPEG_DIAGNOSTIC_UPLOAD if diagnostic_jpeg else None}

    def sample(self, state, front, wrist, request_id, *, timings=None, on_actions=None, rtc=None,
               vision_audit=False):
        def now():
            if timings is not None and self.device == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter()

        started = now()
        batch = {"observation.state": torch.from_numpy(state.copy()), "task": TASK}
        for name, rgb in (("front", front), ("wrist", wrist)):
            batch[f"observation.images.{name}"] = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        batch = self.pre(batch)
        if rtc is not None:
            prefix, window = rtc
            batch[PREFIX] = torch.from_numpy(prefix.copy()).unsqueeze(0).to(self.device)
            batch[WINDOW] = torch.from_numpy(window.copy()).to(self.device)
        preprocessed = now()
        context = None
        if self.verify_replay and self.reuse_replay_prefix:
            context = self.adapter.prefix(batch)
            normalized, trace = self.adapter.sample_from_prefix(batch, context)
        else:
            normalized, trace = self.adapter.sample(batch)
        sampled = now()
        physical = self.post(normalized.clone())
        if on_actions is not None:
            args = (normalized.detach().cpu(),) if self.adapter.cfg.rtc else ()
            physical_cpu = physical.detach().cpu()
            if timings is not None:
                timings.update(preprocess_s=preprocessed-started, flow_sample_s=sampled-preprocessed,
                               postprocess_s=now()-sampled)
            on_actions(physical_cpu, *args)
        record = pack_record(batch, trace, physical, self.health["pipeline_id"], request_id)
        if vision_audit:
            views, masks = self.policy._preprocess_images(batch)
            tensors = {f"observation.encoder.{i}": value for i,value in enumerate([*views,*masks])}
            record["vision_input_digest"] = observation_digest(tensors)
        packed = now()
        if self.verify_replay:
            with torch.no_grad():
                if context is not None:
                    # pack_record already binds the trace to this exact batch. Re-evaluate
                    # the recorded denoise step; only the frozen prefix is shared.
                    replay = self.adapter.recompute_from_prefix(batch, trace, context)
                else:
                    replay = self.adapter.recompute(batch, trace)
                error = float((replay-trace.old_logprobs).abs().max())
                ratio_error = float(((replay-trace.old_logprobs).sum((1,2)).exp()-1).abs().max())
            if error > 1e-5 or ratio_error > 1e-3:
                raise ValueError("Sampler runtime failed old-policy probability replay")
            record["replay_check"] = {"max_abs_error":error,"max_ratio_error":ratio_error}
        if timings is not None:
            timings.update(preprocess_s=preprocessed-started, flow_sample_s=sampled-preprocessed,
                           record_pack_s=packed-sampled, replay_s=now()-packed)
        return record


def make_server(engine, port=8095):
    lock = threading.Lock()
    cache = (DeferredTraces(capacity=96, max_bytes=192*1024*1024)
             if engine.health.get("rtc_mode") == RTC_MODE else DeferredTraces())
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/trace/") and engine.health.get("deferred_transport") == DEFERRED_TRANSPORT:
                status, payload = cache.get(self.path[len("/trace/"):])
            elif self.path == "/health":
                status, payload = 200, json.dumps(engine.health).encode()
            else:
                status, payload = 404, b"Not found"
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        def do_POST(self):
            entered = time.perf_counter()
            timings = {} if getattr(engine, "profile", False) else None
            deferred = self.path == "/sample-deferred"
            split = self.path == "/sample-split" or deferred
            sent = False
            token = None
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not (split or self.path == "/sample") or not 0 < size <= 8*1024*1024:
                    raise ValueError("Invalid request")
                if split and engine.health.get("split_transport") != SPLIT_TRANSPORT:
                    raise ValueError("Split transport unavailable")
                if deferred and engine.health.get("deferred_transport") != DEFERRED_TRANSPORT:
                    raise ValueError("Deferred trace transport unavailable")
                body = self.rfile.read(size)
                uploaded = time.perf_counter()
                with np.load(io.BytesIO(body), allow_pickle=False) as data:
                    if str(data["protocol"]) != PROTOCOL: raise ValueError("Protocol mismatch")
                    request_id = str(data["request_id"])
                    if split and (len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id)):
                        raise ValueError("Invalid split request ID")
                    compact = "transport" in data
                    if compact and (str(data["transport"]) != LOSSLESS_TRANSPORT or
                                    engine.health.get("lossless_transport") != LOSSLESS_TRANSPORT):
                        raise ValueError("Transport mismatch")
                    state, front, wrist = data["state"], data["front"], data["wrist"]
                    guided = engine.health.get("rtc_mode") == RTC_MODE
                    rtc = None
                    sparse = "sparse_rgb" in data
                    audit = "vision_audit" in data
                    if (sparse or audit) and (not guided or not compact or
                            engine.health.get("sparse_rgb") != sparse_rgb.CODEC):
                        raise ValueError("Sparse RGB/Audit is unavailable")
                    if guided and not split:
                        raise ValueError("Guided Flow-SDE requires the actions-first stream")
                    if "rtc_prefix" in data or "rtc_window" in data:
                        if not guided or "rtc_prefix" not in data or "rtc_window" not in data:
                            raise ValueError("Unexpected or incomplete RTC conditioning")
                        rtc = validate_rtc_request(data["rtc_prefix"], data["rtc_window"], engine.health["config"])
                    if compact:
                        codec = str(data["upload_codec"]) if "upload_codec" in data else None
                        webp = codec == WEBP_UPLOAD and engine.health.get("webp_upload") == WEBP_UPLOAD
                        jpeg = (deferred and not guided and codec == JPEG_DIAGNOSTIC_UPLOAD and
                                engine.health.get("diagnostic_jpeg_upload") == JPEG_DIAGNOSTIC_UPLOAD)
                        if codec is not None and not (webp or jpeg):
                            raise ValueError("Upload codec mismatch")
                        front, wrist = (decode_rgb(image.tobytes(), webp=webp, diagnostic_jpeg=jpeg)
                                        for image in (front, wrist))
                        if sparse:
                            if not webp or str(data["sparse_rgb"]) != sparse_rgb.CODEC:
                                raise ValueError("Invalid sparse RGB encoding")
                            front, wrist = sparse_rgb.unpack(front), sparse_rgb.unpack(wrist)
                    else:
                        jpeg = False
                    if state.shape != (6,) or state.dtype != np.float32 or not np.isfinite(state).all():
                        raise ValueError("Invalid state")
                    for image in (front, wrist):
                        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or max(image.shape[:2]) > 1200:
                            raise ValueError("Invalid RGB image")
                parsed = time.perf_counter()
                if split:
                    if deferred:
                        token = cache.reserve(request_id)

                    def emit(actions, normalized=None):
                        nonlocal sent
                        cfg = engine.health["config"]
                        if (actions.shape != (1, cfg["action_chunk"], cfg["action_dim"]) or
                                not torch.isfinite(actions).all()):
                            raise ValueError("Invalid sampled actions")
                        fields = dict(request_id=request_id, actions=actions[0].tolist(),
                                               pipeline_id=engine.health["pipeline_id"],
                                               policy_version=engine.health["policy_version"])
                        if guided:
                            if normalized is None or normalized.shape != actions.shape or not torch.isfinite(normalized).all():
                                raise ValueError("Invalid RTC model-space actions")
                            fields["normalized"] = normalized[0].tolist()
                        head = json.dumps(fields).encode()
                        sent = True
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-pi05-actions-first")
                        if timings is not None:
                            self.send_header("X-PI05-Action-Timings", json.dumps(timings))
                        self.send_header("Connection", "close")
                        if deferred:
                            self.send_header("Content-Length", str(4+len(head)))
                        self.end_headers()
                        self.close_connection = True
                        self.wfile.write(struct.pack("!I", len(head)) + head)
                        self.wfile.flush()

                    with lock:
                        options = {"rtc": rtc} if guided else {}
                        if audit:
                            options["vision_audit"] = True
                        if timings is not None:
                            timings.update(body_read_s=uploaded-entered, parse_s=parsed-uploaded,
                                           gpu_queue_s=time.perf_counter()-parsed)
                            options["timings"] = timings
                        record = engine.sample(state, front, wrist, request_id, on_actions=emit, **options)
                    if jpeg:
                        record["diagnostic_upload"] = JPEG_DIAGNOSTIC_UPLOAD
                    if sparse:
                        record["rgb_transport"] = sparse_rgb.CODEC
                    if not sent:
                        raise ValueError("Sampler did not emit actions")
                    payload = encode_transport_record(record) if compact else encode_record(record)
                    if len(payload) > MAX_RECORD:
                        raise ValueError("Sampler response too large")
                    if deferred:
                        cache.finish(request_id, token, payload)
                    else:
                        self.wfile.write(struct.pack("!I", len(payload)))
                        self.wfile.write(payload)
                    return
                with lock:
                    acquired = time.perf_counter()
                    args = (state, front, wrist, request_id)
                    record = engine.sample(*args) if timings is None else engine.sample(*args, timings=timings)
                    prepared = time.perf_counter()
                    payload = encode_transport_record(record) if compact else encode_record(record)
                if timings is not None:
                    timings.update(read_and_parse_s=parsed-entered, queue_s=acquired-parsed,
                                   encode_s=time.perf_counter()-prepared, response_bytes=len(payload))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                if timings is not None:
                    self.send_header("X-PI05-Timings", json.dumps(timings))
            except Exception as error:
                if token is not None:
                    cache.finish(request_id, token)
                if sent:
                    self.log_error("Split response failed: %r", error)
                    return
                payload = json.dumps({"error": repr(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def validate_rtc_request(prefix, window, cfg):
    if (prefix.dtype != np.float32 or prefix.ndim != 2 or prefix.shape[1] != cfg["action_dim"]
            or not np.isfinite(prefix).all() or window.dtype != np.int64 or window.shape != (2,)
            or not 0 <= window[0] < window[1] <= len(prefix) <= cfg["action_chunk"]):
        raise ValueError("Invalid RTC prefix or overlap")
    return prefix, window


class RemoteSampler:
    def __init__(self, url, *, timeout=120, use_compact=True, guided=False):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path not in ("", "/"):
            raise ValueError("Use a loopback SSH-forwarded sampler URL")
        self.url, self.timeout = url.rstrip("/"), timeout
        with urllib.request.urlopen(self.url+"/health", timeout=timeout) as response:
            self.health = json.load(response)
        if (self.health.get("protocol") != PROTOCOL or not self.health.get("ready") or
            self.health.get("rtc_enabled") is not guided or self.health.get("units") != "body_degrees_gripper_percent"
            or (guided and (self.health.get("rtc_mode") != RTC_MODE or not self.health.get("verify_replay")
                            or self.health.get("config", {}).get("rtc") != RTC_MODE))):
            raise ValueError("Wrong flow-SDE sampler deployment")
        self.compact = use_compact and self.health.get("lossless_transport") == LOSSLESS_TRANSPORT

    def _request(self, state, rgb_images, path, *, webp=False, diagnostic_jpeg=False, rtc=None, webp_method=2,
                 sparse=False, vision_audit=False):
        request_id, stream = uuid.uuid4().hex, io.BytesIO()
        fields = dict(protocol=PROTOCOL, request_id=request_id, state=np.asarray(state, np.float32))
        if sparse or vision_audit:
            if not self.compact or not webp or self.health.get("sparse_rgb") != sparse_rgb.CODEC:
                raise ValueError("Sparse RGB/Audit is unavailable")
            if sparse:
                rgb_images = tuple(sparse_rgb.pack(img) for img in rgb_images)
                fields["sparse_rgb"] = sparse_rgb.CODEC
            if vision_audit:
                fields["vision_audit"] = True
        if rtc is not None:
            fields["rtc_prefix"], fields["rtc_window"] = validate_rtc_request(*rtc, self.health["config"])
        if webp:
            if not self.compact or not features.check("webp") or self.health.get("webp_upload") != WEBP_UPLOAD:
                raise ValueError("Lossless WebP unavailable")
            fields["upload_codec"] = WEBP_UPLOAD
        if diagnostic_jpeg:
            if (webp or not self.compact or
                    self.health.get("diagnostic_jpeg_upload") != JPEG_DIAGNOSTIC_UPLOAD):
                raise ValueError("Diagnostic JPEG unavailable")
            fields["upload_codec"] = JPEG_DIAGNOSTIC_UPLOAD
        if self.compact:
            np.savez(stream, **fields, transport=LOSSLESS_TRANSPORT,
                     front=np.frombuffer(encode_rgb(rgb_images[0], webp=webp, diagnostic_jpeg=diagnostic_jpeg,
                                                    webp_method=webp_method), np.uint8),
                     wrist=np.frombuffer(encode_rgb(rgb_images[1], webp=webp, diagnostic_jpeg=diagnostic_jpeg,
                                                    webp_method=webp_method), np.uint8))
        else:
            np.savez_compressed(stream, **fields, front=rgb_images[0], wrist=rgb_images[1])
        request = urllib.request.Request(self.url+path, data=stream.getvalue(),
                                         headers={"Content-Type": "application/octet-stream"})
        return request_id, request

    def sample_split(self, state, rgb_images, on_actions, *, webp=False):
        """Publish actions before receiving and verifying their complete record."""
        if self.health.get("split_transport") != SPLIT_TRANSPORT:
            raise ValueError("Sampler does not support actions-first streaming")
        request_id, request = self._request(state, rgb_images, "/sample-split", webp=webp)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            actions = self._action_packet(response, request_id)
            on_actions(request_id, actions.copy())
            record = self._decode(read_frame(response, MAX_RECORD), request_id)
        return self._check_actions(actions, record)

    def sample_guided(self, state, rgb_images, on_actions, *, rtc=None, webp=False, webp_method=2,
                      sparse=False, vision_audit=False):
        if not self.health.get("rtc_enabled") or self.health.get("split_transport") != SPLIT_TRANSPORT:
            raise ValueError("Expected a guided actions-first Flow-SDE sampler")
        started = time.perf_counter()
        if webp_method not in (0, 2):
            raise ValueError("Expected WebP method 0 or 2")
        request_id, request = self._request(state, rgb_images, "/sample-split", webp=webp, rtc=rtc,
                                            webp_method=webp_method, sparse=sparse, vision_audit=vision_audit)
        encoded = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            server_timings = response.headers.get("X-PI05-Action-Timings")
            actions, normalized = self._action_packet(response, request_id, guided=True)
            arrived = time.perf_counter()
            on_actions(request_id, actions.copy(), normalized.copy())
            payload = read_frame(response, MAX_RECORD)
            received = time.perf_counter()
            record = self._decode(payload, request_id)
        record = self._check_guided_record(actions, normalized, rtc, record)
        self._check_rgb_transport(record, sparse)
        if server_timings is not None:
            record["transport_profile"] = dict(server=json.loads(server_timings), upload_bytes=len(request.data),
                trace_bytes=len(payload), encode_s=encoded-started, http_to_action_s=arrived-encoded,
                action_to_trace_s=received-arrived, verify_s=time.perf_counter()-received)
        return record

    def request_guided(self, state, rgb_images, *, rtc=None, webp=False, webp_method=2, sparse=False):
        if not self.health.get("rtc_enabled") or self.health.get("deferred_transport") != DEFERRED_TRANSPORT:
            raise ValueError("Expected guided deferred Flow-SDE support")
        started = time.perf_counter()
        request_id, request = self._request(state, rgb_images, "/sample-deferred", webp=webp, rtc=rtc,
                                            webp_method=webp_method, sparse=sparse)
        encoded = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            server = response.headers.get("X-PI05-Action-Timings")
            actions, normalized = self._action_packet(response, request_id, guided=True)
        profile = dict(server=json.loads(server) if server else {}, upload_bytes=len(request.data),
                       encode_s=encoded-started, http_to_action_s=time.perf_counter()-encoded)
        return request_id, actions, normalized, profile

    def fetch_guided_trace(self, request_id, actions, normalized, rtc, *, sparse=False):
        record = self._check_guided_record(actions, normalized, rtc, self.fetch_trace(request_id, actions))
        self._check_rgb_transport(record, sparse)
        return record

    @staticmethod
    def _check_rgb_transport(record, sparse):
        if record.get("rgb_transport") != (sparse_rgb.CODEC if sparse else None):
            raise ValueError("Trace RGB transport differs from the request")

    def _check_guided_record(self, actions, normalized, rtc, record):
        batch = record["batch"]
        if rtc is None:
            if PREFIX in batch or WINDOW in batch:
                raise ValueError("Bootstrap trace unexpectedly uses RTC conditioning")
        elif (not np.array_equal(batch[PREFIX].numpy()[0], rtc[0])
              or not np.array_equal(batch[WINDOW].numpy(), rtc[1])):
            raise ValueError("Replay conditioning differs from the RTC request")
        cfg = self.health["config"]
        if not np.array_equal(normalized, record["trace"]["chains"][0,-1,:cfg["action_chunk"],:cfg["action_dim"]].numpy()):
            raise ValueError("Model-space action packet differs from the recorded chain")
        return self._check_actions(actions, record)

    def _action_packet(self, response, request_id, *, guided=False):
        if response.headers.get("Content-Type") != "application/x-pi05-actions-first":
            raise ValueError("Wrong action response type")
        head = json.loads(read_frame(response, 64 * 1024))
        actions = np.asarray(head.pop("actions"), np.float32)
        normalized = np.asarray(head.pop("normalized"), np.float32) if guided else None
        cfg = self.health["config"]
        if (head != dict(request_id=request_id, pipeline_id=self.health["pipeline_id"],
                         policy_version=self.health["policy_version"]) or
                actions.shape != (cfg["action_chunk"], cfg["action_dim"]) or
                not np.isfinite(actions).all()):
            raise ValueError("Invalid action packet")
        if guided:
            if normalized.shape != actions.shape or not np.isfinite(normalized).all():
                raise ValueError("Invalid model-space action packet")
            return actions, normalized
        return actions

    @staticmethod
    def _check_actions(actions, record):
        if not np.array_equal(actions, record["physical_actions"][0].numpy()):
            raise ValueError("Action packet and replay record differ")
        return record

    def request_actions(self, state, rgb_images, *, webp=False, diagnostic_jpeg=False):
        if self.health.get("deferred_transport") != DEFERRED_TRANSPORT:
            raise ValueError("Sampler does not support deferred traces")
        start = time.perf_counter()
        request_id, request = self._request(state, rgb_images, "/sample-deferred", webp=webp,
                                            diagnostic_jpeg=diagnostic_jpeg)
        encoded = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            actions = self._action_packet(response, request_id)
        return request_id, actions, {"upload_bytes": len(request.data), "encode_s": encoded-start,
                                     "http_action_s": time.perf_counter()-encoded}

    def fetch_trace(self, request_id, actions):
        if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
            raise ValueError("Invalid trace ID")
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            remaining = deadline-time.monotonic()
            with urllib.request.urlopen(self.url+"/trace/"+request_id, timeout=max(.001, remaining)) as response:
                if response.status == 200:
                    payload = response.read(MAX_RECORD+1)
                    if len(payload) > MAX_RECORD:
                        raise ValueError("Sampler response too large")
                    return self._check_actions(actions, self._decode(payload, request_id))
                if response.status != 202:
                    raise ValueError("Unexpected deferred trace status")
            time.sleep(.05)
        raise TimeoutError("Deferred trace was not ready")

    def _decode(self, payload, request_id):
        record = decode_transport_record(payload) if self.compact else decode_record(payload)
        if (record["request_id"] != request_id or record["pipeline_id"] != self.health["pipeline_id"] or
            record["trace"]["policy_version"] != self.health["policy_version"]):
            raise ValueError("Mismatched sampler response")
        return record

    def sample(self, state, rgb_images, *, webp=False):
        self.last_timings = None
        started = time.perf_counter()
        request_id, request = self._request(state, rgb_images, "/sample", webp=webp)
        encoded = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            headers_received = time.perf_counter()
            timings = response.headers.get("X-PI05-Timings")
            payload = response.read(MAX_RECORD+1)
        received = time.perf_counter()
        if len(payload) > MAX_RECORD: raise ValueError("Sampler response too large")
        record = self._decode(payload, request_id)
        if timings is not None:
            self.last_timings = {"server": json.loads(timings), "upload_bytes": len(request.data),
                                 "request_encode_s": encoded-started, "http_to_headers_s": headers_received-encoded,
                                 "response_read_s": received-headers_received, "decode_s": time.perf_counter()-received}
        return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--guided-rtc", action="store_true", help="Serve replayable guided Flow-SDE on a separate port")
    p.add_argument("--num-steps", type=int, choices=range(2, 21), metavar="2..20", default=10)
    p.add_argument("--sparse-rgb", action="store_true", help="Enable fixed-shape lossless bilinear taps for probes")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--verify-replay", action="store_true",help="Recompute old probabilities for each diagnostic request")
    p.add_argument("--profile", action="store_true", help="Return per-stage timings and synchronize CUDA at stage boundaries")
    p.add_argument("--diagnostic-jpeg", action="store_true", help="Allow lossy JPEG in deferred test probes only")
    p.add_argument("--offload-text-tables", action="store_true", help="Local CUDA probe with text tables stored on CPU")
    p.add_argument("--reuse-replay-prefix", action="store_true", help="Reuse the current request's frozen prefix for online replay")
    args = p.parse_args(); torch.set_num_threads(args.threads)
    if args.guided_rtc and (not args.verify_replay or args.port == 8095):
        p.error("--guided-rtc requires --verify-replay and a separate port")
    engine = SamplerEngine(args.base, args.adapter, args.tokenizer, args.device,
                           FlowRLConfig(num_steps=args.num_steps, rtc=RTC_MODE if args.guided_rtc else None),
                           verify_replay=args.verify_replay, profile=args.profile, diagnostic_jpeg=args.diagnostic_jpeg,
                           offload_text_tables=args.offload_text_tables, reuse_replay_prefix=args.reuse_replay_prefix,
                           sparse_upload=args.sparse_rgb)
    server = make_server(engine, args.port)
    print("FLOW_SDE_SAMPLER_READY", json.dumps(engine.health), flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__": main()
