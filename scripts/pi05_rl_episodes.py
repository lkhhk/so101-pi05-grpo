"""Versioned, auditable flow-RL episode storage. No robot or model imports."""
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import uuid

import torch

from pi05_rl import FlowRLConfig, FlowTrace, observation_digest, group_advantages

SCHEMA = "pi05-flow-episode-v1"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    tmp.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def pack_record(batch, trace, physical_actions, pipeline_id, request_id):
    record = cpu_tree({"schema": SCHEMA, "batch": batch, "trace": asdict(trace),
                       "physical_actions": physical_actions, "pipeline_id": pipeline_id,
                       "request_id": request_id})
    validate_record(record)
    return record


def validate_record(record):
    if record.get("schema") != SCHEMA: raise ValueError("Wrong rollout schema")
    trace = FlowTrace(**record["trace"])
    cfg = FlowRLConfig(**trace.config)
    trace.validate(cfg, trace.policy_version)
    from pi05_rl_rtc import guidance_from_batch
    guidance_from_batch(record["batch"], cfg)
    if trace.chains.shape[0] != 1: raise ValueError("Store one environment chunk per record")
    if trace.observation_digest != observation_digest(record["batch"]):
        raise ValueError("Corrupt or mismatched replay observation")
    actions = record["physical_actions"]
    if actions.shape != (1, cfg.action_chunk, cfg.action_dim) or not torch.isfinite(actions).all():
        raise ValueError("Invalid physical actions")
    if not record.get("pipeline_id") or not record.get("request_id"):
        raise ValueError("Missing pipeline/request identity")
    return trace, cfg


def encode_record(record):
    validate_record(record)
    return _compress(record)


def _compress(record):
    stream = io.BytesIO()
    torch.save(record, stream)
    return gzip.compress(stream.getvalue(), compresslevel=1)


def _decompress(payload):
    # Bound decompression before torch.load. Accept only tensor/primitive trees.
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024: raise ValueError("Rollout record too large")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def decode_record(payload):
    record = _decompress(payload)
    validate_record(record)
    return record


def open_rgb_png(payload):
    from PIL import Image
    with Image.open(io.BytesIO(payload), formats=["PNG"]) as image:
        if image.mode != "RGB" or not all(0 < side <= 1200 for side in image.size):
            raise ValueError("Invalid RGB PNG")
        return image.copy()


def encode_transport_record(record):
    """Use PNG only for images that reconstruct bit for bit; keep episode files unchanged."""
    from PIL import Image
    validate_record(record)
    batch = dict(record["batch"])
    images = {}
    for key, tensor in record["batch"].items():
        if (not key.startswith("observation.images.") or tensor.dtype != torch.float32 or
                tensor.ndim != 4 or tensor.shape[:2] != (1, 3) or max(tensor.shape[2:]) > 1200):
            continue
        pixels = (tensor * 255).round().clamp(0, 255).to(torch.uint8)
        if not torch.equal(pixels.float().div(255).contiguous().view(torch.int32),
                           tensor.contiguous().view(torch.int32)):
            continue
        stream = io.BytesIO()
        Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).save(stream, format="PNG", compress_level=1)
        images[key] = stream.getvalue()
        del batch[key]
    return _compress({**record, "batch": batch, "transport_images": images})


def decode_transport_record(payload):
    import numpy as np
    record = _decompress(payload)
    images = record.pop("transport_images")
    for key, png in images.items():
        if not key.startswith("observation.images.") or key in record["batch"]:
            raise ValueError("Invalid transport image key")
        pixels = torch.from_numpy(np.array(open_rgb_png(png)))
        record["batch"][key] = pixels.permute(2, 0, 1).unsqueeze(0).float().div(255)
    validate_record(record)
    return record


class EpisodeWriter:
    def __init__(self, directory, *, group_id, scene_id, health, source="real", reset_state=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema": SCHEMA, "episode_id": uuid.uuid4().hex,
                         "group_id": group_id, "scene_id": scene_id, "source": source,
                         "health": health, "reset_state": reset_state,
                         "phase": "collecting", "chunks": []}
        self.flush()

    def flush(self): write_json(self.directory / "episode.json", self.manifest)

    def append(self, record, *, executed_steps, clipped_steps=0, execution_start=0):
        if self.manifest["phase"] != "collecting": raise ValueError("Episode closed")
        if record.get("diagnostic_upload"):
            raise ValueError("Diagnostic upload cannot be used as episode data")
        trace, cfg = validate_record(record)
        health = self.manifest["health"]
        if record["pipeline_id"] != health["pipeline_id"] or trace.policy_version != health["policy_version"]:
            raise ValueError("Behavior policy/pipeline changed inside episode")
        if trace.config != health["config"]: raise ValueError("Sampler config changed inside episode")
        if not valid_execution_window(execution_start, executed_steps, cfg.action_chunk) or not 0 <= clipped_steps <= executed_steps:
            raise ValueError("Invalid executed-step count")
        if any(row["request_id"] == record["request_id"] for row in self.manifest["chunks"]):
            raise ValueError("Duplicate chunk request")
        name = f"chunk-{len(self.manifest['chunks']):05d}.pt.gz"
        payload = encode_record(record)
        with open(self.directory / name, "xb") as stream: stream.write(payload)
        self.manifest["chunks"].append({"file": name, "sha256": hashlib.sha256(payload).hexdigest(),
                                        "executed_steps": executed_steps, "clipped_steps": clipped_steps,
                                        "request_id": record["request_id"]})
        if execution_start:
            self.manifest["chunks"][-1]["execution_start"] = execution_start
        self.flush()

    def finish(self, **metadata):
        self.manifest.update(metadata)
        self.manifest["phase"] = "awaiting_label"
        self.flush()


def valid_execution_window(start, count, length):
    return (type(start) is int and type(count) is int
            and 0 <= start < length and 0 < count <= length-start)


def label_episode(directory, outcome, *, reason=None):
    directory = Path(directory)
    m = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    if m["phase"] != "awaiting_label": raise ValueError("Episode is running or already labelled")
    if outcome not in ("success", "failure", "infrastructure"):
        raise ValueError("Invalid outcome")
    if outcome == "infrastructure" and not reason: raise ValueError("Infrastructure exclusion needs a reason")
    if m.get("trace_integrity") == "incomplete" and outcome != "infrastructure":
        raise ValueError("Incomplete async collection can only be labelled infrastructure")
    if outcome == "success" and not m["chunks"]: raise ValueError("Cannot mark empty episode successful")
    if m.get("collection_error") and (outcome == "success" or not reason):
        raise ValueError("Interrupted collection needs a failure/fault diagnosis and reason")
    m.update(phase="complete", outcome=outcome, reason=reason,
             reward=None if outcome == "infrastructure" else int(outcome == "success"))
    write_json(directory / "episode.json", m)


def read_episode(directory):
    directory = Path(directory)
    m = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("phase") != "complete":
        raise ValueError("Episode is incomplete/unlabelled")
    if m.get("execution_mode") in ("async-flow-sde", "guided-flow-sde", "mujoco-flow-sde") and m.get("outcome") != "infrastructure":
        count_key = "simulated_commands" if m["source"] in ("test", "sim") else "motor_commands_sent"
        if m.get("trace_integrity") != "complete" or sum(row["executed_steps"] for row in m["chunks"]) != m.get(count_key):
            raise ValueError("Async trace coverage does not match the executed actions")
    records = []
    seen = set()
    for row in m["chunks"]:
        if Path(row["file"]).name != row["file"]: raise ValueError("Invalid record path")
        payload = (directory / row["file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]: raise ValueError("Corrupt chunk file")
        record = decode_record(payload)
        trace, cfg = validate_record(record)
        if record["request_id"] in seen or record["request_id"] != row["request_id"]:
            raise ValueError("Duplicate/mismatched request")
        seen.add(record["request_id"])
        if (record["pipeline_id"] != m["health"]["pipeline_id"] or
            trace.policy_version != m["health"]["policy_version"] or trace.config != m["health"]["config"]):
            raise ValueError("Mixed policy/pipeline/config in episode")
        if not valid_execution_window(row.get("execution_start", 0), row["executed_steps"], cfg.action_chunk):
            raise ValueError("Invalid execution mask")
        records.append((row, record))
    return m, records


def training_set(root, group_size, *, allow_test=False, allow_sim=False):
    root = Path(root)
    paths = list(root.glob("*/episode.json"))
    if allow_sim:
        paths.extend(root.glob("group-*/*/episode.json"))
    episodes, ids = [], set()
    for path in sorted(paths):
        m, records = read_episode(path.parent)
        if m["episode_id"] in ids: raise ValueError("Duplicate episode ID")
        ids.add(m["episode_id"])
        if m["source"] != "real" and not (allow_test or (allow_sim and m["source"] == "sim")):
            raise ValueError("Test data or simulation data requires explicit opt-in")
        if any(record.get("diagnostic_upload") for _, record in records):
            raise ValueError("Diagnostic upload cannot be used as training data")
        if m["outcome"] == "infrastructure": continue
        if m["outcome"] not in ("success", "failure") or m["reward"] != int(m["outcome"] == "success"):
            raise ValueError("Invalid terminal reward")
        if not records: raise ValueError("No executed chunks in scored episode")
        episodes.append((m, records))
    if not episodes: raise ValueError("No scored episodes")
    signatures = {(m["health"]["pipeline_id"], m["health"]["policy_version"]) for m, _ in episodes}
    if len(signatures) != 1: raise ValueError("Mixed behavior policies/pipelines in training batch")
    groups, group_scenes, group_resets, group_modes = {}, {}, {}, {}
    for m, _ in episodes:
        gid = m["group_id"]
        groups.setdefault(gid, len(groups))
        group_scenes.setdefault(gid, m["scene_id"])
        if group_scenes[gid] != m["scene_id"]: raise ValueError("Group contains different initial scenes")
        group_modes.setdefault(gid, m.get("execution_mode"))
        if group_modes[gid] != m.get("execution_mode"): raise ValueError("Group contains different execution modes")
        if m["source"] == "sim":
            key = json.dumps([m.get("scene_seed"), m.get("reset_state")])
            group_resets.setdefault(gid, key)
            if group_resets[gid] != key or m.get("reset_state") is None:
                raise ValueError("Group contains different simulator resets")
        if m["source"] == "real":
            reference = m.get("reset_reference")
            if not reference or m.get("reset_state") is None: raise ValueError("Real episode lacks reset-pose evidence")
            key = json.dumps(reference,sort_keys=True)
            group_resets.setdefault(gid,key)
            if group_resets[gid] != key: raise ValueError("Group contains different reset references")
            current, target, tol = (torch.tensor(x) for x in (m["reset_state"],reference["state"],reference["tolerance"]))
            if any(t.shape != (6,) for t in (current,target,tol)) or not all(torch.isfinite(t).all() for t in (current,target,tol)):
                raise ValueError("Invalid reset evidence")
            if (tol <= 0).any() or ((current-target).abs() > tol).any(): raise ValueError("Reset-pose mismatch")
    adv, keep = group_advantages(torch.tensor([m["reward"] for m, _ in episodes]),
                                 torch.tensor([groups[m["group_id"]] for m, _ in episodes]), group_size)
    if not keep.any(): raise ValueError("No mixed-outcome group; collect additional groups")
    return [(m, records, float(a)) for (m, records), a, k in zip(episodes, adv, keep) if bool(k)]
