"""Strict local IRA + existing LoRA loader for the no-hardware RL adapter."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ira_lora(base: Path, adapter: Path, *, device="cpu", allow_test_checkpoint=False,
                 offload_text_tables=False):
    """Load every base/adapter key explicitly; never accept random fallback.

    Meta construction + per-tensor casts avoid holding two full fp32 models.
    Known Transformers 4->5 wrapper renames are matched against actual keys.
    Checkpoints on disk are never rewritten or merged.
    """
    local_dependencies = Path(__file__).resolve().parents[1] / "vendor/pi05-rl-deps"
    if local_dependencies.is_dir():
        sys.path.insert(0, str(local_dependencies))
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open

    if offload_text_tables and device != "cuda":
        raise ValueError("Text table offload requires the local CUDA path")

    def placement(name):
        if offload_text_tables and (".lm_head." in name or ".language_model.embed_tokens." in name):
            return "cpu"
        return device

    base, adapter = Path(base), Path(adapter)
    training_metadata = adapter / "rl_training.json"
    if training_metadata.exists() and json.loads(training_metadata.read_text(encoding="utf-8")).get("test_only") and not allow_test_checkpoint:
        raise ValueError("Test-only checkpoint cannot be used for normal collection/deployment")
    weights, lora = base / "model.safetensors", adapter / "adapter_model.safetensors"
    for file in (weights, lora, adapter / "config.json", adapter / "adapter_config.json"):
        if not file.is_file():
            raise FileNotFoundError(file)
    base_hash, adapter_hash = sha256(weights), sha256(lora)
    cfg = PreTrainedConfig.from_pretrained(str(adapter), local_files_only=True)
    cfg.device, cfg.rtc_config, cfg.gradient_checkpointing = "meta", None, False
    cfg.pretrained_path, cfg.use_peft = str(base), False
    with torch.device("meta"):
        policy = PI05Policy(cfg)
    expected = policy.state_dict()
    state, renamed = {}, {}
    with safe_open(weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            target = key
            if target not in expected:
                target = target.replace("paligemma.language_model.", "paligemma.model.language_model.")
                target = target.replace("paligemma.vision_tower.", "paligemma.model.vision_tower.")
                target = target.replace("paligemma.multi_modal_projector.", "paligemma.model.multi_modal_projector.")
                target = target.replace("vision_tower.vision_model.", "vision_tower.")
            if target not in expected or target in state:
                raise ValueError(f"Unmapped or duplicate base weight: {key}")
            value = source.get_tensor(key)
            if value.shape != expected[target].shape:
                raise ValueError(f"Base weight shape mismatch: {key}")
            state[target] = value.to(dtype=expected[target].dtype, device=placement(target))
            if target != key:
                renamed[key] = target
    policy.load_state_dict(state, strict=True, assign=True)
    del state, expected
    # Nonpersistent buffers are absent from safetensors. Rebuild their exact
    # constructor values after meta initialization, failing on unknown buffers.
    for name, module in list(policy.named_modules()):
        if name.endswith("rotary_emb") and any(b.is_meta for b in module.buffers()):
            parent_name, child_name = name.rsplit(".", 1)
            with torch.device("cpu"):
                replacement = type(module)(module.config)
            policy.get_submodule(parent_name).add_module(child_name, replacement.to(device))
    for name, buffer in list(policy.named_buffers()):
        if not buffer.is_meta:
            continue
        parent_name, attr = name.rsplit(".", 1)
        parent = policy.get_submodule(parent_name)
        if attr == "position_ids":
            value = torch.arange(buffer.numel(), device=placement(name)).reshape(buffer.shape)
        elif attr == "embed_scale":
            value = torch.tensor(math.sqrt(parent.embedding_dim), device=placement(name), dtype=buffer.dtype)
        else:
            raise ValueError(f"Unrecognized meta buffer: {name}")
        setattr(parent, attr, value)
    policy.config.device = device
    policy = PeftModel.from_pretrained(policy, str(adapter), is_trainable=True,
                                       local_files_only=True).eval()
    loaded = get_peft_model_state_dict(policy, save_embedding_layers=False)
    with safe_open(lora, framework="pt", device="cpu") as source:
        if set(source.keys()) != set(loaded):
            raise ValueError("Adapter keys differ after loading")
        for key in source.keys():
            value = source.get_tensor(key)
            if not torch.equal(value, loaded[key].detach().cpu().to(value.dtype)):
                raise ValueError(f"Adapter value mismatch: {key}")
    if any(t.is_meta for t in list(policy.parameters()) + list(policy.buffers())):
        raise ValueError("Model contains unmaterialized tensors")
    if offload_text_tables:
        joint = policy.get_base_model().model.paligemma_with_expert
        language = joint.paligemma.model.language_model
        embedding = language.get_input_embeddings()
        heads = (joint.paligemma.lm_head, joint.gemma_expert.lm_head)
        if embedding.weight.device.type != "cpu" or any(head.weight.device.type != "cpu" for head in heads):
            raise ValueError("Expected CPU text tables after strict loading")
        def reject_logits(module, args):
            raise RuntimeError("Text logits are unavailable on the local flow sampler")

        for head in heads:
            head.register_forward_pre_hook(reject_logits)
        embedding.register_forward_pre_hook(lambda module, args: (args[0].to("cpu"),))
        embedding.register_forward_hook(lambda module, args, output: output.to(device))

    provenance = {"base": str(base.resolve()), "adapter": str(adapter.resolve()),
                  "base_sha256": base_hash, "adapter_sha256": adapter_hash,
                  "policy_version": hashlib.sha256(f"{base_hash}:{adapter_hash}".encode()).hexdigest(),
                  "base_size_bytes": weights.stat().st_size,
                  "strict_base_load": True, "exact_adapter_values": True,
                  "renamed_base_keys": len(renamed), "device": device,
                  "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad)}
    if offload_text_tables:
        provenance["placement"] = "cpu_text_tables_cuda_flow"
    return policy, provenance


def make_processors(policy, adapter, tokenizer, device):
    from lerobot.policies.factory import make_pre_post_processors
    return make_pre_post_processors(
        policy.config, pretrained_path=str(adapter),
        preprocessor_overrides={"device_processor": {"device": device},
                                "tokenizer_processor": {"tokenizer_name": str(tokenizer)}})


def load_saved_observation(directory):
    """Use existing RGB snapshots and original six physical state values."""
    import cv2
    directory = Path(directory)
    first = next(item for line in (directory / "telemetry.jsonl").read_text().splitlines()
                 if "state" in (item := json.loads(line)))
    batch = {"observation.state": torch.tensor(first["state"], dtype=torch.float32),
             "task": "Pick up the block and place it inside the box."}
    for cam in ("front", "wrist"):
        bgr = cv2.imread(str(directory / f"{cam}-start.jpg"))
        if bgr is None:
            raise FileNotFoundError(directory / f"{cam}-start.jpg")
        batch[f"observation.images.{cam}"] = torch.from_numpy(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 255
    return batch
