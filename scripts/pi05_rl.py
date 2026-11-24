"""RLinf-style single stochastic denoising-step RL for LeRobot PI05.

No robot I/O. Log probabilities are transition surrogates on the recorded
denoising chain, NOT exact marginal densities of the final physical action.
The flow-SDE equations are used verbatim from the pinned Apache-2.0 RLinf
reference; see vendor/rlinf-reference/manifest.json and LICENSE.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import importlib.util
import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rlinf_flow_reference",
    ROOT / "vendor/rlinf-reference/rlinf/models/embodiment/openpi/sampling/rl_sampler.py",
)
flow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flow)


@dataclass(frozen=True)
class FlowRLConfig:
    num_steps: int = 10
    noise_level: float = 0.5
    action_dim: int = 6
    action_chunk: int = 50
    clip_low: float = 0.2
    clip_high: float = 0.28
    rtc: str | None = None

    def __post_init__(self):
        if self.num_steps < 2 or not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("flow_sde requires num_steps >= 2 and positive finite noise_level")
        if self.action_dim < 1 or self.action_chunk < 1:
            raise ValueError("Invalid action shape")
        if not 0 < self.clip_low < 1 or not 0 < self.clip_high < 1:
            raise ValueError("Invalid clipping range")
        if self.rtc not in (None, "lerobot-exp-v1"):
            raise ValueError("Unknown Flow-SDE RTC guidance version")

    def serialized(self):
        result = asdict(self)
        if self.rtc is None:
            del result["rtc"]  # Keep prior unguided pipeline IDs and traces unchanged.
        return result


@dataclass
class FlowTrace:
    chains: Tensor                 # [B, N+1, full_horizon, padded_dim]
    denoise_index: Tensor          # [B], independent random choice for each row
    old_logprobs: Tensor           # [B, executed_horizon, physical_dim]
    config: dict
    policy_version: str
    observation_digest: str | None = None

    def validate(self, cfg: FlowRLConfig, policy_version: str):
        if self.config != cfg.serialized() or self.policy_version != policy_version:
            raise ValueError("Trace sampler configuration or behavior policy version mismatch")
        b, n, h, d = self.chains.shape
        if n != cfg.num_steps + 1 or h < cfg.action_chunk or d < cfg.action_dim:
            raise ValueError("Invalid denoising chain shape")
        if self.denoise_index.shape != (b,) or self.denoise_index.dtype != torch.long:
            raise ValueError("Invalid denoising indices")
        if ((self.denoise_index < 0) | (self.denoise_index >= cfg.num_steps)).any():
            raise ValueError("Denoising index out of bounds")
        if self.old_logprobs.shape != (b, cfg.action_chunk, cfg.action_dim):
            raise ValueError("Invalid old logprob shape")
        if not torch.isfinite(self.chains).all() or not torch.isfinite(self.old_logprobs).all():
            raise ValueError("Non-finite rollout")
        if self.chains.requires_grad or self.old_logprobs.requires_grad:
            raise ValueError("Behavior data must be detached")


@torch.no_grad()
def sample_flow(velocity, shape, cfg, *, device, policy_version, generator=None, noise=None,
                denoise_index=None):
    """RLinf non-joint mode: one SDE transition, ODE at all other steps."""
    if len(shape) != 3 or shape[1] < cfg.action_chunk or shape[2] < cfg.action_dim:
        raise ValueError("Sampler shape does not cover physical action dimensions")
    x = torch.randn(shape, device=device, generator=generator) if noise is None else noise.detach().float().clone()
    if tuple(x.shape) != tuple(shape) or not torch.isfinite(x).all():
        raise ValueError("Invalid initial noise")
    b = shape[0]
    chosen = (torch.randint(cfg.num_steps, (b,), device=device, generator=generator)
              if denoise_index is None else denoise_index.to(device))
    if chosen.shape != (b,) or chosen.dtype != torch.long or ((chosen < 0) | (chosen >= cfg.num_steps)).any():
        raise ValueError("Invalid stochastic step selection")
    times = flow.get_timesteps(cfg.num_steps, device)
    chain, old = [x.clone()], torch.empty((b, cfg.action_chunk, cfg.action_dim), device=device)
    for i in range(cfg.num_steps):
        index = torch.full((b,), i, device=device, dtype=torch.long)
        v = velocity(x, times[i].expand(b)).float()
        mu, sigma = flow.sample_mean_var(x, v, index, noise_method="flow_sde",
                                       noise_level=cfg.noise_level, num_steps=cfg.num_steps)
        selected = chosen == i
        ode = x + (times[i+1] - times[i]) * v
        sde = mu + sigma * torch.randn(shape, device=device, generator=generator)
        x = torch.where(selected[:, None, None], sde, ode)
        lp = flow.gaussian_logprob(x, mu, sigma)[:, :cfg.action_chunk, :cfg.action_dim]
        old[selected] = lp[selected]
        chain.append(x.clone())
    trace = FlowTrace(torch.stack(chain, dim=1).detach(), chosen.detach().clone(), old.detach(),
                      cfg.serialized(), policy_version)
    trace.validate(cfg, policy_version)
    return x[:, :cfg.action_chunk, :cfg.action_dim], trace


def recompute_flow(velocity, trace, cfg, *, policy_version):
    """Re-evaluate the SAME recorded transition; never resample during update."""
    trace.validate(cfg, policy_version)
    row = torch.arange(trace.chains.shape[0], device=trace.chains.device)
    index = trace.denoise_index
    x = trace.chains[row, index].detach()
    target = trace.chains[row, index + 1].detach()
    time = flow.get_timesteps(cfg.num_steps, x.device)[index]
    v = velocity(x, time).float()
    mu, sigma = flow.sample_mean_var(x, v, index, noise_method="flow_sde",
                                   noise_level=cfg.noise_level, num_steps=cfg.num_steps)
    lp = flow.gaussian_logprob(target, mu, sigma)[:, :cfg.action_chunk, :cfg.action_dim]
    if not torch.isfinite(lp).all():
        raise ValueError("Non-finite recomputed logprob")
    return lp


def group_advantages(rewards: Tensor, group_ids: Tensor, group_size: int):
    """Terminal binary rewards; homogeneous groups are explicitly excluded.

    IDs represent repeated episodes of the same initial scene, never action
    chunks from one episode. Caller broadcasts the episode advantage to chunks.
    """
    if rewards.ndim != 1 or rewards.shape != group_ids.shape or group_size < 2:
        raise ValueError("Expected one reward and group ID per episode")
    if group_ids.dtype != torch.long or not torch.isfinite(rewards).all():
        raise ValueError("Invalid reward/group IDs")
    if not ((rewards == 0) | (rewards == 1)).all():
        raise ValueError("This adapter expects binary terminal rewards")
    rewards = rewards.detach().float()
    adv, keep = torch.zeros_like(rewards), torch.zeros_like(rewards, dtype=torch.bool)
    for gid in group_ids.unique():
        mask = group_ids == gid
        if int(mask.sum()) != group_size:
            raise ValueError("Incomplete or oversized episode group")
        values = rewards[mask]
        if values.min() == values.max():
            continue
        adv[mask] = (values - values.mean()) / (values.std() + 1e-6)
        keep[mask] = True
    return adv, keep


def grpo_loss(new_logprobs, old_logprobs, advantages, keep, cfg):
    """RLinf chunk-level ratio: sum physical action/chunk logprobs.

    Each row is one chunk. For variable-length episodes the caller must select
    its intended chunk/episode weighting; this function averages valid rows.
    """
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.ndim != 3:
        raise ValueError("Expected matching [chunks, horizon, action_dim] logprobs")
    b, h, d = new_logprobs.shape
    if (h, d) != (cfg.action_chunk, cfg.action_dim) or advantages.shape != (b,) or keep.shape != (b,):
        raise ValueError("Loss shape mismatch")
    if keep.dtype != torch.bool or not keep.any():
        raise ValueError("No mixed-outcome groups; collect more episodes, do not update")
    if not all(torch.isfinite(t).all() for t in (new_logprobs, old_logprobs, advantages)):
        raise ValueError("Non-finite GRPO input")
    delta = (new_logprobs - old_logprobs.detach()).sum(dim=(1, 2))[keep]
    ratio = delta.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Non-finite ratio; reject batch instead of silently clipping logs")
    adv = advantages.detach()[keep]
    loss = -torch.minimum(ratio * adv, ratio.clamp(1-cfg.clip_low, 1+cfg.clip_high) * adv).mean()
    return loss, {"ratio_mean": float(ratio.detach().mean()),
                  "clip_fraction": float(((ratio < 1-cfg.clip_low) | (ratio > 1+cfg.clip_high)).float().mean()),
                  "valid_chunks": int(keep.sum())}


class LeRobotFlowRL(nn.Module):
    """Frozen visual/language prefix + trainable existing action-expert LoRA."""
    def __init__(self, policy, cfg: FlowRLConfig, *, policy_version: str):
        super().__init__()
        self.policy, self.cfg, self.policy_version = policy, cfg, policy_version
        if policy._rtc_enabled():
            raise ValueError("Disable policy-native RTC; Flow-SDE applies and records its own RTC guidance")
        if getattr(policy.config, "use_visual_memory", False) or getattr(policy.config, "use_proprioceptive_memory", False):
            raise ValueError("Memory models require a separate observation replay adapter")
        if cfg.action_chunk > policy.config.chunk_size or cfg.action_dim > policy.config.max_action_dim:
            raise ValueError("RL shape exceeds PI05 output")
        policy.eval()  # disables dropout; autograd remains enabled for LoRA
        trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n or not ("gemma_expert" in n or "action_in_proj" in n or "action_out_proj" in n) for n in trainable):
            raise ValueError("Only action-expert/projection LoRA parameters may be trainable")

    @property
    def core(self):
        # PeftModel.model is the wrapped PI05Policy, not PI05Pytorch.
        base = self.policy.get_base_model() if hasattr(self.policy, "get_base_model") else self.policy
        return base.model

    @torch.no_grad()
    def prefix(self, batch):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        self.policy.eval()
        core = self.core
        images, masks = self.policy._preprocess_images(batch)
        # LeRobot uses the first parameter to place its image preprocessing.
        # With the text embedding on CPU, move the processed images to the vision device.
        token_device = batch["observation.language.tokens"].device
        images = [image.to(token_device) for image in images]
        masks = [mask.to(token_device) for mask in masks]
        emb, pad, att = core.embed_prefix(images, masks, batch["observation.language.tokens"],
                                           batch["observation.language.attention_mask"])
        attention = make_att_2d_masks(pad, att)
        if getattr(core, "use_typed_attention_masks", False):
            attention = core._prepare_attention_masks_4d(attention, dtype=emb.dtype)
        else:
            attention = core._prepare_attention_masks_4d(attention)
        pg = core.paligemma_with_expert.paligemma
        language_model = pg.model.language_model if hasattr(pg, "model") else pg.language_model
        language_model.config._attn_implementation = "eager"
        _, cache = core.paligemma_with_expert.forward(
            attention_mask=attention, position_ids=pad.long().cumsum(1)-1,
            past_key_values=None, inputs_embeds=[emb, None], use_cache=True)
        return pad.detach(), cache

    def velocity(self, context, guidance=None):
        pad, cache = context
        def evaluate(x, t):
            # Some old Transformers caches mutate even with use_cache=False.
            v = self.core.denoise_step(pad, copy.deepcopy(cache), x, t)
            return v if guidance is None else guidance.apply(x, t, v)
        return evaluate

    def sample(self, batch, *, generator=None, noise=None, denoise_index=None):
        return self.sample_from_prefix(batch, self.prefix(batch), generator=generator, noise=noise,
                                       denoise_index=denoise_index)

    def sample_from_prefix(self, batch, context, *, generator=None, noise=None, denoise_index=None):
        """Sample with the already-computed prefix for this same observation."""
        shape = (batch["observation.language.tokens"].shape[0], self.policy.config.chunk_size,
                 self.policy.config.max_action_dim)
        from pi05_rl_rtc import guidance_from_batch
        guidance = guidance_from_batch(batch, self.cfg)
        if guidance is not None: guidance = guidance.for_model(shape)
        actions, trace = sample_flow(self.velocity(context, guidance), shape, self.cfg,
                                    device=context[0].device, policy_version=self.policy_version,
                                    generator=generator, noise=noise, denoise_index=denoise_index)
        trace.observation_digest = observation_digest(batch)
        return actions, trace

    def recompute(self, batch, trace):
        return self.recompute_from_prefix(batch, trace)

    def recompute_from_prefix(self, batch, trace, context=None):
        if trace.observation_digest != observation_digest(batch):
            raise ValueError("Replay observation differs from the rollout observation")
        from pi05_rl_rtc import guidance_from_batch
        guidance = guidance_from_batch(batch, self.cfg)
        if guidance is not None:
            guidance = guidance.for_model((trace.chains.shape[0], *trace.chains.shape[2:]))
        if context is None:
            context = self.prefix(batch)
        return recompute_flow(self.velocity(context, guidance), trace, self.cfg,
                              policy_version=self.policy_version)


def observation_digest(batch):
    """Bind the probability trace to exact processed pixels/state/tokens."""
    digest = hashlib.sha256()
    for key in sorted(k for k in batch if k.startswith("observation.")):
        value = batch[key]
        if not isinstance(value, Tensor):
            raise ValueError(f"Processed observation must contain tensors: {key}")
        cpu = value.detach().cpu().contiguous()
        digest.update(f"{key}:{cpu.dtype}:{tuple(cpu.shape)}".encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
