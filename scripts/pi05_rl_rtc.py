"""Versioned LeRobot EXP guidance for a replayable Flow-SDE transition."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PREFIX = "observation.rtc.prefix"
WINDOW = "observation.rtc.window"
MODE = "lerobot-exp-v1"


@dataclass(frozen=True)
class ModelGuidance:
    prefix: torch.Tensor
    weights: torch.Tensor

    def apply(self, x, t, velocity):
        """Match the installed LeRobot's identity-Jacobian guidance in value.

        In this LeRobot version the denoiser is called before setting x.requires_grad:
        its correction equals the weighted residual. Expand that residual explicitly
        so the trainer differentiates the guided mean with respect to the LoRA weights.
        """
        t = t[:, None, None]
        x1 = x - t * velocity
        correction = (self.prefix - x1) * self.weights
        cap = torch.as_tensor(10., device=t.device, dtype=t.dtype)
        tau = 1 - t
        one_minus_tau = (1 - tau).square()
        inverse_r2 = (one_minus_tau + tau.square()) / one_minus_tau
        c = torch.nan_to_num((1 - tau) / tau, posinf=cap)
        gain = torch.minimum(torch.nan_to_num(c * inverse_r2, posinf=cap), cap)
        return velocity - gain * correction


@dataclass(frozen=True)
class Guidance:
    prefix: torch.Tensor
    delay: int
    horizon: int

    def for_model(self, shape):
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor

        batch, horizon, dim = shape
        values = self.prefix
        if values.shape[0] != batch or values.shape[1] > horizon or values.shape[2] > dim:
            raise ValueError("RTC prefix exceeds the model action shape")
        padded = F.pad(values, (0, dim-values.shape[2], 0, horizon-values.shape[1]))
        rtc = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP))
        weights = rtc.get_prefix_weights(self.delay, self.horizon, horizon).to(values.device)
        return ModelGuidance(padded, weights[None, :, None])

def guidance_from_batch(batch, cfg):
    """The optional bootstrap has no prefix; a continuation stores both tensors."""
    if PREFIX not in batch and WINDOW not in batch:
        return None
    if cfg.rtc != MODE or PREFIX not in batch or WINDOW not in batch:
        raise ValueError("Incomplete or unexpected RTC guidance data")
    prefix, window = batch[PREFIX], batch[WINDOW]
    if (not isinstance(prefix, torch.Tensor) or prefix.dtype != torch.float32 or prefix.ndim != 3
            or prefix.shape[0] != batch["observation.language.tokens"].shape[0]
            or prefix.shape[2] != cfg.action_dim or prefix.requires_grad or not torch.isfinite(prefix).all()):
        raise ValueError("Invalid normalized RTC prefix")
    if not isinstance(window, torch.Tensor) or window.dtype != torch.int64 or window.shape != (2,):
        raise ValueError("Invalid RTC delay/horizon tensor")
    delay, horizon = window.tolist()
    if not 0 <= delay < horizon <= prefix.shape[1] <= cfg.action_chunk:
        raise ValueError("Invalid RTC overlap")
    return Guidance(prefix, delay, horizon)
