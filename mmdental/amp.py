"""Automatic mixed precision helpers compatible with old and new PyTorch."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Union

import torch


DeviceLike = Union[str, torch.device]


def autocast_context(enabled: bool, device: DeviceLike) -> Any:
    """Return a CUDA autocast context without PyTorch 2.4+ deprecation warnings."""
    device_type = torch.device(device).type
    if not enabled or device_type != "cuda":
        return nullcontext()

    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        return amp_module.autocast(device_type="cuda")
    # PyTorch <= 1.x fallback.
    return torch.cuda.amp.autocast()


def make_grad_scaler(enabled: bool) -> Any:
    """Create a CUDA GradScaler across PyTorch 1.x and 2.x signatures."""
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        try:
            return amp_module.GradScaler("cuda", enabled=enabled)
        except TypeError:
            # Some intermediate PyTorch releases expose torch.amp.GradScaler
            # without the device argument.
            return amp_module.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
