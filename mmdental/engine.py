"""Training and validation loops shared by command-line scripts."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .amp import autocast_context
from .losses import MultilabelMeter, MultiTaskLoss


def move_targets(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in targets.items()}


def move_model_inputs(
    batch: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Move the input tensors used by either supported architecture.

    The legacy model consumes ``image`` plus optional 2.5-D ``tooth_views``.
    The dental-ROI model consumes ``image`` plus true-3-D ``tooth_bboxes``.
    Keeping this dispatch in one place prevents training and inference from
    silently feeding different modalities to the same checkpoint.
    """

    image = batch["image"].to(device, non_blocking=True)
    optional: Dict[str, torch.Tensor] = {}
    for name in ("tooth_views", "tooth_quality", "tooth_bboxes"):
        if name in batch:
            optional[name] = batch[name].to(device, non_blocking=True)
    return image, optional


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: MultiTaskLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    amp: bool = False,
    grad_accumulation: int = 1,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    head_names = ["teeth", "diagnosis", "actions", "medications"]
    if bool(getattr(model, "use_tooth_pair_head", False)):
        head_names.append("tooth_diagnosis")
    meter = MultilabelMeter(head_names)
    totals: Dict[str, float] = {}
    sample_count = 0
    num_batches = len(loader)

    for batch_index, batch in enumerate(loader):
        images, model_kwargs = move_model_inputs(batch, device)
        targets = move_targets(batch["targets"], device)
        batch_size = int(images.shape[0])
        with torch.set_grad_enabled(training):
            with autocast_context(amp, device):
                outputs = model(images, **model_kwargs)
                losses = criterion(outputs, targets)
                scaled_loss = losses["total"] / max(1, grad_accumulation)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

        should_step = training and (
            (batch_index + 1) % max(1, grad_accumulation) == 0
            or (batch_index + 1) == num_batches
        )
        if should_step:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_size
        meter.update(outputs, targets)
        sample_count += batch_size

    metrics = {"loss_{}".format(name): value / max(1, sample_count) for name, value in totals.items()}
    metrics.update(meter.compute())
    return metrics
