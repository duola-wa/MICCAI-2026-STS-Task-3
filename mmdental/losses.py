"""Masked, class-balanced losses and lightweight multilabel metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LOSS_WEIGHTS = {
    "diagnosis": 1.0,
    "teeth": 0.8,
    "actions": 0.7,
    "medications": 0.3,
    "text_embedding": 0.5,
    "sex": 0.05,
    "age": 0.05,
    "tooth_diagnosis": 0.8,
}


def compute_pos_weights(
    encoded_targets: Sequence[Dict[str, np.ndarray]],
    head_names: Sequence[str] = ("teeth", "diagnosis", "actions", "medications"),
    maximum: float = 10.0,
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for name in head_names:
        values = []
        for target in encoded_targets:
            if float(target["{}_mask".format(name)]) > 0:
                values.append(np.asarray(target[name], dtype=np.float32))
        if not values:
            raise ValueError("No valid targets for head {}".format(name))
        matrix = np.stack(values, axis=0)
        positive = matrix.sum(axis=0)
        negative = matrix.shape[0] - positive
        weight = np.sqrt(negative / np.maximum(positive, 1.0))
        weight = np.clip(weight, 1.0, maximum).astype(np.float32)
        output[name] = torch.from_numpy(weight)
    return output


def compute_tooth_diagnosis_pos_weight(
    encoded_targets: Sequence[Dict[str, np.ndarray]],
    maximum: float = 10.0,
) -> torch.Tensor:
    values = [
        np.asarray(target["tooth_diagnosis"], dtype=np.float32)
        for target in encoded_targets
        if float(target["tooth_diagnosis_mask"]) > 0
    ]
    if not values:
        raise ValueError("No tooth-diagnosis pair targets were extracted")
    matrix = np.concatenate(values, axis=0)
    positive = matrix.sum(axis=0)
    negative = matrix.shape[0] - positive
    weight = np.sqrt(negative / np.maximum(positive, 1.0))
    return torch.from_numpy(np.clip(weight, 1.0, maximum).astype(np.float32))


def _masked_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    element_loss = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        pos_weight=pos_weight.to(logits.device),
        reduction="none",
    )
    sample_loss = element_loss.mean(dim=1)
    mask = mask.float().reshape(-1)
    return (sample_loss * mask).sum() / mask.sum().clamp_min(1.0)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        pos_weights: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float] = None,
    ) -> None:
        super().__init__()
        self.pos_weights = {name: value.float() for name, value in pos_weights.items()}
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        if loss_weights:
            self.loss_weights.update(loss_weights)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        for name in ("teeth", "diagnosis", "actions", "medications"):
            losses[name] = _masked_binary_loss(
                outputs[name],
                targets[name],
                targets["{}_mask".format(name)],
                self.pos_weights[name],
            )

        if "tooth_diagnosis" in outputs:
            pair_logits = outputs["tooth_diagnosis"]
            pair_target = targets["tooth_diagnosis"].float()
            pair_weight = self.pos_weights["tooth_diagnosis"].to(pair_logits.device)
            pair_element = F.binary_cross_entropy_with_logits(
                pair_logits,
                pair_target,
                pos_weight=pair_weight,
                reduction="none",
            )
            pair_sample = pair_element.mean(dim=(1, 2))
            pair_mask = targets["tooth_diagnosis_mask"].float().reshape(-1)
            if "segmentation_quality" in outputs:
                pair_mask = pair_mask * (
                    outputs["segmentation_quality"].detach() > 0
                ).float()
            losses["tooth_diagnosis"] = (
                pair_sample * pair_mask
            ).sum() / pair_mask.sum().clamp_min(1.0)

        text_mask = targets["text_embedding_mask"].float().reshape(-1)
        cosine = 1.0 - F.cosine_similarity(
            outputs["text_embedding"], targets["text_embedding"].float(), dim=-1
        )
        losses["text_embedding"] = (cosine * text_mask).sum() / text_mask.sum().clamp_min(1.0)

        sex_mask = targets["sex_mask"].float().reshape(-1)
        safe_sex = targets["sex"].long().clamp_min(0)
        sex_loss = F.cross_entropy(outputs["sex"], safe_sex, reduction="none")
        losses["sex"] = (sex_loss * sex_mask).sum() / sex_mask.sum().clamp_min(1.0)

        age_mask = targets["age_mask"].float().reshape(-1)
        age_loss = F.smooth_l1_loss(outputs["age"], targets["age"].float(), reduction="none")
        losses["age"] = (age_loss * age_mask).sum() / age_mask.sum().clamp_min(1.0)

        total = sum(self.loss_weights[name] * loss for name, loss in losses.items())
        losses["total"] = total
        return losses


class MultilabelMeter:
    def __init__(self, head_names: Iterable[str], threshold: float = 0.5) -> None:
        self.head_names = list(head_names)
        self.threshold = float(threshold)
        self.counts = {
            name: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            for name in self.head_names
        }

    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> None:
        for name in self.head_names:
            prediction = torch.sigmoid(outputs[name]) >= self.threshold
            truth = targets[name] >= 0.5
            mask = targets["{}_mask".format(name)].bool()
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(-1)
            if name == "tooth_diagnosis" and "segmentation_quality" in outputs:
                quality_mask = outputs["segmentation_quality"].detach() > 0
                while quality_mask.ndim < prediction.ndim:
                    quality_mask = quality_mask.unsqueeze(-1)
                mask = mask & quality_mask
            prediction = prediction & mask
            truth = truth & mask
            self.counts[name]["tp"] += float((prediction & truth).sum().item())
            self.counts[name]["fp"] += float((prediction & ~truth).sum().item())
            self.counts[name]["fn"] += float((~prediction & truth).sum().item())

    def compute(self) -> Dict[str, float]:
        output = {}
        for name, counts in self.counts.items():
            denominator = 2.0 * counts["tp"] + counts["fp"] + counts["fn"]
            output["{}_micro_f1".format(name)] = (
                2.0 * counts["tp"] / denominator if denominator > 0 else 0.0
            )
        return output
