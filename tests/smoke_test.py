"""End-to-end smoke test using one real CBCT and real patient-level labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmdental.amp import autocast_context, make_grad_scaler
from mmdental.data import CachedViewDataset, load_split_records, preprocess_case
from mmdental.labels import LabelSchema
from mmdental.losses import (
    MultiTaskLoss,
    compute_pos_weights,
    compute_tooth_diagnosis_pos_weight,
)
from mmdental.model import build_model_from_schema
from mmdental.paths import default_data_root, default_segmentation_dir, default_work_dir
from mmdental.segmentation import preprocess_tooth_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument("--work-dir", type=Path, default=default_work_dir() / "smoke")
    parser.add_argument("--segmentation-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segmentation_dir = (
        args.segmentation_dir
        if args.segmentation_dir is not None
        else default_segmentation_dir(args.data_root)
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    records = load_split_records(args.data_root, "Train-Labeled")
    schema_records = records[:8]
    schema = LabelSchema.fit(schema_records, text_dim=8, min_diagnosis_frequency=1)
    schema.save(args.work_dir / "schema", schema_records)

    record = schema_records[0]
    cache_path = preprocess_case(
        record,
        cache_dir=args.work_dir / "cache",
        num_slices=2,
        image_size=64,
        neighbor_offset=1,
        overwrite=True,
    )
    tooth_cache_path = preprocess_tooth_case(
        record,
        segmentation_dir=segmentation_dir,
        cache_dir=args.work_dir / "cache",
        image_size=48,
        padding_mm=8.0,
        neighbor_offset=1,
        overwrite=True,
    )
    dataset = CachedViewDataset(
        [record],
        args.work_dir / "cache",
        schema,
        training=True,
        include_tooth_data=True,
        segmentation_dropout=0.0,
    )
    item = dataset[0]
    image = item["image"].unsqueeze(0).to(device)
    tooth_views = item["tooth_views"].unsqueeze(0).to(device)
    tooth_quality = item["tooth_quality"].unsqueeze(0).to(device)
    targets = {
        name: tensor.unsqueeze(0).to(device)
        for name, tensor in item["targets"].items()
    }
    model = build_model_from_schema(
        schema,
        imagenet_pretrained=False,
        token_dim=64,
        num_transformer_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        max_slices=4,
        spatial_pool_size=1,
        use_tooth_branch=True,
        use_tooth_pair_head=True,
        tooth_transformer_layers=1,
    ).to(device)
    encoded_schema_targets = [schema.encode_record(item) for item in schema_records]
    pos_weights = compute_pos_weights(encoded_schema_targets)
    pos_weights["tooth_diagnosis"] = compute_tooth_diagnosis_pos_weight(
        encoded_schema_targets
    )
    criterion = MultiTaskLoss(pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(args.amp, device):
        output = model(
            image,
            tooth_views=tooth_views,
            tooth_quality=tooth_quality,
        )
        losses = criterion(output, targets)
    if scaler.is_enabled():
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses["total"].backward()
        optimizer.step()

    assert tuple(image.shape) == (1, 3, 2, 3, 64, 64)
    assert output["teeth"].shape[-1] == len(schema.tooth_labels)
    assert tuple(tooth_views.shape[:4]) == (1, 32, 3, 3)
    assert output["segmentation_quality"].shape == (1,)
    assert output["tooth_diagnosis"].shape == (
        1, 32, len(schema.diagnosis_codes)
    )
    assert torch.isfinite(losses["total"])
    print("device={}, amp={}".format(device, scaler.is_enabled()))
    print("cache={}".format(cache_path))
    print("tooth_cache={}".format(tooth_cache_path))
    print("segmentation={}".format(segmentation_dir / (record.case_id + ".nii.gz")))
    print("image_shape={}".format(tuple(image.shape)))
    print("diagnosis_classes={}".format(len(schema.diagnosis_codes)))
    print("total_loss={:.6f}".format(float(losses["total"].detach().item())))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
