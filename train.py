"""Train a patient-level MMDental multi-task baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from mmdental.amp import make_grad_scaler
from mmdental.data import CachedViewDataset, CaseRecord, cache_path_for_case, load_supervised_records
from mmdental.engine import run_epoch
from mmdental.labels import LabelSchema
from mmdental.losses import (
    MultiTaskLoss,
    compute_pos_weights,
    compute_tooth_diagnosis_pos_weight,
)
from mmdental.model import MMDentalModel, build_model_from_schema
from mmdental.paths import default_cache_dir, default_data_root, default_runs_dir
from mmdental.roi3d import DentalROI3DDataset, dental_roi_cache_path_for_case
from mmdental.segmentation import tooth_cache_path_for_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-dir", type=Path, default=default_runs_dir() / "seg_fdi")
    parser.add_argument(
        "--model-type",
        choices=["2p5d", "dental_roi_3d"],
        default="dental_roi_3d",
        help="True 3-D dental ROI model (default) or the legacy 2.5-D baseline.",
    )
    parser.add_argument(
        "--use-unlabeled-records",
        action="store_true",
        help="Use Train-Unlabeled.csv as supervision. Confirm challenge rules first.",
    )
    parser.add_argument("--rebuild-schema", action="store_true")
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--min-diagnosis-frequency", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0, help="0..K-1; use -1 to train on all cases.")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-slices", type=int, default=32)
    parser.add_argument("--spatial-pool-size", type=int, default=2)
    parser.add_argument("--roi3d-base-channels", type=int, default=24)
    parser.add_argument(
        "--use-segmentation",
        action="store_true",
        help="Use compact FDI-aligned tooth views prepared by prepare_teeth.py.",
    )
    parser.add_argument("--segmentation-dropout", type=float, default=0.35)
    parser.add_argument("--tooth-transformer-layers", type=int, default=1)
    parser.add_argument("--max-tooth-delta", type=float, default=2.0)
    parser.add_argument("--max-diagnosis-delta", type=float, default=2.0)
    parser.add_argument("--use-tooth-pair-head", action="store_true")
    parser.add_argument(
        "--freeze-global-model",
        action="store_true",
        help="Keep the warm-started global encoder/heads fixed and train only tooth modules.",
    )
    parser.add_argument("--imagenet-pretrained", action="store_true")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-model-checkpoint",
        type=Path,
        default=None,
        help="Initialize matching global-model weights; unlike --resume, optimizer/epoch are not restored.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=0, help="Development only: keep first N cases.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no CUDA support")
    return device


def split_records(
    records: Sequence[CaseRecord], schema: LabelSchema, fold: int, num_folds: int, seed: int
) -> Tuple[List[CaseRecord], List[CaseRecord]]:
    records = list(records)
    if fold == -1:
        return records, []
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    if fold < 0 or fold >= num_folds:
        raise ValueError("fold must be -1 or in [0, {})".format(num_folds))
    diagnosis_targets = np.stack(
        [schema.encode_record(record)["diagnosis"] for record in records], axis=0
    )
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=seed
        )
        split_iterator = splitter.split(np.zeros(len(records)), diagnosis_targets)
    except ImportError:
        print("WARNING: iterative-stratification is unavailable; falling back to KFold.")
        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iterator = splitter.split(records)
    for current_fold, (train_indices, validation_indices) in enumerate(split_iterator):
        if current_fold == fold:
            return (
                [records[index] for index in train_indices],
                [records[index] for index in validation_indices],
            )
    raise AssertionError("Requested fold was not created")


def ensure_caches(
    records: Sequence[CaseRecord],
    cache_dir: Path,
    model_type: str,
    require_tooth_data: bool = False,
) -> None:
    if model_type == "dental_roi_3d":
        missing_roi = [
            str(dental_roi_cache_path_for_case(cache_dir, record))
            for record in records
            if not dental_roi_cache_path_for_case(cache_dir, record).is_file()
        ]
        if missing_roi:
            raise FileNotFoundError(
                "{} 3-D dental ROI caches are missing. Run prepare_roi3d.py first. "
                "First missing: {}".format(len(missing_roi), missing_roi[0])
            )
        return
    missing = [str(cache_path_for_case(cache_dir, record)) for record in records if not cache_path_for_case(cache_dir, record).is_file()]
    if missing:
        raise FileNotFoundError(
            "{} cached cases are missing. Run prepare_data.py first. First missing: {}".format(
                len(missing), missing[0]
            )
        )
    if require_tooth_data:
        missing_teeth = [
            str(tooth_cache_path_for_case(cache_dir, record))
            for record in records
            if not tooth_cache_path_for_case(cache_dir, record).is_file()
        ]
        if missing_teeth:
            raise FileNotFoundError(
                "{} tooth caches are missing. Run prepare_teeth.py first. First missing: {}".format(
                    len(missing_teeth), missing_teeth[0]
                )
            )


def prepare_schema(args: argparse.Namespace, records: Sequence[CaseRecord]) -> LabelSchema:
    schema_dir = args.output_dir / "schema"
    schema_path = schema_dir / "schema.json"
    if schema_path.is_file() and not args.rebuild_schema:
        schema = LabelSchema.load(schema_dir)
        selected_splits = sorted({record.split for record in records})
        selected_case_ids = sorted("{}:{}".format(record.split, record.case_id) for record in records)
        if schema.source_splits != selected_splits:
            raise RuntimeError(
                "Existing schema uses {}, but selected records use {}. Choose another --output-dir or add --rebuild-schema.".format(
                    schema.source_splits, selected_splits
                )
            )
        if schema.source_case_ids and schema.source_case_ids != selected_case_ids:
            raise RuntimeError(
                "Existing schema was built from different patient IDs. Choose another --output-dir or add --rebuild-schema."
            )
        return schema
    schema = LabelSchema.fit(
        records,
        text_dim=args.text_dim,
        min_diagnosis_frequency=args.min_diagnosis_frequency,
    )
    schema.save(schema_dir, records)
    return schema


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    named_parameters = list(model.named_parameters())
    backbone_parameters = [
        parameter for name, parameter in named_parameters if name.startswith("backbone.")
    ]
    tooth_parameters = [
        parameter for name, parameter in named_parameters if name.startswith("tooth_")
    ]
    excluded = {id(parameter) for parameter in backbone_parameters + tooth_parameters}
    global_parameters = [
        parameter for _, parameter in named_parameters if id(parameter) not in excluded
    ]
    groups = [
        {"params": backbone_parameters, "lr": args.backbone_lr, "name": "backbone"},
        {"params": global_parameters, "lr": args.global_lr, "name": "global"},
    ]
    if tooth_parameters:
        groups.append({"params": tooth_parameters, "lr": args.lr, "name": "tooth"})
    return torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )


def load_encoder_weights(model: MMDentalModel, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("encoder_state_dict", checkpoint.get("model_state", checkpoint))
    if any(key.startswith("backbone.") for key in state):
        state = {key[len("backbone."):]: value for key, value in state.items() if key.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print("Loaded encoder {} (missing={}, unexpected={})".format(path, len(missing), len(unexpected)))


def load_initial_model_weights(
    model: torch.nn.Module,
    path: Path,
    schema: LabelSchema,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema_signature") != schema.signature():
        raise ValueError("Initial model checkpoint uses a different label schema")
    state = checkpoint.get("model_state", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "tooth_projection.",
        "tooth_quality_projection.",
        "tooth_visual_gate.",
        "tooth_position_embedding",
        "tooth_transformer.",
        "tooth_pool_score.",
        "tooth_slot_head.",
        "tooth_diagnosis_head.",
        "tooth_pair_head.",
    )
    disallowed_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise ValueError(
            "Initial checkpoint is not a compatible global model; missing={}, unexpected={}".format(
                disallowed_missing[:10], unexpected[:10]
            )
        )
    print(
        "Initialized global model from {} (new segmentation tensors={})".format(
            path, len(missing)
        )
    )


def validation_score(metrics: Dict[str, float]) -> float:
    names = [
        "diagnosis_micro_f1",
        "teeth_micro_f1",
        "actions_micro_f1",
        "medications_micro_f1",
    ]
    patient_score = float(np.mean([metrics.get(name, 0.0) for name in names]))
    if "tooth_diagnosis_micro_f1" in metrics:
        # Pair prediction is an auxiliary task.  Use its actual F1 (not an
        # inverse loss that can look deceptively good while every pair is
        # wrong) and keep checkpoint selection focused on official entities.
        return 0.95 * patient_score + 0.05 * float(
            metrics["tooth_diagnosis_micro_f1"]
        )
    return patient_score


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    schema: LabelSchema,
    args: argparse.Namespace,
    validation_records: Sequence[CaseRecord],
    epoch: int,
    best_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.model_config,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "schema_dir": str((args.output_dir / "schema").resolve()),
            # POSIX separators make this relocation hint portable even when a
            # Windows-trained checkpoint is copied to Ubuntu.
            "schema_relative_dir": "../schema",
            "schema_signature": schema.signature(),
            "validation_case_keys": [
                "{}:{}".format(record.split, record.case_id)
                for record in validation_records
            ],
            "epoch": epoch,
            "best_score": best_score,
            "args": serializable_args(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    # The legacy ResNet may start from ImageNet/SSL weights, whereas the new
    # Conv3d encoder starts from scratch.  Freezing that random 3-D backbone or
    # training it at the legacy 1e-5 rate would prevent it from learning.
    if args.backbone_lr is None:
        args.backbone_lr = 1e-4 if args.model_type == "dental_roi_3d" else 1e-5
    if args.global_lr is None:
        args.global_lr = 1e-4 if args.model_type == "dental_roi_3d" else 3e-5
    if args.freeze_backbone_epochs is None:
        args.freeze_backbone_epochs = 0 if args.model_type == "dental_roi_3d" else 3
    if not 0.0 <= args.segmentation_dropout <= 1.0:
        raise ValueError("--segmentation-dropout must be in [0, 1]")
    if args.resume and args.init_model_checkpoint:
        raise ValueError("Use either --resume or --init-model-checkpoint, not both")
    if args.encoder_checkpoint and args.init_model_checkpoint:
        raise ValueError("Use either --encoder-checkpoint or --init-model-checkpoint, not both")
    if args.model_type == "2p5d" and args.use_tooth_pair_head and not args.use_segmentation:
        raise ValueError("--use-tooth-pair-head requires --use-segmentation")
    if args.model_type == "dental_roi_3d":
        incompatible = []
        if args.imagenet_pretrained:
            incompatible.append("--imagenet-pretrained")
        if args.encoder_checkpoint:
            incompatible.append("--encoder-checkpoint")
        if args.init_model_checkpoint:
            incompatible.append("--init-model-checkpoint")
        if args.freeze_global_model:
            incompatible.append("--freeze-global-model")
        if incompatible:
            raise ValueError(
                "The true-3-D backbone cannot load old 2-D weights/options: {}".format(
                    ", ".join(incompatible)
                )
            )
        if args.roi3d_base_channels < 8:
            raise ValueError("--roi3d-base-channels must be at least 8")
    if args.freeze_global_model and not (args.init_model_checkpoint or args.resume):
        raise ValueError(
            "--freeze-global-model requires --init-model-checkpoint or --resume"
        )
    set_seed(args.seed)
    device = select_device(args.device)
    if device.type == "cpu":
        print("WARNING: training on CPU. Install a CUDA-enabled PyTorch build for full training.")
    print("Device: {}".format(device))

    records = load_supervised_records(args.data_root, args.use_unlabeled_records)
    if args.limit_cases > 0:
        records = records[: args.limit_cases]
        print("DEVELOPMENT MODE: limiting supervised data to {} cases".format(len(records)))
    ensure_caches(
        records,
        args.cache_dir,
        model_type=args.model_type,
        require_tooth_data=args.use_segmentation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = prepare_schema(args, records)
    train_records, validation_records = split_records(records, schema, args.fold, args.num_folds, args.seed)
    print(
        "Supervised cases: total={}, train={}, validation={}, sources={}".format(
            len(records), len(train_records), len(validation_records), schema.source_splits
        )
    )
    if args.use_unlabeled_records:
        print("WARNING: Train-Unlabeled.csv supervision is ENABLED. Confirm this is permitted by organizers.")

    if args.model_type == "dental_roi_3d":
        train_dataset = DentalROI3DDataset(
            train_records,
            args.cache_dir,
            schema,
            training=True,
            segmentation_dropout=args.segmentation_dropout,
        )
        validation_dataset = DentalROI3DDataset(
            validation_records,
            args.cache_dir,
            schema,
            training=False,
        )
    else:
        train_dataset = CachedViewDataset(
            train_records,
            args.cache_dir,
            schema,
            training=True,
            include_tooth_data=args.use_segmentation,
            segmentation_dropout=args.segmentation_dropout if args.use_segmentation else 0.0,
        )
        validation_dataset = CachedViewDataset(
            validation_records,
            args.cache_dir,
            schema,
            training=False,
            include_tooth_data=args.use_segmentation,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = None
    if validation_records:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    if args.model_type == "dental_roi_3d":
        model = build_model_from_schema(
            schema,
            model_type="dental_roi_3d",
            token_dim=args.token_dim,
            num_attention_heads=args.attention_heads,
            dropout=args.dropout,
            tooth_transformer_layers=args.tooth_transformer_layers,
            roi3d_base_channels=args.roi3d_base_channels,
        ).to(device)
    else:
        model = build_model_from_schema(
            schema,
            model_type="2p5d",
            imagenet_pretrained=args.imagenet_pretrained,
            token_dim=args.token_dim,
            num_transformer_layers=args.transformer_layers,
            num_attention_heads=args.attention_heads,
            dropout=args.dropout,
            max_slices=args.max_slices,
            spatial_pool_size=args.spatial_pool_size,
            use_tooth_branch=args.use_segmentation,
            use_tooth_pair_head=args.use_tooth_pair_head,
            tooth_transformer_layers=args.tooth_transformer_layers,
            max_tooth_delta=args.max_tooth_delta,
            max_diagnosis_delta=args.max_diagnosis_delta,
            segmentation_mapping="nnunet32-fdi-v1",
        ).to(device)
    if args.encoder_checkpoint:
        load_encoder_weights(model, args.encoder_checkpoint)
    if args.init_model_checkpoint:
        load_initial_model_weights(model, args.init_model_checkpoint, schema)

    encoded_targets = [schema.encode_record(record) for record in train_records]
    pos_weights = compute_pos_weights(encoded_targets)
    if bool(getattr(model, "use_tooth_pair_head", False)):
        pos_weights["tooth_diagnosis"] = compute_tooth_diagnosis_pos_weight(
            encoded_targets
        )
    loss_weight_overrides = (
        {"tooth_diagnosis": 0.2}
        if args.model_type == "dental_roi_3d"
        else None
    )
    criterion = MultiTaskLoss(
        pos_weights,
        loss_weights=loss_weight_overrides,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-7
    )
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    best_score = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint.get("schema_signature") != schema.signature():
            raise ValueError("Resume checkpoint uses a different label schema")
        if checkpoint.get("model_config") != model.model_config:
            raise ValueError(
                "Resume checkpoint uses a different model/segmentation configuration. "
                "Use --init-model-checkpoint for transfer instead."
            )
        saved_validation_keys = checkpoint.get("validation_case_keys")
        current_validation_keys = [
            "{}:{}".format(record.split, record.case_id)
            for record in validation_records
        ]
        if saved_validation_keys is not None and saved_validation_keys != current_validation_keys:
            raise ValueError(
                "Resume checkpoint uses a different validation fold. "
                "Keep the same split dependency and data selection."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    fold_name = "all" if args.fold == -1 else "fold_{}".format(args.fold)
    fold_dir = args.output_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "history.jsonl"
    if start_epoch == 0 and log_path.is_file():
        log_path.write_text("", encoding="utf-8")
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_global_model:
            model.freeze_global_model(True)
            frozen = True
        else:
            frozen = epoch < args.freeze_backbone_epochs
            model.freeze_global_model(False)
            model.freeze_backbone(frozen)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            grad_accumulation=args.grad_accumulation,
            grad_clip=args.grad_clip,
        )
        if validation_loader is not None:
            with torch.no_grad():
                validation_metrics = run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    optimizer=None,
                    scaler=None,
                    amp=args.amp,
                )
            score = validation_score(validation_metrics)
        else:
            validation_metrics = {}
            score = -float(train_metrics["loss_total"])
        scheduler.step()

        row = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "score": score,
            "learning_rates": {
                str(group.get("name", index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "Epoch {:03d} train_loss={:.4f} val_loss={} score={:.4f}".format(
                epoch,
                train_metrics["loss_total"],
                "{:.4f}".format(validation_metrics["loss_total"]) if validation_metrics else "n/a",
                score,
            )
        )

        improved = score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            fold_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            schema,
            args,
            validation_records,
            epoch,
            best_score,
        )
        if improved:
            save_checkpoint(
                fold_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                schema,
                args,
                validation_records,
                epoch,
                best_score,
            )
        if validation_loader is not None and epochs_without_improvement >= args.patience:
            print("Early stopping after {} epochs without improvement".format(args.patience))
            break

    print("Training finished in {:.1f} minutes. Best score={:.4f}".format((time.time() - start_time) / 60.0, best_score))
    print("Checkpoint: {}".format((fold_dir / "best.pt").resolve()))


if __name__ == "__main__":
    main()
