"""Optional SimSiam pretraining on cached labeled + unlabeled CBCT slices."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mmdental.amp import autocast_context, make_grad_scaler
from mmdental.data import CaseRecord, cache_path_for_case, load_split_records
from mmdental.model import ResNet18FeatureMap
from mmdental.paths import default_cache_dir, default_data_root, default_runs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output", type=Path, default=default_runs_dir() / "ssl" / "backbone.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--imagenet-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=0, help="Development only: keep first N cases.")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def augment_slice(image: torch.Tensor) -> torch.Tensor:
    scale = 0.8 + 0.4 * torch.rand(1).item()
    bias = -0.1 + 0.2 * torch.rand(1).item()
    image = image * scale + bias
    if torch.rand(1).item() < 0.8:
        image = image + torch.randn_like(image) * (0.01 + 0.025 * torch.rand(1).item())
    if torch.rand(1).item() < 0.7:
        shift_y = int(torch.randint(-12, 13, (1,)).item())
        shift_x = int(torch.randint(-12, 13, (1,)).item())
        image = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
    if torch.rand(1).item() < 0.5:
        height, width = image.shape[-2:]
        crop_fraction = 0.85 + 0.15 * torch.rand(1).item()
        crop_h = max(32, int(height * crop_fraction))
        crop_w = max(32, int(width * crop_fraction))
        top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
        left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
        image = F.interpolate(
            image[:, top:top + crop_h, left:left + crop_w].unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image.clamp(0.0, 1.0)


class RandomCachedSliceDataset(Dataset):
    def __init__(self, records: Sequence[CaseRecord], cache_dir: Path) -> None:
        self.records = list(records)
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        path = cache_path_for_case(self.cache_dir, record)
        if not path.is_file():
            raise FileNotFoundError("Missing {}. Run prepare_data.py first.".format(path))
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        view = int(torch.randint(0, cached.shape[0], (1,)).item())
        slice_index = int(torch.randint(0, cached.shape[1], (1,)).item())
        image = torch.from_numpy(np.array(cached[view, slice_index], dtype=np.float32, copy=True))
        return {"view1": augment_slice(image.clone()), "view2": augment_slice(image.clone())}


class SimSiam(nn.Module):
    def __init__(self, imagenet_pretrained: bool = False, projection_dim: int = 256) -> None:
        super().__init__()
        self.backbone = ResNet18FeatureMap(pretrained=imagenet_pretrained)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(image)
        feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        return self.projector(feature)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        z1 = self.encode(view1)
        z2 = self.encode(view2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return -0.5 * (
            F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
            + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    records: List[CaseRecord] = []
    records.extend(load_split_records(args.data_root, "Train-Labeled"))
    records.extend(load_split_records(args.data_root, "Train-Unlabeled"))
    if args.limit_cases > 0:
        records = records[: args.limit_cases]
        print("DEVELOPMENT MODE: limiting SSL data to {} cases".format(len(records)))
    dataset = RandomCachedSliceDataset(records, args.cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = SimSiam(imagenet_pretrained=args.imagenet_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.amp, device):
                loss = model(view1, view2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * int(view1.shape[0])
            count += int(view1.shape[0])
        scheduler.step()
        print("Epoch {:03d} ssl_loss={:.5f}".format(epoch, total_loss / max(1, count)))
        torch.save(
            {
                "encoder_state_dict": model.backbone.state_dict(),
                "epoch": epoch,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            args.output,
        )
    print("Saved pretrained backbone to {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
