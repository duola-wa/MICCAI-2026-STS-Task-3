"""Neural networks for patient-level MMDental prediction.

``MMDentalModel`` is the original multi-view 2.5-D baseline.  The newer
``MMDental3DROIModel`` consumes a true 3-D dental-arch crop and uses the
nnU-Net tooth boxes to retain one contextual token for every adult FDI slot.
The legacy class deliberately remains unchanged so its checkpoints stay
loadable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


LEGACY_ARCHITECTURE = "multiview_2p5d_resnet18"
DENTAL_ROI_3D_ARCHITECTURE = "dental_roi_3d_v1"


class ResNet18FeatureMap(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        network = self._create_resnet(pretrained)
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.out_channels = 512
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _create_resnet(pretrained: bool) -> nn.Module:
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet18(weights=weights)
        except (ImportError, TypeError):
            # torchvision <= 0.12
            return resnet18(pretrained=pretrained)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.image_mean) / self.image_std
        return self.features(images)


class MMDentalModel(nn.Module):
    def __init__(
        self,
        num_teeth: int,
        num_diagnosis: int,
        num_actions: int,
        num_medications: int,
        text_dim: int,
        token_dim: int = 256,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        max_slices: int = 32,
        spatial_pool_size: int = 2,
        imagenet_pretrained: bool = False,
        use_tooth_branch: bool = False,
        num_adult_teeth: int = 32,
        tooth_quality_dim: int = 10,
        tooth_transformer_layers: int = 1,
        max_tooth_delta: float = 2.0,
        max_diagnosis_delta: float = 2.0,
        segmentation_mapping: str = "nnunet32-fdi-v1",
        use_tooth_pair_head: bool = False,
    ) -> None:
        super().__init__()
        if token_dim % num_attention_heads != 0:
            raise ValueError("token_dim must be divisible by num_attention_heads")
        if use_tooth_branch and num_teeth < num_adult_teeth:
            raise ValueError(
                "num_teeth={} is smaller than num_adult_teeth={}".format(
                    num_teeth, num_adult_teeth
                )
            )
        if tooth_transformer_layers < 1:
            raise ValueError("tooth_transformer_layers must be at least 1")
        if max_tooth_delta <= 0 or max_diagnosis_delta <= 0:
            raise ValueError("Residual delta limits must be positive")
        self.model_config: Dict[str, Any] = {
            "num_teeth": int(num_teeth),
            "num_diagnosis": int(num_diagnosis),
            "num_actions": int(num_actions),
            "num_medications": int(num_medications),
            "text_dim": int(text_dim),
            "token_dim": int(token_dim),
            "num_transformer_layers": int(num_transformer_layers),
            "num_attention_heads": int(num_attention_heads),
            "dropout": float(dropout),
            "max_slices": int(max_slices),
            "spatial_pool_size": int(spatial_pool_size),
            "use_tooth_branch": bool(use_tooth_branch),
            "num_adult_teeth": int(num_adult_teeth),
            "tooth_quality_dim": int(tooth_quality_dim),
            "tooth_transformer_layers": int(tooth_transformer_layers),
            "max_tooth_delta": float(max_tooth_delta),
            "max_diagnosis_delta": float(max_diagnosis_delta),
            "segmentation_mapping": str(segmentation_mapping),
            "use_tooth_pair_head": bool(use_tooth_pair_head),
            # Checkpoints contain the trained weights; do not redownload on load.
            "imagenet_pretrained": False,
        }
        self.max_slices = int(max_slices)
        self.spatial_pool_size = int(spatial_pool_size)
        self.num_spatial_tokens = self.spatial_pool_size ** 2
        self.use_tooth_branch = bool(use_tooth_branch)
        self.num_adult_teeth = int(num_adult_teeth)
        self.tooth_quality_dim = int(tooth_quality_dim)
        self.max_tooth_delta = float(max_tooth_delta)
        self.max_diagnosis_delta = float(max_diagnosis_delta)
        self.use_tooth_pair_head = bool(use_tooth_pair_head)
        self.global_model_frozen = False

        self.backbone = ResNet18FeatureMap(pretrained=imagenet_pretrained)
        self.token_projection = nn.Sequential(
            nn.Linear(self.backbone.out_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.view_embedding = nn.Parameter(torch.zeros(1, 3, 1, 1, token_dim))
        self.slice_embedding = nn.Parameter(torch.zeros(1, 1, max_slices, 1, token_dim))
        self.spatial_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, self.num_spatial_tokens, token_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_attention_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(token_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                "teeth": nn.Linear(token_dim, num_teeth),
                "diagnosis": nn.Linear(token_dim, num_diagnosis),
                "actions": nn.Linear(token_dim, num_actions),
                "medications": nn.Linear(token_dim, num_medications),
                "sex": nn.Linear(token_dim, 2),
                "age": nn.Linear(token_dim, 1),
                "text_embedding": nn.Linear(token_dim, text_dim),
            }
        )
        if self.use_tooth_branch:
            # Each tooth has three orthogonal 2.5D views.  The global and local
            # branches share the ResNet, which keeps the parameter count small
            # enough for the 50-case supervised set.
            self.tooth_projection = nn.Sequential(
                nn.Linear(self.backbone.out_channels * 3, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
            )
            self.tooth_quality_projection = nn.Sequential(
                nn.Linear(self.tooth_quality_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
            )
            self.tooth_visual_gate = nn.Sequential(
                nn.Linear(self.tooth_quality_dim, max(16, token_dim // 4)),
                nn.GELU(),
                nn.Linear(max(16, token_dim // 4), 1),
                nn.Sigmoid(),
            )
            self.tooth_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_adult_teeth, token_dim)
            )
            tooth_encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_attention_heads,
                dim_feedforward=token_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.tooth_transformer = nn.TransformerEncoder(
                tooth_encoder_layer,
                num_layers=tooth_transformer_layers,
                norm=nn.LayerNorm(token_dim),
            )
            self.tooth_pool_score = nn.Linear(token_dim, 1)
            self.tooth_slot_head = nn.Linear(token_dim, 1)
            self.tooth_diagnosis_head = nn.Linear(token_dim, num_diagnosis)
            if self.use_tooth_pair_head:
                self.tooth_pair_head = nn.Linear(token_dim, num_diagnosis)
        self._initialize_tokens()

    def _initialize_tokens(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.slice_embedding, std=0.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        if self.use_tooth_branch:
            nn.init.trunc_normal_(self.tooth_position_embedding, std=0.02)
            # The segmentation branch is a bounded residual and initially
            # reproduces the global-only baseline exactly.
            nn.init.zeros_(self.tooth_slot_head.weight)
            nn.init.zeros_(self.tooth_slot_head.bias)
            nn.init.zeros_(self.tooth_diagnosis_head.weight)
            nn.init.zeros_(self.tooth_diagnosis_head.bias)
            if self.use_tooth_pair_head:
                nn.init.zeros_(self.tooth_pair_head.weight)
                nn.init.zeros_(self.tooth_pair_head.bias)

    def train(self, mode: bool = True) -> "MMDentalModel":
        super().train(mode)
        if self.use_tooth_branch:
            # Batch size is one and the shared encoder sees 36 global images
            # followed by up to 96 local/blank crops.  Updating BatchNorm here
            # destroys the warm-started global representation.
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if mode and self.global_model_frozen:
            self.backbone.eval()
            self.transformer.eval()
            self.dropout.eval()
        return self

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def freeze_global_model(self, frozen: bool = True) -> None:
        """Freeze the warm-started global model while training tooth modules."""
        self.global_model_frozen = bool(frozen)
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (
                not frozen or name.startswith("tooth_")
            )

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        """
        Args:
            views: [batch, view=3, slices, channel=3, height, width]
        """
        if views.ndim != 6:
            raise ValueError("Expected [B,3,S,3,H,W], got {}".format(tuple(views.shape)))
        batch_size, num_views, num_slices, channels, height, width = views.shape
        if num_views != 3 or channels != 3:
            raise ValueError("Expected three views and three channels, got {}".format(tuple(views.shape)))
        if num_slices > self.max_slices:
            raise ValueError("{} slices exceed max_slices={}".format(num_slices, self.max_slices))

        flat = views.reshape(batch_size * num_views * num_slices, channels, height, width)
        feature_map = self.backbone(flat)
        pooled = F.adaptive_avg_pool2d(
            feature_map,
            output_size=(self.spatial_pool_size, self.spatial_pool_size),
        )
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        tokens = tokens.reshape(
            batch_size,
            num_views,
            num_slices,
            self.num_spatial_tokens,
            -1,
        )
        tokens = (
            tokens
            + self.view_embedding[:, :num_views]
            + self.slice_embedding[:, :, :num_slices]
            + self.spatial_embedding
        )
        tokens = tokens.reshape(batch_size, -1, tokens.shape[-1])
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        encoded = self.transformer(sequence)
        return encoded[:, 0]

    def encode_teeth(
        self,
        tooth_views: torch.Tensor,
        tooth_quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode 32 fixed FDI slots while retaining absent-tooth information."""
        if not self.use_tooth_branch:
            raise RuntimeError("encode_teeth called while use_tooth_branch=False")
        if tooth_views.ndim != 6:
            raise ValueError(
                "Expected tooth_views [B,32,3,3,H,W], got {}".format(
                    tuple(tooth_views.shape)
                )
            )
        batch_size, num_teeth, num_views, channels, height, width = tooth_views.shape
        if num_teeth != self.num_adult_teeth or num_views != 3 or channels != 3:
            raise ValueError(
                "Expected [B,{},3,3,H,W], got {}".format(
                    self.num_adult_teeth, tuple(tooth_views.shape)
                )
            )
        if tuple(tooth_quality.shape) != (
            batch_size,
            self.num_adult_teeth,
            self.tooth_quality_dim,
        ):
            raise ValueError(
                "Expected tooth_quality [B,{},{}], got {}".format(
                    self.num_adult_teeth,
                    self.tooth_quality_dim,
                    tuple(tooth_quality.shape),
                )
            )

        flat = tooth_views.reshape(
            batch_size * num_teeth * num_views,
            channels,
            height,
            width,
        )
        feature_map = self.backbone(flat)
        visual = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        visual = visual.reshape(batch_size, num_teeth, num_views * self.backbone.out_channels)
        visual = self.tooth_projection(visual)

        # Column 0 is the deterministic presence flag.  No visual feature is
        # admitted for a missing/failed slot, but its FDI and quality token remain
        # available so the network can reason about missing teeth.
        presence = tooth_quality[..., :1].clamp(0.0, 1.0)
        fill = tooth_quality[..., 2:3].clamp(0.0, 1.0)
        component_purity = tooth_quality[..., 3:4].clamp(0.0, 1.0)
        deterministic_reliability = presence * torch.sqrt(fill * component_purity + 1e-8)
        visual_gate = self.tooth_visual_gate(tooth_quality) * deterministic_reliability
        tokens = (
            self.tooth_position_embedding
            + self.tooth_quality_projection(tooth_quality)
            + visual * visual_gate
        )
        tokens = self.tooth_transformer(tokens)
        pooling_weights = torch.softmax(self.tooth_pool_score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * pooling_weights.unsqueeze(-1), dim=1)
        present_count = presence.sum(dim=1).clamp_min(1.0)
        case_quality = (
            deterministic_reliability.sum(dim=1) / present_count
        ).clamp(0.0, 1.0)
        return tokens, pooled, visual_gate.squeeze(-1), case_quality

    def forward(
        self,
        views: torch.Tensor,
        tooth_views: Optional[torch.Tensor] = None,
        tooth_quality: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        global_feature = self.encode(views)
        tooth_tokens: Optional[torch.Tensor] = None
        visual_gate: Optional[torch.Tensor] = None
        tooth_feature: Optional[torch.Tensor] = None
        case_quality: Optional[torch.Tensor] = None
        if self.use_tooth_branch:
            if tooth_views is None or tooth_quality is None:
                raise ValueError(
                    "This checkpoint requires tooth_views and tooth_quality. "
                    "Run prepare_teeth.py and enable tooth data in the dataset."
                )
            tooth_tokens, tooth_feature, visual_gate, case_quality = self.encode_teeth(
                tooth_views, tooth_quality
            )
        patient_feature = self.dropout(global_feature)
        output = {name: head(patient_feature) for name, head in self.heads.items()}
        if self.use_tooth_branch:
            assert tooth_tokens is not None
            assert tooth_feature is not None
            assert case_quality is not None
            adult_slot_logits = self.max_tooth_delta * torch.tanh(
                self.tooth_slot_head(tooth_tokens).squeeze(-1) / self.max_tooth_delta
            )
            pair_logits: Optional[torch.Tensor] = None
            if self.use_tooth_pair_head:
                pair_logits = self.tooth_pair_head(tooth_tokens)
                output["tooth_diagnosis"] = pair_logits
                pair_tooth_evidence = pair_logits.max(dim=-1).values
                adult_slot_logits = adult_slot_logits + 0.5 * self.max_tooth_delta * torch.tanh(
                    pair_tooth_evidence / self.max_tooth_delta
                )
            teeth_logits = output["teeth"].clone()
            teeth_logits[:, : self.num_adult_teeth] = (
                teeth_logits[:, : self.num_adult_teeth]
                + case_quality * adult_slot_logits
            )
            output["teeth"] = teeth_logits
            diagnosis_source = (
                pair_logits.max(dim=1).values
                if pair_logits is not None
                else self.tooth_diagnosis_head(tooth_feature)
            )
            diagnosis_delta = self.max_diagnosis_delta * torch.tanh(
                diagnosis_source / self.max_diagnosis_delta
            )
            output["diagnosis"] = output["diagnosis"] + case_quality * diagnosis_delta
            assert visual_gate is not None
            output["tooth_visual_gate"] = visual_gate
            output["segmentation_quality"] = case_quality.squeeze(-1)
        output["age"] = output["age"].squeeze(-1)
        output["text_embedding"] = F.normalize(output["text_embedding"], dim=-1)
        output["patient_feature"] = patient_feature
        return output


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest small GroupNorm divisor for ``channels``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    """A compact residual block that is stable with batch size one."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.skip(image)
        image = self.activation(self.norm1(self.conv1(image)))
        image = self.norm2(self.conv2(image))
        return self.activation(image + residual)


class DentalROIBackbone3D(nn.Module):
    """Small true-3-D encoder with an intermediate map for tooth ROI pooling."""

    def __init__(self, input_channels: int = 2, base_channels: int = 24) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = ResidualBlock3D(base_channels, base_channels)
        self.stage2 = ResidualBlock3D(base_channels, base_channels * 2, stride=2)
        self.stage3 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.stage4 = ResidualBlock3D(base_channels * 4, base_channels * 8, stride=2)
        self.roi_channels = base_channels * 2
        self.out_channels = base_channels * 8
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.stage1(self.stem(image))
        roi_feature_map = self.stage2(image)
        global_feature_map = self.stage4(self.stage3(roi_feature_map))
        return roi_feature_map, global_feature_map


class MMDental3DROIModel(nn.Module):
    """Segmentation-guided true-3-D dental-arch model.

    The first input channel is the windowed CBCT crop and the second is the
    union tooth mask. ``tooth_bboxes`` has shape ``[B, 32, 6]`` and stores
    normalized coordinates in ``[z0, z1, y0, y1, x0, x1]`` order.  Boxes are
    contextual (tooth plus a physical margin), so ROI pooling can see lesions
    around a tooth rather than just voxels inside the segmentation.
    """

    bbox_order = ("z0", "z1", "y0", "y1", "x0", "x1")

    def __init__(
        self,
        num_teeth: int,
        num_diagnosis: int,
        num_actions: int,
        num_medications: int,
        text_dim: int,
        token_dim: int = 192,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        input_channels: int = 2,
        base_channels: int = 24,
        num_adult_teeth: int = 32,
        tooth_quality_dim: int = 10,
        roi_pool_size: int = 2,
        max_tooth_delta: float = 2.0,
        segmentation_mapping: str = "nnunet32-fdi-v1",
        use_tooth_pair_head: bool = True,
    ) -> None:
        super().__init__()
        if token_dim % num_attention_heads != 0:
            raise ValueError("token_dim must be divisible by num_attention_heads")
        if num_transformer_layers < 1:
            raise ValueError("num_transformer_layers must be at least 1")
        if num_teeth < num_adult_teeth:
            raise ValueError(
                "num_teeth={} is smaller than num_adult_teeth={}".format(
                    num_teeth, num_adult_teeth
                )
            )
        if tooth_quality_dim < 1:
            raise ValueError("tooth_quality_dim must be positive")
        if roi_pool_size < 1:
            raise ValueError("roi_pool_size must be positive")
        if max_tooth_delta <= 0:
            raise ValueError("max_tooth_delta must be positive")

        self.model_config: Dict[str, Any] = {
            "architecture": DENTAL_ROI_3D_ARCHITECTURE,
            "num_teeth": int(num_teeth),
            "num_diagnosis": int(num_diagnosis),
            "num_actions": int(num_actions),
            "num_medications": int(num_medications),
            "text_dim": int(text_dim),
            "token_dim": int(token_dim),
            "num_transformer_layers": int(num_transformer_layers),
            "num_attention_heads": int(num_attention_heads),
            "dropout": float(dropout),
            "input_channels": int(input_channels),
            "base_channels": int(base_channels),
            "num_adult_teeth": int(num_adult_teeth),
            "tooth_quality_dim": int(tooth_quality_dim),
            "roi_pool_size": int(roi_pool_size),
            "max_tooth_delta": float(max_tooth_delta),
            "segmentation_mapping": str(segmentation_mapping),
            "use_tooth_pair_head": bool(use_tooth_pair_head),
        }
        self.input_channels = int(input_channels)
        self.num_adult_teeth = int(num_adult_teeth)
        self.tooth_quality_dim = int(tooth_quality_dim)
        self.roi_pool_size = int(roi_pool_size)
        self.max_tooth_delta = float(max_tooth_delta)
        self.use_tooth_pair_head = bool(use_tooth_pair_head)
        self.use_tooth_branch = True
        self.uses_roi3d = True
        self.global_model_frozen = False

        self.backbone = DentalROIBackbone3D(
            input_channels=self.input_channels,
            base_channels=base_channels,
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.backbone.out_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.tooth_roi_projection = nn.Sequential(
            nn.Linear(self.backbone.roi_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.tooth_quality_projection = nn.Sequential(
            nn.Linear(self.tooth_quality_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.tooth_visual_gate = nn.Sequential(
            nn.Linear(self.tooth_quality_dim, max(16, token_dim // 4)),
            nn.GELU(),
            nn.Linear(max(16, token_dim // 4), 1),
            nn.Sigmoid(),
        )
        self.tooth_position_embedding = nn.Parameter(
            torch.zeros(1, self.num_adult_teeth, token_dim)
        )
        self.tooth_presence_embedding = nn.Embedding(2, token_dim)
        tooth_encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_attention_heads,
            dim_feedforward=token_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tooth_transformer = nn.TransformerEncoder(
            tooth_encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(token_dim),
        )
        self.tooth_pool_score = nn.Linear(token_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                "teeth": nn.Linear(token_dim, num_teeth),
                "diagnosis": nn.Linear(token_dim, num_diagnosis),
                "actions": nn.Linear(token_dim, num_actions),
                "medications": nn.Linear(token_dim, num_medications),
                "sex": nn.Linear(token_dim, 2),
                "age": nn.Linear(token_dim, 1),
                "text_embedding": nn.Linear(token_dim, text_dim),
            }
        )
        self.tooth_slot_head = nn.Linear(token_dim, 1)
        if self.use_tooth_pair_head:
            self.tooth_pair_head = nn.Linear(token_dim, num_diagnosis)
        self._initialize_tokens()

    def _initialize_tokens(self) -> None:
        nn.init.trunc_normal_(self.tooth_position_embedding, std=0.02)
        nn.init.normal_(self.tooth_presence_embedding.weight, std=0.02)
        # Start as a patient-level model; local FDI corrections are learned
        # gradually rather than destabilising the first supervised epoch.
        nn.init.zeros_(self.tooth_slot_head.weight)
        nn.init.zeros_(self.tooth_slot_head.bias)
        if self.use_tooth_pair_head:
            nn.init.zeros_(self.tooth_pair_head.weight)
            nn.init.zeros_(self.tooth_pair_head.bias)

    def train(self, mode: bool = True) -> "MMDental3DROIModel":
        super().train(mode)
        if mode and self.global_model_frozen:
            self.backbone.eval()
            self.global_projection.eval()
        return self

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def freeze_global_model(self, frozen: bool = True) -> None:
        """Freeze the volume/global path while fitting FDI-specific modules."""
        self.global_model_frozen = bool(frozen)
        local_prefixes = ("tooth_", "fusion.")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = not frozen or name.startswith(local_prefixes)

    def _validate_inputs(
        self,
        image: torch.Tensor,
        tooth_bboxes: Optional[torch.Tensor],
        tooth_quality: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 5:
            raise ValueError("Expected image [B,C,D,H,W], got {}".format(tuple(image.shape)))
        if image.shape[1] != self.input_channels:
            raise ValueError(
                "Expected {} 3-D input channels, got {}".format(
                    self.input_channels, image.shape[1]
                )
            )
        batch_size = image.shape[0]
        if tooth_bboxes is None:
            tooth_bboxes = image.new_zeros((batch_size, self.num_adult_teeth, 6))
        if tuple(tooth_bboxes.shape) != (batch_size, self.num_adult_teeth, 6):
            raise ValueError(
                "Expected tooth_bboxes [B,{},6], got {}".format(
                    self.num_adult_teeth, tuple(tooth_bboxes.shape)
                )
            )
        finite_boxes = torch.isfinite(tooth_bboxes).all(dim=-1)
        tooth_bboxes = torch.nan_to_num(tooth_bboxes.float()).clamp(0.0, 1.0)
        starts = tooth_bboxes[..., (0, 2, 4)]
        ends = tooth_bboxes[..., (1, 3, 5)]
        valid_boxes = finite_boxes & ((ends - starts) > 1e-4).all(dim=-1)

        if tooth_quality is None:
            tooth_quality = image.new_zeros(
                (batch_size, self.num_adult_teeth, self.tooth_quality_dim)
            )
            tooth_quality[..., 0] = valid_boxes.to(image.dtype)
        if tuple(tooth_quality.shape) != (
            batch_size,
            self.num_adult_teeth,
            self.tooth_quality_dim,
        ):
            raise ValueError(
                "Expected tooth_quality [B,{},{}], got {}".format(
                    self.num_adult_teeth,
                    self.tooth_quality_dim,
                    tuple(tooth_quality.shape),
                )
            )
        tooth_quality = torch.nan_to_num(tooth_quality.float())
        presence = (
            tooth_quality[..., 0].clamp(0.0, 1.0)
            * valid_boxes.to(tooth_quality.dtype)
        )
        return tooth_bboxes, tooth_quality, presence

    def _pool_tooth_rois(
        self,
        feature_map: torch.Tensor,
        tooth_bboxes: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        """Trilinearly sample all 32 variable-size contextual boxes."""
        batch_size, channels = feature_map.shape[:2]
        num_teeth = tooth_bboxes.shape[1]
        pool_size = self.roi_pool_size
        coordinate = (
            torch.arange(
                pool_size,
                dtype=feature_map.dtype,
                device=feature_map.device,
            )
            + 0.5
        ) / float(pool_size)
        pooled_batches = []
        for batch_index in range(batch_size):
            boxes = tooth_bboxes[batch_index].to(dtype=feature_map.dtype)
            z0, z1, y0, y1, x0, x1 = boxes.unbind(dim=-1)
            tz = coordinate.view(1, pool_size, 1, 1)
            ty = coordinate.view(1, 1, pool_size, 1)
            tx = coordinate.view(1, 1, 1, pool_size)
            z = z0[:, None, None, None] + (z1 - z0)[:, None, None, None] * tz
            y = y0[:, None, None, None] + (y1 - y0)[:, None, None, None] * ty
            x = x0[:, None, None, None] + (x1 - x0)[:, None, None, None] * tx
            z = z.expand(num_teeth, pool_size, pool_size, pool_size)
            y = y.expand(num_teeth, pool_size, pool_size, pool_size)
            x = x.expand(num_teeth, pool_size, pool_size, pool_size)
            # grid_sample uses x/y/z ordering and the [-1, 1] coordinate range.
            grid = torch.stack((x * 2.0 - 1.0, y * 2.0 - 1.0, z * 2.0 - 1.0), dim=-1)
            source = feature_map[batch_index : batch_index + 1].expand(
                num_teeth, channels, *feature_map.shape[2:]
            )
            sampled = F.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            pooled = sampled.mean(dim=(2, 3, 4))
            pooled_batches.append(pooled)
        pooled_features = torch.stack(pooled_batches, dim=0)
        return pooled_features * presence.unsqueeze(-1).to(pooled_features.dtype)

    def encode(
        self,
        image: torch.Tensor,
        tooth_bboxes: Optional[torch.Tensor],
        tooth_quality: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tooth_bboxes, tooth_quality, presence = self._validate_inputs(
            image, tooth_bboxes, tooth_quality
        )
        roi_feature_map, global_feature_map = self.backbone(image)
        global_feature = F.adaptive_avg_pool3d(global_feature_map, 1).flatten(1)
        global_feature = self.global_projection(global_feature)

        roi_features = self._pool_tooth_rois(
            roi_feature_map, tooth_bboxes, presence
        )
        roi_tokens = self.tooth_roi_projection(roi_features)
        if self.tooth_quality_dim >= 4:
            fill = tooth_quality[..., 2].clamp(0.0, 1.0)
            purity = tooth_quality[..., 3].clamp(0.0, 1.0)
            deterministic_reliability = presence * (
                0.25 + 0.75 * torch.sqrt(fill * purity + 1e-8)
            )
        else:
            deterministic_reliability = presence
        learned_gate = self.tooth_visual_gate(tooth_quality).squeeze(-1)
        visual_gate = presence * (0.25 + 0.75 * learned_gate) * (
            0.5 + 0.5 * deterministic_reliability
        )
        presence_index = (presence >= 0.5).long()
        tooth_tokens = (
            self.tooth_position_embedding
            + self.tooth_presence_embedding(presence_index)
            + self.tooth_quality_projection(tooth_quality)
            + roi_tokens * visual_gate.unsqueeze(-1)
        )
        tooth_tokens = self.tooth_transformer(tooth_tokens)
        pooling_weights = torch.softmax(
            self.tooth_pool_score(tooth_tokens).squeeze(-1), dim=1
        )
        tooth_feature = torch.sum(
            tooth_tokens * pooling_weights.unsqueeze(-1), dim=1
        )
        patient_feature = self.fusion(
            torch.cat((global_feature, tooth_feature), dim=-1)
        )
        present_count = presence.sum(dim=1).clamp_min(1.0)
        case_quality = (
            deterministic_reliability.sum(dim=1) / present_count
        ).clamp(0.0, 1.0)
        return patient_feature, tooth_tokens, visual_gate, case_quality

    def forward(
        self,
        image: torch.Tensor,
        tooth_bboxes: Optional[torch.Tensor] = None,
        tooth_quality: Optional[torch.Tensor] = None,
        tooth_views: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # ``tooth_views`` is accepted only so the shared engine can keep one
        # signature.  Supplying legacy crops to a 3-D checkpoint is an error.
        if tooth_views is not None:
            raise ValueError(
                "MMDental3DROIModel does not consume legacy tooth_views; "
                "provide tooth_bboxes from the 3-D ROI cache"
            )
        patient_feature, tooth_tokens, visual_gate, case_quality = self.encode(
            image, tooth_bboxes, tooth_quality
        )
        patient_feature = self.dropout(patient_feature)
        output = {name: head(patient_feature) for name, head in self.heads.items()}

        adult_delta = self.max_tooth_delta * torch.tanh(
            self.tooth_slot_head(tooth_tokens).squeeze(-1) / self.max_tooth_delta
        )
        teeth_logits = output["teeth"].clone()
        teeth_logits[:, : self.num_adult_teeth] = (
            teeth_logits[:, : self.num_adult_teeth] + adult_delta
        )
        output["teeth"] = teeth_logits
        if self.use_tooth_pair_head:
            output["tooth_diagnosis"] = self.tooth_pair_head(tooth_tokens)
        output["tooth_visual_gate"] = visual_gate
        output["segmentation_quality"] = case_quality
        output["age"] = output["age"].squeeze(-1)
        output["text_embedding"] = F.normalize(output["text_embedding"], dim=-1)
        output["patient_feature"] = patient_feature
        return output


def normalize_model_architecture(value: Optional[str]) -> str:
    """Normalize user-facing aliases while keeping checkpoint tags stable."""
    normalized = str(value or LEGACY_ARCHITECTURE).strip().lower().replace("_", "-")
    if normalized in {
        "2p5d",
        "2.5d",
        "legacy",
        "multiview-2p5d-resnet18",
    }:
        return LEGACY_ARCHITECTURE
    if normalized in {
        "3d",
        "roi3d",
        "dental-roi-3d",
        "dental-roi-3d-v1",
    }:
        return DENTAL_ROI_3D_ARCHITECTURE
    raise ValueError("Unknown model architecture: {!r}".format(value))


def build_model_from_schema(
    schema: Any,
    imagenet_pretrained: bool = False,
    model_type: str = LEGACY_ARCHITECTURE,
    architecture: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    common = {
        "num_teeth": len(schema.tooth_labels),
        "num_diagnosis": len(schema.diagnosis_codes),
        "num_actions": len(schema.action_labels),
        "num_medications": len(schema.medication_labels),
        "text_dim": schema.text_dim,
    }
    selected = normalize_model_architecture(architecture or model_type)
    if selected == DENTAL_ROI_3D_ARCHITECTURE:
        parameters = dict(kwargs)
        if "tooth_transformer_layers" in parameters:
            parameters["num_transformer_layers"] = parameters.pop(
                "tooth_transformer_layers"
            )
        if "roi3d_base_channels" in parameters:
            parameters["base_channels"] = parameters.pop("roi3d_base_channels")
        # The 3-D encoder is trained from the challenge volumes; ImageNet is a
        # 2-D RGB pretraining source and is intentionally not applied here.
        return MMDental3DROIModel(**common, **parameters)
    return MMDentalModel(
        **common,
        imagenet_pretrained=imagenet_pretrained,
        **kwargs,
    )


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Instantiate either generation from its serialized ``model_config``."""
    parameters = dict(config)
    architecture = parameters.pop(
        "architecture", parameters.pop("model_type", LEGACY_ARCHITECTURE)
    )
    selected = normalize_model_architecture(str(architecture))
    if selected == DENTAL_ROI_3D_ARCHITECTURE:
        # Older inference code defensively forces this legacy flag to False.
        # It has no meaning for a Conv3d encoder, so tolerate and discard it.
        parameters.pop("imagenet_pretrained", None)
        return MMDental3DROIModel(**parameters)
    return MMDentalModel(**parameters)


def load_model_from_checkpoint(
    checkpoint_or_path: Union[Mapping[str, Any], str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load a legacy or 3-D model without the caller branching on its class."""
    if isinstance(checkpoint_or_path, (str, Path)):
        try:
            checkpoint = torch.load(
                checkpoint_or_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:  # torch < 2.0
            checkpoint = torch.load(checkpoint_or_path, map_location=map_location)
    else:
        checkpoint = dict(checkpoint_or_path)
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain model_config and model_state")
    model = build_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    return model
