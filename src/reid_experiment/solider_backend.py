"""Official SOLIDER Swin-Base inference wrapper for the isolated A/B test.

The implementation imports the pinned upstream SOLIDER-REID source without
modifying it. It intentionally has no dependency on NvDCF, MV3DT, MQTT, or the
production gallery. A caller supplies person crops and receives L2-normalized
1024-D global features.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "vendor/reid_experiment/SOLIDER-REID"
DEFAULT_CHECKPOINT = ROOT / "models/reid_experiment/solider_swin_base_msmt17.pth"


class SoliderBackend:
    """Lazy PyTorch backend using the official Swin-Base MSMT17 checkpoint."""

    input_height = 384
    input_width = 128
    embedding_dim = 1024

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = "cuda"):
        try:
            import numpy as np
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "SOLIDER experimental dependencies are missing: install the isolated "
                "PyTorch environment; production DeepStream dependencies were not changed."
            ) from error
        if not UPSTREAM.exists():
            raise FileNotFoundError(f"pinned SOLIDER source is missing: {UPSTREAM}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"SOLIDER checkpoint is missing: {checkpoint}")
        self.np = np
        self.torch = torch
        self.Image = Image
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        sys.path.insert(0, str(UPSTREAM))
        from .compat import install_legacy_shims
        install_legacy_shims()
        from config import cfg
        from model import make_model

        cfg.defrost()
        cfg.MODEL.NAME = "transformer"
        cfg.MODEL.TRANSFORMER_TYPE = "swin_base_patch4_window7_224"
        cfg.MODEL.JPM = False
        cfg.MODEL.SEMANTIC_WEIGHT = 0.2
        cfg.INPUT.SIZE_TRAIN = [self.input_height, self.input_width]
        cfg.INPUT.SIZE_TEST = [self.input_height, self.input_width]
        cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
        cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
        cfg.TEST.NECK_FEAT = "before"
        cfg.TEST.FEAT_NORM = "yes"
        cfg.freeze()
        self.config = cfg
        self.model = make_model(cfg, num_class=1041, camera_num=0, view_num=0,
                                semantic_weight=0.2)
        self.model.load_param(str(checkpoint))
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.input_height, self.input_width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _tensor(self, crop_bgr):
        rgb = self.np.asarray(crop_bgr)[:, :, ::-1]
        return self.transform(self.Image.fromarray(rgb)).unsqueeze(0).to(self.device)

    @staticmethod
    def _normalize(features):
        return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def embed(self, crop_bgr) -> list[float]:
        started = time.perf_counter()
        with self.torch.inference_mode():
            feature = self.model(self._tensor(crop_bgr))
            # The pinned SOLIDER model returns (feature, intermediate feature
            # maps) during evaluation; only the global feature is used here.
            if isinstance(feature, tuple):
                feature = feature[0]
            feature = self._normalize(feature).squeeze(0).detach().float().cpu().numpy()
        if feature.shape != (self.embedding_dim,) or not self.np.isfinite(feature).all():
            raise RuntimeError(f"unexpected SOLIDER feature shape/content: {feature.shape}")
        return feature.tolist()

    def embed_batch(self, crops_bgr: Iterable) -> tuple[list[list[float]], float]:
        crops = list(crops_bgr)
        if not crops:
            return [], 0.0
        started = time.perf_counter()
        with self.torch.inference_mode():
            batch = self.torch.cat([self._tensor(crop) for crop in crops], dim=0)
            features = self.model(batch)
            if isinstance(features, tuple):
                features = features[0]
            features = self._normalize(features).detach().float().cpu().numpy()
        if features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise RuntimeError(f"unexpected SOLIDER batch feature shape: {features.shape}")
        return features.tolist(), (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate isolated SOLIDER checkpoint loading")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    backend = SoliderBackend(args.checkpoint.resolve(), args.device)
    print(json.dumps({
        "backend": "pytorch",
        "checkpoint": str(args.checkpoint.resolve()),
        "input_size": [backend.input_height, backend.input_width],
        "embedding_dim": backend.embedding_dim,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "distance": "cosine on L2-normalized features",
        "device": str(backend.device),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
