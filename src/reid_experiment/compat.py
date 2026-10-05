"""Runtime shims for the pinned SOLIDER code on current PyTorch/MMCV.

These modules are injected only immediately before importing SOLIDER.  The
upstream checkout remains unmodified and production DeepStream does not load
this module.
"""

from __future__ import annotations

import collections.abc
import sys
import types


def install_legacy_shims() -> None:
    """Provide the small legacy APIs imported by SOLIDER's upstream source."""
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.container_abcs = collections.abc
        torch_six.string_classes = (str,)
        torch_six.int_classes = (int,)
        sys.modules["torch._six"] = torch_six

    # SOLIDER imports mmcv.runner.load_checkpoint only for legacy Swin code;
    # the checkpoint itself is loaded by SOLIDER's model.load_param().
    if "mmcv.runner" not in sys.modules:
        runner = types.ModuleType("mmcv.runner")

        def load_checkpoint(model, filename, *args, **kwargs):
            import torch

            checkpoint = torch.load(filename, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(state_dict, dict):
                state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            return checkpoint

        runner.load_checkpoint = load_checkpoint
        sys.modules["mmcv.runner"] = runner
