"""Loading Meta's released SAM 2.1 checkpoints into the reimplemented modules.

The checkpoint is a dict ``{"model": state_dict}``. Every reimplemented module keeps the
parameter names of its checkpoint prefix, so loading is a strict ``load_state_dict`` on the
prefix-stripped slice. Any mismatch is a bug in the reimplementation, never something to
paper over with ``strict=False``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def load_checkpoint(path: str | Path) -> dict[str, Tensor]:
    """Return the flat model state dict from a SAM 2.1 ``.pt`` file (weights only)."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(sd, dict):
        raise ValueError(f"unexpected checkpoint structure in {path}")
    return {str(k): v for k, v in sd.items()}


def slice_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Keys under ``prefix`` with the prefix removed (``prefix`` without trailing dot)."""
    p = prefix + "."
    return {k[len(p) :]: v for k, v in state_dict.items() if k.startswith(p)}


def load_prefix(module: nn.Module, state_dict: dict[str, Tensor], prefix: str) -> None:
    """Strictly load the ``prefix`` slice of ``state_dict`` into ``module``.

    Raises ``KeyError`` listing missing and unexpected keys so the diff is readable.
    """
    part = slice_prefix(state_dict, prefix)
    if not part:
        raise KeyError(f"no checkpoint keys under prefix {prefix!r}")
    result = module.load_state_dict(part, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise KeyError(
            f"state dict mismatch under {prefix!r}:\n"
            f"  missing ({len(result.missing_keys)}): {sorted(result.missing_keys)[:20]}\n"
            f"  unexpected ({len(result.unexpected_keys)}): "
            f"{sorted(result.unexpected_keys)[:20]}"
        )
    for name, param in module.state_dict().items():
        if param.shape != part[name].shape:  # load_state_dict already checks; belt and braces
            raise ValueError(f"{prefix}.{name}: shape {tuple(param.shape)} != checkpoint")
