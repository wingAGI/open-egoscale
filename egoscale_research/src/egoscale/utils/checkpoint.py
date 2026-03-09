from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: str = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("extra", {})
