from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrainingHistory:
    train: List[Dict[str, float]] = field(default_factory=list)
    val: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if self.train:
            payload["last_train_loss"] = self.train[-1]["loss"]
            payload["last_train_step"] = self.train[-1]["step"]
        if self.val:
            payload["last_val_loss"] = self.val[-1]["loss"]
            payload["last_val_step"] = self.val[-1]["step"]
            payload["best_val_loss"] = min(item["loss"] for item in self.val)
        return payload


class WandbLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str,
        run_name: str,
        mode: str,
        config: Dict[str, object],
    ) -> None:
        self.enabled = enabled and bool(project)
        self._run = None
        if not self.enabled:
            return
        import wandb

        init_kwargs = {
            "project": project,
            "config": config,
            "mode": mode,
        }
        if entity:
            init_kwargs["entity"] = entity
        if run_name:
            init_kwargs["name"] = run_name
        self._run = wandb.init(**init_kwargs)

    def log(self, payload: Dict[str, float]) -> None:
        if self._run is None:
            return
        self._run.log(payload)

    @property
    def active(self) -> bool:
        return self._run is not None

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None
