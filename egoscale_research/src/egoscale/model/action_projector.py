from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from egoscale.config import ActionSemanticsSpec
from egoscale.model.embodiment_adapter import EmbodimentAdapter


class EmbodimentActionProjector(nn.Module):
    def __init__(self, action_dim: int, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        return self.network(noisy_actions)


class ActionProjector(nn.Module):
    def __init__(self, action_semantics: Mapping[str, ActionSemanticsSpec], d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.specs = {spec.embodiment_id: spec for spec in action_semantics.values()}
        self.adapter = EmbodimentAdapter(self.specs)
        self.projectors = nn.ModuleDict(
            {
                embodiment_id: EmbodimentActionProjector(spec.action_dim, d_model, hidden_dim)
                for embodiment_id, spec in self.specs.items()
            }
        )

    def encode(self, noisy_actions: torch.Tensor, embodiment_id: list[str]) -> torch.Tensor:
        batch_size, action_horizon, _ = noisy_actions.shape
        d_model = next(iter(self.projectors.values())).network[-1].out_features
        output = noisy_actions.new_zeros((batch_size, action_horizon, d_model))

        def apply_projection(name: str, indices: torch.Tensor) -> None:
            action_slice = noisy_actions.index_select(0, indices.to(noisy_actions.device))
            projected = self.projectors[name](action_slice)
            output.index_copy_(0, indices.to(output.device), projected)

        self.adapter.route_batch(embodiment_id, apply_projection)
        return output
