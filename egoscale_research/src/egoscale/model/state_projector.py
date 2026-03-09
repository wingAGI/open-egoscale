from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from egoscale.config import StateSemanticsSpec
from egoscale.model.embodiment_adapter import EmbodimentAdapter


class EmbodimentStateProjector(nn.Module):
    def __init__(self, state_dim: int, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.placeholder_adapter = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.proprio_adapter = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.shared_state_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, raw_state: torch.Tensor, has_proprio: torch.Tensor) -> torch.Tensor:
        placeholder_features = self.placeholder_adapter(raw_state)
        proprio_features = self.proprio_adapter(raw_state)
        selector = has_proprio.to(dtype=raw_state.dtype).unsqueeze(-1)
        mixed = placeholder_features * (1.0 - selector) + proprio_features * selector
        return self.shared_state_trunk(mixed).unsqueeze(1)


class StateProjector(nn.Module):
    def __init__(self, state_semantics: Mapping[str, StateSemanticsSpec], d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.specs = {spec.embodiment_id: spec for spec in state_semantics.values()}
        self.adapter = EmbodimentAdapter(self.specs)
        self.projectors = nn.ModuleDict(
            {
                embodiment_id: EmbodimentStateProjector(spec.state_dim, d_model, hidden_dim)
                for embodiment_id, spec in self.specs.items()
            }
        )

    def forward(self, raw_state: torch.Tensor, has_proprio: torch.Tensor, embodiment_id: list[str]) -> torch.Tensor:
        batch_size = raw_state.shape[0]
        output = raw_state.new_zeros((batch_size, 1, next(iter(self.projectors.values())).shared_state_trunk[-1].out_features))

        def apply_projection(name: str, indices: torch.Tensor) -> None:
            state_slice = raw_state.index_select(0, indices.to(raw_state.device))
            proprio_slice = has_proprio.index_select(0, indices.to(has_proprio.device))
            projected = self.projectors[name](state_slice, proprio_slice)
            output.index_copy_(0, indices.to(output.device), projected)

        self.adapter.route_batch(embodiment_id, apply_projection)
        return output

    def set_placeholder_trainable(self, trainable: bool) -> None:
        for projector in self.projectors.values():
            projector.placeholder_adapter.requires_grad_(trainable)

    def set_proprio_trainable(self, trainable: bool) -> None:
        for projector in self.projectors.values():
            projector.proprio_adapter.requires_grad_(trainable)

    def set_shared_trainable(self, trainable: bool) -> None:
        for projector in self.projectors.values():
            projector.shared_state_trunk.requires_grad_(trainable)
