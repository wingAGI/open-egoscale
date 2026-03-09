from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from egoscale.config import ActionSemanticsSpec
from egoscale.model.embodiment_adapter import EmbodimentAdapter


class EmbodimentActionDecoder(nn.Module):
    def __init__(self, d_model: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, pred_action_latents: torch.Tensor) -> torch.Tensor:
        return self.network(pred_action_latents)


class ActionDecoder(nn.Module):
    def __init__(self, action_semantics: Mapping[str, ActionSemanticsSpec], d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.specs = {spec.embodiment_id: spec for spec in action_semantics.values()}
        self.adapter = EmbodimentAdapter(self.specs)
        self.decoders = nn.ModuleDict(
            {
                embodiment_id: EmbodimentActionDecoder(d_model, spec.action_dim, hidden_dim)
                for embodiment_id, spec in self.specs.items()
            }
        )

    def decode(self, pred_action_latents: torch.Tensor, embodiment_id: list[str]) -> torch.Tensor:
        batch_size, action_horizon, _ = pred_action_latents.shape
        max_action_dim = max(spec.action_dim for spec in self.specs.values())
        output = pred_action_latents.new_zeros((batch_size, action_horizon, max_action_dim))

        def apply_projection(name: str, indices: torch.Tensor) -> None:
            latents = pred_action_latents.index_select(0, indices.to(pred_action_latents.device))
            decoded = self.decoders[name](latents)
            target = output.index_select(0, indices.to(output.device))
            target[..., : decoded.shape[-1]] = decoded
            output.index_copy_(0, indices.to(output.device), target)

        self.adapter.route_batch(embodiment_id, apply_projection)
        return output
