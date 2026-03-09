from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor, context_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        residual = query_tokens
        query_tokens = self.self_norm(query_tokens)
        query_tokens = residual + self.self_attn(query_tokens, query_tokens, query_tokens, need_weights=False)[0]

        residual = query_tokens
        normalized = self.cross_norm(query_tokens)
        query_tokens = residual + self.cross_attn(
            normalized,
            context_tokens,
            context_tokens,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )[0]

        residual = query_tokens
        query_tokens = residual + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class FlowMatchingActionHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, num_timestep_buckets: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.num_timestep_buckets = num_timestep_buckets
        self.timestep_embedding = nn.Embedding(num_timestep_buckets, d_model)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = action_tokens.shape[0]
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(batch_size)
        if timesteps.dim() != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [B]")
        timesteps = timesteps.clamp_(0, self.num_timestep_buckets - 1).long()

        query_tokens = torch.cat([state_tokens, action_tokens], dim=1)
        query_tokens = query_tokens + self.timestep_embedding(timesteps).unsqueeze(1)
        key_padding_mask = ~context_mask.to(dtype=torch.bool)
        for block in self.blocks:
            query_tokens = block(query_tokens, context_tokens, key_padding_mask)
        query_tokens = self.output_norm(query_tokens)
        return query_tokens[:, -action_tokens.shape[1] :, :]
