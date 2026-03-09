from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from egoscale.config import ExperimentConfig
from egoscale.data.transforms import AffineNormalizer
from egoscale.model.action_decoder import ActionDecoder
from egoscale.model.action_head import FlowMatchingActionHead
from egoscale.model.action_projector import ActionProjector
from egoscale.model.state_projector import StateProjector
from egoscale.model.vlm_backbone import build_vlm_backbone


class EgoScalePolicy(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.vlm_backbone = build_vlm_backbone(config.model)
        self.state_projector = StateProjector(config.state_semantics, d_model=config.model.vlm_token_dim)
        self.action_projector = ActionProjector(config.action_semantics, d_model=config.model.vlm_token_dim)
        self.action_head = FlowMatchingActionHead(
            d_model=config.model.vlm_token_dim,
            num_heads=config.model.num_attention_heads,
            num_layers=config.model.num_dit_layers,
            num_timestep_buckets=config.model.num_timestep_buckets,
            mlp_ratio=config.model.mlp_ratio,
        )
        self.action_decoder = ActionDecoder(config.action_semantics, d_model=config.model.vlm_token_dim)
        self.max_action_dim = max(spec.action_dim for spec in config.action_semantics.values())

    def forward(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor,
        text: List[str],
        raw_state: torch.Tensor,
        has_proprio: torch.Tensor,
        actions: torch.Tensor,
        embodiment_id: List[str],
    ) -> Dict[str, torch.Tensor]:
        backbone_output = self.vlm_backbone(images=images, image_mask=image_mask, text=text, device=actions.device, dtype=actions.dtype)
        state_tokens = self.state_projector(raw_state=raw_state, has_proprio=has_proprio, embodiment_id=embodiment_id)
        batch_size = actions.shape[0]
        noise = torch.randn_like(actions)
        time = torch.rand(batch_size, device=actions.device, dtype=actions.dtype)
        noisy_actions = (1.0 - time[:, None, None]) * noise + time[:, None, None] * actions
        target_velocity = actions - noise
        taus = torch.clamp((time * self.config.model.num_timestep_buckets).floor().long(), max=self.config.model.num_timestep_buckets - 1)
        action_tokens = self.action_projector.encode(noisy_actions, embodiment_id)
        pred_action_latents = self.action_head(
            context_tokens=backbone_output.context_tokens,
            context_mask=backbone_output.context_mask,
            state_tokens=state_tokens,
            action_tokens=action_tokens,
            timesteps=taus,
        )
        pred_velocities = self.action_decoder.decode(pred_action_latents, embodiment_id)[..., : actions.shape[-1]]
        loss = F.mse_loss(pred_velocities, target_velocity)
        return {
            "loss": loss,
            "context_tokens": backbone_output.context_tokens,
            "context_mask": backbone_output.context_mask.to(dtype=torch.float32),
            "state_tokens": state_tokens,
            "pred_action_latents": pred_action_latents,
            "pred_velocities": pred_velocities,
            "target_velocity": target_velocity,
            "taus": taus,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor,
        text: List[str],
        raw_state: torch.Tensor,
        has_proprio: torch.Tensor,
        embodiment_id: List[str],
        action_dim: int,
        action_normalizer: Optional[AffineNormalizer] = None,
    ) -> torch.Tensor:
        backbone_output = self.vlm_backbone(images=images, image_mask=image_mask, text=text, device=raw_state.device, dtype=raw_state.dtype)
        state_tokens = self.state_projector(raw_state=raw_state, has_proprio=has_proprio, embodiment_id=embodiment_id)
        batch_size = raw_state.shape[0]
        actions = torch.randn(
            batch_size,
            self.config.model.action_horizon,
            action_dim,
            device=raw_state.device,
            dtype=raw_state.dtype,
        )
        dt = 1.0 / float(self.config.model.num_inference_timesteps)
        for step in range(self.config.model.num_inference_timesteps):
            current_time = step / float(self.config.model.num_inference_timesteps)
            tau = int(current_time * self.config.model.num_timestep_buckets)
            timestep = torch.full((batch_size,), tau, device=raw_state.device, dtype=torch.long)
            action_tokens = self.action_projector.encode(actions, embodiment_id)
            pred_action_latents = self.action_head(
                context_tokens=backbone_output.context_tokens,
                context_mask=backbone_output.context_mask,
                state_tokens=state_tokens,
                action_tokens=action_tokens,
                timesteps=timestep,
            )
            pred_velocity = self.action_decoder.decode(pred_action_latents, embodiment_id)[..., :action_dim]
            actions = actions + dt * pred_velocity
        return action_normalizer.inverse_transform(actions) if action_normalizer is not None else actions

    def module_groups(self) -> Dict[str, nn.Module]:
        return {
            "vlm_visual_encoder": self.vlm_backbone.visual_encoder_group,
            "vlm_multimodal_adapter": self.vlm_backbone.multimodal_adapter_group,
            "vlm_language_backbone": self.vlm_backbone.language_backbone_group,
            "state_projector_bundle": self.state_projector,
            "action_projector": self.action_projector,
            "flow_matching_action_head": self.action_head,
            "action_decoder": self.action_decoder,
        }
