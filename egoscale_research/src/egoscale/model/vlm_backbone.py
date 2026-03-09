from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import nn

from egoscale.config import ModelConfig


@dataclass
class VLMBackboneOutput:
    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    aux: Dict[str, torch.Tensor]


class BaseVLMBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def visual_encoder_group(self) -> nn.Module:
        raise NotImplementedError

    @property
    def multimodal_adapter_group(self) -> nn.Module:
        raise NotImplementedError

    @property
    def language_backbone_group(self) -> nn.Module:
        raise NotImplementedError


class DummyVLMBackbone(BaseVLMBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        hidden_dim = config.vlm_token_dim
        self.visual_encoder = nn.Sequential(nn.Conv2d(3, hidden_dim, kernel_size=1), nn.GELU())
        self.image_projector = nn.Linear(hidden_dim, hidden_dim)
        self.text_embedding = nn.Embedding(4096, hidden_dim)
        self.text_projector = nn.Linear(hidden_dim, hidden_dim)

    @property
    def visual_encoder_group(self) -> nn.Module:
        return self.visual_encoder

    @property
    def multimodal_adapter_group(self) -> nn.Module:
        return self.image_projector

    @property
    def language_backbone_group(self) -> nn.Module:
        return nn.Sequential(self.text_embedding, self.text_projector)

    def forward(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor,
        text: List[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> VLMBackboneOutput:
        batch_size, num_views, t_visual, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * num_views * t_visual, channels, height, width)
        encoded = self.visual_encoder(flat_images)
        encoded = encoded.mean(dim=(-2, -1))
        image_tokens = self.image_projector(encoded).reshape(
            batch_size,
            num_views * t_visual,
            self.config.vlm_token_dim,
        )
        image_tokens = image_tokens.repeat_interleave(self.config.dummy_tokens_per_view_step, dim=1)
        image_token_mask = image_mask.reshape(batch_size, num_views * t_visual)
        image_token_mask = image_token_mask.repeat_interleave(self.config.dummy_tokens_per_view_step, dim=1)

        text_tokens, text_mask = self._encode_text(text, device=images.device, dtype=images.dtype)
        context_tokens = torch.cat([image_tokens, text_tokens], dim=1)
        context_mask = torch.cat([image_token_mask, text_mask], dim=1)
        return VLMBackboneOutput(context_tokens=context_tokens, context_mask=context_mask, aux={})

    def _encode_text(self, text: List[str], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.zeros((len(text), self.config.dummy_text_tokens), dtype=torch.long, device=device)
        token_mask = torch.zeros((len(text), self.config.dummy_text_tokens), dtype=torch.bool, device=device)
        for row, sentence in enumerate(text):
            pieces = sentence.lower().split()[: self.config.dummy_text_tokens]
            if not pieces:
                continue
            token_mask[row, : len(pieces)] = True
            token_ids[row, : len(pieces)] = torch.tensor(
                [abs(hash(piece)) % 4096 for piece in pieces],
                dtype=torch.long,
                device=device,
            )
        embedded = self.text_embedding(token_ids)
        return self.text_projector(embedded).to(dtype=dtype), token_mask


class Qwen25VLBackbone(BaseVLMBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            from transformers import AutoProcessor, Qwen2_5_VLModel
        except ImportError as exc:
            raise ImportError("transformers with Qwen2.5-VL support is required for qwen2_5_vl backbone") from exc

        backbone_name = os.getenv("EGOSCALE_VLM_BACKBONE_NAME", config.vlm_backbone_name)
        config.vlm_backbone_name = backbone_name
        load_kwargs = {"local_files_only": True} if Path(backbone_name).exists() else {}
        self.processor = AutoProcessor.from_pretrained(
            backbone_name,
            min_pixels=config.qwen_min_pixels,
            max_pixels=config.qwen_max_pixels,
            **load_kwargs,
        )
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLModel.from_pretrained(backbone_name, torch_dtype=model_dtype, **load_kwargs)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is not None and config.vlm_token_dim != hidden_size:
            warnings.warn(
                f"Overriding vlm_token_dim from {config.vlm_token_dim} to Qwen hidden_size {hidden_size}.",
                stacklevel=2,
            )
            config.vlm_token_dim = int(hidden_size)
        self.visual_encoder = self._resolve_attr(config.vision_encoder_attr)
        multimodal_modules = [self._resolve_attr(path) for path in config.multimodal_adapter_attrs]
        self.multimodal_adapter = nn.ModuleList(multimodal_modules)
        self.language_backbone = self._resolve_attr(config.language_backbone_attr)
        vision_config = getattr(self.model.config, "vision_config", None)
        self.image_token_divisor = int(getattr(vision_config, "spatial_merge_size", 1)) ** 2

    @property
    def visual_encoder_group(self) -> nn.Module:
        return self.visual_encoder

    @property
    def multimodal_adapter_group(self) -> nn.Module:
        return self.multimodal_adapter

    @property
    def language_backbone_group(self) -> nn.Module:
        return self.language_backbone

    def forward(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor,
        text: List[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> VLMBackboneOutput:
        device = device or next(self.model.parameters()).device
        messages = self._build_messages(images, text)
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
        model_dtype = next(self.model.parameters()).dtype
        normalized_inputs: Dict[str, torch.Tensor | object] = {}
        for key, value in model_inputs.items():
            if not torch.is_tensor(value):
                normalized_inputs[key] = value
                continue
            if value.is_floating_point():
                normalized_inputs[key] = value.to(device=device, dtype=model_dtype)
            else:
                normalized_inputs[key] = value.to(device=device)
        model_inputs = normalized_inputs
        outputs = self.model(**model_inputs, output_hidden_states=True, return_dict=True)
        context_tokens = outputs.last_hidden_state
        if dtype is not None:
            context_tokens = context_tokens.to(dtype=dtype)
        context_mask = self._build_context_mask(
            attention_mask=model_inputs["attention_mask"].to(dtype=torch.bool),
            mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            image_mask=image_mask.to(device=device),
        )
        return VLMBackboneOutput(context_tokens=context_tokens, context_mask=context_mask, aux=model_inputs)

    def _resolve_attr(self, path: str) -> nn.Module:
        current = self.model
        for name in path.split("."):
            if not hasattr(current, name) and current is self.model and name == "model" and hasattr(current, "language_model"):
                name = "language_model"
            current = getattr(current, name)
        return current

    def _build_messages(self, images: torch.Tensor, text: List[str]) -> List[Dict[str, object]]:
        batch_size, num_views, t_visual = images.shape[:3]
        messages = []
        for batch_index in range(batch_size):
            content: List[Dict[str, object]] = []
            for slot_index in range(num_views * t_visual):
                view_index = slot_index // t_visual
                time_index = slot_index % t_visual
                pil_image = _tensor_to_pil(images[batch_index, view_index, time_index])
                content.append({"type": "image", "image": pil_image})
            content.append({"type": "text", "text": text[batch_index]})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _build_context_mask(
        self,
        attention_mask: torch.Tensor,
        mm_token_type_ids: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        context_mask = attention_mask.clone()
        if mm_token_type_ids is None or image_grid_thw is None:
            return context_mask

        batch_size = image_mask.shape[0]
        flat_slots = image_mask.reshape(batch_size, -1)
        token_counts = torch.div(
            image_grid_thw.prod(dim=-1),
            self.image_token_divisor,
            rounding_mode="floor",
        ).tolist()
        expected_images = flat_slots.numel()
        if len(token_counts) != expected_images:
            return context_mask

        token_counts_by_batch = [
            token_counts[row * flat_slots.shape[1] : (row + 1) * flat_slots.shape[1]]
            for row in range(batch_size)
        ]

        for batch_index in range(batch_size):
            image_positions = (mm_token_type_ids[batch_index] == 1).nonzero(as_tuple=False).flatten()
            cursor = 0
            for slot_index, valid in enumerate(flat_slots[batch_index].tolist()):
                count = int(token_counts_by_batch[batch_index][slot_index])
                span = image_positions[cursor : cursor + count]
                if len(span) != count:
                    return attention_mask
                if not valid:
                    context_mask[batch_index, span] = False
                cursor += count
        return context_mask


def build_vlm_backbone(config: ModelConfig) -> BaseVLMBackbone:
    if config.backbone_impl == "dummy":
        return DummyVLMBackbone(config)
    if config.backbone_impl == "qwen2_5_vl":
        return Qwen25VLBackbone(config)
    raise ValueError("Unsupported backbone_impl")


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 255)
    if image.dtype != torch.uint8:
        image = image.to(dtype=torch.uint8)
    array = image.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array)
