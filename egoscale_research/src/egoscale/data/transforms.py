from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch

from egoscale.config import ActionSemanticsSpec, DataConfig, NormalizationSpec, StateSemanticsSpec
from egoscale.data.schema import EgoScaleSample


@dataclass
class AffineNormalizer:
    spec: NormalizationSpec

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        if self.spec.scheme == "none":
            return value
        if self.spec.scheme == "meanstd":
            mean = _expand_stats(self.spec.mean, value)
            std = _expand_stats(self.spec.std or [1.0] * value.shape[-1], value)
            return (value - mean) / torch.clamp(std, min=1e-6)
        if self.spec.scheme == "minmax":
            minimum = _expand_stats(self.spec.min, value)
            maximum = _expand_stats(self.spec.max, value)
            denom = torch.clamp(maximum - minimum, min=1e-6)
            return 2.0 * (value - minimum) / denom - 1.0
        raise ValueError("Unsupported normalization scheme")

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        if self.spec.scheme == "none":
            return value
        if self.spec.scheme == "meanstd":
            mean = _expand_stats(self.spec.mean, value)
            std = _expand_stats(self.spec.std or [1.0] * value.shape[-1], value)
            return value * std + mean
        if self.spec.scheme == "minmax":
            minimum = _expand_stats(self.spec.min, value)
            maximum = _expand_stats(self.spec.max, value)
            return 0.5 * (value + 1.0) * (maximum - minimum) + minimum
        raise ValueError("Unsupported normalization scheme")


class EgoScaleTransforms:
    def __init__(
        self,
        data_config: DataConfig,
        state_semantics: Mapping[str, StateSemanticsSpec],
        action_semantics: Mapping[str, ActionSemanticsSpec],
    ) -> None:
        self.data_config = data_config
        self.state_semantics = state_semantics
        self.action_semantics = action_semantics
        self.state_normalizers = {
            name: AffineNormalizer(spec.normalization) for name, spec in state_semantics.items()
        }
        self.action_normalizers = {
            name: AffineNormalizer(spec.normalization) for name, spec in action_semantics.items()
        }

    def __call__(self, sample: EgoScaleSample) -> EgoScaleSample:
        sample = self._materialize_views(sample)
        sample = self._normalize_state(sample)
        sample = self._normalize_actions(sample)
        return sample

    def action_normalizer(self, action_semantics_name: str) -> AffineNormalizer:
        return self.action_normalizers[action_semantics_name]

    def _materialize_views(self, sample: EgoScaleSample) -> EgoScaleSample:
        if sample.images is not None and sample.image_mask is not None:
            return sample
        if sample.view_images is None:
            raise ValueError("Sample must provide either materialized images or view_images")

        slots = []
        masks = []
        for view_name in self.data_config.canonical_view_order:
            if view_name in sample.view_images:
                frames = sample.view_images[view_name].to(dtype=torch.float32)
                if frames.shape[0] != sample.obs_horizon:
                    raise ValueError("Each view must carry T_visual = obs_horizon frames")
                slots.append(frames)
                masks.append(torch.ones(sample.obs_horizon, dtype=torch.bool))
            else:
                reference = next(iter(sample.view_images.values()))
                dummy = torch.zeros_like(reference, dtype=torch.float32)
                if dummy.shape[0] != sample.obs_horizon:
                    raise ValueError("Dummy slots must align with obs_horizon")
                slots.append(dummy)
                masks.append(torch.zeros(sample.obs_horizon, dtype=torch.bool))
        return sample.replace(images=torch.stack(slots, dim=0), image_mask=torch.stack(masks, dim=0))

    def _normalize_state(self, sample: EgoScaleSample) -> EgoScaleSample:
        if not sample.has_proprio:
            zero_state = torch.zeros_like(sample.raw_state, dtype=torch.float32)
            if not torch.equal(sample.raw_state.to(dtype=torch.float32), zero_state):
                raise ValueError("Placeholder raw_state must remain an exact zero vector")
            return sample.replace(raw_state=zero_state)
        normalizer = self.state_normalizers[sample.state_semantics_name]
        return sample.replace(raw_state=normalizer.transform(sample.raw_state.to(dtype=torch.float32)))

    def _normalize_actions(self, sample: EgoScaleSample) -> EgoScaleSample:
        normalizer = self.action_normalizers[sample.action_semantics_name]
        return sample.replace(actions=normalizer.transform(sample.actions.to(dtype=torch.float32)))


def _expand_stats(values: Iterable[float], target: torch.Tensor) -> torch.Tensor:
    if not list(values):
        return torch.zeros(target.shape[-1], device=target.device, dtype=target.dtype)
    stats = torch.tensor(list(values), device=target.device, dtype=target.dtype)
    while stats.dim() < target.dim():
        stats = stats.unsqueeze(0)
    return stats
