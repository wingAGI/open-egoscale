from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from egoscale.data.schema import EgoScaleSample


class EgoScaleCollator:
    def __call__(self, samples: Sequence[EgoScaleSample]) -> Dict[str, object]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        first = samples[0]
        for sample in samples[1:]:
            if sample.bucket_key != first.bucket_key:
                raise ValueError("Single batch must contain exactly one bucket key")
            if sample.state_dim != first.state_dim:
                raise ValueError("Single batch must keep state_dim constant")
            if sample.action_dim != first.action_dim:
                raise ValueError("Single batch must keep action_dim constant")
            if sample.obs_horizon != first.obs_horizon or sample.action_horizon != first.action_horizon:
                raise ValueError("Single batch must keep obs_horizon and action_horizon constant")
        if first.images is None or first.image_mask is None:
            raise ValueError("Transforms must materialize images and image_mask before collate")

        return {
            "images": torch.stack([sample.images for sample in samples], dim=0),
            "image_mask": torch.stack([sample.image_mask for sample in samples], dim=0),
            "text": [sample.instruction for sample in samples],
            "raw_state": torch.stack([sample.raw_state for sample in samples], dim=0).to(dtype=torch.float32),
            "has_proprio": torch.tensor([sample.has_proprio for sample in samples], dtype=torch.bool),
            "actions": torch.stack([sample.actions for sample in samples], dim=0).to(dtype=torch.float32),
            "embodiment_id": [sample.embodiment_id for sample in samples],
            "action_semantics_name": [sample.action_semantics_name for sample in samples],
            "state_semantics_name": [sample.state_semantics_name for sample in samples],
            "bucket_key": first.bucket_key.as_string(),
        }
