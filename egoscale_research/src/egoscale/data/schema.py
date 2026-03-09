from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional

import torch

from egoscale.config import StageRecipe


@dataclass(frozen=True)
class BatchBucketKey:
    embodiment_id: str
    state_semantics_name: str
    action_semantics_name: str
    has_proprio: bool

    def as_string(self) -> str:
        return "|".join(
            [self.embodiment_id, self.state_semantics_name, self.action_semantics_name, str(self.has_proprio)]
        )


@dataclass(frozen=True)
class EgoScaleSample:
    instruction: str
    embodiment_id: str
    data_source: str
    has_proprio: bool
    task_id: str
    state_semantics_name: str
    action_semantics_name: str
    state_dim: int
    action_dim: int
    raw_state: torch.Tensor
    actions: torch.Tensor
    obs_timestamps: List[float]
    action_timestamps: List[float]
    obs_horizon: int
    action_horizon: int
    camera_views: List[str]
    images: Optional[torch.Tensor] = None
    image_mask: Optional[torch.Tensor] = None
    view_images: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EgoScaleSample":
        view_images = raw.get("view_images")
        coerced_views = None
        if view_images is not None:
            coerced_views = {key: _to_tensor(value) for key, value in view_images.items()}
        return cls(
            instruction=str(raw["instruction"]),
            embodiment_id=str(raw["embodiment_id"]),
            data_source=str(raw["data_source"]),
            has_proprio=bool(raw["has_proprio"]),
            task_id=str(raw["task_id"]),
            state_semantics_name=str(raw["state_semantics_name"]),
            action_semantics_name=str(raw["action_semantics_name"]),
            state_dim=int(raw["state_dim"]),
            action_dim=int(raw["action_dim"]),
            raw_state=_to_tensor(raw["raw_state"]).to(dtype=torch.float32),
            actions=_to_tensor(raw["actions"]).to(dtype=torch.float32),
            obs_timestamps=list(raw["obs_timestamps"]),
            action_timestamps=list(raw["action_timestamps"]),
            obs_horizon=int(raw["obs_horizon"]),
            action_horizon=int(raw["action_horizon"]),
            camera_views=list(raw.get("camera_views", [])),
            images=_optional_tensor(raw.get("images")),
            image_mask=_optional_tensor(raw.get("image_mask"), dtype=torch.bool),
            view_images=coerced_views,
        )

    @property
    def bucket_key(self) -> BatchBucketKey:
        return BatchBucketKey(
            embodiment_id=self.embodiment_id,
            state_semantics_name=self.state_semantics_name,
            action_semantics_name=self.action_semantics_name,
            has_proprio=self.has_proprio,
        )

    def replace(self, **changes: Any) -> "EgoScaleSample":
        return replace(self, **changes)


def validate_stage_sample(stage_recipe: StageRecipe, sample: EgoScaleSample, stage3_allow_aligned_human: bool) -> None:
    if sample.action_dim != sample.actions.shape[-1]:
        raise ValueError("action_dim must match actions.shape[-1]")
    if sample.state_dim != sample.raw_state.shape[-1]:
        raise ValueError("state_dim must match raw_state.shape[-1]")
    if sample.actions.shape[0] != sample.action_horizon:
        raise ValueError("action_horizon must match actions.shape[0]")
    if len(sample.obs_timestamps) != sample.obs_horizon:
        raise ValueError("obs_timestamps must match obs_horizon")
    if len(sample.action_timestamps) != sample.action_horizon:
        raise ValueError("action_timestamps must match action_horizon")
    if sample.obs_timestamps[-1] != sample.action_timestamps[0]:
        raise ValueError("obs_timestamps[-1] must align with action_timestamps[0]")

    if stage_recipe == "stage1":
        allowed_views = {
            ("head",),
            ("head", "left_wrist"),
            ("head", "right_wrist"),
            ("head", "left_wrist", "right_wrist"),
        }
        if sample.embodiment_id != "sharpa" or sample.data_source != "human_retargeted" or sample.has_proprio:
            raise ValueError("Stage I only allows sharpa human-retargeted samples without proprio")
        if sample.state_semantics_name != "sharpa_proprio_v1" or sample.action_semantics_name != "sharpa_wristdelta_hand22_v1":
            raise ValueError("Stage I semantics must match sharpa defaults")
        if sample.action_dim != 28:
            raise ValueError("Stage I action_dim must be 28")
        if tuple(sample.camera_views) not in allowed_views:
            raise ValueError("Stage I camera views must follow the canonical subsets")
    elif stage_recipe == "stage2":
        allowed = {
            ("sharpa", "human_retargeted", False),
            ("sharpa", "robot_native", True),
            ("g1", "robot_native", True),
        }
        if (sample.embodiment_id, sample.data_source, sample.has_proprio) not in allowed:
            raise ValueError("Stage II sample shape is not allowed by spec")
    elif stage_recipe == "stage3":
        if sample.data_source == "human_retargeted":
            if not stage3_allow_aligned_human:
                raise ValueError("Stage III aligned human is disabled")
            if sample.embodiment_id != "sharpa" or sample.has_proprio:
                raise ValueError("Stage III aligned human must be sharpa without proprio")
        else:
            if sample.data_source != "robot_native" or not sample.has_proprio:
                raise ValueError("Stage III robot buckets must be robot_native with proprio")
    else:
        raise ValueError("Unknown stage_recipe")


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def _optional_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = _to_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor
