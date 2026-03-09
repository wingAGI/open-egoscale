from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import yaml

StageRecipe = Literal["stage1", "stage2", "stage3"]

DEFAULT_CANONICAL_VIEW_ORDER = ["head", "left_wrist", "right_wrist"]


@dataclass
class NormalizationSpec:
    scheme: Literal["none", "meanstd", "minmax"] = "none"
    stats_id: str = "identity"
    stats_scope: str = "train_split_only"
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    min: List[float] = field(default_factory=list)
    max: List[float] = field(default_factory=list)


@dataclass
class StateSemanticsSpec:
    name: str
    embodiment_id: str
    state_dim: int
    field_order: List[str]
    units: List[str]
    normalization: NormalizationSpec


@dataclass
class ActionSemanticsSpec:
    name: str
    embodiment_id: str
    action_dim: int
    control_frequency_hz: float
    normalization: NormalizationSpec
    translation_frame: str = "base"
    rotation_frame: str = "base"
    rotation_parameterization: str = "axis_angle"
    translation_unit: str = "meter"
    rotation_unit: str = "radian"
    hand_joint_order: List[str] = field(default_factory=list)


@dataclass
class TrainableModuleGroups:
    vlm_visual_encoder: bool = True
    vlm_multimodal_adapter: bool = True
    vlm_language_backbone: bool = True
    state_projector_bundle: bool = True
    action_projector: bool = True
    flow_matching_action_head: bool = True
    action_decoder: bool = True


@dataclass
class ModelConfig:
    vlm_backbone_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    backbone_impl: Literal["dummy", "qwen2_5_vl"] = "dummy"
    vlm_token_dim: int = 2048
    vlm_max_views: int = 3
    obs_horizon: int = 2
    action_horizon: int = 8
    num_inference_timesteps: int = 8
    num_timestep_buckets: int = 1000
    num_attention_heads: int = 8
    num_dit_layers: int = 4
    mlp_ratio: int = 4
    dummy_tokens_per_view_step: int = 2
    dummy_text_tokens: int = 8
    qwen_min_pixels: int = 256 * 28 * 28
    qwen_max_pixels: int = 1024 * 28 * 28
    vision_encoder_attr: str = "visual"
    multimodal_adapter_attrs: List[str] = field(default_factory=lambda: ["visual.merger"])
    language_backbone_attr: str = "language_model"
    trainable_module_groups: TrainableModuleGroups = field(default_factory=TrainableModuleGroups)


@dataclass
class DataConfig:
    canonical_view_order: List[str] = field(default_factory=lambda: list(DEFAULT_CANONICAL_VIEW_ORDER))
    obs_horizon: int = 2
    action_horizon: int = 8
    sample_stride: int = 1
    bucket_sampling_policy: Literal["proportional_to_active_samples", "manual_weights"] = (
        "proportional_to_active_samples"
    )
    bucket_sampling_weights: Dict[str, float] = field(default_factory=dict)
    embodiment_buckets: List[str] = field(default_factory=lambda: ["sharpa", "g1"])
    data_source_buckets: List[str] = field(default_factory=lambda: ["human_retargeted", "robot_native"])
    allowed_action_semantics_names: List[str] = field(
        default_factory=lambda: ["sharpa_wristdelta_hand22_v1", "g1_lowdim_native_v1"]
    )
    allowed_state_semantics_names: List[str] = field(
        default_factory=lambda: ["sharpa_proprio_v1", "g1_proprio_v1"]
    )
    stage3_allow_aligned_human: bool = False
    stage_sample_rules: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    stage_recipe: StageRecipe = "stage1"
    batch_size: int = 2
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    max_steps: int = 1000
    eval_interval: int = 0
    log_interval: int = 1
    max_val_batches: int = 0
    weight_decay: float = 0.01
    device: str = "cuda"
    seed: int = 7
    run_g1_path: bool = True
    state_adapter_mode: str = "placeholder_or_proprio"
    use_mid_training: bool = True
    lightweight_vlm_freeze: bool = False
    wandb_enabled: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"


@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    state_semantics: Dict[str, StateSemanticsSpec]
    action_semantics: Dict[str, ActionSemanticsSpec]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        model_raw = dict(raw.get("model", {}))
        env_backbone_name = os.getenv("EGOSCALE_VLM_BACKBONE_NAME")
        if env_backbone_name:
            model_raw["vlm_backbone_name"] = env_backbone_name
        if "trainable_module_groups" in model_raw and not isinstance(
            model_raw["trainable_module_groups"], TrainableModuleGroups
        ):
            model_raw["trainable_module_groups"] = TrainableModuleGroups(**model_raw["trainable_module_groups"])
        model = ModelConfig(**model_raw)
        data = DataConfig(**raw.get("data", {}))
        training = TrainingConfig(**raw.get("training", {}))

        state_specs = {
            item["name"]: StateSemanticsSpec(
                name=item["name"],
                embodiment_id=item["embodiment_id"],
                state_dim=item["state_dim"],
                field_order=list(item.get("field_order", [])),
                units=list(item.get("units", [])),
                normalization=NormalizationSpec(**item.get("normalization", {})),
            )
            for item in raw.get("state_semantics", _default_state_semantics())
        }
        action_specs = {
            item["name"]: ActionSemanticsSpec(
                name=item["name"],
                embodiment_id=item["embodiment_id"],
                action_dim=item["action_dim"],
                control_frequency_hz=item.get("control_frequency_hz", 10.0),
                translation_frame=item.get("translation_frame", "base"),
                rotation_frame=item.get("rotation_frame", "base"),
                rotation_parameterization=item.get("rotation_parameterization", "axis_angle"),
                translation_unit=item.get("translation_unit", "meter"),
                rotation_unit=item.get("rotation_unit", "radian"),
                hand_joint_order=list(item.get("hand_joint_order", [])),
                normalization=NormalizationSpec(**item.get("normalization", {})),
            )
            for item in raw.get("action_semantics", _default_action_semantics())
        }
        return cls(
            model=model,
            data=data,
            training=training,
            state_semantics=state_specs,
            action_semantics=action_specs,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_mapping(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                **self.model.__dict__,
                "trainable_module_groups": self.model.trainable_module_groups.__dict__,
            },
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "state_semantics": [spec.__dict__ for spec in self.state_semantics.values()],
            "action_semantics": [spec.__dict__ for spec in self.action_semantics.values()],
        }


def _default_state_semantics() -> List[Dict[str, Any]]:
    return [
        {
            "name": "sharpa_proprio_v1",
            "embodiment_id": "sharpa",
            "state_dim": 32,
            "field_order": [f"sharpa_joint_{index}" for index in range(32)],
            "units": ["radian"] * 32,
            "normalization": {"scheme": "meanstd", "stats_id": "sharpa_state_train_v1"},
        },
        {
            "name": "g1_proprio_v1",
            "embodiment_id": "g1",
            "state_dim": 16,
            "field_order": [f"g1_joint_{index}" for index in range(16)],
            "units": ["radian"] * 16,
            "normalization": {"scheme": "meanstd", "stats_id": "g1_state_train_v1"},
        },
    ]


def _default_action_semantics() -> List[Dict[str, Any]]:
    return [
        {
            "name": "sharpa_wristdelta_hand22_v1",
            "embodiment_id": "sharpa",
            "action_dim": 28,
            "control_frequency_hz": 10.0,
            "translation_frame": "wrist",
            "rotation_frame": "wrist",
            "rotation_parameterization": "axis_angle",
            "translation_unit": "meter",
            "rotation_unit": "radian",
            "hand_joint_order": [f"hand_joint_{index}" for index in range(22)],
            "normalization": {"scheme": "meanstd", "stats_id": "sharpa_action_train_v1"},
        },
        {
            "name": "g1_lowdim_native_v1",
            "embodiment_id": "g1",
            "action_dim": 13,
            "control_frequency_hz": 10.0,
            "normalization": {"scheme": "meanstd", "stats_id": "g1_action_train_v1"},
        },
    ]


def default_stage_sample_rules() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "stage1": [
            {
                "embodiment_id": "sharpa",
                "data_source": "human_retargeted",
                "has_proprio": False,
                "state_semantics_names": ["sharpa_proprio_v1"],
                "action_semantics_names": ["sharpa_wristdelta_hand22_v1"],
                "action_dim": 28,
                "allowed_camera_view_sets": [
                    ["head"],
                    ["head", "left_wrist"],
                    ["head", "right_wrist"],
                    ["head", "left_wrist", "right_wrist"],
                ],
            }
        ],
        "stage2": [
            {
                "embodiment_id": "sharpa",
                "data_source": "human_retargeted",
                "has_proprio": False,
            },
            {
                "embodiment_id": "sharpa",
                "data_source": "robot_native",
                "has_proprio": True,
            },
            {
                "embodiment_id": "g1",
                "data_source": "robot_native",
                "has_proprio": True,
            },
        ],
        "stage3": [
            {
                "data_source": "robot_native",
                "has_proprio": True,
            }
        ],
    }
