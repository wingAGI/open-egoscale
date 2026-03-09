from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from egoscale.config import ExperimentConfig
from egoscale.data import BucketedBatchSampler, EgoScaleCollator, EgoScaleDataset, EgoScaleSample, EgoScaleTransforms
from egoscale.model.policy import EgoScalePolicy
from egoscale.model.vlm_backbone import Qwen25VLBackbone
from egoscale.trainer.stage1 import Stage1Trainer
from egoscale.trainer.stage2 import Stage2Trainer
from egoscale.trainer.stage3 import Stage3Trainer


def build_config() -> ExperimentConfig:
    config = ExperimentConfig.from_mapping(
        {
            "model": {
                "backbone_impl": "dummy",
                "vlm_token_dim": 64,
                "obs_horizon": 2,
                "action_horizon": 4,
                "num_inference_timesteps": 4,
                "num_timestep_buckets": 32,
                "num_attention_heads": 4,
                "num_dit_layers": 2,
                "dummy_tokens_per_view_step": 2,
                "dummy_text_tokens": 4,
            },
            "data": {
                "obs_horizon": 2,
                "action_horizon": 4,
                "bucket_sampling_policy": "proportional_to_active_samples",
            },
            "training": {"stage_recipe": "stage2", "batch_size": 2, "max_steps": 1, "device": "cpu"},
            "state_semantics": [
                {
                    "name": "sharpa_proprio_v1",
                    "embodiment_id": "sharpa",
                    "state_dim": 6,
                    "field_order": ["a"] * 6,
                    "units": ["u"] * 6,
                    "normalization": {"scheme": "none"},
                },
                {
                    "name": "g1_proprio_v1",
                    "embodiment_id": "g1",
                    "state_dim": 4,
                    "field_order": ["a"] * 4,
                    "units": ["u"] * 4,
                    "normalization": {"scheme": "none"},
                },
            ],
            "action_semantics": [
                {
                    "name": "sharpa_wristdelta_hand22_v1",
                    "embodiment_id": "sharpa",
                    "action_dim": 28,
                    "control_frequency_hz": 10,
                    "normalization": {"scheme": "none"},
                },
                {
                    "name": "g1_lowdim_native_v1",
                    "embodiment_id": "g1",
                    "action_dim": 13,
                    "control_frequency_hz": 10,
                    "normalization": {"scheme": "none"},
                },
            ],
        }
    )
    return config


def make_sample(embodiment_id: str, data_source: str, has_proprio: bool) -> EgoScaleSample:
    if embodiment_id == "sharpa":
        state_dim = 6
        action_dim = 28
        state_semantics = "sharpa_proprio_v1"
        action_semantics = "sharpa_wristdelta_hand22_v1"
    else:
        state_dim = 4
        action_dim = 13
        state_semantics = "g1_proprio_v1"
        action_semantics = "g1_lowdim_native_v1"
    raw_state = torch.zeros(state_dim) if not has_proprio else torch.arange(state_dim, dtype=torch.float32)
    actions = torch.randn(4, action_dim)
    head_frames = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8)
    sample = EgoScaleSample(
        instruction="pick up block",
        embodiment_id=embodiment_id,
        data_source=data_source,
        has_proprio=has_proprio,
        task_id="task",
        state_semantics_name=state_semantics,
        action_semantics_name=action_semantics,
        state_dim=state_dim,
        action_dim=action_dim,
        raw_state=raw_state,
        actions=actions,
        obs_timestamps=[0.0, 1.0],
        action_timestamps=[1.0, 2.0, 3.0, 4.0],
        obs_horizon=2,
        action_horizon=4,
        camera_views=["head"],
        view_images={"head": head_frames},
    )
    return sample


class SpecContractTests(unittest.TestCase):
    def test_placeholder_state_and_missing_views_contract(self) -> None:
        config = build_config()
        transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
        sample = transforms(make_sample("sharpa", "human_retargeted", False))
        self.assertTrue(torch.equal(sample.raw_state, torch.zeros_like(sample.raw_state)))
        self.assertEqual(tuple(sample.images.shape), (3, 2, 3, 8, 8))
        self.assertTrue(torch.equal(sample.image_mask[0], torch.tensor([True, True])))
        self.assertTrue(torch.equal(sample.image_mask[1], torch.tensor([False, False])))
        self.assertTrue(torch.equal(sample.image_mask[2], torch.tensor([False, False])))

    def test_bucket_sampler_keeps_batches_single_bucket(self) -> None:
        config = build_config()
        transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
        dataset = EgoScaleDataset(
            [
                make_sample("sharpa", "human_retargeted", False),
                make_sample("sharpa", "human_retargeted", False),
                make_sample("g1", "robot_native", True),
                make_sample("g1", "robot_native", True),
            ],
            data_config=config.data,
            stage_recipe="stage2",
            transform=transforms,
        )
        sampler = BucketedBatchSampler(dataset, batch_size=2, seed=0)
        for batch_indices in sampler:
            keys = {dataset.samples[index].bucket_key.as_string() for index in batch_indices}
            self.assertEqual(len(keys), 1)

    def test_policy_forward_matches_tensor_contract(self) -> None:
        config = build_config()
        transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
        sample = transforms(make_sample("sharpa", "robot_native", True))
        batch = EgoScaleCollator()([sample, sample])
        model = EgoScalePolicy(config)
        output = model(
            images=batch["images"],
            image_mask=batch["image_mask"],
            text=batch["text"],
            raw_state=batch["raw_state"],
            has_proprio=batch["has_proprio"],
            actions=batch["actions"],
            embodiment_id=batch["embodiment_id"],
        )
        self.assertEqual(tuple(output["state_tokens"].shape), (2, 1, config.model.vlm_token_dim))
        self.assertEqual(tuple(output["pred_action_latents"].shape), (2, config.model.action_horizon, config.model.vlm_token_dim))
        self.assertEqual(tuple(output["pred_velocities"].shape), (2, config.model.action_horizon, 28))
        self.assertGreaterEqual(float(output["loss"].item()), 0.0)

    def test_stage3_freeze_matrix(self) -> None:
        config = build_config()
        config.training.stage_recipe = "stage3"
        config.training.use_mid_training = True
        config.data.stage3_allow_aligned_human = False
        trainer = Stage3Trainer(config)
        visual_trainable = any(parameter.requires_grad for parameter in trainer.model.vlm_backbone.visual_encoder_group.parameters())
        placeholder_trainable = any(
            parameter.requires_grad
            for projector in trainer.model.state_projector.projectors.values()
            for parameter in projector.placeholder_adapter.parameters()
        )
        language_trainable = any(parameter.requires_grad for parameter in trainer.model.vlm_backbone.language_backbone_group.parameters())
        self.assertFalse(visual_trainable)
        self.assertFalse(placeholder_trainable)
        self.assertFalse(language_trainable)

    def test_lightweight_vlm_freeze_applies_to_all_stages(self) -> None:
        trainer_specs = [
            ("stage1", Stage1Trainer),
            ("stage2", Stage2Trainer),
            ("stage3", Stage3Trainer),
        ]
        for stage_recipe, trainer_cls in trainer_specs:
            config = build_config()
            config.training.stage_recipe = stage_recipe
            config.training.lightweight_vlm_freeze = True
            trainer = trainer_cls(config)

            visual_trainable = any(
                parameter.requires_grad for parameter in trainer.model.vlm_backbone.visual_encoder_group.parameters()
            )
            adapter_trainable = any(
                parameter.requires_grad for parameter in trainer.model.vlm_backbone.multimodal_adapter_group.parameters()
            )
            language_trainable = any(
                parameter.requires_grad for parameter in trainer.model.vlm_backbone.language_backbone_group.parameters()
            )
            action_head_trainable = any(parameter.requires_grad for parameter in trainer.model.action_head.parameters())

            self.assertFalse(visual_trainable, stage_recipe)
            self.assertFalse(adapter_trainable, stage_recipe)
            self.assertFalse(language_trainable, stage_recipe)
            self.assertTrue(action_head_trainable, stage_recipe)

    def test_qwen_context_mask_respects_image_mask(self) -> None:
        backbone = object.__new__(Qwen25VLBackbone)
        backbone.image_token_divisor = 4
        backbone.model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
        attention_mask = torch.ones((1, 8), dtype=torch.bool)
        mm_token_type_ids = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 2, 4], [1, 2, 4]], dtype=torch.long)
        image_mask = torch.tensor([[[True], [False]]], dtype=torch.bool)

        context_mask = backbone._build_context_mask(attention_mask, mm_token_type_ids, image_grid_thw, image_mask)

        self.assertTrue(torch.equal(context_mask[0], torch.tensor([True, True, False, False, True, True, True, True])))


if __name__ == "__main__":
    unittest.main()
