from __future__ import annotations

from itertools import cycle
from typing import Dict

import torch
from torch.utils.data import DataLoader

from egoscale.config import ExperimentConfig, StageRecipe
from egoscale.data import BucketedBatchSampler, EgoScaleCollator, EgoScaleDataset
from egoscale.model.policy import EgoScalePolicy
from egoscale.utils.seed import seed_everything


class BaseTrainer:
    stage_recipe: StageRecipe = "stage1"

    def __init__(self, config: ExperimentConfig, model: EgoScalePolicy | None = None) -> None:
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        seed_everything(config.training.seed)
        self.model = model or EgoScalePolicy(config)
        self.model.to(self.device)
        self.apply_freeze_recipe()
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def apply_freeze_recipe(self) -> None:
        groups = self.model.module_groups()
        spec = self.config.training.stage_recipe
        mid_training = self.config.training.use_mid_training
        allow_aligned_human = self.config.data.stage3_allow_aligned_human
        lightweight_vlm_freeze = self.config.training.lightweight_vlm_freeze

        for name, module in groups.items():
            module.requires_grad_(True)

        self.model.state_projector.set_shared_trainable(True)
        self.model.state_projector.set_placeholder_trainable(True)
        self.model.state_projector.set_proprio_trainable(True)

        if spec == "stage1":
            groups["vlm_visual_encoder"].requires_grad_(True)
            groups["vlm_multimodal_adapter"].requires_grad_(True)
            groups["vlm_language_backbone"].requires_grad_(True)
            self.model.state_projector.set_proprio_trainable(False)
        elif spec == "stage2":
            groups["vlm_visual_encoder"].requires_grad_(True)
            groups["vlm_multimodal_adapter"].requires_grad_(True)
            groups["vlm_language_backbone"].requires_grad_(False)
        elif spec == "stage3":
            visual_trainable = not mid_training
            groups["vlm_visual_encoder"].requires_grad_(visual_trainable)
            groups["vlm_multimodal_adapter"].requires_grad_(visual_trainable)
            groups["vlm_language_backbone"].requires_grad_(False)
            self.model.state_projector.set_placeholder_trainable(allow_aligned_human)
        else:
            raise ValueError("Unknown stage recipe")

        if lightweight_vlm_freeze:
            groups["vlm_visual_encoder"].requires_grad_(False)
            groups["vlm_multimodal_adapter"].requires_grad_(False)
            groups["vlm_language_backbone"].requires_grad_(False)

    def make_dataloader(self, dataset: EgoScaleDataset) -> DataLoader:
        batch_sampler = BucketedBatchSampler(
            dataset=dataset,
            batch_size=self.config.training.batch_size,
            sampling_policy=self.config.data.bucket_sampling_policy,
            bucket_sampling_weights=self.config.data.bucket_sampling_weights,
            seed=self.config.training.seed,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=EgoScaleCollator())

    def fit(self, dataset: EgoScaleDataset) -> Dict[str, float]:
        self.model.train()
        dataloader = self.make_dataloader(dataset)
        last_loss = 0.0
        step_iterator = cycle(dataloader)
        for _ in range(self.config.training.max_steps):
            batch = next(step_iterator)
            loss = self.train_step(batch)
            last_loss = float(loss.detach().cpu())
        return {"loss": last_loss}

    def train_step(self, batch: Dict[str, object]) -> torch.Tensor:
        tensor_batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = self.model(
            images=tensor_batch["images"],
            image_mask=tensor_batch["image_mask"],
            text=tensor_batch["text"],
            raw_state=tensor_batch["raw_state"],
            has_proprio=tensor_batch["has_proprio"],
            actions=tensor_batch["actions"],
            embodiment_id=tensor_batch["embodiment_id"],
        )
        loss = output["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss
