from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from egoscale.config import ExperimentConfig, StageRecipe
from egoscale.data import BucketedBatchSampler, EgoScaleCollator, EgoScaleDataset
from egoscale.model.policy import EgoScalePolicy
from egoscale.utils.metrics import TrainingHistory, WandbLogger
from egoscale.utils.seed import seed_everything


class BaseTrainer:
    stage_recipe: StageRecipe = "stage1"

    def __init__(self, config: ExperimentConfig, model: EgoScalePolicy | None = None) -> None:
        self.config = config
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        seed_everything(config.training.seed + self.rank)
        self.model = model or EgoScalePolicy(config)
        self.model.to(self.device)
        self.apply_freeze_recipe()
        if self.is_distributed:
            ddp_kwargs = {"device_ids": [self.local_rank]} if self.device.type == "cuda" else {}
            self.model = DDP(self.model, find_unused_parameters=False, **ddp_kwargs)
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.history = TrainingHistory()
        self.wandb = WandbLogger(
            enabled=config.training.wandb_enabled and self.is_main_process,
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            run_name=config.training.wandb_run_name,
            mode=config.training.wandb_mode,
            config=config.to_dict(),
        )

    def unwrap_model(self) -> EgoScalePolicy:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def apply_freeze_recipe(self) -> None:
        model = self.unwrap_model()
        groups = model.module_groups()
        spec = self.config.training.stage_recipe
        mid_training = self.config.training.use_mid_training
        allow_aligned_human = self.config.data.stage3_allow_aligned_human
        lightweight_vlm_freeze = self.config.training.lightweight_vlm_freeze

        for name, module in groups.items():
            module.requires_grad_(True)

        model.state_projector.set_shared_trainable(True)
        model.state_projector.set_placeholder_trainable(True)
        model.state_projector.set_proprio_trainable(True)

        if spec == "stage1":
            groups["vlm_visual_encoder"].requires_grad_(True)
            groups["vlm_multimodal_adapter"].requires_grad_(True)
            groups["vlm_language_backbone"].requires_grad_(True)
            model.state_projector.set_proprio_trainable(False)
        elif spec == "stage2":
            groups["vlm_visual_encoder"].requires_grad_(True)
            groups["vlm_multimodal_adapter"].requires_grad_(True)
            groups["vlm_language_backbone"].requires_grad_(False)
        elif spec == "stage3":
            visual_trainable = not mid_training
            groups["vlm_visual_encoder"].requires_grad_(visual_trainable)
            groups["vlm_multimodal_adapter"].requires_grad_(visual_trainable)
            groups["vlm_language_backbone"].requires_grad_(False)
            model.state_projector.set_placeholder_trainable(allow_aligned_human)
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
            num_replicas=self.world_size,
            rank=self.rank,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=EgoScaleCollator())

    def fit(self, dataset: EgoScaleDataset, val_dataset: Optional[EgoScaleDataset] = None) -> Dict[str, float]:
        self.model.train()
        dataloader = self.make_dataloader(dataset)
        last_loss = 0.0
        if len(dataloader) == 0:
            raise ValueError("Training dataset produced no batches for this rank")
        self._active_train_dataloader = dataloader
        self._active_train_epoch = 0
        train_sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(self._active_train_epoch)
        self._active_train_iterator = iter(dataloader)
        for step in range(1, self.config.training.max_steps + 1):
            loss = self.train_step()
            last_loss = float(loss.detach().cpu())
            self.history.train.append({"step": float(step), "loss": last_loss})
            if self.is_main_process and step % max(1, self.config.training.log_interval) == 0:
                payload = {"step": float(step), "train/loss": last_loss}
                self.wandb.log(payload)
                print(f"[train] step={step} loss={last_loss:.6f}")
            if val_dataset is not None and self.config.training.eval_interval > 0 and step % self.config.training.eval_interval == 0:
                val_loss = self.evaluate(val_dataset)
                if self.is_main_process:
                    self.history.val.append({"step": float(step), "loss": val_loss})
                    payload = {"step": float(step), "val/loss": val_loss}
                    self.wandb.log(payload)
                    print(f"[val] step={step} loss={val_loss:.6f}")
        self.wandb.finish()
        if self.is_distributed:
            dist.barrier()
        return {"loss": last_loss, **self.history.summary()}

    def train_step(self) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        grad_accum_steps = max(1, int(self.config.training.grad_accum_steps))
        total_loss = 0.0
        dataloader = getattr(self, "_active_train_dataloader", None)
        if dataloader is None:
            raise RuntimeError("Active train dataloader is not set")
        if getattr(self, "_active_train_iterator", None) is None:
            raise RuntimeError("Active train iterator is not set")
        for micro_step in range(grad_accum_steps):
            batch = self._next_batch(dataloader, iterator_attr="_active_train_iterator", epoch_attr="_active_train_epoch")
            should_sync = micro_step == grad_accum_steps - 1 or not self.is_distributed
            sync_context = self.model.no_sync() if self.is_distributed and not should_sync else nullcontext()
            with sync_context:
                output = self.forward_batch(batch)
                raw_loss = output["loss"]
                total_loss += float(raw_loss.detach().cpu())
                (raw_loss / grad_accum_steps).backward()
        self.optimizer.step()
        loss = torch.tensor(total_loss / grad_accum_steps, device=self.device)
        if self.is_distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / self.world_size
        return loss

    def evaluate(self, dataset: EgoScaleDataset) -> float:
        dataloader = self.make_dataloader(dataset)
        self.model.eval()
        losses = 0.0
        batches = 0
        max_batches = self.config.training.max_val_batches
        sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader, start=1):
                output = self.forward_batch(batch)
                losses += float(output["loss"].detach().cpu())
                batches += 1
                if max_batches > 0 and batch_index >= max_batches:
                    break
        self.model.train()
        if self.is_distributed:
            payload = torch.tensor([losses, batches], device=self.device, dtype=torch.float64)
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            losses = float(payload[0].item())
            batches = int(payload[1].item())
        if batches <= 0:
            raise ValueError("Validation dataset produced no batches")
        return losses / batches

    def _next_batch(self, dataloader: DataLoader, *, iterator_attr: str, epoch_attr: str) -> Dict[str, object]:
        iterator = getattr(self, iterator_attr)
        try:
            return next(iterator)
        except StopIteration:
            epoch = getattr(self, epoch_attr, 0) + 1
            setattr(self, epoch_attr, epoch)
            sampler = getattr(dataloader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            iterator = iter(dataloader)
            setattr(self, iterator_attr, iterator)
            return next(iterator)

    def forward_batch(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        tensor_batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return self.model(
            images=tensor_batch["images"],
            image_mask=tensor_batch["image_mask"],
            text=tensor_batch["text"],
            raw_state=tensor_batch["raw_state"],
            has_proprio=tensor_batch["has_proprio"],
            actions=tensor_batch["actions"],
            embodiment_id=tensor_batch["embodiment_id"],
        )
