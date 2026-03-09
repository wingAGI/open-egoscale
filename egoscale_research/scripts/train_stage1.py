from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from _common import PROJECT_ROOT  # noqa: F401
from egoscale.config import ExperimentConfig
from egoscale.data import EgoScaleDataset, EgoScaleTransforms
from egoscale.trainer.stage1 import Stage1Trainer
from egoscale.utils.checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage1.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val-dataset", default="", help="Optional held-out validation dataset jsonl")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA in the current implementation")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    start = time.time()
    try:
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1
        is_main_process = rank == 0
        if is_main_process:
            print(f"[train_stage1] loading config from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
        config.training.stage_recipe = "stage1"
        if is_main_process:
            effective_batch = config.training.batch_size * world_size * max(1, config.training.grad_accum_steps)
            print(f"[train_stage1] world_size = {world_size}, micro_batch = {config.training.batch_size}, grad_accum = {config.training.grad_accum_steps}, effective_batch = {effective_batch}")
            print(f"[train_stage1] building dataset from {args.dataset}")
        transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
        dataset = EgoScaleDataset.from_jsonl(args.dataset, config.data, "stage1", transform=transforms)
        val_dataset = None
        if args.val_dataset:
            if is_main_process:
                print(f"[train_stage1] building validation dataset from {args.val_dataset}")
            val_dataset = EgoScaleDataset.from_jsonl(args.val_dataset, config.data, "stage1", transform=transforms)
            if is_main_process:
                print(f"[train_stage1] validation dataset size = {len(val_dataset)}")
        if is_main_process:
            print(f"[train_stage1] dataset size = {len(dataset)}")
            print(f"[train_stage1] creating trainer with backbone_impl = {config.model.backbone_impl}")
        trainer = Stage1Trainer(config)
        if is_main_process:
            print(f"[train_stage1] starting fit for max_steps = {config.training.max_steps}")
        metrics = trainer.fit(dataset, val_dataset=val_dataset)
        if is_main_process:
            save_checkpoint(
                args.checkpoint,
                trainer.unwrap_model(),
                trainer.optimizer,
                extra={"metrics": metrics, "history": trainer.history.__dict__},
            )
            print(f"[train_stage1] saved checkpoint to {args.checkpoint}")
            print(f"[train_stage1] done in {time.time() - start:.1f}s with metrics = {metrics}")
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
