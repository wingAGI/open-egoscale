from __future__ import annotations

import argparse

from _common import PROJECT_ROOT  # noqa: F401
from egoscale.config import ExperimentConfig
from egoscale.data import EgoScaleDataset, EgoScaleTransforms
from egoscale.trainer.stage2 import Stage2Trainer
from egoscale.utils.checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage2.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.training.stage_recipe = "stage2"
    transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
    dataset = EgoScaleDataset.from_jsonl(args.dataset, config.data, "stage2", transform=transforms)
    trainer = Stage2Trainer(config)
    metrics = trainer.fit(dataset)
    save_checkpoint(args.checkpoint, trainer.model, trainer.optimizer, extra={"metrics": metrics})


if __name__ == "__main__":
    main()
