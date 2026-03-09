from __future__ import annotations

import argparse

from _common import PROJECT_ROOT  # noqa: F401
from egoscale.config import ExperimentConfig
from egoscale.data import EgoScaleCollator, EgoScaleDataset, EgoScaleTransforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage2.yaml"))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
    dataset = EgoScaleDataset.from_jsonl(args.dataset, config.data, config.training.stage_recipe, transform=transforms)
    collator = EgoScaleCollator()
    batch = collator([dataset[0]])
    print(
        {
            "images": tuple(batch["images"].shape),
            "image_mask": tuple(batch["image_mask"].shape),
            "raw_state": tuple(batch["raw_state"].shape),
            "actions": tuple(batch["actions"].shape),
            "bucket_key": batch["bucket_key"],
        }
    )


if __name__ == "__main__":
    main()
