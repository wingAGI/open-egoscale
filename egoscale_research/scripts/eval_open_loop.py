from __future__ import annotations

import argparse

import torch

from _common import PROJECT_ROOT  # noqa: F401
from egoscale.config import ExperimentConfig
from egoscale.data import EgoScaleCollator, EgoScaleDataset, EgoScaleTransforms
from egoscale.model.policy import EgoScalePolicy
from egoscale.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage3.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    transforms = EgoScaleTransforms(config.data, config.state_semantics, config.action_semantics)
    dataset = EgoScaleDataset.from_jsonl(args.dataset, config.data, config.training.stage_recipe, transform=transforms)
    collator = EgoScaleCollator()
    batch = collator([dataset[0]])
    model = EgoScalePolicy(config)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    action_semantics_name = batch["action_semantics_name"][0]
    action_dim = config.action_semantics[action_semantics_name].action_dim
    normalizer = transforms.action_normalizer(action_semantics_name)
    predicted = model.sample_actions(
        images=batch["images"],
        image_mask=batch["image_mask"],
        text=batch["text"],
        raw_state=batch["raw_state"],
        has_proprio=batch["has_proprio"],
        embodiment_id=batch["embodiment_id"],
        action_dim=action_dim,
        action_normalizer=normalizer,
    )
    print({"pred_shape": tuple(predicted.shape), "pred_mean_abs": float(predicted.abs().mean().item())})


if __name__ == "__main__":
    main()
