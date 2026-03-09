from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from _common import PROJECT_ROOT  # noqa: F401
from egoscale.data.schema import EgoScaleSample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jsonl dataset")
    parser.add_argument("--output", required=True, help="Output YAML stats file")
    args = parser.parse_args()

    stats = {
        "state_semantics": defaultdict(list),
        "action_semantics": defaultdict(list),
    }
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = EgoScaleSample.from_mapping(json.loads(line))
            if sample.has_proprio:
                stats["state_semantics"][sample.state_semantics_name].append(sample.raw_state)
            stats["action_semantics"][sample.action_semantics_name].append(sample.actions.reshape(-1, sample.action_dim))

    payload = {"state_semantics": {}, "action_semantics": {}}
    for name, values in stats["state_semantics"].items():
        stacked = torch.stack(values, dim=0)
        payload["state_semantics"][name] = {
            "mean": stacked.mean(dim=0).tolist(),
            "std": stacked.std(dim=0, unbiased=False).clamp_min(1e-6).tolist(),
        }
    for name, values in stats["action_semantics"].items():
        stacked = torch.cat(values, dim=0)
        payload["action_semantics"][name] = {
            "mean": stacked.mean(dim=0).tolist(),
            "std": stacked.std(dim=0, unbiased=False).clamp_min(1e-6).tolist(),
            "min": stacked.min(dim=0).values.tolist(),
            "max": stacked.max(dim=0).values.tolist(),
        }

    with Path(args.output).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


if __name__ == "__main__":
    main()
