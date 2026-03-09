from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from torch.utils.data import Dataset, Sampler

from egoscale.config import DataConfig, StageRecipe
from egoscale.data.schema import EgoScaleSample, validate_stage_sample


class EgoScaleDataset(Dataset[EgoScaleSample]):
    def __init__(
        self,
        samples: Iterable[EgoScaleSample | dict],
        data_config: DataConfig,
        stage_recipe: StageRecipe,
        transform=None,
    ) -> None:
        self.data_config = data_config
        self.stage_recipe = stage_recipe
        self.transform = transform
        self.samples: List[EgoScaleSample] = []
        for raw in samples:
            sample = raw if isinstance(raw, EgoScaleSample) else EgoScaleSample.from_mapping(raw)
            validate_stage_sample(stage_recipe, sample, data_config.stage3_allow_aligned_human)
            if sample.action_semantics_name not in data_config.allowed_action_semantics_names:
                continue
            if sample.state_semantics_name not in data_config.allowed_state_semantics_names:
                continue
            if sample.data_source not in data_config.data_source_buckets:
                continue
            if sample.embodiment_id not in data_config.embodiment_buckets:
                continue
            self.samples.append(sample)

    @classmethod
    def from_jsonl(cls, path: str | Path, data_config: DataConfig, stage_recipe: StageRecipe, transform=None) -> "EgoScaleDataset":
        with Path(path).open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return cls(rows, data_config=data_config, stage_recipe=stage_recipe, transform=transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EgoScaleSample:
        sample = self.samples[index]
        return self.transform(sample) if self.transform is not None else sample

    def bucket_indices(self) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for index, sample in enumerate(self.samples):
            key = sample.bucket_key.as_string()
            buckets.setdefault(key, []).append(index)
        return buckets


class BucketedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: EgoScaleDataset,
        batch_size: int,
        drop_last: bool = False,
        sampling_policy: str = "proportional_to_active_samples",
        bucket_sampling_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampling_policy = sampling_policy
        self.bucket_sampling_weights = bucket_sampling_weights or {}
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        generator = random.Random(self.seed)
        buckets = {key: list(indices) for key, indices in self.dataset.bucket_indices().items()}
        for indices in buckets.values():
            generator.shuffle(indices)

        while buckets:
            key = self._pick_bucket(generator, buckets)
            indices = buckets[key]
            if len(indices) < self.batch_size and self.drop_last:
                buckets.pop(key)
                continue
            batch = indices[: self.batch_size]
            del indices[: self.batch_size]
            if not indices:
                buckets.pop(key)
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.dataset.bucket_indices().values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total

    def _pick_bucket(self, generator: random.Random, buckets: Dict[str, List[int]]) -> str:
        keys = list(buckets)
        if self.sampling_policy == "manual_weights":
            weights = [self.bucket_sampling_weights.get(key, 0.0) for key in keys]
            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("manual_weights policy requires positive bucket_sampling_weights")
        else:
            weights = [float(len(buckets[key])) for key in keys]
            weight_sum = sum(weights)
        threshold = generator.random() * weight_sum
        running = 0.0
        for key, weight in zip(keys, weights):
            running += weight
            if running >= threshold:
                return key
        return keys[-1]
