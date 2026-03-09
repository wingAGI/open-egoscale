from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
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
        asset_root: str | Path | None = None,
    ) -> None:
        self.data_config = data_config
        self.stage_recipe = stage_recipe
        self.transform = transform
        self.asset_root = Path(asset_root).resolve() if asset_root is not None else None
        self.samples: List[EgoScaleSample] = []
        self._rows: List[EgoScaleSample | dict] = []
        for raw in samples:
            sample = raw if isinstance(raw, EgoScaleSample) else self._coerce_sample_metadata(raw)
            validate_stage_sample(stage_recipe, sample, data_config)
            if sample.action_semantics_name not in data_config.allowed_action_semantics_names:
                continue
            if sample.state_semantics_name not in data_config.allowed_state_semantics_names:
                continue
            if sample.data_source not in data_config.data_source_buckets:
                continue
            if sample.embodiment_id not in data_config.embodiment_buckets:
                continue
            self.samples.append(sample)
            self._rows.append(raw)

    @classmethod
    def from_jsonl(cls, path: str | Path, data_config: DataConfig, stage_recipe: StageRecipe, transform=None) -> "EgoScaleDataset":
        jsonl_path = Path(path)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        asset_root = _infer_asset_root(jsonl_path, rows)
        return cls(rows, data_config=data_config, stage_recipe=stage_recipe, transform=transform, asset_root=asset_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EgoScaleSample:
        row = self._rows[index]
        sample = self.samples[index] if isinstance(row, EgoScaleSample) else self._materialize_sample(row)
        return self.transform(sample) if self.transform is not None else sample

    def bucket_indices(self) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for index, sample in enumerate(self.samples):
            key = sample.bucket_key.as_string()
            buckets.setdefault(key, []).append(index)
        return buckets

    def _coerce_sample_metadata(self, raw: dict) -> EgoScaleSample:
        if "episode_file" in raw and "obs_index_start" in raw and "action_index_start" in raw:
            return _manifest_row_to_sample_metadata(raw)
        return EgoScaleSample.from_mapping(raw)

    def _materialize_sample(self, raw: dict) -> EgoScaleSample:
        if "episode_file" in raw and "obs_index_start" in raw and "action_index_start" in raw:
            if self.asset_root is None:
                raise ValueError("Manifest-style dataset requires asset_root")
            return _materialize_manifest_row(raw, self.asset_root)
        return EgoScaleSample.from_mapping(raw)


class BucketedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: EgoScaleDataset,
        batch_size: int,
        drop_last: bool = False,
        sampling_policy: str = "proportional_to_active_samples",
        bucket_sampling_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampling_policy = sampling_policy
        self.bucket_sampling_weights = bucket_sampling_weights or {}
        self.seed = seed
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        generator = random.Random(self.seed + self.epoch)
        buckets = {key: list(indices) for key, indices in self.dataset.bucket_indices().items()}
        for indices in buckets.values():
            generator.shuffle(indices)

        batch_index = 0
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
            if batch_index % self.num_replicas == self.rank:
                yield batch
            batch_index += 1

    def __len__(self) -> int:
        total = 0
        for indices in self.dataset.bucket_indices().values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        if total <= self.rank:
            return 0
        return (total - self.rank + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

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


def _infer_asset_root(jsonl_path: Path, rows: Sequence[dict]) -> Path | None:
    if not rows:
        return None
    first = rows[0]
    if not isinstance(first, dict) or "episode_file" not in first:
        return None
    candidates = [jsonl_path.parent, jsonl_path.parent.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / str(first["episode_file"])).exists():
            return candidate.resolve()
    return jsonl_path.parent.resolve()


def _manifest_row_to_sample_metadata(raw: Dict[str, Any]) -> EgoScaleSample:
    state_dim = int(raw["state_dim"])
    action_dim = int(raw["action_dim"])
    obs_horizon = int(raw["obs_horizon"])
    action_horizon = int(raw["action_horizon"])
    return EgoScaleSample(
        instruction=str(raw["instruction"]),
        embodiment_id=str(raw["embodiment_id"]),
        data_source=str(raw["data_source"]),
        has_proprio=bool(raw["has_proprio"]),
        task_id=str(raw["task_id"]),
        state_semantics_name=str(raw["state_semantics_name"]),
        action_semantics_name=str(raw["action_semantics_name"]),
        state_dim=state_dim,
        action_dim=action_dim,
        raw_state=torch.zeros(state_dim, dtype=torch.float32),
        actions=torch.zeros((action_horizon, action_dim), dtype=torch.float32),
        obs_timestamps=list(raw["obs_timestamps"]),
        action_timestamps=list(raw["action_timestamps"]),
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        camera_views=list(raw.get("camera_views", [])),
    )


def _materialize_manifest_row(raw: Dict[str, Any], asset_root: Path) -> EgoScaleSample:
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    metadata = _manifest_row_to_sample_metadata(raw)
    episode_path = _resolve_asset_path(asset_root, str(raw["episode_file"]))
    obs_index_start = int(raw["obs_index_start"])
    action_index_start = int(raw["action_index_start"])
    obs_horizon = metadata.obs_horizon
    action_horizon = metadata.action_horizon

    with h5py.File(episode_path, "r") as handle:
        signals = handle["signals"]
        raw_state_ds = signals["raw_state"]
        actions_ds = signals["actions"]
        if metadata.has_proprio:
            raw_state = torch.tensor(raw_state_ds[action_index_start], dtype=torch.float32)
        else:
            raw_state = torch.zeros(metadata.state_dim, dtype=torch.float32)
        actions = torch.tensor(actions_ds[action_index_start : action_index_start + action_horizon], dtype=torch.float32)

    view_images: Dict[str, torch.Tensor] = {}
    video_refs = raw.get("video_refs", {})
    for view_name in metadata.camera_views:
        video_ref = video_refs.get(view_name)
        if video_ref is None:
            continue
        video_path = _resolve_asset_path(asset_root, str(video_ref))
        frames = _read_video_frames(
            imageio,
            video_path,
            metadata.obs_timestamps[:obs_horizon],
        )
        view_images[view_name] = torch.from_numpy(frames)

    return metadata.replace(raw_state=raw_state, actions=actions, view_images=view_images)


def _read_video_frames(imageio_module: Any, video_path: Path, timestamps: Sequence[float]) -> Any:
    import numpy as np

    reader = imageio_module.get_reader(str(video_path), format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 30.0)
        frames = []
        for timestamp in timestamps:
            frame_index = max(0, int(round(float(timestamp) * fps)))
            frame = reader.get_data(frame_index)
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)
            frames.append(np.asarray(frame, dtype=np.uint8).transpose(2, 0, 1))
        return np.stack(frames, axis=0)
    finally:
        reader.close()


def _resolve_asset_path(asset_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else (asset_root / path).resolve()
