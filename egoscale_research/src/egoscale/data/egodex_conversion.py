from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import h5py
import numpy as np


EXPECTED_FINGERS = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class EgoDexEpisode:
    task_name: str
    episode_name: str
    video_path: Path
    h5_path: Path

    @property
    def episode_id(self) -> str:
        return f"{_slugify(self.task_name)}_{_slugify(self.episode_name)}"

    @property
    def task_id(self) -> str:
        return f"egodex/{_slugify(self.task_name)}"


@dataclass(frozen=True)
class EpisodeConversionResult:
    episode_id: str
    task_id: str
    num_steps: int
    num_chunks: int
    available_views: List[str]
    output_h5_path: Path
    output_video_path: Path
    instruction: str


def discover_egodex_episodes(root: str | Path) -> List[EgoDexEpisode]:
    base = Path(root)
    episodes: List[EgoDexEpisode] = []
    for task_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        mp4_files = {path.stem: path for path in task_dir.glob("*.mp4")}
        h5_files = {path.stem: path for path in task_dir.glob("*.hdf5")}
        for stem, h5_path in sorted(h5_files.items()):
            video_path = mp4_files.get(stem)
            if video_path is None:
                continue
            episodes.append(
                EgoDexEpisode(
                    task_name=task_dir.name,
                    episode_name=stem,
                    video_path=video_path,
                    h5_path=h5_path,
                )
            )
    return episodes


def convert_episode(
    episode: EgoDexEpisode,
    output_root: str | Path,
    *,
    target_fps: float = 10.0,
    source_fps: float = 30.0,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    sample_stride: int = 1,
    video_mode: str = "symlink",
    stage_action_semantics_name: str = "sharpa_wristdelta_hand22_v1",
    stage_state_semantics_name: str = "sharpa_proprio_v1",
    stage_state_dim: int = 32,
) -> EpisodeConversionResult:
    output_root = Path(output_root)
    meta_dir = output_root / "meta"
    data_dir = output_root / "data" / "episodes"
    videos_dir = output_root / "videos" / "egodex"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(episode.h5_path, "r") as handle:
        transforms_group = _require_group(handle, "transforms")
        hand_stream = _select_active_hand_stream(transforms_group, handle.get("confidences"))
        instruction = _extract_instruction(handle, episode.task_name)

        hand_name = hand_stream["hand_name"]
        wrist_transforms = np.asarray(hand_stream["wrist"], dtype=np.float32)
        finger_transforms = {
            finger: {
                "tip": np.asarray(parts["tip"], dtype=np.float32),
                "knuckle": np.asarray(parts["knuckle"], dtype=np.float32),
            }
            for finger, parts in hand_stream["fingers"].items()
        }

    if wrist_transforms.ndim != 3 or wrist_transforms.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected wrist transform shape for {episode.h5_path}: {wrist_transforms.shape}")

    step = _source_to_target_stride(source_fps, target_fps)
    sampled_indices = np.arange(0, wrist_transforms.shape[0], step, dtype=np.int64)
    if sampled_indices.size < obs_horizon + action_horizon:
        raise ValueError(f"Episode {episode.episode_id} is too short after resampling")

    wrist_sampled = wrist_transforms[sampled_indices]
    descriptors = compute_hand_descriptors(wrist_sampled, finger_transforms, sampled_indices, source_fps)
    actions = build_pseudo_actions(wrist_sampled, descriptors)
    timestamps = sampled_indices.astype(np.float32) / float(source_fps)

    episode_h5_path = data_dir / f"{episode.episode_id}.h5"
    _write_episode_h5(
        episode_h5_path,
        episode=episode,
        instruction=instruction,
        hand_name=hand_name,
        timestamps=timestamps,
        wrist_transforms=wrist_sampled,
        hand_descriptors=descriptors,
        actions=actions,
    )

    output_video_path = videos_dir / f"{episode.episode_id}_head.mp4"
    _materialize_video(episode.video_path, output_video_path, mode=video_mode)

    episode_record = {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "source_dataset": "egodex",
        "data_source": "human_retargeted",
        "embodiment_id": "sharpa",
        "available_views": ["head"],
        "language": instruction,
        "episode_file": str(episode_h5_path.relative_to(output_root)),
        "video_paths": {"head": str(output_video_path.relative_to(output_root))},
        "num_steps": int(actions.shape[0]),
        "duration_sec": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "control_frequency_hz": float(target_fps),
        "active_hand": hand_name,
        "source_video_path": str(episode.video_path),
        "source_h5_path": str(episode.h5_path),
    }
    _append_jsonl(meta_dir / "episodes.jsonl", episode_record)

    chunk_count = 0
    for chunk in iter_stage1_chunks(
        episode=episode,
        output_root=output_root,
        instruction=instruction,
        timestamps=timestamps,
        num_actions=actions.shape[0],
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        sample_stride=sample_stride,
        stage_action_semantics_name=stage_action_semantics_name,
        stage_state_semantics_name=stage_state_semantics_name,
        stage_state_dim=stage_state_dim,
    ):
        _append_jsonl(meta_dir / "chunks_stage1.jsonl", chunk)
        chunk_count += 1

    return EpisodeConversionResult(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        num_steps=int(actions.shape[0]),
        num_chunks=chunk_count,
        available_views=["head"],
        output_h5_path=episode_h5_path,
        output_video_path=output_video_path,
        instruction=instruction,
    )


def iter_stage1_chunks(
    *,
    episode: EgoDexEpisode,
    output_root: Path,
    instruction: str,
    timestamps: np.ndarray,
    num_actions: int,
    obs_horizon: int,
    action_horizon: int,
    sample_stride: int,
    stage_action_semantics_name: str,
    stage_state_semantics_name: str,
    stage_state_dim: int,
) -> Iterator[Dict[str, object]]:
    for obs_index_start in range(0, num_actions - action_horizon - obs_horizon + 2, sample_stride):
        obs_index_end = obs_index_start + obs_horizon
        action_index_start = obs_index_end - 1
        action_index_end = action_index_start + action_horizon
        if action_index_end > num_actions:
            break
        obs_ts = timestamps[obs_index_start:obs_index_end].tolist()
        act_ts = timestamps[action_index_start:action_index_end].tolist()
        yield {
            "chunk_id": f"{episode.episode_id}:{obs_index_start}",
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "stage": "stage1",
            "instruction": instruction,
            "embodiment_id": "sharpa",
            "data_source": "human_retargeted",
            "has_proprio": False,
            "state_semantics_name": stage_state_semantics_name,
            "action_semantics_name": stage_action_semantics_name,
            "state_dim": int(stage_state_dim),
            "action_dim": 28,
            "obs_horizon": int(obs_horizon),
            "action_horizon": int(action_horizon),
            "obs_timestamps": obs_ts,
            "action_timestamps": act_ts,
            "camera_views": ["head"],
            "episode_file": str((Path("data") / "episodes" / f"{episode.episode_id}.h5").as_posix()),
            "video_refs": {"head": str((Path("videos") / "egodex" / f"{episode.episode_id}_head.mp4").as_posix())},
            "obs_index_start": int(obs_index_start),
            "action_index_start": int(action_index_start),
        }


def compute_hand_descriptors(
    wrist_transforms: np.ndarray,
    finger_transforms: Mapping[str, Mapping[str, np.ndarray]],
    sampled_indices: np.ndarray,
    source_fps: float,
) -> np.ndarray:
    tip_positions_world = {
        finger: _translations(parts["tip"])[sampled_indices]
        for finger, parts in finger_transforms.items()
    }
    knuckle_positions_world = {
        finger: _translations(parts["knuckle"])[sampled_indices]
        for finger, parts in finger_transforms.items()
    }
    palm_positions_world = _translations(wrist_transforms)
    wrist_inverse = np.linalg.inv(wrist_transforms)

    tip_positions_local = {
        finger: _transform_points(wrist_inverse, positions)
        for finger, positions in tip_positions_world.items()
    }
    knuckle_positions_local = {
        finger: _transform_points(wrist_inverse, positions)
        for finger, positions in knuckle_positions_world.items()
    }

    curls = []
    spreads = []
    tip_distances = []
    tip_speeds = []
    knuckle_spread_anchors = _compute_spread_anchors(knuckle_positions_local)
    dt = max(1.0 / source_fps, 1e-6)

    for finger in EXPECTED_FINGERS:
        tip = tip_positions_local[finger]
        knuckle = knuckle_positions_local[finger]
        tip_vector = tip - knuckle
        curls.append(np.linalg.norm(tip_vector, axis=1))
        anchor = knuckle_spread_anchors[finger]
        spreads.append(_angle_to_anchor(knuckle, anchor))
        tip_distances.append(np.linalg.norm(tip, axis=1))
        velocities = np.zeros((tip.shape[0],), dtype=np.float32)
        if tip.shape[0] > 1:
            velocities[1:] = np.linalg.norm(np.diff(tip, axis=0), axis=1) / dt
        tip_speeds.append(velocities)

    pinch = np.linalg.norm(tip_positions_local["thumb"] - tip_positions_local["index"], axis=1)
    openness = np.mean(np.stack(tip_distances, axis=1), axis=1)

    descriptors = np.stack(
        curls
        + spreads
        + tip_distances
        + tip_speeds
        + [pinch.astype(np.float32), openness.astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    return descriptors


def build_pseudo_actions(wrist_transforms: np.ndarray, hand_descriptors: np.ndarray) -> np.ndarray:
    if wrist_transforms.shape[0] != hand_descriptors.shape[0]:
        raise ValueError("wrist_transforms and hand_descriptors must have the same length")
    if wrist_transforms.shape[0] < 2:
        raise ValueError("Need at least 2 timesteps to build pseudo-actions")

    rel_actions = []
    for index in range(wrist_transforms.shape[0] - 1):
        relative = np.linalg.inv(wrist_transforms[index]) @ wrist_transforms[index + 1]
        translation = relative[:3, 3].astype(np.float32)
        rotation = rotation_matrix_to_axis_angle(relative[:3, :3]).astype(np.float32)
        rel_actions.append(np.concatenate([translation, rotation, hand_descriptors[index + 1]], axis=0))
    return np.asarray(rel_actions, dtype=np.float32)


def rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cosine = max(min((trace - 1.0) / 2.0, 1.0), -1.0)
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    denom = 2.0 * math.sin(angle)
    axis = np.array(
        [
            (rotation[2, 1] - rotation[1, 2]) / denom,
            (rotation[0, 2] - rotation[2, 0]) / denom,
            (rotation[1, 0] - rotation[0, 1]) / denom,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (axis / norm * angle).astype(np.float32)


def write_dataset_info(
    output_root: str | Path,
    *,
    dataset_name: str,
    version: str,
    obs_horizon: int,
    action_horizon: int,
    control_frequency_hz: float,
) -> None:
    output_root = Path(output_root)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "dataset_name": dataset_name,
        "version": version,
        "stages": ["stage1"],
        "embodiments": ["sharpa"],
        "canonical_view_order": ["head", "left_wrist", "right_wrist"],
        "control_frequency_hz": float(control_frequency_hz),
        "obs_horizon": int(obs_horizon),
        "action_horizon": int(action_horizon),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_splits(output_root: str | Path, train_episode_ids: Sequence[str], val_episode_ids: Sequence[str]) -> None:
    payload = {
        "train_episode_ids": list(train_episode_ids),
        "val_episode_ids": list(val_episode_ids),
        "test_episode_ids": [],
    }
    output_root = Path(output_root)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "splits.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _select_active_hand_stream(
    transforms_group: h5py.Group,
    confidences_group: h5py.Group | None,
) -> Dict[str, object]:
    candidates = []
    for hand_name in ("left", "right"):
        wrist_key = _find_transform_key(transforms_group, hand_name, "hand")
        finger_data = _find_finger_keys(transforms_group, hand_name)
        wrist = np.asarray(transforms_group[wrist_key])
        if len(finger_data) != len(EXPECTED_FINGERS):
            continue
        confidence_score = _confidence_score(confidences_group, hand_name)
        motion_score = _motion_energy(wrist)
        candidates.append(
            {
                "hand_name": hand_name,
                "wrist": wrist,
                "fingers": {finger: {"tip": transforms_group[parts["tip"]], "knuckle": transforms_group[parts["knuckle"]]} for finger, parts in finger_data.items()},
                "score": confidence_score + motion_score,
            }
        )
    if not candidates:
        raise ValueError("No valid hand stream found in EgoDex transforms group")
    return max(candidates, key=lambda item: float(item["score"]))


def _confidence_score(confidences_group: h5py.Group | None, hand_name: str) -> float:
    if confidences_group is None:
        return 0.0
    values: List[np.ndarray] = []
    prefix = hand_name.lower()
    for key, dataset in confidences_group.items():
        if prefix in key.lower():
            values.append(np.asarray(dataset, dtype=np.float32).reshape(-1))
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def _motion_energy(wrist: np.ndarray) -> float:
    positions = _translations(wrist)
    if positions.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).mean())


def _find_transform_key(transforms_group: h5py.Group, hand_name: str, suffix: str) -> str:
    prefix = hand_name.lower()
    target_suffix = suffix.lower()
    matches = [key for key in transforms_group.keys() if prefix in key.lower() and key.lower().endswith(target_suffix)]
    if not matches:
        raise KeyError(f"Could not find transform key for {hand_name}/{suffix}")
    return sorted(matches)[0]


def _find_finger_keys(transforms_group: h5py.Group, hand_name: str) -> Dict[str, Dict[str, str]]:
    prefix = hand_name.lower()
    result: Dict[str, Dict[str, str]] = {}
    for key in transforms_group.keys():
        lowered = key.lower()
        if prefix not in lowered:
            continue
        for finger in EXPECTED_FINGERS:
            if finger not in lowered:
                continue
            if lowered.endswith("tip"):
                result.setdefault(finger, {})["tip"] = key
            elif lowered.endswith("knuckle"):
                result.setdefault(finger, {})["knuckle"] = key
    return {finger: parts for finger, parts in result.items() if {"tip", "knuckle"} <= set(parts)}


def _extract_instruction(handle: h5py.File, default_task_name: str) -> str:
    candidate_attrs = ("llm_description", "description", "instruction", "task_description")
    for name in candidate_attrs:
        if name in handle.attrs:
            return _normalize_text(handle.attrs[name])
    if "meta" in handle and isinstance(handle["meta"], h5py.Group):
        for name in candidate_attrs:
            if name in handle["meta"].attrs:
                return _normalize_text(handle["meta"].attrs[name])
    return f"perform {default_task_name.replace('_', ' ')}"


def _normalize_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar = value.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8", "ignore").strip()
        return str(scalar).strip()
    return str(value).strip()


def _source_to_target_stride(source_fps: float, target_fps: float) -> int:
    stride = int(round(source_fps / target_fps))
    if stride <= 0:
        raise ValueError("Invalid FPS ratio")
    return stride


def _translations(transforms: np.ndarray) -> np.ndarray:
    return np.asarray(transforms[..., :3, 3], dtype=np.float32)


def _transform_points(inverse_transforms: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    local = np.einsum("nij,nj->ni", inverse_transforms.astype(np.float32), homogeneous)
    return local[:, :3]


def _compute_spread_anchors(knuckle_positions_local: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    anchors: Dict[str, np.ndarray] = {}
    for finger in EXPECTED_FINGERS:
        anchors[finger] = knuckle_positions_local[finger]
    return anchors


def _angle_to_anchor(knuckle: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    dot = np.sum(knuckle * anchor, axis=1)
    denom = np.linalg.norm(knuckle, axis=1) * np.linalg.norm(anchor, axis=1)
    cosine = np.clip(dot / np.clip(denom, 1e-6, None), -1.0, 1.0)
    return np.arccos(cosine).astype(np.float32)


def _write_episode_h5(
    path: Path,
    *,
    episode: EgoDexEpisode,
    instruction: str,
    hand_name: str,
    timestamps: np.ndarray,
    wrist_transforms: np.ndarray,
    hand_descriptors: np.ndarray,
    actions: np.ndarray,
) -> None:
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["episode_id"] = episode.episode_id
        meta.attrs["task_id"] = episode.task_id
        meta.attrs["instruction"] = instruction
        meta.attrs["embodiment_id"] = "sharpa"
        meta.attrs["data_source"] = "human_retargeted"
        meta.attrs["active_hand"] = hand_name

        timestamps_group = handle.create_group("timestamps")
        timestamps_group.create_dataset("step", data=timestamps, compression="gzip")

        signals = handle.create_group("signals")
        signals.create_dataset("raw_state", data=np.zeros((actions.shape[0], 32), dtype=np.float32), compression="gzip")
        signals.create_dataset("actions", data=actions, compression="gzip")
        signals.create_dataset("active_wrist_pose", data=wrist_transforms[:-1], compression="gzip")
        signals.create_dataset("hand_descriptor", data=hand_descriptors[:-1], compression="gzip")


def _materialize_video(source: Path, destination: Path, *, mode: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    if mode == "hardlink":
        destination.hardlink_to(source.resolve())
        return
    if mode == "copy":
        import shutil

        shutil.copy2(source, destination)
        return
    raise ValueError(f"Unsupported video_mode: {mode}")


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _require_group(handle: h5py.File, key: str) -> h5py.Group:
    if key not in handle or not isinstance(handle[key], h5py.Group):
        raise KeyError(f"Missing HDF5 group: {key}")
    return handle[key]


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "item"
