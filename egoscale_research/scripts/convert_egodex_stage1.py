from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Set

from _common import PROJECT_ROOT  # noqa: F401
from egoscale.data.egodex_conversion import (
    convert_episode,
    discover_egodex_episodes,
    write_dataset_info,
    write_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Root directory of the extracted EgoDex split")
    parser.add_argument("--output-root", required=True, help="Output dataset root following meta/data/videos layout")
    parser.add_argument("--dataset-name", default="egodex_stage1_v1")
    parser.add_argument("--dataset-version", default="0.1.0")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--video-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--action-semantics-name", default="sharpa_wristdelta_hand22_v1")
    parser.add_argument("--state-semantics-name", default="sharpa_proprio_v1")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume from existing manifests instead of resetting output")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    existing_episode_ids = load_existing_episode_ids(output_root) if args.resume else set()
    if not args.resume:
        reset_output_manifests(output_root)
    write_dataset_info(
        output_root,
        dataset_name=args.dataset_name,
        version=args.dataset_version,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        control_frequency_hz=args.target_fps,
    )

    episodes = discover_egodex_episodes(input_root)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise ValueError(f"No EgoDex episodes found under {input_root}")

    existing_chunk_counts = load_existing_chunk_counts(output_root) if args.resume else Counter()
    results = load_existing_results(output_root, existing_chunk_counts) if args.resume else []
    task_index = {}
    skipped = []
    for episode in episodes:
        if episode.episode_id in existing_episode_ids:
            continue
        try:
            result = convert_episode(
                episode,
                output_root,
                target_fps=args.target_fps,
                source_fps=args.source_fps,
                obs_horizon=args.obs_horizon,
                action_horizon=args.action_horizon,
                sample_stride=args.sample_stride,
                video_mode=args.video_mode,
                stage_action_semantics_name=args.action_semantics_name,
                stage_state_semantics_name=args.state_semantics_name,
                stage_state_dim=args.state_dim,
            )
        except Exception as exc:
            skipped.append({"episode_id": episode.episode_id, "task_id": episode.task_id, "reason": str(exc)})
            print(json.dumps({"episode_id": episode.episode_id, "status": "skipped", "reason": str(exc)}))
            continue
        results.append(result)
        existing_episode_ids.add(result.episode_id)
        task_index[result.task_id] = {"task_id": result.task_id, "task_name": episode.task_name, "source_dataset": "egodex"}
        print(json.dumps({"episode_id": result.episode_id, "num_steps": result.num_steps, "num_chunks": result.num_chunks}))

    tasks_path = output_root / "meta" / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as handle:
        for payload in sorted(task_index.values(), key=lambda item: item["task_id"]):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    train_ids, val_ids = split_episode_ids([result.episode_id for result in results], val_ratio=args.val_ratio, seed=args.seed)
    write_splits(output_root, train_ids, val_ids)

    summary = {
        "num_episodes": len(results),
        "num_train_episodes": len(train_ids),
        "num_val_episodes": len(val_ids),
        "num_chunks": sum(result.num_chunks for result in results),
        "num_skipped": len(skipped),
        "output_root": str(output_root),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if skipped:
        skipped_path = output_root / "meta" / "skipped_episodes.jsonl"
        with skipped_path.open("a", encoding="utf-8") as handle:
            for payload in skipped:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def reset_output_manifests(output_root: Path) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("episodes.jsonl", "tasks.jsonl", "chunks_stage1.jsonl", "splits.json", "skipped_episodes.jsonl"):
        path = meta_dir / filename
        if path.exists():
            path.unlink()


def load_existing_episode_ids(output_root: Path) -> Set[str]:
    episodes_path = output_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return set()
    with episodes_path.open("r", encoding="utf-8") as handle:
        return {json.loads(line)["episode_id"] for line in handle if line.strip()}


def load_existing_results(output_root: Path, chunk_counts: Counter[str]):
    episodes_path = output_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return []
    results = []
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            results.append(
                type(
                    "ExistingResult",
                    (),
                    {
                        "episode_id": row["episode_id"],
                        "task_id": row["task_id"],
                        "num_steps": row["num_steps"],
                        "num_chunks": int(chunk_counts.get(row["episode_id"], 0)),
                    },
                )()
            )
    return results


def load_existing_chunk_counts(output_root: Path) -> Counter[str]:
    chunks_path = output_root / "meta" / "chunks_stage1.jsonl"
    counts: Counter[str] = Counter()
    if not chunks_path.exists():
        return counts
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[row["episode_id"]] += 1
    return counts


def split_episode_ids(episode_ids: list[str], *, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = list(episode_ids)
    generator = random.Random(seed)
    generator.shuffle(ids)
    num_val = int(round(len(ids) * val_ratio))
    if num_val <= 0 and len(ids) > 1 and val_ratio > 0:
        num_val = 1
    val_ids = sorted(ids[:num_val])
    train_ids = sorted(ids[num_val:])
    return train_ids, val_ids


if __name__ == "__main__":
    main()
