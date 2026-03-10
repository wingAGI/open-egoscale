# EgoScale Research

Lightweight reference implementation for the `spec_v2.md` EgoScale VLA stack:

- pretrained VLM backbone
- embodiment-specific state and action adapters
- flow-matching action head
- Stage I / II / III training scripts

## Layout

```text
egoscale_research/
  configs/
  scripts/
  src/egoscale/
  tests/
```

## Install

From `egoscale_research/`:

```bash
pip install -e .
```

The Qwen2.5-VL and SmolVLM2 paths require a `transformers` build with the relevant model classes.

## Backbone Configuration

The default stage configs use:

- `model.backbone_impl: qwen2_5_vl`
- `model.vlm_backbone_name: Qwen/Qwen2.5-VL-3B-Instruct`
- `model.vlm_token_dim: 2048`

The repository also supports:

- `model.backbone_impl: smolvlm`
- `model.vlm_backbone_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- `model.vlm_backbone_name: HuggingFaceTB/SmolVLM2-2.2B-Instruct`

For SmolVLM2, the code path uses:

- `vision_model` as the visual encoder freeze group
- `connector` as the multimodal adapter freeze group
- `text_model` as the language backbone freeze group

SmolVLM2-specific resizing knobs live under `model`:

- `smolvlm_do_resize`
- `smolvlm_resize_longest_edge`
- `smolvlm_do_image_splitting`
- `smolvlm_max_image_size_longest_edge`

If the weights already exist on disk, override the backbone path through:

```bash
export EGOSCALE_VLM_BACKBONE_NAME=/path/to/Qwen2.5-VL-3B-Instruct
```

When `EGOSCALE_VLM_BACKBONE_NAME` points to a local directory, the loader switches to offline local-file mode automatically.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Training

Stage scripts:

```bash
PYTHONPATH=src python scripts/train_stage1.py --config configs/stage1.yaml --dataset /path/to/stage1.jsonl --checkpoint /path/to/stage1.pt
PYTHONPATH=src python scripts/train_stage2.py --config configs/stage2.yaml --dataset /path/to/stage2.jsonl --checkpoint /path/to/stage2.pt
PYTHONPATH=src python scripts/train_stage3.py --config configs/stage3.yaml --dataset /path/to/stage3.jsonl --checkpoint /path/to/stage3.pt
```

Stage I can optionally run held-out validation during training:

```bash
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/stage1.yaml \
  --dataset /path/to/stage1_train.jsonl \
  --val-dataset /path/to/stage1_val.jsonl \
  --checkpoint /path/to/stage1.pt
```

Relevant config keys live under `training`:

- `grad_accum_steps`
- `eval_interval`
- `log_interval`
- `max_val_batches`
- `wandb_enabled`
- `wandb_project`
- `wandb_entity`
- `wandb_run_name`
- `wandb_mode`

Stage-specific dataset constraints are configurable in each YAML under `data.stage_sample_rules`.
This is the layer that decides which combinations of `embodiment_id`, `data_source`, `has_proprio`,
semantics names, action dimensions, and camera-view sets are legal for Stage I / II / III.
You can switch a stage from the default Sharpa contract to an SO101 or EgoDex-specific contract
without editing validator code in `src/egoscale/data/schema.py`.

For a real-backbone smoke test with Qwen2.5-VL, use:

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/stage1_qwen_smoke.yaml \
  --dataset /path/to/stage1_manifest.jsonl \
  --checkpoint /path/to/stage1_qwen_smoke.pt
```

For SmolVLM2 smoke tests, use:

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/stage1_smolvlm_500m_smoke.yaml \
  --dataset /path/to/stage1_manifest.jsonl \
  --checkpoint /path/to/stage1_smolvlm_500m_smoke.pt
```

Single-node multi-GPU training uses `torchrun`. The configured `training.batch_size` is the
per-process micro-batch, and the effective global batch is:
`world_size * training.batch_size * training.grad_accum_steps`.

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node=4 scripts/train_stage1.py \
  --config configs/stage1.yaml \
  --dataset /path/to/stage1_train.jsonl \
  --val-dataset /path/to/stage1_val.jsonl \
  --checkpoint /path/to/stage1_ddp.pt
```

Batch inspection:

```bash
PYTHONPATH=src python scripts/inspect_batch.py --config configs/stage2.yaml --dataset /path/to/data.jsonl
```

EgoDex Stage I dataset conversion into the repository's `meta/data/videos` layout:

```bash
PYTHONPATH=src python scripts/convert_egodex_stage1.py \
  --input-root /path/to/egodex/train \
  --output-root /path/to/output/egodex_stage1_v1
```

## Lightweight Smoke Mode

For 3B Qwen smoke tests on limited VRAM, set:

```yaml
training:
  lightweight_vlm_freeze: true
```

This freezes:

- `vlm_visual_encoder`
- `vlm_multimodal_adapter`
- `vlm_language_backbone`

and keeps the stage-specific state/action stack trainable, which is enough to validate:

- dataset loading
- transforms and collate
- forward/backward
- optimizer step
- checkpoint save

## Verified Path

The following path was verified against a real Qwen2.5-VL-3B checkpoint:

- `Qwen25VLBackbone` loads from a local checkpoint directory
- invalid camera slots are masked out correctly in `context_mask`
- `train_stage1.py`, `train_stage2.py`, and `train_stage3.py` complete 1-step smoke runs with `training.lightweight_vlm_freeze: true`
