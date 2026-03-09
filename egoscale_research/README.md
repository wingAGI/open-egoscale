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

The Qwen2.5-VL path requires a `transformers` build with `Qwen2_5_VLModel` support.

## Backbone Configuration

The default stage configs use:

- `model.backbone_impl: qwen2_5_vl`
- `model.vlm_backbone_name: Qwen/Qwen2.5-VL-3B-Instruct`
- `model.vlm_token_dim: 2048`

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

Batch inspection:

```bash
PYTHONPATH=src python scripts/inspect_batch.py --config configs/stage2.yaml --dataset /path/to/data.jsonl
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
