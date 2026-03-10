# HF Mirror Model Download Runbook

更新时间：2026-03-11

## 目标

当服务器直接访问 Hugging Face 不稳定时，使用 `hf-mirror` 先把模型下载到本地目录，再通过本地路径运行 `egoscale_research`。

这份 runbook 基于服务器 `hex@114.212.165.225` 的一次真实排查整理，结论已经验证。

## 已验证可行的路径

### 1. 创建独立环境

建议不要复用服务器已有环境，尤其不要用 Python 3.13。

```bash
~/miniconda3/bin/conda create -y -n egoscale python=3.10
```

安装核心依赖：

```bash
~/miniconda3/bin/conda run -n egoscale pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
~/miniconda3/bin/conda run -n egoscale pip install transformers PyYAML Pillow numpy h5py imageio imageio-ffmpeg num2words wandb
~/miniconda3/bin/conda run -n egoscale pip install -e /home/hex/workspace/open-egoscale/egoscale_research
```

## 2. 用 `hf-mirror` 下载模型

服务器上 `huggingface_hub` 可用，但没有 `huggingface-cli` 可执行文件。最稳的办法是直接调用 `snapshot_download`。

先确认镜像可访问：

```bash
HF_ENDPOINT=https://hf-mirror.com \
~/miniconda3/bin/conda run -n egoscale python -c "import os; from huggingface_hub import HfApi; api=HfApi(endpoint=os.environ['HF_ENDPOINT']); info=api.model_info('HuggingFaceTB/SmolVLM2-500M-Video-Instruct'); print(info.id, len(info.siblings))"
```

再下载训练真正需要的文件，不要先整仓库全量拉取：

```bash
HF_ENDPOINT=https://hf-mirror.com \
~/miniconda3/bin/conda run -n egoscale python -c "from huggingface_hub import snapshot_download; p=snapshot_download(repo_id='HuggingFaceTB/SmolVLM2-500M-Video-Instruct', local_dir='/home/hex/models/SmolVLM2-500M-Video-Instruct', allow_patterns=['model.safetensors','config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','chat_template.json','added_tokens.json']); print(p)"
```

### 为什么不要先全量下载

这次排查里，先前的全量下载把很多 `onnx` 文件也拉下来了，目录膨胀到 `7.1G`，但训练实际只需要：

- `model.safetensors`
- tokenizer / processor / config 文件

真正可用、清理后的目录大小约为 `1.9G`。

## 3. 清理多余文件

如果之前已经误下了 `onnx` 和中间缓存，可以这样清：

```bash
rm -rf /home/hex/models/SmolVLM2-500M-Video-Instruct/onnx
rm -rf /home/hex/models/SmolVLM2-500M-Video-Instruct/.cache
du -sh /home/hex/models/SmolVLM2-500M-Video-Instruct
```

清理后，保留这些文件即可：

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `preprocessor_config.json`
- `processor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.json`
- `merges.txt`
- `chat_template.json`
- `added_tokens.json`

## 4. 验证本地路径可加载

```bash
cd /home/hex/workspace/open-egoscale/egoscale_research
~/miniconda3/bin/conda run -n egoscale env \
  PYTHONPATH=src \
  EGOSCALE_VLM_BACKBONE_NAME=/home/hex/models/SmolVLM2-500M-Video-Instruct \
  python -c "from egoscale.model.vlm_backbone import build_vlm_backbone; from egoscale.config import ExperimentConfig; cfg=ExperimentConfig.from_yaml('configs/stage1_smolvlm_500m_smoke.yaml'); m=build_vlm_backbone(cfg.model); print(type(m).__name__); print('LOAD_OK')"
```

这一步已经验证通过。

## 5. 重跑 smoke 的正确方式

明确绑定一张较空的 GPU，并显式指定本地模型目录：

```bash
cd /home/hex/workspace/open-egoscale/egoscale_research
~/miniconda3/bin/conda run -n egoscale env \
  CUDA_VISIBLE_DEVICES=5 \
  OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=src \
  EGOSCALE_VLM_BACKBONE_NAME=/home/hex/models/SmolVLM2-500M-Video-Instruct \
  python scripts/train_stage1.py \
    --config configs/stage1_smolvlm_500m_smoke.yaml \
    --dataset /home/hex/workspace/mini_egodex/meta/chunks_stage1.jsonl \
    --checkpoint /home/hex/workspace/open-egoscale/egoscale_research/tmp/stage1_smolvlm_500m_smoke.pt
```

## 当前已确认的问题

模型下载问题已经排除，但 `SmolVLM` smoke 仍然会在前向阶段失败：

```text
ValueError: At least one sample has <image> tokens not divisible by patch_size.
```

这说明当前阻塞点已经从“网络 / 模型下载失败”转成了“`SmolVLMBackbone` 输入构造与 processor 契约不匹配”。

换句话说：

- `hf-mirror` 路径是通的
- 本地模型目录是可加载的
- 数据最小样本是可读的
- 当前剩余问题是代码层面的 `SmolVLM` prompt / image token 对齐

## 下次直接照着做

1. 建 `egoscale` 环境。
2. 用 `HF_ENDPOINT=https://hf-mirror.com` + `snapshot_download(..., allow_patterns=[...])` 下载核心文件。
3. 删除 `onnx` 和模型目录下 `.cache`。
4. 用 `EGOSCALE_VLM_BACKBONE_NAME=/home/hex/models/SmolVLM2-500M-Video-Instruct` 先做本地加载验证。
5. 再跑 smoke。

这样可以绕开这次排查里遇到的两个低效路径：

- 直接在线加载，卡在 Hugging Face 访问
- 先全量下载，把大量 `onnx` 冗余文件也拖下来
