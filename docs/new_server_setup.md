# 新服务器环境准备与迁移清单

更新时间：2026-03-09

## 目标

在新 `48G` 服务器上尽快恢复当前 `open-egoscale` 的 `Stage I` 训练环境，避免重复排查：

- 代码版本不一致
- 本地模型路径不一致
- 训练数据路径不一致
- `wandb` 未配置
- 多卡启动命令不一致

## 代码来源

项目已经推到 GitHub：

- `https://github.com/wingAGI/open-egoscale.git`

建议新服务器直接拉最新代码，而不是手工复制工作区。

```bash
cd /root/autodl-tmp/workspace
git clone https://github.com/wingAGI/open-egoscale.git
cd open-egoscale/egoscale_research
```

## 需要迁移的数据

### 训练数据

- `/root/autodl-tmp/egodex_stage1_fullcopy`

这是当前可直接用于 `Stage I` 的 `EgoDex -> meta/data/videos` 数据集。

### 模型权重

- `/root/models/Qwen2.5-VL-3B-Instruct`

训练时建议继续使用相同路径，这样不需要改配置，只需要设置：

```bash
export EGOSCALE_VLM_BACKBONE_NAME=/root/models/Qwen2.5-VL-3B-Instruct
```

## 当前环境建议导出

在旧服务器上先导出一份包版本清单，便于新服务器复现：

```bash
/root/miniconda3/bin/python -V
/root/miniconda3/bin/pip freeze > /root/autodl-nas/egoscale_pip_freeze.txt
```

至少要关注这些包版本：

- `torch`
- `torchvision`
- `transformers`
- `accelerate`
- `wandb`
- `h5py`
- `imageio`
- `imageio-ffmpeg`
- `PyYAML`

## 建议的路径约定

为了减少后续改配置，建议新服务器继续保持这些路径：

- 代码：
  - `/root/autodl-tmp/workspace/open-egoscale/egoscale_research`
- 训练数据：
  - `/root/autodl-tmp/egodex_stage1_fullcopy`
- 模型：
  - `/root/models/Qwen2.5-VL-3B-Instruct`

## wandb

如果需要在线看曲线，建议新服务器尽早完成：

```bash
/root/miniconda3/bin/wandb login
```

或者把旧服务器上的：

- `~/.netrc`

同步到新服务器。

## 多卡启动方式

当前项目已经支持：

- `gradient accumulation`
- 单机多卡 `DDP`

多卡入口统一使用：

```bash
/root/miniconda3/bin/python -m torch.distributed.run --nproc_per_node=<N> ...
```

训练时保留这些环境变量：

```bash
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src
```

## 新服务器上的最小验证顺序

### 1. 检查 GPU

```bash
nvidia-smi
```

确认：

- GPU 数量
- 单卡显存

### 2. 检查模型路径

```bash
ls -lh /root/models/Qwen2.5-VL-3B-Instruct
```

### 3. 检查训练数据

```bash
ls /root/autodl-tmp/egodex_stage1_fullcopy/meta
wc -l /root/autodl-tmp/egodex_stage1_fullcopy/meta/chunks_stage1.jsonl
```

### 4. 跑 1-step smoke

优先先跑：

- `Qwen2.5-VL`
- `DDP`
- `1 step`

确认：

- 模型能加载
- 分布式能启动
- checkpoint 能写出

## 经验结论

### 已确认可用

- `wandb online`
- `Stage I train/val`
- `gradient accumulation`
- `DDP`
- `Qwen2.5-VL` 单卡 smoke
- `Qwen2.5-VL` 2 卡 DDP smoke

### 已确认的限制

- `2 x 32GB` 跑 `Qwen2.5-VL-3B` `Stage I` 全参数训练会 OOM
- 因此新服务器的核心目标是更高显存，而不是继续调整当前 `32G` 机器

