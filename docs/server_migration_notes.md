# 32G Server 保留项与迁移建议

更新时间：2026-03-09

## 不建议删除

### `/root/autodl-tmp/egodex_stage1_fullcopy`

- 大小约 `15G`
- 这是当前已经完成转换、可直接用于 `Stage I` 训练的 `EgoDex -> meta/data/videos` 数据集
- 训练入口实际依赖它下面的：
  - `meta/chunks_stage1.jsonl`
  - `meta/splits.json`
  - `data/episodes/*.h5`
  - `videos/**/*.mp4`

### `/root/models/Qwen2.5-VL-3B-Instruct`

- 大小约 `8.2G`
- 这是已经验证可用的本地 `Qwen2.5-VL-3B-Instruct` 模型目录
- 如果新服务器不想重新下载模型，建议直接迁移

### `/root/autodl-tmp/egodex`

- 大小约 `19G`
- 这是原始 `EgoDex` 解压目录
- 如果只是继续训练，理论上可以不迁移
- 但如果后面还要：
  - 重做转换
  - 修改 `Stage I schema`
  - 重采样或更改 `obs/action horizon`
  那最好保留一份原始数据

## 最小迁移集

如果目标只是把当前 `Stage I` 训练迁移到新 `48G` 服务器，最小需要迁移：

### 代码

- 直接从 GitHub 拉取：
  - `https://github.com/wingAGI/open-egoscale.git`

### 训练数据

- `/root/autodl-tmp/egodex_stage1_fullcopy`

### 模型权重

- `/root/models/Qwen2.5-VL-3B-Instruct`

## 可不迁移

- `/root/autodl-tmp/egodex_stage1_sample`
- `/root/autodl-tmp/checkpoints/...`
- `/root/autodl-tmp/logs/...`
- `/root/autodl-tmp/workspace/egoscale_research`
  - 新服务器直接 `git clone` 即可

