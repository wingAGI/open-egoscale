# Open EgoScale 数据目录设计

这份文档只讨论一件事：

- `open-egoscale` 的训练数据在磁盘上应该怎么组织

它不讨论动作语义本身，也不要求你把 `EgoScale` 的 schema 改成别的格式。  
这里讲的是**存储层**，也就是：

- 文件怎么分目录
- 视频怎么存
- 元数据怎么存
- 时序数组怎么存
- train / val / test 怎么索引

## 1. 设计目标

这套目录设计主要服务下面几件事：

- 先支持 `Stage I / II / III` 三阶段训练
- 先支持 `EgoDex -> Stage I`
- 不把所有内容都塞进一个超大的 `jsonl`
- 便于后续换 action semantics、换 embodiment、换数据源
- 便于后续接 `SO101`、真实机器人和 aligned human data

## 2. 总体原则

建议把数据拆成三层：

1. `meta`
   放索引、schema、统计信息、split、task 信息
2. `data`
   放低维时序数据和 chunk 索引
3. `videos`
   放原始或处理后的视频文件

也就是说：

- `meta` 负责“怎么找到数据”
- `data` 负责“时序数值具体是什么”
- `videos` 负责“图像内容在哪里”

## 3. 推荐目录结构

建议在项目里统一采用下面这套结构：

```text
dataset_name/
  meta/
    info.json
    splits.json
    tasks.jsonl
    episodes.jsonl
    chunks_stage1.jsonl
    chunks_stage2.jsonl
    chunks_stage3.jsonl
    stats/
      action_stats_stage1.json
      action_stats_stage2.json
      state_stats_stage2.json
      state_stats_stage3.json
  data/
    episodes/
      episode_000001.h5
      episode_000002.h5
    shards/
      stage1_chunk_000.parquet
      stage1_chunk_001.parquet
      stage2_chunk_000.parquet
      stage3_chunk_000.parquet
  videos/
    egodex/
      episode_000001_head.mp4
      episode_000002_head.mp4
    aligned/
      episode_100001_head.mp4
      episode_100001_left_wrist.mp4
      episode_100001_right_wrist.mp4
    robot/
      episode_200001_head.mp4
      episode_200001_left_wrist.mp4
      episode_200001_right_wrist.mp4
```

这是推荐结构，不是硬约束。  
关键是三层职责要清楚，不要混在一起。

## 4. `meta` 目录放什么

### 4.1 `info.json`

放整个数据集的全局信息，例如：

- 数据集名称
- 版本号
- 创建时间
- 支持的 stage
- 支持的 embodiment
- 默认 `obs_horizon`
- 默认 `action_horizon`
- 默认控制频率
- canonical view order

示例：

```json
{
  "dataset_name": "egodex_stage1_v1",
  "version": "0.1.0",
  "stages": ["stage1"],
  "embodiments": ["human"],
  "canonical_view_order": ["head", "left_wrist", "right_wrist"],
  "control_frequency_hz": 10,
  "obs_horizon": 2,
  "action_horizon": 8
}
```

### 4.2 `splits.json`

放 train / val / test 的 episode 或 chunk 划分规则。

示例：

```json
{
  "train_episode_ids": ["episode_000001", "episode_000002"],
  "val_episode_ids": ["episode_000101"],
  "test_episode_ids": ["episode_000201"]
}
```

建议按 **episode** 划分，不要按 chunk 随机打散，否则很容易泄漏相邻窗口。

### 4.3 `tasks.jsonl`

每行一个 task，描述 task 级别信息：

- `task_id`
- `task_name`
- `source_dataset`
- `language`
- `stage_tags`

### 4.4 `episodes.jsonl`

每行一个 episode，描述 episode 级别信息：

- `episode_id`
- `task_id`
- `source_dataset`
- `data_source`
- `embodiment_id`
- `video_paths`
- `episode_file`
- `num_frames`
- `duration_sec`
- `control_frequency_hz`
- `available_views`
- `language`

这个文件是全局目录表，后面所有 chunk 都可以回指到 episode。

### 4.5 `chunks_stage*.jsonl`

这是最重要的 manifest 文件。  
每一行表示一个训练样本，也就是一个 chunk。

它不应该塞入完整图像和大数组，而应该只存：

- chunk 对应的 `episode_id`
- 当前 stage
- `embodiment_id`
- `data_source`
- `state_semantics_name`
- `action_semantics_name`
- chunk 的时间范围或 offset
- 对应 `episode.h5` 里的索引位置
- 对应视频文件路径或视频 id

也就是说：

- `jsonl` 在这里是**索引层**
- 不是最终主数据载体

### 4.6 `stats/`

放归一化统计量：

- state mean/std
- action mean/std
- 可选 min/max

这些统计只用 train split 计算，不要混进 val/test。

## 5. `data` 目录放什么

### 5.1 为什么不能全塞进 `jsonl`

因为训练数据里真正大的东西通常有三类：

- 图像帧
- 长时序 pose / state / action 数组
- 分 chunk 后的大量重复窗口

如果都写进 JSONL，会有几个问题：

- 文件很大
- 解析慢
- 随机读差
- 压缩比低
- 很难长期维护

所以 `jsonl` 只适合做索引，不适合做主存储。

### 5.2 `episodes/*.h5`

推荐把每个 episode 的低维时序主数据放到一个 `h5` 文件里。

适合放进 `episode_xxx.h5` 的内容包括：

- timestamps
- wrist poses
- hand descriptors
- raw_state
- actions
- optional confidences
- optional language fields

对 `Stage I EgoDex`，一个 episode h5 可以放：

- 重采样后的 `camera_timestamps`
- active hand wrist poses
- 22D hand descriptor
- chunkable 的 28D pseudo-action 序列

对 `Stage II/III`，也可以继续放：

- robot proprio
- robot native action
- aligned human action

这样训练时不需要每次重新解析原始 `hdf5`。

### 5.3 `shards/*.parquet`

如果后期数据规模上来，可以再加一层按 chunk 预展开的 `parquet` shard。

适合放：

- chunk 级别的小表格字段
- chunk 对应的数组偏移
- 简短的数值特征

但不建议把大图像直接存到 parquet 里。

对于你现在的阶段，这一层可以先不做。  
先有：

- `episodes/*.h5`
- `chunks_stage*.jsonl`

就够了。

## 6. `videos` 目录放什么

视频统一放在 `videos/` 下，按来源或阶段分子目录。

建议原则：

- 原始视频保留 `mp4`
- 每个 view 单独一个视频文件
- 文件名里包含 episode id 和 view name

示例：

```text
videos/
  egodex/
    episode_000001_head.mp4
  aligned/
    episode_100001_head.mp4
    episode_100001_left_wrist.mp4
    episode_100001_right_wrist.mp4
  robot/
    episode_200001_head.mp4
    episode_200001_left_wrist.mp4
    episode_200001_right_wrist.mp4
```

为什么建议保留独立视频，而不是抽帧全存进 h5：

- `mp4` 更省空间
- 更适合长期归档
- 更容易复查原始观测
- 和主流机器人数据集的做法一致

如果后面发现解码瓶颈明显，再考虑缓存抽帧结果。

## 7. 三阶段怎么共用这套结构

### 7.1 Stage I

`Stage I` 的典型组织：

- `meta/chunks_stage1.jsonl`
- `data/episodes/episode_*.h5`
- `videos/egodex/*.mp4`

其中：

- `videos` 保留头戴第一视角视频
- `episode.h5` 保存重采样后的 pose 和 pseudo-action
- `chunks_stage1.jsonl` 只保存 chunk 索引

### 7.2 Stage II

`Stage II` 的典型组织：

- `meta/chunks_stage2.jsonl`
- `data/episodes/episode_*.h5`
- `videos/aligned/*.mp4`
- `videos/robot/*.mp4`

这里一部分 episode 是：

- aligned human

另一部分是：

- robot native

manifest 通过 `data_source` 区分两者。

### 7.3 Stage III

`Stage III` 的典型组织：

- `meta/chunks_stage3.jsonl`
- `data/episodes/episode_*.h5`
- `videos/robot/*.mp4`

这里基本就是目标任务机器人的专用数据。

## 8. 推荐的 chunk manifest 字段

建议 `chunks_stage1.jsonl` / `chunks_stage2.jsonl` / `chunks_stage3.jsonl` 每行至少包含：

- `chunk_id`
- `episode_id`
- `task_id`
- `stage`
- `instruction`
- `embodiment_id`
- `data_source`
- `has_proprio`
- `state_semantics_name`
- `action_semantics_name`
- `state_dim`
- `action_dim`
- `obs_horizon`
- `action_horizon`
- `obs_timestamps`
- `action_timestamps`
- `camera_views`
- `episode_file`
- `video_refs`
- `obs_index_start`
- `action_index_start`

注意：

- `episode_file` 指向 `data/episodes/episode_xxx.h5`
- `video_refs` 指向 `videos/...`
- `obs_index_start` / `action_index_start` 用来从 `h5` 里切出对应窗口

这样一条 manifest 非常轻，但已经足够 loader 定位所有实际数据。

## 9. `episode.h5` 里建议保存的字段

推荐按 episode 存这些数据：

```text
/meta
  episode_id
  task_id
  instruction
  embodiment_id
  data_source

/timestamps
  obs
  action

/views
  head/frame_timestamps
  left_wrist/frame_timestamps
  right_wrist/frame_timestamps

/signals
  raw_state
  actions
  active_wrist_pose
  hand_descriptor
  confidence
```

如果是 `Stage I EgoDex`：

- `raw_state` 可以全零
- `actions` 是你构造的 `28D pseudo-action`
- `active_wrist_pose` 和 `hand_descriptor` 可以保留，方便以后复查

## 10. 为什么这是现在最合适的折中

这套结构的好处是：

- 比纯 JSONL 稳定
- 比一上来做复杂分布式 shard 简单
- 易于 debug
- 易于后续改 schema
- 易于把原始视频和训练信号分离

它特别适合你现在这个阶段：

- 先做 `EgoDex -> Stage I`
- 再做小规模 `Stage II/III`

## 11. 当前项目的建议落地路径

如果现在就开始做，建议目录直接建成这样：

```text
open-egoscale/
  datasets/
    egodex_stage1_v1/
      meta/
      data/
      videos/
    aligned_so101_v1/
      meta/
      data/
      videos/
    robot_so101_tasks_v1/
      meta/
      data/
      videos/
```

其中：

- `egodex_stage1_v1` 只服务 `Stage I`
- `aligned_so101_v1` 只服务 `Stage II`
- `robot_so101_tasks_v1` 只服务 `Stage III`

训练时由各自的 `chunks_stage*.jsonl` 作为入口。

## 12. 最后的结论

建议你现在不要用：

- 一个超大 JSONL，把图像、动作、状态全塞进去

而应该用：

- `meta/` 放索引和统计
- `data/` 放 episode 级数值数据
- `videos/` 放原始或处理后 mp4

对当前 `open-egoscale`，这已经足够兼顾：

- 实验推进速度
- 数据可维护性
- 后续扩展到 `SO101` 和更多 stage 的空间
