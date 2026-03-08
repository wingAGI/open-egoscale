# EgoScale x VLM Spec V2

这份文档用于固定当前讨论结论：EgoScale 的实现应当 `load` 一个 pretrained VLM，而不是直接 `load` 官方 `GR00T-N1.5-3B` 这种完整 VLA。`GR00T` 的价值主要在于提供可参考的 VLA 架构组织、multi-modal 接口和 diffusion-style action head 设计。

## 当前目标

- 以 pretrained VLM 为真实底座，能实际 `load`
- 架构尽量接近 EgoScale 论文
- 输入支持多相机图像 + 文本 + proprio
- 动作头保留 flow-matching / DiT 风格
- 训练流程拆成 Stage I / II / III
- synthetic 数据格式尽量贴 LeRobot / 官方 modality 组织
- 代码设计参考 GR00T，但不要求 checkpoint 兼容

## 设计边界

明确不做的事情：

- 不直接 `load` 官方 `GR00T-N1.5-3B`
- 不要求复用官方 `prepare_input -> backbone -> action_head` 全链路
- 不要求复用官方 pretrained 的 state/action projector
- 不为第一版引入显式 `view embedding`
- 不为第一版引入跨 `action_dim` 的同 batch 混训

## 参考基线

基于 `Isaac-GR00T` `n1.5-release` 源码确认，以下部分值得参考：

- visual tokens -> policy head 的组织方式
- flow-matching action head 的输入接口
- 多视角图像先组织再展平的处理路径
- `embodiment_id` 驱动 embodiment-specific adapter 的思路

需要强调：

- GR00T 是参考架构，不是要直接加载的底模
- EgoScale 的 backbone 应是 pretrained VLM
- EgoScale 的 state/action projector 应独立实现并从头训练

### 关键参考源码位置

- `/tmp/Isaac-GR00T-n15/gr00t/model/backbone/eagle_backbone.py`
- `/tmp/Isaac-GR00T-n15/gr00t/model/action_head/flow_matching_action_head.py`
- `/tmp/Isaac-GR00T-n15/gr00t/model/transforms.py`

## EgoScale 相对参考基线的核心约束

只改这几类：

1. backbone 改成 pretrained VLM，而不是 GR00T 整体 VLA
2. 训练 recipe 改成 Stage I / II / III，而不是单阶段 finetune
3. state projector / action projector 不复用预训练 VLA，而是从头训练
4. human 数据视为 retarget 到 `sharpa` embodiment 的数据
5. `g1` 保留独立本体适配和独立动作维度
6. 冻结策略按阶段配置

不改的地方：

- 动作头仍采用 flow-matching / DiT 风格
- 多视角输入仍沿用先组织再展平的思路
- `embodiment_id` 机制保留
- 第一版不引入显式 `view embedding`

## 代码组织

```text
egoscale_research/
  configs/
    stage1.yaml
    stage2.yaml
    stage3.yaml
    model.yaml
    data.yaml
  scripts/
    prepare_synth_data.py
    train_stage1.py
    train_stage2.py
    train_stage3.py
    eval_open_loop.py
    inspect_batch.py
  src/egoscale/
    model/
      vlm_backbone.py
      state_projector.py
      action_projector.py
      action_head.py
      embodiment_adapter.py
      policy.py
    data/
      dataset.py
      transforms.py
      collate.py
      schema.py
    trainer/
      base.py
      stage1.py
      stage2.py
      stage3.py
    utils/
      checkpoint.py
      dist.py
      seed.py
```

## 模型结构

顶层结构保持为：

```text
pretrained VLM backbone
  -> state projector
  -> action projector / conditioning
  -> flow-matching action head
  -> minimal embodiment-specific adapters
```

### 1. `VLMBackbone`

职责：

- 调用 pretrained VLM `from_pretrained(...)`
- 输出 visual-text context tokens
- 对外暴露稳定的 token interface
- 不耦合 embodiment-specific 逻辑

### 2. `StateProjector`

职责：

- 接收 robot proprio 或 placeholder state
- 投影到 policy 使用的 state token / state feature 空间
- 从头训练，不复用任何预训练 VLA 的 state projector

说明：

- `sharpa` / `g1` 可共享大框架，但允许 embodiment-specific adapter
- 若样本没有真实 proprio，可走 placeholder path

### 3. `ActionProjector`

职责：

- 负责 action target 的输入输出适配
- 负责 noisy action / action token 的统一接口
- 从头训练，不复用任何预训练 VLA 的 action projector

### 4. `FlowMatchingActionHead`

职责：

- 保留 DiT-style flow matching 预测范式
- 输入 visual-text context、state feature、noisy action、timestep、future tokens
- 输出动作预测

### 5. `EmbodimentAdapter`

职责：

- 基于 `embodiment_id` 选择对应路径
- 支持 `sharpa` 与 `g1` 的 state/action 适配
- 不把 `human` 当成独立 embodiment

## embodiment 定义

训练里只保留两类真实 embodiment：

- `sharpa`
- `g1`

其中：

- `human` 数据不是独立 embodiment
- `human` 数据是从人类视频转成的 `sharpa` 数据
- 因此 human-retargeted 样本训练时使用 `embodiment_id=sharpa`
- `is_human` 仅表示数据来源或是否缺失真实 robot proprio，不表示独立动作语义

## 动作空间

第一版按 embodiment 划分：

- `sharpa`: `28 = wrist_delta(6) + hand22`
- `g1`: 原生低维动作，可配置，例如 `13`

结论：

- human-retargeted 与 sharpa-native 数据共享同一 `sharpa` action space
- `g1` 保留独立动作维度
- 单个 batch 内只允许单一 `action_dim`

### batch 约束

必须固定：

- 同一个 batch 只能出现一个 embodiment bucket
- 同一个 batch 只能出现一个 `action_dim`
- `sharpa` 和 `g1` 的混合通过 alternating batches 或 interleaved steps 完成
- gradient accumulation 只用于放大有效 batch size，不用于解决动作维度不一致

## 多视角

第一版沿用官方“先组织、再展平”的范式，不额外加 `view embedding`。

### 固定视角顺序

必须写死 canonical order：

1. `head`
2. `left_wrist`
3. `right_wrist`

### 处理原则

- dataset schema 必须按上述顺序产出
- transform 里先组织成 `[V, T, C, H, W]`
- 再按统一顺序展平后送入 VLM visual processor
- modality config / eval / inference 必须共享同一顺序

由于没有显式 `view embedding`，视角语义完全依赖位置，因此顺序不能漂移。

## 数据 schema

synthetic 数据尽量贴 LeRobot / 官方 modality 配置。

建议目录：

```text
data/synthetic_egoscale/
  meta/
    modality_stage1.json
    modality_stage2.json
    modality_stage3.json
    tasks.json
    embodiments.json
  stage1_human_retarget_to_sharpa/
  stage2_aligned/
  stage3_sharpa/
  stage3_g1/
```

每条 episode 至少包含：

- 3 路图像，顺序固定为 `head, left_wrist, right_wrist`
- instruction
- `state` 或 `use_placeholder_state=true`
- `action`
- `embodiment_id`
- `task_id`

关键字段：

- `is_human`
- `embodiment_id`
- `use_placeholder_state`
- `action_dim`
- `action_mask`

补充约束：

- `is_human=true` 的样本若来自 retarget 数据，仍然使用 `embodiment_id=sharpa`
- `action_mask` 只用于同一 `action_dim` 内的局部缺失或裁剪，不用于在同 batch 混不同动作维度

## 三阶段训练

### Stage I

数据：

- human-retargeted-to-sharpa only

训练目标：

- 学 human manipulation prior
- 学 `sharpa` 28D 动作空间上的 flow matching
- 建立 VLM visual-text tokens 到动作生成的基本映射

训练模块：

- VLM visual/projector: train 或 LoRA
- VLM language backbone: 默认冻结
- state projector: train
- action projector: train
- action head DiT: train
- placeholder state path: train

### Stage II

数据：

- aligned human-retargeted-to-sharpa + sharpa-native mixture

训练目标：

- 将 human prior 锚定到真实 robot sensing/control
- 对齐 retarget sharpa 数据和原生 sharpa 数据

训练模块：

- state projector: train
- action projector: train
- action head DiT: train
- VLM visual/projector: train 或 LoRA
- VLM language backbone: 默认冻结

### Stage III

数据：

- task-specific robot demos
- 可分 `stage3_sharpa` 与 `stage3_g1`

训练目标：

- task specialization
- 针对不同 embodiment 做末端适配

训练模块：

- 对应 embodiment adapter
- state projector
- action projector
- action head DiT
- 视觉侧小幅开放

## 配置原则

保留通用配置思想：

- `tune_visual`
- `tune_llm`
- `tune_projector`
- `tune_diffusion_model`

新增 EgoScale 配置：

- `vlm_backbone_name`
- `use_placeholder_state`
- `stage_recipe`
- `mix_human_robot_ratio`
- `canonical_view_order`
- `embodiment_buckets`
- `run_g1_path`

## 训练实现约束

第一版明确采用：

- sharpa bucket 和 g1 bucket 分开组 batch
- trainer 可以在 step 级别交替采样不同 bucket
- 不在一次 forward 中混合不同 `action_dim`
- gradient accumulation 允许开启，但仅作为优化手段

## 评估

第一版只做 open-loop：

- action MSE
- wrist MSE
- hand MSE
- stage2_on_stage3_val
- g1_stage2 / g1_stage3

不做：

- 闭环环境
- 真实 rollout success
- 真实机器人控制

## 最大风险

1. pretrained VLM 的选择和 token interface 需要本机实测，不能只靠文档推断
2. VLM 依赖版本、显存占用和 processor 接口可能比预期更敏感
3. placeholder state 插入位置需要谨慎，否则会污染 state projector 设计
4. synthetic 数据要足够贴近官方 modality 组织，否则后续替换真实数据会痛苦
5. retarget human data 和 native sharpa data 的统计差异可能会影响 Stage II 对齐

## 当前建议的实现顺序

1. 选定并验证一个 pretrained VLM 的本机加载与最小 forward
2. 打通多视角 + 文本 -> VLM token interface
3. 实现 state projector / action projector / action head
4. 插入 placeholder state path
5. 写 Stage I / II / III trainers
6. 加入 sharpa / g1 bucket 训练逻辑
7. 跑 synthetic 全流程

## 暂不实现的内容

- 真实人手关键点恢复
- 真实 retarget pipeline
- 显式 domain adaptation loss
- FLARE 训练
- RL / 仿真闭环

## 当前决策摘要

- 真实底座是 pretrained VLM，不是 GR00T VLA
- GR00T 主要作为架构参考
- 不引入显式 `view embedding`
- 视角顺序固定为 `head, left_wrist, right_wrist`
- `human` 不作为独立 embodiment
- human-retargeted 数据按 `sharpa` embodiment 训练
- `state projector` 与 `action projector` 从 Stage I 开始训练
- `g1` 保留独立动作维度与独立适配
- batch 内 `action_dim` 必须恒定
