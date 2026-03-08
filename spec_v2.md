# EgoScale x GR00T Spec V2

这份文档用于固定当前讨论结论：以官方 `Isaac-GR00T` / `GR00T-N1.5-3B` 为主线，只在 EgoScale 与官方明确不同的地方做修改。

## 当前目标

- 以官方 `GR00T-N1.5-3B` 为真实底座，能实际 `load`
- 架构尽量接近 EgoScale 论文
- 输入支持多相机图像 + 文本 + proprio
- 动作头保留官方 flow-matching / DiT 风格
- 训练流程拆成 Stage I / II / III
- synthetic 数据格式尽量贴 LeRobot / 官方 modality 组织

## 官方 GR00T 基线

基于 `Isaac-GR00T` `n1.5-release` 源码确认：

- backbone 输出核心是 `backbone_features` 和 `backbone_attention_mask`
- action head 是 `FlowmatchingActionHead`
- action head 的真实输入包括：
  - `backbone_features`
  - `state`
  - `embodiment_id`
  - `noisy action`
  - `timestep`
  - learned `future_tokens`
- multi-view 输入在 transform 中按 `[V, T, C, H, W] -> [(T * V), C, H, W]` 展平后送入 Eagle processor
- 官方没有显式 `view embedding`
- 官方没有单独的 `pooled_context`
- 本体差异主要通过 `embodiment_id` 驱动的 category-specific state/action encoder/decoder 实现

### 关键源码位置

- `/tmp/Isaac-GR00T-n15/gr00t/model/backbone/eagle_backbone.py`
- `/tmp/Isaac-GR00T-n15/gr00t/model/action_head/flow_matching_action_head.py`
- `/tmp/Isaac-GR00T-n15/gr00t/model/transforms.py`

## EgoScale 相对官方需要修改的地方

只改这几类：

1. 训练 recipe 改成 Stage I / II / III，而不是官方单阶段 finetune
2. human 数据没有真实 proprio，不再使用官方默认全零 state 方案，而是替换为 learnable placeholder state path
3. human 与 sharpa 共用 28D 统一动作语义
4. G1 保留独立本体适配
5. Stage II 显式做人类和机器人 aligned mixture
6. 冻结策略按阶段配置，而不是只沿用官方默认开关

不改的地方：

- backbone 仍然沿用官方 Eagle
- action head 仍然沿用官方 flow-matching DiT
- `future_tokens` 保留
- `embodiment_id` 机制保留
- 多视角输入组织尽量沿用官方 transform 路径

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
      gr00t_wrapper.py
      action_head_patch.py
      placeholder_state.py
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
official Eagle backbone
  -> official flow-matching action head
  -> minimal EgoScale patches
```

### 1. `GR00TBackboneWrapper`

职责：

- 调用官方 `GR00T_N1_5.from_pretrained(...)`
- 尽量复用官方 `prepare_input -> backbone -> action_head`
- 不额外发明 `pooled_context` 或 `view_tokens`

### 2. `EgoScaleActionHeadPatch`

基于官方 `FlowmatchingActionHead` 做最小 patch：

- human placeholder state path
- human / sharpa 共享 28D 动作空间
- stage-specific trainable flags

### 3. `PlaceholderStateAdapter`

当 `is_human=True`：

- 不用真实 proprio
- 使用 learnable placeholder state embedding

当 `is_human=False`：

- 走官方 `state_encoder(state, embodiment_id)`

## 动作空间

统一动作为：

- `human`: `28 = wrist_delta(6) + hand22`
- `sharpa`: `28 = wrist_delta(6) + hand22`
- `g1`: 原生低维动作，可配置，例如 `13`

结论：

- `human` 与 `sharpa` 共用 28D action space
- `embodiment_id` 仍不同，但 action 维度一致
- `g1` 保留单独适配

## 多视角

第一版沿用官方范式，不额外加 `view embedding`。

输入视角：

- `head`
- `left_wrist`
- `right_wrist`

处理原则：

- transform 里先组织成 `[V, T, C, H, W]`
- 再按官方方式展平后喂给 Eagle processor
- 不单独做 per-view head
- 不单独做 camera embedding

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
  stage1_human/
  stage2_aligned/
  stage3_sharpa/
  stage3_g1/
```

每条 episode 至少包含：

- 3 路图像
- instruction
- `state` 或 `is_human=true`
- `action`
- `embodiment_id`
- `task_id`

关键字段：

- `is_human`
- `embodiment_id`
- `state_mask`
- `action_mask`

## 三阶段训练

### Stage I

数据：

- human only

训练目标：

- 学 human manipulation prior
- 学 unified action space 上的 flow matching

训练模块：

- backbone visual/projector: train 或 LoRA
- backbone llm: 默认冻结
- action head DiT: train
- placeholder state: train

### Stage II

数据：

- aligned human + robot mixture

训练目标：

- 将 human prior 锚定到 robot sensing/control

训练模块：

- action head DiT: train
- state/action projector: train
- backbone visual/projector: train 或 LoRA
- backbone llm: 默认冻结

### Stage III

数据：

- task-specific robot demos

训练目标：

- task specialization

训练模块：

- 对应 embodiment 路径
- action head DiT
- 视觉侧小幅开放

## 配置原则

保留官方配置思想：

- `tune_visual`
- `tune_llm`
- `tune_projector`
- `tune_diffusion_model`

新增 EgoScale 配置：

- `use_human_placeholder_state`
- `stage_recipe`
- `mix_human_robot_ratio`
- `shared_human_sharpa_action_space`
- `run_g1_path`

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

1. 官方 `GR00T-N1.5-3B` 加载链条需要本机实测，不能只靠文档推断
2. 官方 repo 对依赖版本敏感，尤其是 CUDA / transformers / lerobot
3. human placeholder state 是我们相对官方的论文对齐 patch，需要谨慎插入
4. synthetic 数据要足够贴近官方 modality 组织，否则后续替换真实数据会痛苦

## 当前建议的实现顺序

1. 验证官方 `GR00T-N1.5-3B` 本机加载与最小 forward
2. 复刻官方最小数据流：多视角 + 文本 + state + action
3. 插入 human placeholder state patch
4. 写 Stage I / II / III trainers
5. 加入 human/sharpa shared 28D 与 G1 分支
6. 跑 synthetic 全流程

## 暂不实现的内容

- 真实人手关键点恢复
- 真实 retarget pipeline
- 显式 domain adaptation loss
- FLARE 训练
- RL / 仿真闭环

## 当前决策摘要

- 参考官方实现优先
- 不引入显式 `view embedding`
- 不引入显式 `embodiment token`
- 不保留单独 `pooled_context`
- human / sharpa 共用 28D action space
- g1 保留单独适配
- EgoScale 主要通过 stage recipe 和 placeholder state 体现
