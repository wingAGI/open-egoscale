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
- 不要求第一版在 spec 内固定评测 protocol

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
      action_decoder.py
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
  -> embodiment-specific state projector
  -> embodiment-specific action projector / conditioning
  -> flow-matching action head
  -> embodiment-specific action decoder
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

- `sharpa` / `g1` 各自维护一套独立 `StateProjector`
- 可以共享同一个模块模板，但参数不共享
- 每个 `StateProjector` 内部分成 `placeholder_adapter` / `proprio_adapter` + `shared_state_trunk`
- `placeholder_adapter` 与 `proprio_adapter` 先各自把输入映射到统一 `D_state`
- `shared_state_trunk` 负责把两类输入继续映射到 policy 使用的 state feature 空间
- 若样本没有真实 proprio，则走 `placeholder_adapter`
- 若样本有真实 proprio，则走 `proprio_adapter`
- `use_placeholder_state` 不作为数据集主字段保存，只在 runtime 内由 `has_proprio` 派生：`use_placeholder_state = not has_proprio`

对外接口必须固定为：

- `StateProjector.forward(raw_state, has_proprio, embodiment_id)`
- 输出 `state_tokens`
- `state_tokens` 形状固定为 `[B, N_state, D_model]`
- 第一版固定 `N_state=1`
- `D_model` 必须与 `vlm_token_dim` 一致

### 2.1 `raw_state` contract

第一版必须把 `raw_state` 的张量契约单独固定，否则 dataset / collate / trainer 无法稳定对齐。

- `raw_state` 是进入 `StateProjector` 前的 batch-level 连续值张量
- `raw_state` 形状固定为 `[B, state_dim]`
- `state_dim` 由 `state_semantics_name` 唯一决定，不由 `has_proprio` 动态改变
- `raw_state.dtype` 固定为 `float32`
- `has_proprio` 形状固定为 `[B]`，类型固定为 `bool`
- 第一版 `StateProjector` 不接收 ragged dict；dataset / collate 必须先把 state 整理成定长向量

owner 必须固定：

- dataset schema 负责保存 `state_semantics_name`、`state_dim`、`has_proprio`
- data transforms 负责按 `state_semantics_name` 对真实 proprio 做 state normalization
- 对 `has_proprio=false` 的 placeholder state，data transforms 禁止套用真实 proprio normalization stats，必须把占位零向量原样保留到 projector 输入
- collate 负责把每条样本整理成 `[state_dim]` 并 stack 成 `[B, state_dim]`
- `StateProjector` 只根据 `has_proprio` 选择 `placeholder_adapter` 或 `proprio_adapter`

第一版实现收口为：

- 每个 `embodiment_id` 只绑定一个默认 `state_semantics_name`
- v1 中：
  - `sharpa -> sharpa_proprio_v1`
  - `g1 -> g1_proprio_v1`
- `robot_native` 样本必须提供真实 `raw_state`
- `human_retargeted` 样本必须仍然 materialize 出与对应 embodiment 一致的 `raw_state` 形状，但其数值固定为零向量，占位用途而非物理测量
- `human_retargeted` 样本的 `has_proprio=false` 必须显式写入 schema，runtime 只能依赖该字段切换到 `placeholder_adapter`
- placeholder state 的规范值定义在 `StateProjector` 输入空间；因此进入 `StateProjector` 的 `raw_state[has_proprio=false]` 必须是精确零向量，而不是经 normalization 后得到的非零常量
- 第一版不允许用 `None`、空数组或变长 dict 表示缺失 proprio

推荐 registry 形态：

```yaml
state_semantics:
  name: <state_semantics_name>
  embodiment_id: <sharpa|g1>
  state_dim: <int>
  field_order: [<must-fill>]
  units: [<must-fill>]
  normalization:
    scheme: <meanstd|none>
    stats_id: <must-fill if applicable>
    stats_scope: <train_split_only>
```

补充约束：

- 同一个 batch 内，所有样本必须共享同一个 `state_semantics_name`
- 同一个 batch 内，`state_dim` 必须恒定
- 第一版推荐按 `(embodiment_id, state_semantics_name, action_semantics_name, has_proprio)` 分桶组 batch
- 若同一 embodiment 里同时存在 `has_proprio=true/false`，通过 alternating batches 或 interleaved steps 训练，不在一个 batch 内混合两条 state path
- `field_order` 一旦确定，dataset、normalization stats、checkpoint、eval 全链路必须保持一致

### 3. `ActionProjector`

职责：

- 负责 action target 的输入输出适配
- 负责 noisy action / action token 的统一接口
- 从头训练，不复用任何预训练 VLA 的 action projector

说明：

- `sharpa` / `g1` 各自维护一套独立 `ActionProjector`
- 可以共享同一个模块模板，但参数不共享
- `ActionProjector` 只负责输入侧：把 `noisy_actions` 编码到 latent action token 空间
- `ActionProjector.encode(noisy_actions, embodiment_id)` 输出 `action_tokens`
- `action_tokens` 形状固定为 `[B, action_horizon, D_model]`
- 第一版默认所有训练样本都提供完整有效的 `[action_horizon, action_dim]` 动作窗口
- 若后续保留裁剪或不完整窗口，再额外引入可选 `action_mask`

### 4. `FlowMatchingActionHead`

职责：

- 保留 DiT-style flow matching 预测范式
- 输入 `context_tokens`、`state_tokens`、`action_tokens`、`timestep`
- 输出 `pred_action_latents`

接口必须固定为：

- `FlowMatchingActionHead.forward(context_tokens, context_mask, state_tokens, action_tokens, timesteps)`
- `pred_action_latents` 形状固定为 `[B, action_horizon, D_model]`
- 第一版不再保留额外歧义命名 `future tokens`，统一使用 `action_tokens`

### 5. `ActionDecoder`

职责：

- 负责输出侧：把 latent action token 解码回 embodiment-specific action space
- 将共享 action head 的输出映射成不同 embodiment 的真实动作维度

接口必须固定为：

- `ActionDecoder.decode(pred_action_latents, embodiment_id)`
- 输出 `pred_velocities`
- `pred_velocities` 形状固定为 `[B, action_horizon, action_dim]`
- `sharpa` / `g1` 各自维护独立 `ActionDecoder`

### 6. `Flow matching training contract`

第一版训练目标显式固定，参考 GR00T 当前实现范式：

- 在 normalized action space 做 velocity prediction，不在第一版把主 loss 放到 latent space
- 设样本中的物理动作窗口为 `raw_actions`
- 在进入 policy 前，必须先按 `action_semantics_name` 对 `raw_actions` 做 action normalization，得到 `actions`
- 第一版规定：`ActionProjector`、`FlowMatchingActionHead`、`ActionDecoder`、flow matching noise path 全部工作在 normalized action space
- 采样高斯噪声 `noise ~ N(0, I)`
- 采样连续时间 `t in [0, 1)`
- 构造 `noisy_actions = (1 - t) * noise + t * actions`
- 监督目标固定为 `target_velocity = actions - noise`
- 训练时送入 action-side 网络的离散 timestep 固定为：
  - `tau = min(floor(t * num_timestep_buckets), num_timestep_buckets - 1)`
- `ActionProjector.encode(noisy_actions, embodiment_id)` 生成 `action_tokens`
- `FlowMatchingActionHead` 预测 `pred_action_latents`
- `ActionDecoder.decode(pred_action_latents, embodiment_id)` 解码得到 `pred_velocities`
- 主 loss 固定为 `MSE(pred_velocities, target_velocity)`
- 第一版默认所有样本都提供完整有效的动作窗口，因此主 loss 不要求 `action_mask`
- 若后续引入裁剪或不完整窗口，可额外加入可选 `action_mask`

补充约束：

- 第一版允许在物理动作空间保留原始语义与单位，但训练主链路必须统一切换到 normalized action space
- `target_velocity`、`pred_velocities`、采样初值 `noise` 的量纲都以 normalized action space 为准
- 若后续启用 `action_mask`，loss reduction 必须只在有效元素上做平均
- `ActionDecoder` 属于训练主链路，凡是 `action projector / action head` 被训练的阶段，`ActionDecoder` 默认也参与训练
- 只有在显式冻结整个 action-side projector bundle 时，`ActionDecoder` 才能随之冻结
- 第一版不要求 latent-level auxiliary loss；若后续加入，只能作为可选辅助项，不能替代 action-space 主 loss

### 7. `Flow matching inference contract`

第一版推理接口固定为显式 Euler rollout，并对齐当前 GR00T 实现风格：

- 配置项固定新增：
  - `num_inference_timesteps`
  - `num_timestep_buckets`
- 记 `N = num_inference_timesteps`
- 记 `dt = 1 / N`
- 在 normalized action space 中初始化：
  - `actions_0 ~ N(0, I)`
- 迭代步固定为 `k = 0, 1, ..., N - 1`
- 连续时间网格固定为：
  - `t_k = k / N`
- 因此：
  - `t=0` 被包含
  - `t=1` 不在迭代输入网格内
- 送入 action-side 网络的离散 timestep 固定为：
  - `tau_k = floor(t_k * num_timestep_buckets)`
- 第 `k` 步前向固定为：
  - `action_tokens_k = ActionProjector.encode(actions_k, embodiment_id)`
  - `pred_action_latents_k = FlowMatchingActionHead(context_tokens, context_mask, state_tokens, action_tokens_k, tau_k)`
  - `pred_velocity_k = ActionDecoder.decode(pred_action_latents_k, embodiment_id)`
- 第 `k` 步更新公式固定为：
  - `actions_{k+1} = actions_k + dt * pred_velocity_k`
- 第一版 sampling loop 内不额外做 `clip` / `clamp`
- 若后续要加入采样期裁剪，必须作为显式可配置项写入 config，而不是默认隐式行为
- 迭代结束后得到：
  - `pred_actions_normalized = actions_N`
- 在 policy 输出前，必须按当前 `action_semantics_name` 做 action unnormalization，得到物理空间中的 `pred_actions`

### 8. `EmbodimentAdapter`

职责：

- 基于 `embodiment_id` 选择对应路径
- 在 runtime 选择对应 embodiment 的 state/action projector 与 action decoder
- 不把 `human` 当成独立 embodiment

### 9. `Tensor contract summary`

第一版 policy 内部统一使用以下张量约定：

- `context_tokens`: `[B, N_ctx, D_model]`
- `context_mask`: `[B, N_ctx]`
- `state_tokens`: `[B, 1, D_model]`
- `action_tokens`: `[B, action_horizon, D_model]`
- `pred_action_latents`: `[B, action_horizon, D_model]`
- `pred_velocities`: `[B, action_horizon, action_dim]`
- `pred_actions`: `[B, action_horizon, action_dim]`

owner 必须固定：

- `VLMBackbone` 负责 `context_tokens/context_mask`
- `StateProjector` 负责 `state_tokens`
- `ActionProjector` 负责 `action_tokens`
- `FlowMatchingActionHead` 负责 `pred_action_latents`
- `ActionDecoder` 负责 `pred_velocities`
- sampling / ODE solver 负责从 `pred_velocities` 迭代得到 `pred_actions`

### 10. `VLM token interface`

第一版必须先固定一个最小可替换接口，避免 backbone 接入阶段反复返工。

`VLMBackbone.forward(...)` 对外只承诺：

- 输入：`images`, `image_mask`, `text`, `device`, `dtype`
- `images` 形状固定为 `[B, V, T_visual, C, H, W]`
- `image_mask` 形状固定为 `[B, V, T_visual]`，其中 `1` 表示真实图像，`0` 表示 dummy slot
- 缺失视角允许出现，但必须通过 `image_mask` 显式标识
- 文本输入统一为 instruction string，不在 policy 层直接处理 tokenizer 细节
- 输出：`context_tokens` 与 `context_mask`
- `context_tokens` 形状固定为 `[B, N_ctx, D]`
- `context_mask` 形状固定为 `[B, N_ctx]`
- policy / projector / action head 只能依赖这两个输出，不得依赖 backbone 私有中间层命名

其中 `T_visual` 在第一版必须显式绑定到观测时序配置：

- `T_visual = obs_horizon`
- 等价地，`T_visual = len(video_delta_indices)`
- 不允许把 `T_visual` 作为独立于 `obs_horizon` 的隐式自由变量
- 若后续真的需要“每个观测时刻再叠多帧”的 frame-stack 机制，必须额外引入新名字，不得继续复用 `T_visual`

ownership 必须固定：

- dataset / collate 只负责产出 `images`、`image_mask`、`text` 及其原始 batch 组织
- `VLMBackbone` 负责产出 `context_tokens` 与 `context_mask`
- trainer / eval 只能消费 `context_mask`，不得在 backbone 外部自行构造 `context_mask`

论文约束必须显式记入选型标准：

- Figure 2 明确把 policy 写成 `VLM backbone + DiT action expert + embodiment-specific adapters`
- Section 2.4 明确写到：
  - Stage I `fully unfreezing every parameter of the VLA model`
  - Stage II `freezing the vision-language backbone while only updating the vision encoder and DiT action expert`
  - Stage III `the vision encoder is frozen if mid-training is used and unfrozen otherwise`
- 附录 D.1 对跨 embodiment mid-training 的更细约束是：
  - `only the vision encoder, DiT action expert, and state-action encoder and decoder are updated, while the vision-language backbone remains frozen`

因此第一版 VLM 选型必须满足以下硬约束，而不只是“能跑通 forward”：

- 能本机 `from_pretrained(...)`
- 能接收多图输入或多图展平输入
- 能稳定产出 token-level 输出，而不是仅 pooled embedding
- 必须能把 trainable domain 至少拆成：
  - `vision_encoder`
  - `vlm_multimodal_adapter`
  - `language_backbone`
- 必须支持对上述模块分别 freeze / unfreeze；若模型只能整体 train/freeze，则不满足论文对应的 Stage II / III recipe
- `vlm_multimodal_adapter` 在第一版统一指代所有视觉侧接入参数，包括 visual projector、image-to-token bridge、`merger`、`mm_projector` 及其它把图像特征接到语言栈的适配层；它统一归类到 visual side，而不是 language side
- 必须能稳定给出“哪些 context token 来自哪些图像 slot”的可恢复映射，或给出每张图对应的固定 token span，从而把 `image_mask` 映射到 token-level `context_mask`
- 若某 VLM 只能接收 dummy image，但无法把 dummy slot 生成的 token 从 `context_mask` 中屏蔽，则不作为第一版 backbone
- 能明确区分哪些层可做 LoRA / finetune

第一版建议的选型方案：

1. bring-up backbone：
   - 首选 `Qwen/Qwen2.5-VL-3B-Instruct`
   - 目标是先打通 `from_pretrained -> multi-image input -> hidden states -> token interface`
   - 选择理由：官方 `transformers` 文档公开了 bare model / hidden states / image features 接口，工程接入成本最低
2. scale-up backbone：
   - 若 3B 路线稳定，再升级到 `Qwen/Qwen2.5-VL-7B-Instruct`
   - 仅在显存与吞吐允许时启用，不作为第一版最小可用实现的阻塞项
3. 不作为第一优先的候选：
   - 当前 vendor 里的 `Eagle` 更适合作为架构参考，不作为第一版默认真实底座
   - 原因不是它不能做 VLM，而是当前 spec 的核心目标是“一个可直接 `from_pretrained`、且能严格执行 Stage II / III 冻结策略的真实 backbone”

落地决策：

- v1 默认先实现 `Qwen2.5-VL-3B`
- 若本机实测发现 token span 恢复、多图输入或模块级冻结不满足约束，再更换 backbone，而不是在 policy 层打补丁规避

### 10.1 缺失视角与 token-level masking contract

第一版必须把“缺失视角怎么在 token 级别失效”写死，否则单路 / 多路样本混训会出现语义漂移。

问题本质：

- dataset 侧的 `image_mask` 只知道“这个 view slot 没有真实图像”
- 但很多 VLM 在看到 dummy image 后，仍然会为该图像生成一串正常长度的视觉 token
- 如果这些 token 没被进一步从 `context_mask` 中屏蔽，action head 看到的就不是“缺失视角”，而是“存在两个内容恒为 dummy 的额外相机”

第一版 contract 固定为：

- `image_mask=0` 的 slot 仍可 materialize 成 dummy image，用来满足 processor 的静态输入形状
- 但 `VLMBackbone` 必须继续把该 slot 对应的视觉 token 全部映射为 `context_mask=0`
- 也就是说，dummy image 只服务于张量对齐，不能在语义上充当真实视觉上下文
- 若 backbone 只能输出全 1 的 `context_mask`，则该 backbone 不满足第一版要求

最小例子：

- canonical order 固定为 `head, left_wrist, right_wrist`
- 某条样本只有 `head`
- dataset 会产出：
  - `head = real image`
  - `left_wrist = dummy image`
  - `right_wrist = dummy image`
  - `image_mask = [1, 0, 0]`
- 正确行为不是“让模型看到 3 路图，其中两路是黑图”
- 正确行为是“让模型只把 `head` 对应 token 视为有效上下文，其余两路 token 在 `context_mask` 中失效”

## embodiment 定义

训练里只保留两类真实 embodiment：

- `sharpa`
- `g1`

其中：

- `human` 数据不是独立 embodiment
- `human` 数据是从人类视频转成的 `sharpa` 数据
- 因此 human-retargeted 样本训练时使用 `embodiment_id=sharpa`
- `embodiment_id` 只描述本体 / 动作接口，不描述数据来源
- 第一版新增字段 `data_source` 描述样本来源；推荐取值：
  - `human_retargeted`
  - `robot_native`
- 第一版新增字段 `has_proprio` 描述样本是否真实提供 proprio
- `use_placeholder_state` 不作为数据集主字段保存，而在 runtime 内由 `has_proprio` 派生

第一版推荐的有效组合：

- `embodiment_id=sharpa, data_source=human_retargeted, has_proprio=false`
- `embodiment_id=sharpa, data_source=robot_native, has_proprio=true`
- `embodiment_id=g1, data_source=robot_native, has_proprio=true`

第一版不建议的组合：

- `embodiment_id=g1, data_source=human_retargeted`
- `embodiment_id=human, *`

## 动作空间

第一版按 embodiment 划分：

- `sharpa`: `28 = wrist_delta(6) + hand22`
- `g1`: 原生低维动作，可配置，例如 `13`

结论：

- human-retargeted 与 sharpa-native 数据共享同一 `sharpa` action space
- `g1` 保留独立动作维度
- 单个 batch 内只允许单一 `action_dim`

### 动作语义约束

第一版必须做 action normalization。

所有数据源必须先对齐到同一动作语义定义后再入库；shape 一致不代表动作语义一致。

这里的 `action_semantics_name` 不是“动作维度名字”，而是“整套动作语义定义的唯一标识”。它至少要同时固定：

- `embodiment_id`
- `action_dim`
- `wrist_delta(6)` 的坐标系
- `wrist_delta(6)` 的旋转参数化方式
- 平移单位
- 控制频率
- `hand22` 的 joint 顺序

可理解为：

- `sharpa_wristdelta_hand22_v1`
- `g1_lowdim_native_v1`

这类名字背后对应的是一整套可校验的动作语义真值表，而不只是一个字符串标签。

第一版 action normalization contract 固定为：

- normalization 的 owner 是 data processor / transforms，不是 trainer 内联逻辑
- 归一化发生在 action representation 固定之后；若使用 relative wrist action，则先转 relative，再做 normalization
- 归一化与反归一化都必须按 `action_semantics_name` 查表，不得只按 `action_dim` 或 `embodiment_id` 猜测
- 训练时模型看到的是 normalized `actions`
- 推理时模型输出的是 normalized `pred_actions_normalized`，在 policy 最后一跳反归一化为物理动作
- 第一版默认采用 per-dimension 仿射归一化；可选：
  - min-max 到 `[-1, 1]`
  - mean/std
- 同一个 `action_semantics_name` 在一次训练 run 内只能绑定一种 normalization 方案
- normalization 统计量必须基于训练数据预先计算并固化；改变 `action_horizon`、action representation、动作维度定义或语义版本时，必须重新生成统计量
- 若不同数据源共享同一个 `action_semantics_name`，则它们必须共享同一套 normalization stats
- 第一版不在 spec 内固定 min-max 还是 mean/std，但实现必须显式配置并持久化该选择

第一版实现收口为：

- `action_semantics_name` 必须存在于样本 schema 和 registry 中
- sampler / batcher / validator 按 `action_semantics_name` 分桶
- runtime 的 projector / decoder 路由第一版仍按 `embodiment_id`
- 第一版要求每个 `embodiment_id` 只绑定一个默认 `action_semantics_name`
- 因此 v1 中：
  - `sharpa -> sharpa_wristdelta_hand22_v1`
  - `g1 -> g1_lowdim_native_v1`
- 若后续同一 embodiment 下引入多个 semantics 版本，再把 runtime 路由主键从 `embodiment_id` 升级为 `action_semantics_name`

第一版至少要为每个 `action_semantics_name` 固定以下 registry：

```yaml
action_semantics:
  name: <action_semantics_name>
  embodiment_id: <sharpa|g1>
  action_dim: <int>
  wrist_delta:
    translation_frame: <must-fill>
    rotation_frame: <must-fill>
    rotation_parameterization: <must-fill>
    translation_unit: <must-fill>
    rotation_unit: <must-fill>
  control:
    frequency_hz: <must-fill>
  hand22:
    joint_order: [<must-fill if applicable>]
  normalization:
    scheme: <minmax|meanstd>
    stats_id: <must-fill>
    stats_scope: <train_split_only>
```

说明：

- `sharpa` native 数据
- human-retargeted-to-sharpa 数据
- 后续任何 synthetic 数据

都必须先绑定到某个明确的 `action_semantics_name` 后，才能进入同一 schema 与训练流程。

约束改为：

- 同一个 batch 内，所有样本必须共享同一个 `action_semantics_name`
- 同一次 forward / loss path 内，不允许混入不同 `action_semantics_name`
- 同一次训练 run 可以包含多个 `action_semantics_name`
- 多个 `action_semantics_name` 的混训只能通过 alternating batches 或 interleaved steps 完成

### batch 约束

必须固定：

- 同一个 batch 只能出现一个 embodiment bucket
- 同一个 batch 只能出现一个 `action_dim`
- `sharpa` 和 `g1` 的混合通过 alternating batches 或 interleaved steps 完成
- gradient accumulation 只用于放大有效 batch size，不用于解决动作维度不一致

## 多视角

第一版沿用官方“先组织、再展平”的范式，不额外加 `view embedding`。

### 固定视角顺序

如果样本提供多视角，必须写死 canonical order：

1. `head`
2. `left_wrist`
3. `right_wrist`

### 处理原则

- dataset schema 若提供多视角，必须按上述顺序产出
- transform 里统一组织成固定 canonical slots 的 `[V, T_visual, C, H, W]`
- 第一版 `V` 固定为 `len(canonical_view_order)`，即使样本只提供单路 `head` 也必须 materialize 成完整 slots
- 再按统一顺序展平后送入 VLM visual processor
- 缺失视角通过 dummy image + `image_mask` 标识，不靠位置挪动补齐
- `VLMBackbone` 必须负责把缺失视角对应的 visual token 映射为 `context_mask=0`
- modality config / inference 必须共享同一顺序
- 第一版不额外定义独立视觉时间维；`T_visual` 就是 `obs_horizon`

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

- 1 路或多路图像；若多于 1 路，则顺序固定为 `head, left_wrist, right_wrist`
- instruction
- `raw_state`
- `action`
- `embodiment_id`
- `data_source`
- `has_proprio`
- `task_id`

关键字段：

- `embodiment_id`
- `data_source`
- `has_proprio`
- `camera_views`
- `state_semantics_name`
- `state_dim`
- `action_dim`
- `action_semantics_name`
- `obs_timestamps`
- `action_timestamps`
- `obs_horizon`
- `action_horizon`

字段语义：

- `embodiment_id`: 本体 / 动作接口标识，不表示数据来源
- `data_source`: 样本来源标识；第一版推荐 `human_retargeted` 或 `robot_native`
- `has_proprio`: 当前样本是否真实提供 proprio；这是 dataset schema 的事实字段
- `camera_views`: 当前样本实际提供的视角名列表
- `state_semantics_name`: 当前样本绑定的 state layout / normalization 定义标识
- `state_dim`: 当前样本 `raw_state` 的最后一维长度
- `raw_state`: 始终 materialize 成定长向量；若 `has_proprio=false`，则为与对应 `state_semantics_name` 一致的零向量占位，且该零向量在 transforms 后仍必须保持为精确零值
- `action_semantics_name`: 当前样本绑定的动作语义定义标识，而不是单纯的动作维度名
- `camera_views` 只记录样本真实提供的视角；张量层仍必须 materialize 成完整 canonical slots

补充约束：

- `data_source=human_retargeted` 的样本仍然使用 `embodiment_id=sharpa`
- `has_proprio` 必须由数据准备阶段显式写入，不允许由 trainer 基于 `data_source` 硬编码猜测
- collate / trainer 内部只允许按固定规则派生：`use_placeholder_state = not has_proprio`
- `human_retargeted` 通常默认 `has_proprio=false`，`robot_native` 通常默认 `has_proprio=true`，但这只是 data-prep 默认值，不是 runtime 强约束
- state normalization 统计量只适用于真实 proprio；`has_proprio=false` 的 placeholder state 不参与 state stats 计算，也不应用 state normalization
- 第一版默认所有训练样本都必须提供完整有效的动作窗口，不要求 `action_mask`
- 若后续保留尾部裁剪或局部缺失窗口，`action_mask` 只能作为可选字段扩展引入
- dataset / collate 的 padding 只负责同步产出 `image_mask`，不负责构造 `context_mask`

## 各 Stage 数据形态

第一版不仅要定义“用哪些数据”，还要把每个 stage 允许出现的样本形态写死。

### Stage I data shape

- 只允许：
  - `embodiment_id=sharpa`
  - `data_source=human_retargeted`
  - `has_proprio=false`
  - `state_semantics_name=sharpa_proprio_v1`
  - `action_semantics_name=sharpa_wristdelta_hand22_v1`
- `raw_state` 必须 materialize 为零向量，形状固定为 `[sharpa_state_dim]`
- `action_dim` 固定为 `28`
- `camera_views` 可为：
  - `[head]`
  - `[head, left_wrist]`
  - `[head, right_wrist]`
  - `[head, left_wrist, right_wrist]`
- batch bucket 固定为：
  - `(embodiment_id=sharpa, state_semantics_name=sharpa_proprio_v1, action_semantics_name=sharpa_wristdelta_hand22_v1, has_proprio=false)`

### Stage II data shape

- 允许三类样本：
  - `embodiment_id=sharpa, data_source=human_retargeted, has_proprio=false`
  - `embodiment_id=sharpa, data_source=robot_native, has_proprio=true`
  - `embodiment_id=g1, data_source=robot_native, has_proprio=true`
- 不允许：
  - `embodiment_id=g1, data_source=human_retargeted`
- `state_semantics_name` 只能为：
  - `sharpa_proprio_v1`
  - `g1_proprio_v1`
- `action_semantics_name` 只能为：
  - `sharpa_wristdelta_hand22_v1`
  - `g1_lowdim_native_v1`
- batch bucket 固定为：
  - `(embodiment_id, state_semantics_name, action_semantics_name, has_proprio)`
- 训练时通过 alternating batches 或 interleaved steps 在以下 bucket 间切换：
  - `sharpa + human_retargeted + placeholder path`
  - `sharpa + robot_native + proprio path`
  - `g1 + robot_native + proprio path`

### Stage III data shape

- 新增显式配置：
  - `stage3_allow_aligned_human: bool`
- 当 `stage3_allow_aligned_human=false` 时，只允许：
  - `data_source=robot_native`
  - `has_proprio=true`
  - batch bucket 只能走 proprio path
- 当 `stage3_allow_aligned_human=true` 时，额外允许：
  - `embodiment_id=sharpa, data_source=human_retargeted, has_proprio=false`
- 第一版即使打开 `stage3_allow_aligned_human=true`，仍不允许：
  - `embodiment_id=g1, data_source=human_retargeted`
- Stage III 中 `state_semantics_name` / `action_semantics_name` 仍必须与 embodiment 的默认 registry 保持一致
- Stage III 的 batch bucket 仍固定为：
  - `(embodiment_id, state_semantics_name, action_semantics_name, has_proprio)`
- 若打开 `stage3_allow_aligned_human`，trainer 必须显式交替采样：
  - robot-native bucket
  - aligned-human bucket
- 不能在单个 batch 内同时混合 placeholder path 和 proprio path

## 时序 contract

第一版必须显式固定 action chunk 的时序接口。

- 一个训练样本对应单个当前观测时刻 `t`
- 设该样本的 anchor index 为 `i`，则 `i` 就是“当前观测时刻 `t`”
- 输入观测窗口形状为 `[obs_horizon]`
- 第一版视觉时间维固定为 `T_visual = obs_horizon`
- `video_delta_indices` 必须与 `obs_horizon` 一致，即：
  - `len(video_delta_indices) = obs_horizon`
- 第一版默认观测窗口采用历史采样窗口，约束为：
  - `video_delta_indices` 全部 `<= 0`
  - `video_delta_indices[-1] = 0`
  - 若 `len(video_delta_indices) > 1`，则相邻步长必须恒定且为正
- `obs indices = [i - obs_horizon + 1, ..., i]`
- 监督动作窗口形状为 `[action_horizon, action_dim]`
- `action indices = [i, i + 1, ..., i + action_horizon - 1]`
- 当前观测窗口预测从当前控制步开始执行的动作 chunk
- 第一维监督动作对应当前时刻 `t`，而不是 `t+1`
- `sample_stride` 定义为相邻两个训练样本的 anchor index 间隔，单位是控制步
- `sample_stride` 控制样本抽取密度，不改变单个 chunk 内部的时间间隔
- dataset、collate、trainer、eval 必须共享同一组 `obs_horizon` / `action_horizon` / `sample_stride` 配置
- `obs_timestamps[-1]` 必须与 `action_timestamps[0]` 对齐到同一控制步；若原始数据不是这样，必须在数据准备阶段重排
- 第一版默认训练样本都必须提供完整有效的动作窗口，因此这里不强制要求 `action_mask`

## 三阶段训练

### Stage I

数据：

- human-retargeted-to-sharpa only

说明：

- Stage I 不要求每条样本必须有 3 路图像
- 若只有单路 `head` 视角，也允许进入训练，但仍需 materialize 成完整 canonical slots
- 只要 `camera_views` 与 `image_mask` 语义自洽即可
- Stage I 不允许真实 proprio 路径进入 batch
- Stage I 的 `StateProjector` 只走 `placeholder_adapter -> shared_state_trunk`

训练目标：

- 学 human manipulation prior
- 学 `sharpa` 28D 动作空间上的 flow matching
- 建立 VLM visual-text tokens 到动作生成的基本映射

### Stage II

数据：

- aligned human-retargeted-to-sharpa + embodiment-aligned robot play mixture

说明：

- Stage II 可以包含 `sharpa-native` 数据
- Stage II 也允许引入 `g1` play 数据
- `EmbodimentAdapter` 的 `g1` 分支从 Stage II 起就允许出现
- Stage II 必须按 `(embodiment_id, state_semantics_name, action_semantics_name, has_proprio)` 分桶采样
- Stage II 的 sharpa-human bucket 走 placeholder path；sharpa-native / g1-native bucket 走 proprio path

训练目标：

- 将 human prior 锚定到真实 robot sensing/control
- 对齐 retarget human 数据与 embodiment-aligned robot play 数据

### Stage III

数据：

- task-specific robot demos
- 可分 `stage3_sharpa` 与 `stage3_g1`
- 可选加入 aligned human demos，但必须显式打开 `stage3_allow_aligned_human`

说明：

- Stage III 是 task-specific specialization，不是 `g1` 首次进入训练的阶段
- `stage3_allow_aligned_human=false` 时，Stage III 默认为 pure robot post-training
- `stage3_allow_aligned_human=true` 时，Stage III 允许复现论文 one-shot setting 中的 `robot demo + aligned human demos`

训练目标：

- task specialization
- 针对不同 embodiment 做末端适配

### Freeze matrix

以下 freeze matrix 作为第一版实现约束，尽量贴 EgoScale 论文原文：

| Module | Stage I | Stage II | Stage III |
| --- | --- | --- | --- |
| VLM visual encoder | `train` | `train` | `freeze if mid-training is used; otherwise train` |
| VLM multimodal adapter | `train` | `train` | `freeze if mid-training is used; otherwise train` |
| VLM language backbone | `train` | `freeze` | `freeze` |
| `placeholder_adapter` | `train` | `train` | `train if stage3_allow_aligned_human else freeze` |
| `proprio_adapter` | `inactive` | `train` | `train` |
| `shared_state_trunk` | `train` | `train` | `train` |
| `ActionProjector` | `train` | `train` | `train` |
| `FlowMatchingActionHead` / DiT action expert | `train` | `train` | `train` |
| `ActionDecoder` | `train` | `train` | `train` |

表格解释：

- Stage I 对齐论文正文 `fully unfreezing every parameter of the VLA model`。在本 spec 的拆分实现中，这对应于所有主链路模块均可训练；但由于 Stage I 数据没有真实 proprio，`proprio_adapter` 虽可实例化，实际视为 `inactive`。
- Stage II 对齐论文正文 `freezing the vision-language backbone while only updating the vision encoder and DiT action expert`，并结合附录 D.1 的更细表述：`only the vision encoder, DiT action expert, and state-action encoder and decoder are updated, while the vision-language backbone remains frozen`。因此本 spec 固定为：冻结 `VLM language backbone`，训练 `VLM visual encoder`、`vlm_multimodal_adapter`、state/action encoder-decoder 与共享 DiT。
- Stage III 对齐论文正文 `the vision encoder is frozen if mid-training is used and unfrozen otherwise, to accommodate new embodiments when needed`。论文没有逐项写死 Stage III 其余模块的冻结状态；为落地实现，第一版 spec 补充约束为：`VLM language backbone` 保持冻结，state/action 主链路继续训练。
- Stage III 是否训练 `placeholder_adapter` 不再靠隐式推断，而由 `stage3_allow_aligned_human` 显式决定；只有当 Stage III 真的采样 `has_proprio=false` 的 aligned human bucket 时，该路径才保持可训练。

## 配置原则

第一版 spec 不再定义旧的 `tune_visual / tune_llm / tune_projector / tune_diffusion_model` 开关。配置真值来源统一改为显式模块组：

- `trainable_module_groups.vlm_visual_encoder`
- `trainable_module_groups.vlm_multimodal_adapter`
- `trainable_module_groups.vlm_language_backbone`
- `trainable_module_groups.state_projector_bundle`
- `trainable_module_groups.action_projector`
- `trainable_module_groups.flow_matching_action_head`
- `trainable_module_groups.action_decoder`

说明：

- `vlm_multimodal_adapter` 统一覆盖 visual projector、`merger`、`mm_projector` 以及其它 image-to-token 接入层
- `state_projector_bundle` 统一覆盖 `placeholder_adapter`、`proprio_adapter`、`shared_state_trunk`
- 任何旧 `tune_*` 字段若仍存在于实现代码，只能视为内部兼容细节，不属于 spec 对外公开 contract

新增 EgoScale 配置：

- `vlm_backbone_name`
- `vlm_token_dim`
- `vlm_max_views`
- `stage_recipe`
- `trainable_module_groups`
- `bucket_sampling_policy`
- `bucket_sampling_weights`
- `canonical_view_order`
- `embodiment_buckets`
- `data_source_buckets`
- `run_g1_path`
- `state_adapter_mode`
- `stage3_allow_aligned_human`
- `allowed_action_semantics_names`
- `allowed_state_semantics_names`
- `obs_horizon`
- `action_horizon`
- `sample_stride`
- `num_inference_timesteps`
- `num_timestep_buckets`

其中：

- `allowed_action_semantics_names` 用于显式声明本次训练允许采样哪些动作语义 bucket
- `allowed_state_semantics_names` 用于显式声明本次训练允许采样哪些 state 语义 bucket
- `data_source_buckets` 用于显式声明本次训练会采样哪些数据来源，例如 `human_retargeted` / `robot_native`
- `stage3_allow_aligned_human` 决定 Stage III 是否允许 aligned human bucket 进入训练
- `bucket_sampling_policy` 第一版固定支持：
  - `proportional_to_active_samples`
  - `manual_weights`
- 默认 `bucket_sampling_policy=proportional_to_active_samples`
- 当使用 `proportional_to_active_samples` 时，bucket 采样概率按当前 run 中每个 active bucket 的有效样本数成正比计算，等价于在过滤后的合并 dataset 上做均匀采样
- active bucket 的样本数必须在应用 `stage_recipe`、`stage3_allow_aligned_human`、`allowed_*_semantics_names`、`data_source_buckets` 等过滤条件后再统计，并在 run 内归一化为最终采样概率
- `bucket_sampling_weights` 只在 `bucket_sampling_policy=manual_weights` 时生效；其 key 必须与 sampler 的最小分桶键一一对应
- `sample_stride` 是相邻两个训练样本的 anchor index 间隔，单位是控制步
- `num_inference_timesteps` 控制 flow matching sampling 的 Euler rollout 步数
- `num_timestep_buckets` 用于把连续时间 `t in [0, 1)` 离散化成 action-side timestep bucket

## 训练实现约束

第一版明确采用：

- sharpa bucket 和 g1 bucket 分开组 batch
- placeholder path bucket 和 proprio path bucket 分开组 batch
- sampler 的最小分桶键固定为 `(embodiment_id, state_semantics_name, action_semantics_name, has_proprio)`
- trainer 可以在 step 级别交替采样不同 bucket
- 第一版默认 bucket 采样规则为 `proportional_to_active_samples`；只有显式切到 `manual_weights` 时，才允许手工覆盖 bucket 概率
- 不在一次 forward 中混合不同 `action_dim`
- 不在一次 forward 中混合不同 `state_semantics_name`
- gradient accumulation 允许开启，但仅作为优化手段

## 最大风险

1. pretrained VLM 的选择和 token interface 需要本机实测，不能只靠文档推断
2. VLM 依赖版本、显存占用和 processor 接口可能比预期更敏感
3. placeholder state 插入位置需要谨慎，否则会污染 state projector 设计
4. 单路 / 多路视角混用时，`camera_views` 与 `image_mask` 若处理不一致，token 语义会漂移
5. retarget human data 和 native sharpa data 的统计差异可能会影响 Stage II 对齐

## 当前建议的实现顺序

1. 选定并验证一个 pretrained VLM 的本机加载与最小 forward
2. 打通多视角 + 文本 -> VLM token interface
3. 实现 state projector / action projector / action head
4. 接入 `ActionDecoder` 并固定 flow-matching 训练/采样接口
5. 插入 placeholder state path
6. 写 Stage I / II / III trainers
7. 加入 sharpa / g1 bucket 训练逻辑
8. 跑 synthetic 全流程

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
- 如果提供多视角，顺序固定为 `head, left_wrist, right_wrist`
- 即使只提供单路视角，也必须 materialize 成完整 canonical slots
- `T_visual` 不再是未定义自由变量；第一版固定 `T_visual = obs_horizon = len(video_delta_indices)`
- `human` 不作为独立 embodiment
- human-retargeted 数据按 `sharpa` embodiment 训练
- `sharpa` / `g1` 各自维护独立的 `state projector` 与 `action projector`
- `StateProjector` 内部采用 `placeholder_adapter` / `proprio_adapter` + `shared_state_trunk`
- 第一版样本 schema 以 `data_source` 和 `has_proprio` 为主，不保留 `is_human`
- `use_placeholder_state` 不入库，只由 `has_proprio` 在 runtime 派生
- `has_proprio=false` 的 placeholder `raw_state` 必须在 projector 输入空间保持精确零值，不能套用真实 proprio normalization
- 先固定 `VLMBackbone -> context_tokens/context_mask` 的稳定 interface，再接 policy
- `context_mask` 由 `VLMBackbone` 生成，不由 dataset / collate 构造
- 不再要求每条样本固定 3 路图像
- 第一版必须做 action normalization，训练与采样都在 normalized action space 中进行，policy 输出前再反归一化
- 第一版 flow matching 主 loss 固定为 action-space velocity loss
- 训练时 `t` 的采样区间固定为 `[0, 1)`，离散 timestep 必须 clamp 到合法 bucket
- 第一版推理采样固定使用 GR00T-style Euler rollout：`N` 步、`t_k = k / N`、包含 `t=0` 不包含 `t=1`、更新式 `actions_{k+1} = actions_k + (1/N) * pred_velocity_k`
- `ActionDecoder` 属于训练主链路，默认参与 action-side 训练
- `action_semantics_name` 第一版主要用于数据校验和 batch 分桶，不作为 runtime 主路由键
- `state_semantics_name` 也必须进入 batch 分桶键，避免同 embodiment 下不同 state layout 混到同一 forward
- 视觉侧接入层统一收口为 `vlm_multimodal_adapter`，不再拆成 `vision_projector` 和 `bridge` 两套配置语义
- bucket 采样默认按 active bucket 的有效样本数成正比，等价于在过滤后的合并 dataset 上均匀采样
- `g1` 从 Stage II 起就允许进入 aligned robot play mixture
- `g1` 保留独立动作维度与独立适配
- batch 内 `action_dim` 必须恒定
