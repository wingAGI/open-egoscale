# EgoDex -> Open EgoScale Stage I 映射方案

这份文档定义了如何把 `EgoDex` 数据转换成当前 `open-egoscale` 可以直接用于 `Stage I` 训练的样本格式。

重点不是“完全复刻论文里的 Sharpa retarget”，而是先给出一套**工程上可落地、语义上尽量诚实**的映射方案，让 `Stage I` 能先跑起来，并且后续可以平滑切到更干净的 schema。

## 1. 当前仓库的现实约束

当前仓库里，`Stage I` 样本原先默认按这套语义组织：

- `embodiment_id = sharpa`
- `data_source = human_retargeted`
- `has_proprio = false`
- `state_semantics_name = sharpa_proprio_v1`
- `action_semantics_name = sharpa_wristdelta_hand22_v1`
- `action_dim = 28`

不过现在已经把三阶段样本约束从代码硬编码改成了配置项：

- 在每个 stage 的 YAML 里，通过 `data.stage_sample_rules` 控制样本是否合法

这意味着后面要换 `SO101`、`EgoDex` 专用 action semantics，或者改 `Stage II/III` 的数据组合，不再需要改校验代码，只改配置就够了。

## 2. EgoDex 原始数据里有什么

`EgoDex` 对每条 episode 通常提供：

- 一个 `mp4`
- 一个同名 `hdf5`

其中对 `Stage I` 最有用的原始字段是：

- `transforms/camera`
- `transforms/leftHand`
- `transforms/rightHand`
- `transforms/*` 下的 finger tip / knuckle / joint transforms
- `confidences/*`，如果该条数据带置信度
- `camera/intrinsic`
- episode 级别的文字描述或 task 元数据

它**没有**直接给你：

- 机器人 proprio
- 机器人控制 action
- 已经完成的 Sharpa 手部 retarget 结果

所以 `Stage I` 里，`EgoDex` 的定位应该是：

- 人类第一视角 RGB + 3D hand pose 的监督源
- 需要你自己从 pose trajectory 构造伪动作标签

## 3. 推荐采用两层 schema 思路

建议把设计分成两层：

1. **兼容方案**
   目标是先兼容当前仓库训练路径，尽量少改代码。
2. **干净方案**
   目标是让数据语义更诚实，避免把人类 hand pose descriptor 误叫成 Sharpa hand action。

实际执行建议是：

1. 先按兼容方案导出数据，把 `Stage I` 跑通
2. 等训练链路稳定后，再切到干净方案

## 4. Stage I 输出样本格式

每一行输出一个 action-chunk 训练样本，写成一条 JSONL。

当前仓库期望的核心字段大致是：

```json
{
  "instruction": "pick up the sponge",
  "embodiment_id": "sharpa",
  "data_source": "human_retargeted",
  "has_proprio": false,
  "task_id": "egodex/task_x",
  "state_semantics_name": "sharpa_proprio_v1",
  "action_semantics_name": "sharpa_wristdelta_hand22_v1",
  "state_dim": 32,
  "action_dim": 28,
  "raw_state": [0.0, 0.0, "..."],
  "actions": [["... 28 dims ..."], ["... 28 dims ..."]],
  "obs_timestamps": [1.23, 1.33],
  "action_timestamps": [1.33, 1.43, "..."],
  "obs_horizon": 2,
  "action_horizon": 8,
  "camera_views": ["head"],
  "view_images": {
    "head": ["frame_tensor_t0", "frame_tensor_t1"]
  }
}
```

真正导出时当然是数值数组，不是示意字符串。

## 5. 兼容方案

### 5.1 为什么需要兼容方案

如果你现在的目标是：

- 先跑通 `Stage I`
- 先看 `human data size -> human val loss`

那最稳妥的办法不是先大改 action head，而是先把 `EgoDex` 压到现有 `28D` 入口里。

### 5.2 兼容方案的 action 定义

沿用当前 `28D` 结构，但重新解释其语义：

- `0:3`：active wrist 的相对平移增量
- `3:6`：active wrist 的相对旋转增量，axis-angle
- `6:28`：基于人手 pose 构造的 `22D hand descriptor`

注意这里的 `22D` **不是 Sharpa 22DoF 真实关节动作**，而是一个为了兼容当前训练栈而构造的人手姿态描述向量。

### 5.3 active hand 怎么选

因为当前一条样本只承载一个 `28D` action stream，所以建议每个 chunk 只选一只主手：

- 优先选置信度更高的手
- 如果两只手都有效，选 motion energy 更大的手
- 如果差别很小，固定退化到右手

这样做的原因很直接：

- 先保证 schema 稳定
- 不在 `Stage I` 阶段引入双手输出的结构性改造

### 5.4 6D wrist delta 怎么算

对每个 action step：

1. 取 active wrist 在 `t0` 的位姿 `T_wrist_t0`
2. 取 active wrist 在 `t1` 的位姿 `T_wrist_t1`
3. 计算相对变换

```text
T_rel = inv(T_wrist_t0) * T_wrist_t1
```

4. 从 `T_rel` 取平移部分，得到：

- `dx, dy, dz`

5. 从 `T_rel` 取旋转部分，转成 axis-angle，得到：

- `rx, ry, rz`

这样就得到前 6 维。

### 5.5 22D hand descriptor 怎么定义

这里不建议一开始就做 full dexterous retarget。更现实的是构造一个稳定的低维 descriptor，保留抓握和手型变化。

推荐的 `22D` 布局：

- `0:5`：thumb/index/middle/ring/little 的 curl
- `5:10`：thumb/index/middle/ring/little 的 spread
- `10:15`：五指 tip 到 palm 的距离
- `15:20`：五指 tip 的速度模长
- `20`：thumb-index pinch proxy
- `21`：grasp openness

对应到语义上就是：

- 5 维手指弯曲
- 5 维手指张开程度
- 5 维 fingertip-to-palm 距离
- 5 维 fingertip 速度
- 1 维 pinch 指标
- 1 维整体开合程度

### 5.6 22D descriptor 的计算建议

按帧计算：

- 先定义 palm frame
- 用关节链弯曲角近似 curl
- 用相邻手指的横向夹角近似 spread
- 在 palm frame 下算 tip-to-palm distance
- 用有限差分算 tip speed
- 用 thumb tip 和 index tip 距离定义 pinch proxy
- 用平均 tip-to-palm distance 定义 grasp openness

然后把未来 `action_horizon` 个 step 的这些向量拼成 `actions`

## 6. state 字段怎么填

对 `Stage I human` 样本：

- `has_proprio = false`
- `raw_state = zeros(state_dim)`

如果沿用兼容方案，可以先继续：

- `state_semantics_name = sharpa_proprio_v1`
- `state_dim = 32`

这只是占位，不代表真的有 Sharpa proprio。

## 7. 图像字段怎么填

对 `EgoDex` 的 `Stage I`：

- `camera_views = ["head"]`
- 只填 `view_images["head"]`

当前仓库的 transform 会自动把缺失的 `left_wrist` / `right_wrist` 视角补成 masked dummy tensor。

所以最小方案下，不需要你伪造 wrist camera。

## 8. instruction 字段怎么填

优先级建议：

1. 如果 `EgoDex` 元数据里有自然语言描述，用它
2. 否则用 task folder 名称清洗成短句
3. 再不行，就用模板，例如：
   `perform tabletop manipulation`

## 9. 采样和 chunk 规则

建议先沿用当前仓库默认值：

- `obs_horizon = 2`
- `action_horizon = 8`
- `sample_stride = 1`

如果 `EgoDex` 原始视频和 pose 是 `30 Hz`，而你想按 `10 Hz` 训练，那么先做：

1. 把视频帧和 pose 轨迹统一重采样到目标频率
2. 再切 chunk

不要在不同字段上各自随意下采样，否则时间对齐容易乱。

## 10. 时间戳要求

当前样本校验里有一个硬约束：

- `obs_timestamps[-1] == action_timestamps[0]`

所以实际导出时，建议先定义统一控制时间网格，例如 `10 Hz`，然后：

- 观测帧取前 `obs_horizon`
- 预测动作取后 `action_horizon`
- 确保观测窗口最后一个时间点正好对上动作窗口第一个时间点

## 11. 过滤规则

以下样本建议丢弃：

- active hand 置信度太低
- wrist pose 缺失或突变
- 整个 chunk 几乎没有运动
- 视频帧和 pose 时间戳明显错位
- hand transforms 缺关键 joint，导致 descriptor 无法稳定计算

## 12. 归一化建议

建议先导出完整的 train JSONL，再用仓库现有统计脚本按 train split 计算归一化统计量。

原则是：

- 只用 train split 统计
- val split 不参与统计
- wrist delta 和 hand descriptor 一起做 action normalization

## 13. 更干净的长期方案

兼容方案虽然能跑，但有一个明显问题：

- `sharpa_wristdelta_hand22_v1` 这个名字会让人误以为后 22 维是 Sharpa hand joint action

实际上这里放的是人类 hand pose descriptor。

所以长期更建议新增专门的 semantics，比如：

- `egodex_wristdelta_handpose22_v1`

必要的话，再加一个更诚实的 placeholder state semantics，例如：

- `human_placeholder_v1`

这样文档、配置、统计文件都会更清楚。

## 14. 干净方案需要改什么

因为当前仓库已经支持 `data.stage_sample_rules` 配置化，所以现在切到干净方案的改动已经很小了：

1. 在 config 里注册新的 `action_semantics`
2. 如果需要，注册新的 `state_semantics`
3. 在 `stage1.yaml` 里把 `allowed_action_semantics_names` 和 `stage_sample_rules` 改成新的名字

不需要再改 `schema.py` 里的硬编码判断。

## 15. 这份映射方案能解决什么，不能解决什么

它适合：

- `Stage I` 人类预训练
- `human val loss` 曲线实验
- 小规模 scaling 实验

它暂时不能证明：

- 这 22 维就是机器人可执行 hand action
- 不经过 `Stage II` 也能自然迁移到机器人
- 这已经等价于论文里的完整 Sharpa retarget

所以对这份 schema 的定位应该明确：

- 它是 `Stage I` 的工程落地方案
- 不是完整的 human-to-robot grounding 方案

## 16. 推荐执行顺序

建议按这个顺序推进：

1. 先把 `EgoDex` 导成兼容方案的 JSONL
2. 只跑 `Stage I`
3. 先看 `human data size -> human val loss`
4. 再切到更干净的 semantics 命名
5. 最后再围绕真实目标 embodiment 设计 `Stage II/III`

如果后面的目标是 `SO101`，那更推荐在 `Stage II` 开始时重新定义面向 `SO101` 的 action semantics，而不是把 `Stage I` 的人类 descriptor 生硬地当成机器人 action。
