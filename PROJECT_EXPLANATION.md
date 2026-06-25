# Spatial-CoLaRa: External-Mask-Conditioned Spatial CoT for VLA

## 一、研究思路

### 1.1 核心创新

在 LaRA-VLA 的 latent reasoning 范式基础上，引入**显式 mask 条件化的空间状态转移推理**：

```
RGB image + instruction + current external mask
        ↓
visual tokens + mask tokens + language tokens
        ↓
latent transition tokens
        ↓
future / goal mask supervision + relation supervision + action generation
```

mask 不是后处理模块，而是和 RGB 一样的**输入模态**，被编码为 latent tokens 参与推理。

### 1.2 与 LaRA-VLA 的关系

| LaRA-VLA | 我们的扩展 |
|----------|-----------|
| textual/visual CoT latent | + mask-conditioned spatial transition latent |
| subtask + bbox + motion reasoning | + expected_spatial_transition |
| future visual latent | + future_mask supervision |
| 无 goal 概念 | + goal_mask supervision (subtask_end_idx) |
| 无 relation | + relation_label (7 类) |

### 1.3 数据对齐方案：三层桥接

```
CoT 帧 i  ←→  LeRobot 帧 i  ←→  no-noops HDF5 帧 i  ←→  spatial NPZ
(CoT标注)     (LeRobot parquet)  (clip-rt HDF5)       (state replay)
```

基于 clip-rt 的 no-noops HDF5 与 LeRobot CoT 数据帧对帧一致，采用 identity mapping，无需 DTW。

### 1.4 训练阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage I-A | Explicit Transition-CoT SFT（独立 Qwen3-VL，cot_text_transition 监督） | ✅ 已验证，脚本就绪 |
| Stage I-B | 权重桥接：Stage I-A → Qwen_GR00T full framework | ✅ bridge 验证通过 |
| Stage II | Mask encoder + future/goal mask loss + relation loss | 待开发 |
| Stage III | Transition tokens → action head，latent reasoning | 待开发 |

整体路线：
```
显式 Transition-Aware Spatial CoT SFT (Stage I-A)
        ↓
权重桥接回 Qwen_GR00T (Stage I-B)
        ↓
latent transition reasoning + mask-conditioned (Stage II)
        ↓
latent-conditioned continuous action generation (Stage III)
```

## 二、代码文件清单

### 2.1 核心数据模块 (`lara_vla/data/`)

| 文件 | 用途 |
|------|------|
| `spatial_lara_libero_dataset.py` | 底层 NPZ 读取器。从 index 读取 `hdf5_frame_idx`/`hdf5_future_idx`/`subtask_end_idx`，返回 current/future/goal 的 RGB 和 mask。 |
| `spatial_cot_dataset.py` | 合并 CoT + Spatial 的 Dataset。继承上述 Base Dataset，额外加载 CoT 标注，实现动态 mask 过滤（仅 libero_10），输出 transition 字段。 |
| `__init__.py` | 模块标记 |

### 2.2 数据构建脚本 (`scripts/`)

| 文件 | 用途 |
|------|------|
| `rebuild_noop_suite.py` | 从 no-noops HDF5 重建 spatial NPZ（state replay）。输出到 `output/spatial_lara_libero/{suite}_v2/`。 |
| `build_transition_index.py` | 构建合并 transition index。计算 subtask_end_idx、relation_label、expected_spatial_transition、cot_text_transition。输出到 `spatial_lara_libero_index_cot_transition_all.jsonl`。 |
| `regenerate_suite_indices.py` | 从已有 NPZ meta 重新生成 suite 级别 index。 |
| `make_mask_videos.py` | 生成 mask overlay 视频用于可视化验证。 |
| `stage1_cot_train.py` | [已废弃] Stage I 最小训练 sanity check。已被 `train_stage1_cot.py` 替代。 |
| `train_stage1_cot.py` | Stage I-A 正式训练脚本：Explicit Transition-CoT SFT（Accelerate 多GPU、checkpoint、W&B）。 |
| `bridge_stage1a_to_laravla.py` | Stage I-B 桥接脚本：验证 Stage I-A 权重可回灌到 Qwen_GR00T full framework。 |
| `run_stage1a_server.sh` | 服务器一键启动脚本（Stage I-A）。 |

### 2.3 工具脚本 (`tools/`)

| 文件 | 用途 |
|------|------|
| `verify_alignment.py` | 四联图对齐验证（LeRobot RGB vs HDF5 RGB + mask + info）。 |
| `test_spatial_cot_loader.py` | CoT Dataset batch sanity check。 |
| `test_spatial_lara_loader.py` | 底层 Dataset batch 测试。 |
| `check_spatial_lara_dataset.py` | NPZ 批量检查（mask、pose、future 合法性）。 |
| `visualize_dynamic_mask.py` | 动态 mask 过滤可视化。 |
| `inspect_libero_env.py` | LIBERO 环境检查（Phase 1 使用）。 |

### 2.4 服务端配置

| 文件 | 用途 |
|------|------|
| `scripts/server_env.sh` | 服务器无头渲染环境变量（EGL + MuJoCo）。 |

## 三、数据流

### 3.1 训练 index 结构

每条样本（`spatial_lara_libero_index_cot_transition_all.jsonl`）：

```json
{
  "suite": "libero_10",
  "task_id": 0,
  "demo_id": 12,
  "cot_episode_id": 8,
  "cot_frame_idx": 65,
  "hdf5_frame_idx": 65,
  "cot_future_idx": 73,
  "hdf5_future_idx": 73,
  "subtask_end_idx": 110,
  "future_crosses_subtask": false,
  "mask_mode": "dynamic",
  "mask_switch_rule": "grasp+gripper_open+longest_alias+container_prescan",
  "relation_label": "place_inside",
  "relation_label_id": 3,
  "relation_subject": "block",
  "relation_object": "basket",
  "expected_spatial_transition": "the block should become inside the basket",
  "cot_text_transition": "Subtask: ... Reasoning: ... Spatial transition: ...",
  "alignment_method": "identity_no_noops_hdf5",
  "episode_path": "libero_10_v2/task_00/demo_000012/episode_000012.npz"
}
```

### 3.2 Dataset 输出（训练时）

```python
sample = {
    # 模型输入
    "image": current_rgb,           # [3, 224, 224]  from hdf5_frame_idx
    "image_next": future_rgb,        # [3, 224, 224]  from hdf5_future_idx
    "current_affordance_mask_agentview": current_mask,  # [1, 224, 224]
    "current_affordance_mask_wrist": current_mask_wrist,

    # 监督信号
    "future_affordance_mask_agentview": future_mask,
    "goal_affordance_mask_agentview": goal_mask,        # from subtask_end_idx
    "goal_image_debug": goal_rgb,                        # visualization only

    # CoT 文本
    "cot_text_transition": "...",     # Stage I 训练目标
    "expected_spatial_transition": "...",

    # Relation
    "relation_label": "grasp_object",
    "relation_label_id": 1,

    # 动作
    "actions": action_chunk,          # [8, 7]

    # 元数据
    "subtask_end_idx": 110,
    "mask_mode": "dynamic",
    ...
}
```

## 四、关键设计决策

### 4.1 Mask 过滤规则

| Suite | 方式 | 规则 |
|-------|------|------|
| libero_10 | 动态 | grasp事件切换 + gripper_open 允许切换 + 最长别名匹配 + 容器前扫 |
| spatial/object/goal | union | 所有 objects_of_interest 合集，无过滤 |

### 4.2 依赖字段状态

| 字段 | 状态 |
|------|------|
| primary_pose_world / eef | 保留在 NPZ 和 Dataset 输出中，但不参与训练 |
| future_pose | 同上 |
| 6D pose | 已从方案中移除 |
| relation | 已构造，Stage II+ 使用 |

### 4.3 已知排除

- `libero_10/task_4/demo_2, demo_3`：夹爪不稳定导致 mask 跳变，从训练集排除

## 五、运行命令

### 5.1 数据重建

```bash
source scripts/server_env.sh
python scripts/rebuild_noop_suite.py --suite libero_10
python scripts/build_transition_index.py
```

### 5.2 视频验证

```bash
python scripts/make_mask_videos.py --suite libero_10 --task-id 4
```

### 5.3 Stage I 训练验证

```bash
LARAVLA_CKPT=/path/to/checkpoint.pt \
python scripts/stage1_cot_train.py --max-steps 100
```

## 六、目录结构

```
output/
├── spatial_lara_libero/           # NPZ 数据
│   ├── libero_10_v2/              # no-noops 重建
│   ├── libero_spatial_v2/
│   ├── libero_object_v2/
│   └── libero_goal_v2/
├── spatial_lara_libero_no_noops/  # 过渡 index
│   └── spatial_lara_libero_index_cot_transition_all.jsonl
└── mask_videos/                   # 验证视频

datasets/
├── clip-rt/modified_libero_hdf5/  # no-noops HDF5
└── lovejuly/libero_lerobot_all/   # LaRA-VLA CoT 数据
```



claude --resume 68a310cf-337a-46dd-a0a4-bee220857be6
