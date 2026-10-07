# R4：8 卡官方多阶段训练手册

本手册只运行官方 LaRA-VLA、官方 LIBERO LeRobot 数据和已验证的正式
forward。不使用小模型、假数据、dummy action 或简化 loss。

## 固定资源

服务器目录规划：

```text
/data/peixingxing/codevla/LaRA/
├── LaRA-VLA/
├── StarVLA-Qwen3-VL-4B-Instruct-Action/
├── checkpoints/LaRA-VLA-libero/
├── datasets/libero_lerobot_all/
├── dependencies/physical-intelligence-fast/
│   └── ec4d7aa71691cac0b8bed6942be45684db2110f4/
└── runs/repro_r4_official/
```

版本锁定：

- 数据集：`lovejuly/libero_lerobot_all`；
- dataset revision：`fbe4f71c2fd5a6e4f9171c78e51e2f6567277fff`；
- FAST revision：`ec4d7aa71691cac0b8bed6942be45684db2110f4`；
- FAST `tokenizer.json` SHA256：
  `6507dd709287fd018882120c0071787f1f62bad9f180f1e8c5235bda1b71fa78`；
- backbone 与 R2 使用同一份本地
  `StarVLA-Qwen3-VL-4B-Instruct-Action`。

本机数据集共 2.1 GB，包含四个 suite 的 `data/`、`videos/`、`meta/` 和
`annotations/`。复制到服务器时要保留顶层 `.cache/huggingface/download/`，
R4 预检会用其 metadata 确认 dataset revision。

## 训练配方

| 论文阶段 | 代码 stage | optimizer steps | 每卡 batch | 8 卡全局 batch | 主要目标 |
|---|---:|---:|---:|---:|---|
| Stage I | 1 | 5000 | 12 | 96 | CoT + image-next + discrete action token |
| Stage II-1 | 2 | 2000 | 16 | 128 | 替换 Subtask latent |
| Stage II-2 | 3 | 2000 | 16 | 128 | 再替换 BBox latent |
| Stage II-3 | 4 | 2000 | 16 | 128 | 再替换 Reasoning latent |
| Stage III | 4 / full | 40000 | 16 | 128 | continuous action / flow matching |

Stage I 的 image-next weight 为 0.1；Stage II 三段均为 0.2，并使用 EMA
teacher。Stage III 只从 Stage II 最终 checkpoint 加载 `qwen_vl_interface`，
关闭 discrete action token 和 VLM loss，训练新初始化的 continuous action head。

48 GB GPU 首次必须尝试上表的论文 batch。如果发生真实 OOM，优先使用：

| 阶段 | 每卡 batch | accumulation | 全局 batch |
|---|---:|---:|---:|
| Stage I | 6 | 2 | 96 |
| Stage II / III | 8 | 2 | 128 |

如仍不足，开启 gradient checkpointing；不改模型、数据、loss 或阶段步数。

## 服务器基础变量

```bash
source /data/peixingxing/miniconda3/etc/profile.d/conda.sh
conda activate lara-vla

export REPRO_ROOT="$HOME/codevla/LaRA"
export REPO="$REPRO_ROOT/LaRA-VLA"
export LARA_REPRO_ROOT="$REPRO_ROOT"
export LARAVLA_PYTHON="$(which python)"
export LARA_DATASET_ROOT="$REPRO_ROOT/datasets/libero_lerobot_all"
export LARA_BACKBONE="$REPRO_ROOT/StarVLA-Qwen3-VL-4B-Instruct-Action"
export LARA_FAST_TOKENIZER="$REPRO_ROOT/dependencies/physical-intelligence-fast/ec4d7aa71691cac0b8bed6942be45684db2110f4"
export R4_RUN_ROOT="$REPRO_ROOT/runs/repro_r4_official"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

同步代码：

```bash
cd "$REPO"
git checkout main
git pull --ff-only origin main
git status --short
git rev-parse HEAD
```

`git status --short` 必须为空。正式训练开始前记录当时的 commit SHA，
不在训练中途 pull 新代码。

## 数据与 FAST 准备

可以将 3090 主机上的两个目录完整下载或复制到上述服务器路径：

```text
/data/CodePWC/lara_datasets/libero_lerobot_all
/home/robot/codePWC/LaRA/dependencies/physical-intelligence-fast/ec4d7aa71691cac0b8bed6942be45684db2110f4
```

如果选择在服务器从 Hugging Face 下载，必须固定 revision：

```bash
mkdir -p "$REPRO_ROOT/datasets"
mkdir -p "$REPRO_ROOT/dependencies/physical-intelligence-fast"

hf download lovejuly/libero_lerobot_all \
  --repo-type dataset \
  --revision fbe4f71c2fd5a6e4f9171c78e51e2f6567277fff \
  --local-dir "$LARA_DATASET_ROOT"

hf download physical-intelligence/fast \
  --revision ec4d7aa71691cac0b8bed6942be45684db2110f4 \
  --local-dir "$LARA_FAST_TOKENIZER"
```

该下载不应在训练进程内发生。R4 训练强制
`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

## 第一步：数据与环境预检

该步骤不执行模型训练，8 张 GPU 需要全部可见，但无需空闲。后续分布式预检和
正式训练才要求所有训练 GPU 空闲。

先单独验证刚传输的两个目录（读取约 2.1 GB）：

```bash
cd "$REPO"
python scripts/reproduction/verify_r4_assets.py \
  --dataset-root "$LARA_DATASET_ROOT" \
  --fast-tokenizer "$LARA_FAST_TOKENIZER" \
  --output "$REPRO_ROOT/r4-assets.json"
```

成功时最后输出 `R4 ASSET VERIFICATION PASS`。数据集应为 5163 个官方文件、
2,011,668,877 字节，聚合 SHA256 为
`99e134b7fda131b0d33b83baaec820131909be190c76a3cecf613ae30248d719`；FAST 应为
7 个官方文件、698,459 字节，聚合 SHA256 为
`7cd45fc4ea68c30f3ec2ecee636feb0bb145d0d21414c222a0d3069ee7d389e2`。

该脚本会忽略训练后生成的 steps cache 和 lock，只校验 Hugging Face metadata
对应的官方发布文件。因此服务器使用复制或固定 revision 重新下载，
应得到同一指纹。

然后执行完整 R4 预检：

```bash
cd "$REPO"
scripts/reproduction/run_r4_training.sh preflight \
  2>&1 | tee "$REPRO_ROOT/r4-preflight.stdout.log"
```

预检必须完成：

- worktree 干净；
- 恰好 8 张可见 GPU；
- `pip check` 通过；
- dataset revision 恰好为锁定 revision；
- 四个 suite 都能解码真实图像和 8×7 action；
- Stage 1–4 分别生成 0/1/2/3 个 thinking token；
- CoT、bbox、FAST action token 和 16 个 `<img_next>` 存在；
- 单进程生成 4 个 steps cache，后续 8 卡训练只读这些 cache。

预检结果保存在：

```text
$R4_RUN_ROOT/preflight/stage_1.json ... stage_4.json
$R4_RUN_ROOT/metadata/preflight/
```

## 第二步：审查五段实际命令

```bash
scripts/reproduction/run_r4_training.sh dry-run \
  | tee "$REPRO_ROOT/r4-dry-run.txt"
```

该命令不创建训练目录，应输出 Stage 1、2、3、4 和 Stage III 共五行完整
`torchrun` 命令。逐项确认路径、steps、batch、loss weight 和 checkpoint 链。

## 第三步：8 卡分布式预检

这一步使用正式 Stage I 模型、数据、forward、loss、backward、optimizer
和 checkpoint，只将 optimizer step 缩短为 1：

启动前 8 张训练 GPU 应由本次实验独占。包装脚本默认要求每张卡已用显存不超过
1024 MiB；这是启动保护线，不是模型或论文超参数。

```bash
tmux new -s lara-r4
cd "$REPO"
scripts/reproduction/run_r4_training.sh distributed-preflight \
  2>&1 | tee "$REPRO_ROOT/r4-distributed-preflight.stdout.log"
```

默认直接测试论文的每卡 batch 12。验收项：

- 8 个 rank 都初始化；
- effective global batch 为 96；
- VLM 和 image-next loss 有限；
- grad norm 有限且非零；
- optimizer/scheduler 完成一步；
- 模型 checkpoint 和 8 卡完整 training state 均可保存；
- 无 OOM、NaN/Inf、NCCL 错误或 rank 丢失。

首次预检通过后，必须从 step 1 的完整 state 恢复并完成 step 2：

```bash
scripts/reproduction/run_r4_training.sh distributed-resume-preflight \
  2>&1 | tee "$REPRO_ROOT/r4-distributed-resume-preflight.stdout.log"
```

日志必须明确记录 `completed_steps=1`、`batches_seen`、dataset epoch，并在跳过
已消费 batch 后完成 step 2。这才是 8 卡 save → load → resume 验收；仅检查目录
存在不算通过。

另一终端监控：

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -l 2 \
  > "$REPRO_ROOT/r4-distributed-preflight-gpu.csv"
```

如果在 batch 12 上真实 OOM，保留原始日志，然后用等效 batch 复测：

```bash
R4_PREFLIGHT_PER_DEVICE_BATCH=6 \
R4_PREFLIGHT_GRADIENT_ACCUMULATION=2 \
R4_GRADIENT_CHECKPOINTING=true \
scripts/reproduction/run_r4_training.sh distributed-preflight
```

## 第四步：Stage I 和 Stage II 正式训练

8 卡预检通过后，在 tmux 中执行：

```bash
cd "$REPO"
scripts/reproduction/run_r4_training.sh reasoning \
  2>&1 | tee "$REPRO_ROOT/r4-reasoning.stdout.log"
```

该命令顺序执行 5k → 2k → 2k → 2k，每段结束 checkpoint 严格传给下一段。
如 48 GB 上需要等效小 batch，四个值必须成对设置：

```bash
export R4_STAGE1_PER_DEVICE_BATCH=6
export R4_STAGE1_GRADIENT_ACCUMULATION=2
export R4_STAGE2_PER_DEVICE_BATCH=8
export R4_STAGE2_GRADIENT_ACCUMULATION=2
export R4_GRADIENT_CHECKPOINTING=true

scripts/reproduction/run_r4_training.sh reasoning
```

包装脚本会拒绝静默改变全局 batch。中断后必须从对应
`steps_<N>_training_state/` 完整恢复 model、optimizer、scheduler、RNG 和数据进度；
不能把只加载 `.pt` 当作 resume。

## 第五步：Stage III 正式训练

```bash
cd "$REPO"
scripts/reproduction/run_r4_training.sh stage3 \
  2>&1 | tee "$REPRO_ROOT/r4-stage3.stdout.log"
```

脚本固定从以下 Stage II 最终权重加载 Qwen 接口：

```text
$R4_RUN_ROOT/reasoning/libero_vlm_stage_4/checkpoints/steps_2000_pytorch_model.pt
```

Stage III 从 step 16000 开始每 4000 步保存模型 checkpoint；所有 `.pt` 保留，
大体积完整 training state 只滚动保留最新两份。只有新 state 已完整写入
`trainer_state.json` 后才删除旧 state。

若需使用等效 batch：

```bash
export R4_STAGE3_PER_DEVICE_BATCH=8
export R4_STAGE3_GRADIENT_ACCUMULATION=2
export R4_GRADIENT_CHECKPOINTING=true

scripts/reproduction/run_r4_training.sh stage3
```

最终 checkpoint：

```text
$R4_RUN_ROOT/action/libero_all_vla/checkpoints/steps_40000_pytorch_model.pt
```

## 最终验收

每个阶段都要检查：

- `config.yaml` 中的阶段、batch、accumulation、loss weight 和加载 checkpoint；
- `training_validation.json` 中必需 loss 出现，且存在有限非零梯度步；
- `metrics.jsonl` 无 NaN/Inf，学习率和显存记录完整；
- Stage I/II 的 VLM/image-next loss 与 Stage III 的 continuous action loss 符合配置；
- 只有 rank 0 写模型摘要，所有 rank 参与 DeepSpeed state 保存；
- 至少从一个正式 training state 完成一次 save → load → resume 验证。

训练完成后，使用与 R2 完全相同的 4 suite × 10 task × 50 rollout
协议评测 `steps_40000_pytorch_model.pt`，生成 reproduced checkpoint 的 JSON/CSV，
再与论文和 R2 作者 checkpoint 放入同一张表。
