# LaRA-VLA 官方复现记录

## 范围与基线

2026-10-03 删除旧 checkout 和旧环境后，从零重建。当前阶段定义如下：

- R0：代码、依赖、CUDA、LIBERO 和 EGL 环境验收；
- R1：使用作者发布 checkpoint 完成真实单卡 LIBERO rollout；
- R2：在服务器上执行官方 4 suite、2000 rollout 正式评估；
- R3/R4：只有 R2 评估链路一致后才开始训练验证和完整训练。

当前没有加入 Spatial-LaRA 或我们自己的模型修改。服务器从零部署步骤见
[R2 服务器复现手册](R2_SERVER_RUNBOOK.md)。

R2 服务器首次并行启动时发现，robosuite 1.4.0 要求
`MUJOCO_EGL_DEVICE_ID` 是 `CUDA_VISIBLE_DEVICES` 中的物理 GPU ID。旧 wrapper
固定使用 GPU 0；当实际 GPU 池为 `1,2,6,7` 时，四个 client 会在导入阶段
触发断言。复现 wrapper 现默认选择可见 GPU 池中的第一张，并在启动前验证
映射；模型、checkpoint、任务和 rollout 协议均未改变。

同一次启动还暴露了官方并行脚本 readiness probe 的 Shell 问题：Python
代码使用未加引号的 heredoc，错误提示中的反引号 Pip 示例会在 Shell 展开
heredoc 时被执行，并把输出重定向到仓库根目录的 `=11`。现将 heredoc 引号
固定，并通过 `sys.argv` 传入端口，防止任何命令替换；端口探测语义不变。

- Fork：<https://github.com/husterPWC/LaRA-VLA>
- Upstream：<https://github.com/LoveJu1y/LaRA-VLA.git>
- 初始 fork/origin/main/upstream/main：`93b5c03c691e38a3c9e90878e76557009e2df261`
- LIBERO：<https://github.com/Lifelong-Robot-Learning/LIBERO.git>
- LIBERO commit：`8f1084e3132a39270c3a13ebe37270a43ece2a01`
- 初始 tracked worktree 干净，没有执行 merge、rebase 或 reset。
- 主机：Ubuntu 22.04.5，两张 RTX 3090 24 GiB；R0/R1 只使用 GPU 0。
- Driver 570.144；系统 CUDA toolkit 12.4（nvcc V12.4.99）。
- `nvidia-smi` 中的 CUDA 12.8 是驱动能力，不是 torch runtime。

## 环境设计

使用两个新建 Conda 环境，均为 Python 3.10：

- `lara-vla`：policy server、模型加载、训练代码；
- `libero`：MuJoCo/LIBERO client。

关键版本：

- server：torch 2.6.0+cu124、torchvision 0.21.0+cu124、transformers
  4.57.0、accelerate 1.5.2、DeepSpeed 0.16.9、NumPy 1.26.4；
- client：torch 2.5.1+cpu、torchvision 0.20.1+cpu、NumPy 1.24.4、
  MuJoCo 2.3.7、robosuite 1.4.0；
- client 不运行神经网络 policy，CPU torch 不妨碍 EGL 使用 GPU 渲染；
- 使用 torch 2.5.1 读取 LIBERO 可信 init-state 文件，可避免修改 benchmark
  来适配 torch 2.6 的 `weights_only` 默认变化；
- Numba 0.60.0 与 NumPy 1.24 配套；setuptools 75.8.0 保留旧依赖所需 API。

这次重建没有改变模型、loss、curriculum、action 语义或 LIBERO benchmark。

## 主机本地资源

```text
工作区：/home/robot/codePWC/LaRA
代码：  /home/robot/codePWC/LaRA/LaRA-VLA
LIBERO：/home/robot/codePWC/LaRA/LIBERO
训练集：/data/CodePWC/lara_datasets/libero_lerobot_all
backbone：../StarVLA-Qwen3-VL-4B-Instruct-Action
官方运行目录：../checkpoints/LaRA-VLA-libero
官方 checkpoint：../checkpoints/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt
```

`environment/local-weights.sha256` 记录本地权重字节哈希，但它本身不能证明
对应哪个 Hugging Face revision。checkpoint 文件名显示 25k，配置写的是
最大 40k，summary 又包含 16k–40k；不能仅凭文件名推断完整训练来源。

生成本机 LIBERO 配置：

```bash
python scripts/reproduction/configure_libero.py \
  --libero-home ../LIBERO \
  --datasets /data/CodePWC/lara_datasets/clip-rt/modified_libero_hdf5
```

该命令生成未跟踪的 `../LIBERO/libero/config.yaml`。R0 reset/render 和 R1
评估不读取 demonstration HDF5。运行 client 时设置：

```bash
export LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
```

旧 fork 中的 `LARAVLA_VLM_PATH` 补丁在干净 fork 中不存在。使用官方
`framework.qwenvl.base_vlm` 配置机制创建隔离 checkpoint 视图：

```bash
python scripts/reproduction/prepare_official_checkpoint.py \
  --source-run ../checkpoints/LaRA-VLA-libero \
  --backbone ../StarVLA-Qwen3-VL-4B-Instruct-Action \
  --output-run ../checkpoints/repro_r1_official
```

生成的 manifest 会证明只修改 `framework.qwenvl.base_vlm`。checkpoint 使用
硬链接而非符号链接：官方 loader 会先解析 checkpoint 路径再寻找相邻配置；
符号链接会跳回原始目录并读到远端 backbone 名称。硬链接不会复制第二份
10.3 GB 权重，同时能保证加载隔离目录中的配置。

## R0 环境验收

```bash
bash scripts/reproduction/env_check.sh
```

该脚本执行：两个环境的 `pip check`、正式训练入口/模型/dataset loader/Qwen3
导入、BF16 CUDA 计算、PyTorch3D native CUDA、真实 AV1 视频解码、全部
40 个 LIBERO 任务 init states，以及 Goal task 0 的两路 256×256 EGL
reset/render。它不会替换 policy、运行 rollout 或换成小模型。

**R0 于 2026-10-03 通过。** R0 只能证明环境正常，不能替代 checkpoint
加载或 R1 rollout。

## R1 官方 checkpoint 单卡 smoke test

**R1 于 2026-10-04 通过。**

- LaRA commit：`8fbb10816b8ecf62274771392d1d46af38b4ed23`
- LIBERO commit：`8f1084e3132a39270c3a13ebe37270a43ece2a01`
- checkpoint state dict 严格匹配；missing/unexpected keys 都为空；
- 没有触发 optional-module `strict=False` fallback；
- 模型参数量 4.6045B；加载后 PyTorch 统计显存约 9.22 GB；
- Qwen3-VL model/processor、真实 LIBERO Goal 视频帧预处理通过；
- action/state 配置维度为 7，action horizon 为 8；
- 正式 prompt 含 1 个 start、3 个 thinking、1 个 end 和 16 个
  `<img_next>` token。

checkpoint 严格预检命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD" \
/home/robot/miniconda3/envs/lara-vla/bin/python \
  scripts/reproduction/check_official_checkpoint.py \
  --checkpoint ../checkpoints/repro_r1_official/checkpoints/steps_25000_pytorch_model.pt \
  --dataset-root /data/CodePWC/lara_datasets/libero_lerobot_all \
  --output results/r1_checkpoint_preflight.json
```

真实 rollout 使用两个终端：

```bash
# 终端 A
scripts/reproduction/smoke_inference.sh server \
  2>&1 | tee logs/repro_r1/server.log
```

```bash
# 终端 B，等待终端 A 打印 server listening
scripts/reproduction/smoke_inference.sh client \
  2>&1 | tee logs/repro_r1/client_stdout.log
```

实际执行 `libero_goal` task 0、seed 7、1 rollout，任务为 “open the middle
drawer of the cabinet”，结果 1/1 成功。client exit 0，总耗时 15.72 秒。
policy 执行 16 个 action chunks，每次都记录 4 个 iterative latent reasoning
passes。200 ms 外部采样得到 GPU 0 峰值 9767 MiB、峰值利用率 93%，无
fallback、shape error、NaN/Inf、server error 或 OOM。

两路环境图像为 256×256×3 uint8，client 发送前缩放到 224×224×3。返回
normalized action shape 为 1×8×7。官方 evaluator 构造 1×8 robot state，
但不把它发送给 server，因此 checkpoint 使用 `state=None`。这是官方行为，
R1 不静默修改，作为 R2 一致性风险记录。

可审查证据：

- `results/r1_checkpoint_preflight.json`
- `results/r1_official_checkpoint_smoke.json`
- 原始日志位于被 Git 忽略的 `logs/repro_r1/`

LIBERO 会打印可信文件的 `torch.load` future warning。episode 成功写入后，
robosuite EGL 析构器可能重复释放 context 并打印 warning，但 client 仍 exit 0。
没有通过源码补丁隐藏这些 warning。

R1 对 evaluator 只增加可选 `task_id`；默认 `None` 保留官方遍历全部任务的
行为。commit `f44406e` 只把 checkpoint 准备方式从符号链接改为硬链接。

## R0 中定位的依赖打包问题

1. `pipablepytorch3d==0.7.6` 宣称是通用 wheel，却包含
   `cp311-cp311-linux_x86_64` 标签和 CPython 3.11 `.so`；Python 3.10
   只能导入纯 Python 部分，无法使用 native extension。
2. `decord==0.6.0` 与 `eva-decord==0.6.1` 共享并覆盖 94 个安装路径；
   decord 文件名标记为 `py3-none-manylinux2010`，内部 WHEEL 却错误标成
   `cp36-cp36m-manylinux2010`。

修复没有降级 pip，也没有忽略 `pip check`：

- 从官方 PyTorch3D v0.7.6 commit
  `f34104cf6ebefacd7b7e07955ee7aaa823e616ac` 源码构建适配当前 Python、
  torch、CUDA 和 GPU 架构的 wheel；
- 只保留一个 decord provider；解包官方 0.6.0 wheel，只修内部 ABI tag，
  重新生成 RECORD；
- 原始 decord wheel SHA256：
  `51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976`；
- 主机修复后 wheel SHA256：
  `991dd62b390a2393ad0e531f2dd8c6bc20dd26dc89ad85cff363b0de0323952c`；
- decord 内置旧 FFmpeg 无法解码当前 AV1 数据，但官方 loader 实际使用
  `torchvision_av`，该路径已正确解码真实 256×256 AV1 帧。

构建脚本：`scripts/reproduction/build_dependency_wheels.sh`。native wheel
是机器相关产物，不进入 Git；8 GPU 服务器必须按真实 GPU 架构重新编译。

## 后续阶段需要持续追踪的问题

- evaluator 构造 robot state 但不发送给 policy；
- LIBERO client 使用 `min/max` 反归一化，base helper 使用 `q01/q99`；
- 重复定义的 `baseframework.get_action_stats` 被标成 classmethod，却访问
  instance `norm_stats`；LIBERO 正式评估绕过该 helper，因此 R1 未修模型；
- checkpoint loader 有 optional-module `strict=False` 兼容逻辑；每次加载
  都要记录 missing/unexpected keys；
- README 的 `LIBERO_LEROBOT_ROOT` 不被训练路径读取，应使用已有的
  `--datasets.vla_data.data_root_dir` override；
- multistage launcher 只覆盖 5k/2k/2k/2k reasoning 阶段；Stage III 是
  独立脚本，默认 60k、单进程、pretrained path 为空；
- `img_next.use_teacher=false` 会关闭当前 forward 中的 visual supervision
  loss，而论文描述 Stage I/II 使用 visual supervision；
- Stage III YAML 默认仍保留 discrete action supervision 和 VLM weight 1，
  发布 checkpoint 配置则关闭 action tokens 并设置 VLM weight 0；
- gradient accumulation、gradient checkpointing、num_workers、DDP device
  placement、optimizer/scheduler/RNG resume 都要在 R3/R4 前逐项验证。

## R3 训练数据预检

R3 使用的数据根目录是：

```text
/data/CodePWC/lara_datasets/libero_lerobot_all
```

旧目录 `clip-rt/modified_libero_hdf5` 中的四个 HDF5 文件均为 0 字节，
不是可用训练数据，也不是官方当前 README 要求的 LeRobot 数据布局。

离散 action supervision 使用官方配置指定的
`physical-intelligence/fast`。为防止 Hugging Face `main` 漂移，本次固定到提交：

```text
ec4d7aa71691cac0b8bed6942be45684db2110f4
```

其中 `tokenizer.json` 的 SHA256 为：

```text
6507dd709287fd018882120c0071787f1f62bad9f180f1e8c5235bda1b71fa78
```

依赖文件保存在仓库外的
`/home/robot/codePWC/LaRA/dependencies/physical-intelligence-fast/<revision>`，
不进入 Git。FAST 的 8×7 action 编解码测试通过；由于其 DCT 量化，测试样本
最大绝对重建误差约为 0.043。

`scripts/reproduction/check_training_data.py` 直接调用官方 dataset、transform、
视频解码和 reasoning formatter。Stage 1–4 均已逐 suite 验证通过：

| 数据集 | transitions | trajectories | 图像 | action |
|---|---:|---:|---|---|
| LIBERO Object | 66,984 | 454 | 当前/下一帧各 2×224×224×3 | 8×7 |
| LIBERO Goal | 52,042 | 428 | 当前/下一帧各 2×224×224×3 | 8×7 |
| LIBERO Spatial | 52,970 | 432 | 当前/下一帧各 2×224×224×3 | 8×7 |
| LIBERO-10 | 101,469 | 379 | 当前/下一帧各 2×224×224×3 | 8×7 |

四套数据的 mixture sampling weight 均为 0.25。抽取的真实样本均有 CoT、
bbox、FAST action tokens 和 16 个 `<img_next>`；Stage 1/2/3/4 的
`<|thinking|>` 数量依次为 0/1/2/3，与课程替换顺序一致。机器可读结果位于
`results/r3_training_data_stage1.json` 至
`results/r3_training_data_stage4.json`。

### R3 训练配置生效修复

官方 `train.py` 在读取 YAML/CLI 前便以默认参数创建 `Accelerator`，因此
`trainer.gradient_accumulation_steps` 和混合精度配置没有传给
Accelerate/DeepSpeed。训练步又在每次 micro-batch forward 前调用
`optimizer.zero_grad()`；在同步更新的 micro-batch 上，这会清掉此前累积的
梯度。两者共同导致配置中的 gradient accumulation 实际不成立。

最小修复如下：

- 在配置完成合并后创建 `Accelerator` 和 `DeepSpeedPlugin`；
- 明确传入 gradient accumulation、bf16、ZeRO stage 和 offload 配置；
- 将 `zero_grad()` 移到 optimizer step 后，由 AcceleratedOptimizer 仅在同步
  更新步清梯度；
- 将 dataloader 的 `num_workers` 改为配置项，默认仍为官方的 4；
- 默认仍为 ZeRO-2、无 offload；R3 单卡 smoke 才显式覆盖为 CPU optimizer
  offload。

独立配置探针已验证：accumulation=3、bf16、ZeRO-2、CPU optimizer offload
均进入最终 DeepSpeed 配置。该修复只使已有工程配置真正生效，不改变模型、
数据、loss 或 forward。

### R3 checkpoint / resume 修复

官方 `_save_checkpoint` 只写
`steps_<N>_pytorch_model.pt` 和一行 step summary；它没有调用
`accelerator.save_state()`。而 `_load_checkpoint` 却调用
`accelerator.load_state()`，也没有恢复 `completed_steps`。因此原实现只能在
阶段之间传递模型权重，不能恢复 optimizer、scheduler、RNG 或训练进度。

修复后保留两类产物，各自用途明确：

- `steps_<N>_pytorch_model.pt`：保持官方既有命名，供 Stage I → Stage II →
  Stage III 传递权重；
- `steps_<N>_training_state/`：由 `Accelerator.save_state()` 集体保存模型、
  optimizer、scheduler、RNG 和 DeepSpeed 状态，并额外保存
  `trainer_state.json` 中的 completed steps、已读取 batch 数、dataset epoch 和
  image-next EMA 更新计数。

`trainer.is_resume=true` 现在必须同时给出
`trainer.resume_from_checkpoint=<training_state目录>`；若 metadata 缺失会直接
失败。恢复后跳过当前 epoch 已消费的真实 batch，progress bar 从已完成 step
继续。scheduler 也一并交给 `accelerator.prepare()`，从而进入状态保存。

同时修复了 accumulation 下的重复副作用：evaluation、logging 和 checkpoint
只在真正完成 optimizer update 的同步步执行。LeRobot mixture 的 epoch 也会在
dataloader 重建时递增，避免跨 epoch 重复同一批 `(epoch, index, seed)` 样本。

### R3 训练验收指标

官方 trainer 原来只记录 `action_loss`、`vlm_loss` 和 `total_loss`，没有记录
`img_next_loss`、梯度范数和显存；image-next 内部异常又会被 warning 后跳过。
这会使 Stage I/II 在缺失论文视觉预测监督时仍可能正常退出。

现在每个 logging step 额外写入 `metrics.jsonl`：

- 当前阶段实际返回的 `action_loss`、`vlm_loss`、`img_next_loss`；
- `total_loss`；
- optimizer update 前的全局 `grad_norm`；
- 每个 optimizer parameter group 的 learning rate；
- CUDA allocated、reserved 和 peak allocated memory。

所有出现的 loss 和 grad norm 都做 NaN/Inf 检查。训练结束时根据
`training_stage` 检查 VLM 或 action loss；当 image-next teacher 与 loss weight
启用时，还要求运行中至少实际出现一次 `img_next_loss`。此外至少需要一次有限
非零梯度更新。`training_validation.json` 保存 loss 观测次数、非零梯度步数及
total/trainable/frozen 参数量。该检查只验证官方 forward 的真实输出，不增加
或替换 loss。

### R3 模型初始化 seed 修复

官方训练器原来在模型和 optimizer 创建完成后才调用 `set_seed`。Stage III 只从
Stage II checkpoint 加载 `qwen_vl_interface`，连续 action head 需要重新随机
初始化；因此原顺序使 action head 不受配置中的 `seed` 控制。现在在
`Accelerator` 完成 rank/device 初始化后、构建模型前按
`cfg.seed + process_index` 设置随机种子。训练准备阶段仍会用同一规则再次设置，
不改变后续数据和训练随机流的约定。

### 论文训练配方与官方 launcher 的修复

论文 Table 6 与公开脚本逐项比较后，原 launcher 存在会改变训练目标的差异：

- `run_libero_multistage.sh` 把 `IMG_NEXT_USE_TEACHER` 设为 `false`；当前
  forward 在该值为 false 时完全不计算 `img_next_loss`，所以不是一种等价的
  teacher 实现；
- 论文 Stage II 的三个 2k 子阶段均为 image-next weight 0.2、batch 16，原脚本
  第一个子阶段仍为 0.1，前两个子阶段仍为 batch 12；
- `run_laravla_libero.sh` 没有 Stage II checkpoint，使用 60k、batch 8，且沿用
  action tokens 和 VLM weight 1，不能实现论文 Stage III 的连续 action/flow
  matching 目标。

最小修复后：

- 代码 stage 1（论文 Stage I）：5k、batch 12、image-next 0.1、teacher 开启；
- 代码 stage 2/3/4（论文 Stage II）：各 2k、batch 16、image-next 0.2、teacher
  开启，并依次替换 Subtask/BBox/Reasoning；
- 论文 Stage III：必须传入 Stage II 最终 checkpoint，只加载
  `qwen_vl_interface`，关闭离散 action tokens，VLM loss weight 为 0，执行
  40k continuous action/flow matching；默认 8 进程，单卡 R3 可通过环境变量
  `NUM_GPUS=1` 覆盖。

论文 Table 6 的 Stage III batch 为 16；官方发布 checkpoint 的 run config
记录 per-device batch 14。R4 默认遵循论文的 16，并将这项差异保留在最终结果
元数据中，不把二者静默混用。

### R3 单卡正式 smoke 驱动

`scripts/reproduction/smoke_train_r3.sh` 只调用正式
`laravla/training/train.py`，不包含替代模型、假数据或另一套 forward。默认使用：

- GPU 0；
- 正式 `libero_all` LeRobot 数据；
- 本地 Qwen3-VL 4B backbone；
- 固定 revision 的官方 FAST tokenizer；
- batch 1、num_workers 0、bf16、ZeRO-2 CPU optimizer offload；
- 每个阶段 2 个 optimizer steps；
- image-next teacher 在 Stage I/II 开启，在 Stage III 关闭；
- W&B disabled，但完整写入本地 log、metrics 和 validation JSON。

按以下顺序逐项执行和验收：

```bash
bash scripts/reproduction/smoke_train_r3.sh stage1
bash scripts/reproduction/smoke_train_r3.sh resume-stage1
bash scripts/reproduction/smoke_train_r3.sh stage2-1
bash scripts/reproduction/smoke_train_r3.sh stage2-2
bash scripts/reproduction/smoke_train_r3.sh stage2-3
bash scripts/reproduction/smoke_train_r3.sh stage3
```

Stage I 首次运行保存完整 training state；第二条命令必须从 step 2 恢复并完成
step 3。后续阶段保存正式 `.pt` 供链式加载，但关闭重复 training-state 和
final-model 副本，以控制 3090 主机磁盘使用。正式 R4 默认配置仍保存完整状态。

## 修改原则

每个源码修复必须记录根因、修改原因、最小 diff，并独立提交。数据集、cache、
权重、大日志和视频不进入 Git；Git 只保存代码、配置、环境锁、小型结果摘要
和复现元数据。环境检查不得引入替代模型、fake data 或另一套 forward。
