#!/usr/bin/env bash
# Libero-all :: train.py — repository root: bash scripts/run_laravla_libero.sh
set -euo pipefail
export TOKENIZERS_PARALLELISM=false


PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
RELOAD_MODULES="${RELOAD_MODULES:-qwen_vl_interface}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"

if [[ -z "${PRETRAINED_CKPT}" || ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "Stage III 需要有效的 Stage II 最终 checkpoint: PRETRAINED_CKPT" >&2
  exit 1
fi

# 若改 run_root_dir / run_id，请同步改下面 mkdir 路径
mkdir -p results/Libero_VLA/libero_all_vla

args=(
  --config_yaml laravla/config/training/libero.yaml
  --run_root_dir results/Libero_VLA
  --run_id libero_all_vla
  --wandb_project libero_vla
  --framework.training_stage full
  --datasets.vla_data.bridge_reasoning.stage 4
  --datasets.vla_data.bridge_reasoning.include_action_tokens false
  --framework.latent_reasoning.vlm_loss_weight 0
  --trainer.max_train_steps 40000
  --trainer.save_interval 4000
  --trainer.eval_interval 20000000
  --trainer.min_save_step 16000
  --datasets.vla_data.per_device_batch_size 16
  --framework.img_next.use_teacher false
  --framework.action_model.diffusion_model_cfg.dropout 0.1
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}"
  --trainer.reload_modules "${RELOAD_MODULES}"
)

exec torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  laravla/training/train.py \
  "${args[@]}" \
  "$@"
