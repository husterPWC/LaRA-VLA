# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
Latent Reasoning Training Script for LaRA-VLA

Training with implicit latent reasoning (thinking tokens + KV-Cache iterative forward).
Supports VLM loss computation and img_next alignment loss.
"""

# Standard Library
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin, skip_first_batches
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from laravla.training.trainer_utils.trainer_tools import normalize_dotlist_args
from laravla.model.framework import build_framework
from laravla.training.trainer_utils.trainer_tools import TrainerUtils
from laravla.training.trainer_utils.trainer_tools import build_param_lr_groups
from laravla.training.trainer_utils.cot_mode_utils import get_implicit_flags


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def build_accelerator(cfg) -> Accelerator:
    """Build Accelerate/DeepSpeed after the training config is available."""
    gradient_accumulation_steps = int(
        cfg.trainer.get("gradient_accumulation_steps", 1)
    )
    gradient_clipping = cfg.trainer.get("gradient_clipping", None)
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=gradient_clipping,
        zero_stage=int(cfg.trainer.get("deepspeed_zero_stage", 2)),
        offload_optimizer_device=cfg.trainer.get(
            "deepspeed_offload_optimizer_device", "none"
        ),
        offload_param_device=cfg.trainer.get(
            "deepspeed_offload_param_device", "none"
        ),
        zero3_init_flag=bool(
            cfg.trainer.get("deepspeed_zero3_init_flag", False)
        ),
        zero3_save_16bit_model=bool(
            cfg.trainer.get("deepspeed_zero3_save_16bit_model", False)
        ),
    )
    mixed_precision = (
        "bf16"
        if cfg.trainer.get("enable_mixed_precision_training", True)
        else "no"
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    accelerator.print(accelerator.state)
    return accelerator


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from laravla.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    # Safely extract data_mix for logging (could be in ecot.* or vla_data.*)
    cot_mode = getattr(getattr(cfg, "framework", {}), "cot_mode", "implicit")
    mode_flags = getattr(getattr(cfg, "framework", {}), "cot_mode_flags", {}) or {}
    generate_thinking = mode_flags.get("generate_thinking", cot_mode in ("explicit", "implicit"))
    reasoning_stage = getattr(cfg.datasets.vla_data.bridge_reasoning, "stage", "unknown")

    dataset_py = cfg.datasets.vla_data.dataset_py
    data_mix = getattr(cfg.datasets.vla_data, "data_mix", "unknown")
    logger.info(
        "Creating VLA Dataset with dataset_py=`%s`, data_mix=`%s`, cot_mode=`%s`, stage=`%s`, generate_thinking=%s",
        dataset_py,
        data_mix,
        cot_mode,
        reasoning_stage,
        generate_thinking,
    )
    if generate_thinking and isinstance(reasoning_stage, int) and reasoning_stage < 1:
        logger.warning(
            "cot_mode=%s 需要 reasoning 数据，但当前 stage=%s 可能缺少 reasoning 标注，请检查配置/数据混合。",
            cot_mode,
            reasoning_stage,
        )
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Determine warmup steps: use warmup_ratio if provided, otherwise use num_warmup_steps
    if hasattr(cfg.trainer, 'warmup_ratio') and cfg.trainer.warmup_ratio > 0:
        num_warmup_steps = int(cfg.trainer.max_train_steps * cfg.trainer.warmup_ratio)
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"Using warmup_ratio={cfg.trainer.warmup_ratio}: num_warmup_steps={num_warmup_steps} (from {cfg.trainer.max_train_steps} total steps)")
    else:
        num_warmup_steps = cfg.trainer.num_warmup_steps
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"Using num_warmup_steps={num_warmup_steps}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


def validate_ecot_config(cfg):
    """Validate latent reasoning configuration consistency."""
    latent_cfg = cfg.framework.get("latent_reasoning", {})
    if not latent_cfg:
        logger.warning("latent_reasoning config is missing, using defaults")
        return

    vlm_loss_weight = latent_cfg.get("vlm_loss_weight", 0.1)
    compute_language_loss = latent_cfg.get("compute_language_loss", False)
    logger.info(f"Latent reasoning: compute_language_loss={compute_language_loss}, "
                f"vlm_loss_weight={vlm_loss_weight}")

    reasoning_stage = getattr(cfg.datasets.vla_data.bridge_reasoning, "stage", 4)
    logger.info(f"Bridge reasoning stage: {reasoning_stage}")

    img_next_cfg = cfg.framework.get("img_next", {})
    if img_next_cfg and img_next_cfg.get("enable", False):
        logger.info(f"img_next: res={img_next_cfg.get('res')}, "
                    f"loss_weight={img_next_cfg.get('loss_weight')}, "
                    f"use_teacher={img_next_cfg.get('use_teacher', True)}")

    logger.info("Config validation passed")


def sync_bridge_reasoning_to_framework(cfg):
    """
    Sync thinking token definitions from bridge_reasoning (dataloader side)
    to framework.latent_reasoning (model side), ensuring both use the same tokens.
    """
    try:
        bridge_cfg = cfg.datasets.vla_data.bridge_reasoning
    except AttributeError:
        return

    if not getattr(bridge_cfg, "enable", False):
        return

    cfg.framework.enable_latent_reasoning = True

    latent_cfg = getattr(cfg.framework, "latent_reasoning", None)
    if latent_cfg is None:
        cfg.framework.latent_reasoning = {}
        latent_cfg = cfg.framework.latent_reasoning

    def _set(key, value):
        if isinstance(latent_cfg, dict):
            latent_cfg[key] = value
        else:
            setattr(latent_cfg, key, value)

    # Sync token definitions: bridge_reasoning is the single source of truth
    _set("thinking_token", getattr(bridge_cfg, "thinking_token", "<|thinking|>"))
    _set("start_of_thinking_token", getattr(bridge_cfg, "start_token", "<|start_of_thinking|>"))
    _set("end_of_thinking_token", getattr(bridge_cfg, "end_token", "<|end_of_thinking|>"))

    tag2think = getattr(bridge_cfg, "tag2think_count", None)
    if tag2think is not None:
        _set("tag2think_count", tag2think)

    _set("compute_language_loss", True)


class LaRA_VLA_Trainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.batches_seen = 0
        self.vla_epoch_count = 0
        self._resume_batches_to_skip = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.min_save_step = getattr(self.config.trainer, "min_save_step", 0)
        self._img_next_ema_updates = 0
        self._observed_loss_counts = {
            "action_loss": 0,
            "vlm_loss": 0,
            "img_next_loss": 0,
        }
        self._finite_nonzero_grad_steps = 0
        self.parameter_counts = {}

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        is_resume = bool(self.config.trainer.get("is_resume", False))
        if (
            not is_resume
            and hasattr(self.config.trainer, "pretrained_checkpoint")
            and self.config.trainer.pretrained_checkpoint
        ):
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        parameter_counts = self.print_trainable_parameters(self.model)
        if parameter_counts is not None:
            total, trainable = parameter_counts
            self.parameter_counts = {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
            }

        # initialize distributed training components
        (
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        ) = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="ecot-qwengrt00t-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_resume = bool(self.config.trainer.get("is_resume", False))
        resume_from_checkpoint = self.config.trainer.get(
            "resume_from_checkpoint", None
        )

        # resume training state
        print(f"is_resume: {is_resume}")
        if is_resume:
            if not resume_from_checkpoint:
                raise ValueError(
                    "trainer.is_resume=true 时必须设置 "
                    "trainer.resume_from_checkpoint"
                )
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self._load_checkpoint(resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """Restore model, optimizer, scheduler, RNG and trainer progress."""
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        metadata_path = checkpoint_path / "trainer_state.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"训练状态缺少 metadata，不能执行完整 resume: {metadata_path}"
            )

        self.accelerator.load_state(str(checkpoint_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.completed_steps = int(metadata["completed_steps"])
        self.batches_seen = int(metadata["batches_seen"])
        self.vla_epoch_count = int(metadata.get("vla_epoch_count", 0))
        self._img_next_ema_updates = int(
            metadata.get("img_next_ema_updates", 0)
        )
        self._observed_loss_counts = metadata.get(
            "observed_loss_counts", self._observed_loss_counts
        )
        self._finite_nonzero_grad_steps = int(
            metadata.get("finite_nonzero_grad_steps", 0)
        )
        self._resume_batches_to_skip = self.batches_seen % len(
            self.vla_train_dataloader
        )
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
            dataset.set_epoch(self.vla_epoch_count)
        self.accelerator.print(
            f"Resumed from {checkpoint_path}: "
            f"completed_steps={self.completed_steps}, "
            f"batches_seen={self.batches_seen}, "
            f"epoch={self.vla_epoch_count}"
        )

    def _save_checkpoint(self):
        """Save a stage-transfer model plus a fully resumable training state."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"steps_{self.completed_steps}"
        )
        state_path = checkpoint_path + "_training_state"

        save_training_state = bool(
            self.config.trainer.get("save_training_state", True)
        )
        if save_training_state:
            # DeepSpeed state saving is collective and must run on every rank.
            self.accelerator.save_state(
                output_dir=state_path,
                safe_serialization=False,
            )
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            trainer_state = {
                "completed_steps": self.completed_steps,
                "batches_seen": self.batches_seen,
                "vla_epoch_count": self.vla_epoch_count,
                "img_next_ema_updates": self._img_next_ema_updates,
                "observed_loss_counts": self._observed_loss_counts,
                "finite_nonzero_grad_steps": self._finite_nonzero_grad_steps,
            }
            if save_training_state:
                Path(state_path, "trainer_state.json").write_text(
                    json.dumps(trainer_state, indent=2) + "\n",
                    encoding="utf-8",
                )

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
                "training_state": state_path if save_training_state else None,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(
                f"✅ Model checkpoint saved at {checkpoint_path}_pytorch_model.pt"
            )
            if save_training_state:
                self.accelerator.print(f"✅ Training state saved at {state_path}")
                self._prune_training_states()
        self.accelerator.wait_for_everyone()

    def _prune_training_states(self):
        """Keep only the newest complete resume states when a limit is set."""
        keep = self.config.trainer.get(
            "max_training_state_checkpoints", None
        )
        if keep is None:
            return
        keep = int(keep)
        if keep < 1:
            raise ValueError(
                "trainer.max_training_state_checkpoints must be >= 1 or null"
            )

        completed_states = []
        for path in Path(self.checkpoint_dir).glob(
            "steps_*_training_state"
        ):
            match = re.fullmatch(
                r"steps_(\d+)_training_state", path.name
            )
            if match and (path / "trainer_state.json").is_file():
                completed_states.append((int(match.group(1)), path))

        completed_states.sort(reverse=True)
        for _, old_path in completed_states[keep:]:
            shutil.rmtree(old_path)
            self.accelerator.print(
                "Removed old complete training state due to retention: "
                f"{old_path}"
            )

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rates for every optimizer parameter group
                learning_rates = self.lr_scheduler.get_last_lr()
                metrics["learning_rate"] = learning_rates[0]
                for index, group in enumerate(self.optimizer.param_groups):
                    group_name = group.get("name", f"group_{index}")
                    metrics[f"learning_rate/{group_name}"] = learning_rates[index]

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                if torch.cuda.is_available():
                    metrics["gpu_memory/allocated_gib"] = (
                        torch.cuda.memory_allocated() / 1024**3
                    )
                    metrics["gpu_memory/reserved_gib"] = (
                        torch.cuda.memory_reserved() / 1024**3
                    )
                    metrics["gpu_memory/peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated() / 1024**3
                    )

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                metrics_path = Path(self.config.output_dir) / "metrics.jsonl"
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"step": self.completed_steps, **metrics},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        if self._resume_batches_to_skip:
            resumed_dataloader = skip_first_batches(
                self.vla_train_dataloader, self._resume_batches_to_skip
            )
            self.vla_iter = iter(resumed_dataloader)
            self.accelerator.print(
                f"Skipping {self._resume_batches_to_skip} already consumed batches"
            )
            self._resume_batches_to_skip = 0
        else:
            self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        self.batches_seen += 1

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # # evaluate model
            if (
                self.accelerator.sync_gradients
                and self.completed_steps % self.config.trainer.eval_interval == 0
            ):
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            if self.accelerator.sync_gradients:
                self._log_metrics(step_metrics)

            # save checkpoint
            if (
                self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps >= self.min_save_step
                and self.completed_steps > 0
                and self.accelerator.sync_gradients
            ):
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if self.accelerator.is_main_process:

            examples = self._get_next_batch()

            score = 0.0
            num_samples = len(examples)

            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label

            # Predict actions using the model
            output_dict = self.model.predict_action(
                batch_images=batch_images, instructions=instructions, use_ddim=True, num_ddim_steps=20
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D

            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score
        pass
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

            latent_cfg = self.config.framework.get("latent_reasoning", {})
            reasoning_stage = getattr(self.config.datasets.vla_data.bridge_reasoning, "stage", 4)
            logger.info("***** Latent Reasoning Configuration *****")
            logger.info(f"  Reasoning Stage: {reasoning_stage}")
            logger.info(f"  Compute Language Loss: {latent_cfg.get('compute_language_loss', False)}")
            logger.info(f"  VLM Loss Weight: {latent_cfg.get('vlm_loss_weight', 0.1)}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)

                # Use model-computed total_loss if available (includes vlm_loss if enabled)
                # Otherwise fallback to action_loss only (backward compatibility)
                if "total_loss" in output_dict:
                    total_loss = output_dict["total_loss"]
                else:
                    total_loss = output_dict["action_loss"]

            for loss_name in (
                "action_loss",
                "vlm_loss",
                "img_next_loss",
                "total_loss",
            ):
                loss = total_loss if loss_name == "total_loss" else output_dict.get(loss_name)
                if loss is not None and not torch.isfinite(loss.detach()).all():
                    raise FloatingPointError(
                        f"{loss_name} contains NaN/Inf at step {self.completed_steps}"
                    )

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            grad_norm_value = None
            if (
                self.accelerator.sync_gradients
                and self.config.trainer.gradient_clipping is not None
            ):
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
                if grad_norm is not None:
                    grad_norm_value = float(grad_norm.detach().float().item())

            # optimizer step
            self.optimizer.step()
            if (
                self.accelerator.sync_gradients
                and grad_norm_value is None
                and callable(getattr(self.model, "get_global_grad_norm", None))
            ):
                deepspeed_grad_norm = self.model.get_global_grad_norm()
                if deepspeed_grad_norm is not None:
                    grad_norm_value = float(deepspeed_grad_norm)
            self.lr_scheduler.step()
            # AcceleratedOptimizer only clears gradients on synchronized update
            # steps, preserving gradients across accumulation micro-batches.
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients and grad_norm_value is not None:
                if not np.isfinite(grad_norm_value):
                    raise FloatingPointError(
                        f"grad_norm contains NaN/Inf at step {self.completed_steps}"
                    )
                if grad_norm_value > 0:
                    self._finite_nonzero_grad_steps += 1

            # EMA update for img_next target vision encoder (if enabled)
            if self.accelerator.sync_gradients:
                self._img_next_ema_updates += 1
                if self._img_next_ema_updates % 2 == 0:
                    try:
                        qwen_iface = getattr(self.model, "qwen_vl_interface", None)
                        if (
                            qwen_iface is not None
                            and getattr(qwen_iface, "use_img_next_teacher", True)
                            and hasattr(qwen_iface, "update_img_next_ema")
                        ):
                            qwen_iface.update_img_next_ema()
                    except Exception as e:
                        logger.warning(f"[img_next_ema] update skipped due to error: {e}")

        # Build metrics dictionary
        metrics = {}
        for loss_name in ("action_loss", "vlm_loss", "img_next_loss"):
            loss = output_dict.get(loss_name)
            if loss is not None:
                metrics[loss_name] = loss.detach().float().item()
                self._observed_loss_counts[loss_name] += 1
        
        # Add total_loss for monitoring (ensures consistency with backward pass)
        metrics["total_loss"] = total_loss.detach().float().item()
        if grad_norm_value is not None:
            metrics["grad_norm"] = grad_norm_value
        
        return metrics

    def _finalize_training(self):
        """training end processing"""
        training_stage = self.config.framework.training_stage
        required_losses = (
            ("vlm_loss",)
            if training_stage == "reasoning_only"
            else ("action_loss",)
        )
        img_next_cfg = self.config.framework.get("img_next", {})
        if (
            training_stage in ("reasoning_only", "full")
            and img_next_cfg.get("enable", False)
            and img_next_cfg.get("use_teacher", True)
            and img_next_cfg.get("loss_weight", 0) > 0
        ):
            required_losses += ("img_next_loss",)

        missing_losses = [
            name
            for name in required_losses
            if self._observed_loss_counts.get(name, 0) == 0
        ]
        if missing_losses:
            raise RuntimeError(
                f"Training finished without required losses: {missing_losses}; "
                f"observed={self._observed_loss_counts}"
            )
        if self._finite_nonzero_grad_steps == 0:
            raise RuntimeError("Training finished without a finite non-zero gradient step")

        if self.accelerator.is_main_process:
            validation = {
                "completed_steps": self.completed_steps,
                "observed_loss_counts": self._observed_loss_counts,
                "finite_nonzero_grad_steps": self._finite_nonzero_grad_steps,
                "parameter_counts": self.parameter_counts,
            }
            Path(self.config.output_dir, "training_validation.json").write_text(
                json.dumps(validation, indent=2) + "\n",
                encoding="utf-8",
            )

        # save final model
        if (
            self.accelerator.is_main_process
            and self.config.trainer.get("save_final_model", True)
        ):
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("ECoT VLA Training :: Warming Up")

    accelerator = build_accelerator(cfg)
    seed = int(getattr(cfg, "seed", 42)) + accelerator.process_index
    set_seed(seed)
    accelerator.print(f"Seed set before model initialization: {seed}")

    mode_flags = get_implicit_flags()

    # Inject derived flags into cfg
    cfg.framework.enable_latent_reasoning = True
    cfg.framework.emit_thinking_tokens = False
    cfg.framework.cot_mode = "implicit"
    cfg.framework.cot_mode_flags = mode_flags

    training_stage = getattr(cfg.framework, "training_stage", "full")
    if training_stage == "full":
        cfg.datasets.vla_data.bridge_reasoning.stage = mode_flags["reasoning_stage"]

    logger.info(f"[Implicit Reasoning] training_stage={training_stage}, flags={mode_flags}")

    sync_bridge_reasoning_to_framework(cfg)

    # Validate ECoT configuration
    validate_ecot_config(cfg)

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run ECoT VLA Training
    trainer = LaRA_VLA_Trainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECoT Implicit Reasoning Training Script")
    parser.add_argument("--config_yaml", type=str, default="laravla/config/training/bridge.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
