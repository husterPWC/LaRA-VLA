# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image



from laravla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from laravla.model.framework.base_framework import baseframework
from laravla.model.framework.latent_analysis_mixin import LatentAnalysisMixin
from laravla.model.modules.vlm import get_vlm_model
from laravla.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from laravla.training.trainer_utils.trainer_tools import resize_images
from laravla.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(LatentAnalysisMixin, baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ── Spatial Transition Modules (bottleneck) ──────────
        trans_cfg = config.framework.get("mask_conditioned_transition", {}) if hasattr(config.framework, "get") else {}
        self.transition_enabled = trans_cfg.get("enable", False)
        if self.transition_enabled:
            from laravla.model.modules.spatial_transition import (
                VLMProjector, MaskTokenEncoder, MaskConditionedTransitionModule,
                MaskDecoder, RelationHead, TransitionToActionProjector
            )
            vlm_dim = self.qwen_vl_interface.model.config.hidden_size  # 2560
            self.transition_dim = trans_cfg.get("transition_dim", 512)
            num_mask_tokens = trans_cfg.get("num_mask_tokens", 8)
            num_transition_tokens = trans_cfg.get("num_transition_tokens", 6)
            num_relation_labels = trans_cfg.get("num_relation_labels", 6)

            self.vlm_projector = VLMProjector(vlm_dim=vlm_dim, transition_dim=self.transition_dim)
            self.mask_token_encoder = MaskTokenEncoder(
                in_channels=1, transition_dim=self.transition_dim, num_tokens=num_mask_tokens
            )
            self.transition_module = MaskConditionedTransitionModule(
                transition_dim=self.transition_dim, num_transition_tokens=num_transition_tokens
            )
            self.future_mask_decoder = MaskDecoder(
                transition_dim=self.transition_dim, num_transition_tokens=num_transition_tokens,
                output_res=trans_cfg.get("mask_res", 56)
            )
            self.goal_mask_decoder = MaskDecoder(
                transition_dim=self.transition_dim, num_transition_tokens=num_transition_tokens,
                output_res=trans_cfg.get("mask_res", 56)
            )
            self.relation_head = RelationHead(
                transition_dim=self.transition_dim, num_transition_tokens=num_transition_tokens,
                num_classes=num_relation_labels
            )
            # P2: project transition tokens back to VLM dim for action head
            self.transition_to_action = TransitionToActionProjector(
                transition_dim=self.transition_dim, vlm_dim=vlm_dim
            )
            from laravla.model.modules.spatial_transition import GatedTransitionActionAdapter
            self.transition_action_adapter = GatedTransitionActionAdapter(
                transition_dim=self.transition_dim, num_transition_tokens=num_transition_tokens,
                vlm_dim=vlm_dim
            )
            self.transition_loss_weights = trans_cfg.get("loss_weights", {
                "future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05
            })
        else:
            self.vlm_projector = None
            self.mask_token_encoder = None
            self.transition_module = None
            self.future_mask_decoder = None
            self.goal_mask_decoder = None
            self.relation_head = None
            self.transition_to_action = None
            self.transition_action_adapter = None
            self.transition_loss_weights = {}

        # Training stage control: "reasoning_only", "action_only", or "full"
        self.training_stage = config.framework.get("training_stage", "full")

        # Apply parameter freezing based on training stage
        if self.training_stage == "reasoning_only":
            print(f"[Training Stage] reasoning_only mode - Freezing action_model parameters")
            for param in self.action_model.parameters():
                param.requires_grad = False
        elif self.training_stage == "action_only":
            print(f"[Training Stage] action_only mode - Freezing VLM parameters")
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad = False
        else:
            print(f"[Training Stage] full mode - All parameters trainable")
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        action_tokens = [example.get("action_tokens", "") for example in examples]
        # img_next: List of PIL list (primary view), fallback flags
        image_next = [example.get("image_next", None) for example in examples]
        image_next_fallback = torch.tensor(
            [bool(example.get("image_next_fallback", False)) for example in examples],
            device=self.qwen_vl_interface.model.device,
        )
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # ── Trainable param guard for transition stages ───────
        # These stages freeze VLM; transition modules hold trainable params.
        # Only create the full qwen_inputs (with alignment logic) for stages that need them.

        # Deferred: build_qwenvl_inputs is called inside branches that need it
        # (reasoning_only, action_only, full). New stages (explicit_transition_cot,
        # latent_transition, transition_action) use encode_observation instead.

        if self.training_stage in ("reasoning_only", "action_only", "full"):
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                action_tokens=action_tokens,
            )
            enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
            use_iterative_forward = (
                enable_latent_reasoning
                and hasattr(self.qwen_vl_interface, "forward_latent")
            )
        else:
            qwen_inputs = None
            enable_latent_reasoning = False
            use_iterative_forward = False

        if use_iterative_forward:
            # Step 2: Iterative forward with KV-Cache for implicit reasoning
            vlm_outputs = self.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=qwen_inputs.get("labels"),  # May contain masked labels
                position_ids=qwen_inputs.get("position_ids"),
            )
            
            last_hidden = vlm_outputs['hidden_states']  # [B, L, H]
            vlm_loss = vlm_outputs.get('loss')  # May be None if no labels
        elif qwen_inputs is not None:
            # Step 2: Normal forward pass (no iterative reasoning)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
                vlm_loss = qwenvl_outputs.loss if hasattr(qwenvl_outputs, 'loss') else None
        else:
            last_hidden = None
            vlm_loss = None

        # Step 3: Compute losses based on training stage
        result = {}

        img_next_loss = None
        img_next_cfg = getattr(self.config.framework, "img_next", {}) if hasattr(self.config, "framework") else {}
        enable_img_next = img_next_cfg.get("enable", False) and qwen_inputs is not None
        img_next_loss_weight = img_next_cfg.get("loss_weight", 0.5)
        img_next_res = img_next_cfg.get("res", 112)
        img_next_token_id = getattr(self.qwen_vl_interface, "img_next_token_id", None)

        use_img_next_teacher = img_next_cfg.get("use_teacher", True)
        img_next_mask_for_action = (
            (qwen_inputs["input_ids"] == img_next_token_id) if (img_next_token_id is not None and qwen_inputs is not None) else None
        )

        if (
            enable_img_next
            and use_img_next_teacher
            and img_next_token_id is not None
            and img_next_loss_weight is not None
            and img_next_loss_weight > 0
        ):
            img_next_mask = (qwen_inputs["input_ids"] == img_next_token_id)
            try:
                img_next_loss = self._compute_img_next_loss(
                    last_hidden,
                    image_next,
                    img_next_mask,
                    image_next_fallback,
                    target_res=img_next_res,
                )
            except Exception as e:
                logger.warning(f"[img_next_loss] skipped due to error: {e}")
                img_next_loss = None
        
        if self.training_stage == "reasoning_only":
            # Stage 1: Only train VLM reasoning, skip action head
            if vlm_loss is None:
                raise ValueError(
                    "training_stage='reasoning_only' requires VLM loss, but vlm_loss is None. "
                    "Please ensure enable_latent_reasoning=True and labels are provided."
                )
            result["vlm_loss"] = vlm_loss
            if img_next_loss is not None:
                result["img_next_loss"] = img_next_loss
                result["total_loss"] = vlm_loss + img_next_loss_weight * img_next_loss
            else:
                result["total_loss"] = vlm_loss
            return result

        elif self.training_stage == "explicit_transition_cot":
            # Stage I: Explicit Transition-CoT text supervision.
            # Uses cot_text_transition as assistant labels; standard VLM forward
            # (NOT latent reasoning). Does NOT use action head or img_next.
            cot_texts = [example.get("cot_text_transition", "") for example in examples]
            # Build inputs with CoT labels (user portion masked, assistant = labels)
            cot_qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                solutions=cot_texts if any(cot_texts) else None,
                cot_mode=True,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cot_outputs = self.qwen_vl_interface(
                    **cot_qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            cot_loss = cot_outputs.loss
            if cot_loss is None:
                raise ValueError(
                    "training_stage='explicit_transition_cot' requires CoT loss, "
                    "but cot_loss is None. Check that cot_text_transition is provided "
                    "in the batch and cot_mode=True produces valid labels."
                )
            result["vlm_loss"] = cot_loss
            result["total_loss"] = cot_loss
            return result

        elif self.training_stage == "latent_transition":
            # Stage II: Mask-conditioned latent transition reasoning.
            # Frozen: Qwen-VL + action_model. Train: transition modules only.
            if not self.transition_enabled:
                raise ValueError("training_stage='latent_transition' requires "
                                 "mask_conditioned_transition.enable=true in config.")
            from laravla.model.modules.spatial_transition import transition_total_loss
            import torch.nn.functional as F

            # Extract current masks from batch
            cur_masks = torch.from_numpy(
                np.stack([ex["current_affordance_mask_agentview"] for ex in examples])
            ).unsqueeze(1).to(self.qwen_vl_interface.model.device).float()  # [B,1,H,W]

            future_masks = torch.from_numpy(
                np.stack([ex.get("future_affordance_mask_agentview",
                                 np.zeros((224,224), dtype=np.float32)) for ex in examples])
            ).to(self.qwen_vl_interface.model.device).float()

            goal_masks = torch.from_numpy(
                np.stack([ex.get("goal_affordance_mask_agentview",
                                 np.zeros((224,224), dtype=np.float32)) for ex in examples])
            ).to(self.qwen_vl_interface.model.device).float()

            rel_ids = torch.tensor(
                [ex.get("relation_label_id", -1) for ex in examples],
                dtype=torch.long, device=self.qwen_vl_interface.model.device
            )

            # Frozen VLM forward → hidden states (clean path, no warnings)
            with torch.no_grad():
                qwen_out = self.qwen_vl_interface.encode_observation(
                    images=batch_images, instructions=instructions,
                    output_hidden_states=True,
                )
                vlm_hidden = qwen_out.hidden_states[-1].float()  # [B, L, 2560]

            # Project VLM hidden to bottleneck
            vlm_proj = self.vlm_projector(vlm_hidden)  # [B, L, transition_dim]

            # Mask → mask_tokens (in bottleneck space)
            mask_tokens = self.mask_token_encoder(cur_masks)  # [B, K, transition_dim]

            # Transition module (bottleneck)
            transition_tokens = self.transition_module(vlm_proj, mask_tokens)

            # Decode
            future_logits = self.future_mask_decoder(transition_tokens)
            goal_logits = self.goal_mask_decoder(transition_tokens)
            rel_logits = self.relation_head(transition_tokens)

            # Resize GT masks to match decoder output
            R = future_logits.shape[-1]
            future_gt = F.interpolate(
                future_masks.unsqueeze(1), size=(R, R), mode='nearest'
            ).squeeze(1)
            goal_gt = F.interpolate(
                goal_masks.unsqueeze(1), size=(R, R), mode='nearest'
            ).squeeze(1)

            # Compute losses
            w = self.transition_loss_weights
            losses = transition_total_loss(
                future_logits=future_logits, future_target=future_gt,
                goal_logits=goal_logits, goal_target=goal_gt,
                relation_logits=rel_logits, relation_target=rel_ids,
                w_future=w.get("future_mask", 0.05),
                w_goal=w.get("goal_mask", 0.10),
                w_relation=w.get("relation", 0.05),
            )

            losses["transition_tokens"] = transition_tokens
            return losses

        elif self.training_stage == "transition_action":
            # Stage III: Gated transition → action generation.
            # Frozen: Qwen-VL. Train: transition modules + action adapter + action model.
            if not self.transition_enabled:
                raise ValueError("training_stage='transition_action' requires "
                                 "mask_conditioned_transition.enable=true.")
            from laravla.model.modules.spatial_transition import transition_total_loss
            import torch.nn.functional as F

            # ── Transition branch (same as latent_transition) ──
            cur_masks = torch.from_numpy(
                np.stack([ex["current_affordance_mask_agentview"] for ex in examples])
            ).unsqueeze(1).to(self.qwen_vl_interface.model.device).float()
            future_masks = torch.from_numpy(
                np.stack([ex.get("future_affordance_mask_agentview",
                                 np.zeros((224,224), dtype=np.float32)) for ex in examples])
            ).to(self.qwen_vl_interface.model.device).float()
            goal_masks = torch.from_numpy(
                np.stack([ex.get("goal_affordance_mask_agentview",
                                 np.zeros((224,224), dtype=np.float32)) for ex in examples])
            ).to(self.qwen_vl_interface.model.device).float()
            rel_ids = torch.tensor(
                [ex.get("relation_label_id", -1) for ex in examples],
                dtype=torch.long, device=self.qwen_vl_interface.model.device)

            with torch.no_grad():
                qwen_out = self.qwen_vl_interface.encode_observation(
                    images=batch_images, instructions=instructions,
                    output_hidden_states=True,
                )
                vlm_hidden = qwen_out.hidden_states[-1].float()

            vlm_proj = self.vlm_projector(vlm_hidden)
            mask_tokens = self.mask_token_encoder(cur_masks)
            transition_tokens = self.transition_module(vlm_proj, mask_tokens)

            # ── Transition losses ─────────────────────────────
            future_logits = self.future_mask_decoder(transition_tokens)
            goal_logits = self.goal_mask_decoder(transition_tokens)
            rel_logits = self.relation_head(transition_tokens)
            R = future_logits.shape[-1]
            future_gt = F.interpolate(future_masks.unsqueeze(1), size=(R,R), mode='nearest').squeeze(1)
            goal_gt = F.interpolate(goal_masks.unsqueeze(1), size=(R,R), mode='nearest').squeeze(1)
            w = self.transition_loss_weights
            trans_losses = transition_total_loss(
                future_logits=future_logits, future_target=future_gt,
                goal_logits=goal_logits, goal_target=goal_gt,
                relation_logits=rel_logits, relation_target=rel_ids,
                w_future=w.get("future_mask",0.05), w_goal=w.get("goal_mask",0.10),
                w_relation=w.get("relation",0.05))

            # ── Gated adapter: inject transition into action context ──
            conditioned_vl = self.transition_action_adapter(vlm_hidden, transition_tokens)

            # ── Action loss ──────────────────────────────────
            with torch.autocast("cuda", dtype=torch.float32):
                actions_t = torch.tensor(np.array(actions), device=conditioned_vl.device, dtype=conditioned_vl.dtype)
                actions_target = actions_t[:, -(self.future_action_window_size+1):, :]
                repeated_diffusion_steps = getattr(self.config.trainer, "repeated_diffusion_steps", 4) if self.config and hasattr(self.config, "trainer") else 4
                actions_target_rep = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                conditioned_rep = conditioned_vl.repeat(repeated_diffusion_steps, 1, 1)
                state_rep = None
                if state is not None:
                    st = torch.tensor(np.array(state), device=conditioned_vl.device, dtype=conditioned_vl.dtype)
                    if st.ndim == 2: st = st.unsqueeze(1)
                    state_rep = st.repeat(repeated_diffusion_steps, 1, 1)
                action_loss = self.action_model(conditioned_rep, actions_target_rep, state_rep)

            result = {
                "action_loss": action_loss,
                "future_mask_loss": trans_losses.get("future_mask_loss", torch.tensor(0.0)),
                "goal_mask_loss": trans_losses.get("goal_mask_loss", torch.tensor(0.0)),
                "relation_loss": trans_losses.get("relation_loss", torch.tensor(0.0)),
                "total_loss": action_loss + trans_losses["total_loss"],
                "transition_tokens": transition_tokens,
            }
            return result

        elif self.training_stage == "action_only":
            # action_only mode: Only train action head, VLM is frozen
            with torch.autocast("cuda", dtype=torch.float32):
                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                
                state_repeated = None
                if state is not None:
                    state = torch.tensor(
                        np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                    )  # [B, state_dim] or [B, 1, state_dim]
                    
                    # Ensure state is 3D: [B, 1, state_dim]
                    if state.ndim == 2:
                        state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
                    
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)  # [B*repeated_diffusion_steps, 1, state_dim]

                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                )

                result["action_loss"] = action_loss
                result["total_loss"] = action_loss  # Only action loss
                if vlm_loss is not None:
                    result["vlm_loss"] = vlm_loss
                return result
        else:
            # full mode: Train both VLM and action head
            with torch.autocast("cuda", dtype=torch.float32):
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                
                state_repeated = None
                if state is not None:
                    state = torch.tensor(
                        np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                    )  # [B, state_dim] or [B, 1, state_dim]
                    
                    if state.ndim == 2:
                        state = state.unsqueeze(1)
                    
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                )

            result["action_loss"] = action_loss
            
            # Combine with VLM loss if available
        if vlm_loss is not None:
            vlm_loss_weight = self.config.framework.get("latent_reasoning", {}).get("vlm_loss_weight", 0.5)
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = action_loss + vlm_loss_weight * vlm_loss
        else:
            result["total_loss"] = action_loss

        if (
            img_next_loss is not None
            and enable_img_next
            and use_img_next_teacher
            and img_next_loss_weight > 0
        ):
            result["img_next_loss"] = img_next_loss
            result["total_loss"] = result["total_loss"] + img_next_loss_weight * img_next_loss

        return result

    def _compute_img_next_loss(
        self,
        last_hidden: torch.Tensor,
        image_next: List,
        img_next_mask: torch.Tensor,
        fallback_mask: torch.Tensor,
        target_res: int = 112,
    ) -> Optional[torch.Tensor]:
        """
        Compute L1 loss between img_next token hidden states and visual encoder features of next frame.
        """
        if last_hidden is None or image_next is None or len(image_next) == 0:
            return None

        # shape check for mask
        if img_next_mask is None or not torch.any(img_next_mask):
            return None

        device = last_hidden.device
        dtype = last_hidden.dtype

        # Extract predicted embeddings at img_next positions
        try:
            # mask shape [B, L]; expect count per sample = img_next_count (16)
            B = last_hidden.shape[0]
            img_next_count = img_next_mask.sum(dim=1).max().item()
            pred = last_hidden[img_next_mask].view(B, img_next_count, -1)
        except Exception as e:
            logger.warning(f"[img_next_loss] mask reshape failed: {e}")
            return None

        try:
            # 获取 processor
            proc = getattr(self.qwen_vl_interface, "processor", None)
            if proc is None and hasattr(self.qwen_vl_interface, "model"):
                proc = getattr(self.qwen_vl_interface.model, "processor", None)
            
            if proc is None:
                logger.warning("[img_next_loss] processor is None, skip img_next_loss")
                return None
            
            # Use only the primary (first) view for img_next loss to match the single-view Bridge setup,
            # while remaining compatible with single-view datasets (non-list entries).
            flat_images = []
            for sample_imgs in image_next:
                if isinstance(sample_imgs, list):
                    flat_images.append(sample_imgs[0] if len(sample_imgs) > 0 else None)
                else:
                    flat_images.append(sample_imgs)
            
            if len(flat_images) == 0:
                logger.warning("[img_next_loss] no images to process")
                return None

            # Resize next-frame images before processor to ensure `res` takes effect.
            if target_res is not None and int(target_res) > 0:
                resized = []
                for img in flat_images:
                    try:
                        resized.append(img.resize((int(target_res), int(target_res))))
                    except Exception:
                        resized.append(img)
                flat_images = resized
            
           
            img_processor = getattr(proc, "image_processor", None)
            if img_processor is None:
                logger.warning("[img_next_loss] processor.image_processor is None, skip img_next_loss")
                return None
            with torch.no_grad():
                proc_out = img_processor(images=flat_images, return_tensors="pt")
                proc_out = dict(proc_out)
                pixel_values = proc_out.get("pixel_values", None)
                image_grid_thw = proc_out.get("image_grid_thw", None)
                if pixel_values is None:
                    logger.warning("[img_next_loss] processor returned None pixel_values")
                    return None
                pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)

                main_model = getattr(self.qwen_vl_interface, "model", None)
                if main_model is None:
                    logger.warning("[img_next_loss] main_model.get_image_features not available")
                    return None
                
                # Prefer EMA teacher vision encoder when available.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if hasattr(self.qwen_vl_interface, "get_image_features_target"):
                        img_embeds, _ = self.qwen_vl_interface.get_image_features_target(
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                    else:
                        if not hasattr(main_model, "get_image_features"):
                            logger.warning("[img_next_loss] main_model.get_image_features not available")
                            return None
                        img_embeds, _ = main_model.get_image_features(
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                
              
                if isinstance(img_embeds, (list, tuple)):
                    feats = torch.stack([emb for emb in img_embeds], dim=0).to(device, dtype)
                else:
                    feats = img_embeds.to(device, dtype)
                
                if feats is None or feats.numel() == 0:
                    logger.warning("[img_next_loss] extracted features are empty")
                    return None
                
                if feats.dim() == 2:
                    feats = feats.unsqueeze(0)

                grid_side = int(feats.shape[1] ** 0.5)
                target_side = int(img_next_count ** 0.5)
                if grid_side * grid_side != feats.shape[1] or target_side * target_side != img_next_count:
                    logger.warning(f"[img_next_loss] unexpected token grid: tokens={feats.shape[1]}, target={img_next_count}")
                    return None

                feats_2d = feats.transpose(1, 2).reshape(feats.shape[0], feats.shape[2], grid_side, grid_side)
                feats_2d = F.adaptive_avg_pool2d(feats_2d, output_size=(target_side, target_side))
                target_feats = feats_2d.flatten(2).transpose(1, 2)  # [B, target_tokens, C]
        except Exception as e:
            logger.warning(f"[img_next_loss] visual encoding failed: {e}")
            return None

        valid_mask = (~fallback_mask).float().view(-1, 1, 1)
        if valid_mask.sum() <= 0:
            return None

        l1 = torch.nn.functional.l1_loss(pred, target_feats, reduction="none")  # [B, tokens, C]
        mask_full = valid_mask.expand_as(l1)
        l1 = (l1 * mask_full).sum() / mask_full.sum()
        return l1

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Inference: predict future actions via latent reasoning + diffusion sampling.

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL
             - forward_latent for implicit reasoning (iterative KV-Cache)
             - Fallback to normal forward if forward_latent unavailable
          3. Action model prediction from hidden states

        Returns:
            dict with normalized_actions (np.ndarray [B, T, action_dim]).
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        use_iterative_forward = hasattr(self.qwen_vl_interface, 'forward_latent')

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # Step 2: Forward pass
        if use_iterative_forward:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vlm_outputs = self.qwen_vl_interface.forward_latent(
                    input_ids=qwen_inputs["input_ids"],
                    attention_mask=qwen_inputs["attention_mask"],
                    pixel_values=qwen_inputs.get("pixel_values"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                )
                # forward_latent returns a dict with 'hidden_states', 'num_reasoning_passes', etc.
                last_hidden = vlm_outputs['hidden_states']  # [B, L, H]
                
                # Optional: Log reasoning passes for debugging
                num_passes = vlm_outputs.get('num_reasoning_passes', 0)
                if num_passes > 0:
                    logger.info(f" Completed {num_passes} reasoning passes in predict_action")
        else:
            # Baseline mode: Normal forward pass (no iterative reasoning)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "thinking_gen_time": 0.0}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./laravla/config/training/bridge.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    # Smoke test with fake data
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")

    predict_output = model.predict_action(
        batch_images=[batch[0]["image"]],
        instructions=[batch[0]["lang"]],
        state=[batch[0]["state"]],
    )
    print(f"Predicted Action: {predict_output['normalized_actions']}")
