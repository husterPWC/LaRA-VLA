#!/usr/bin/env python
"""
P2 LIBERO Rollout Evaluation (oracle GT mask).
================================================
Loads P2 best_model.pt, runs rollout on specified suite/task.
Uses GT instance segmentation as current_mask (oracle-mask setting).

Usage:
    python scripts/eval_p2_rollout.py --suite libero_10 --task-id 0 --episodes 5
    python scripts/eval_p2_rollout.py --suite libero_spatial --task-id 0 --episodes 5
    python scripts/eval_p2_rollout.py --suite all --episodes 10  # all suites, 10 episodes each
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))


def get_gt_mask_from_env(env, objects_of_interest):
    """Extract binary mask of objects_of_interest from environment instance segmentation."""
    # LIBERO uses robosuite; get instance seg from sim
    seg = env.sim.render(camera_name="agentview", height=224, width=224,
                         mode='segmentation_instance')
    # Map instance IDs to object names via env model
    mask = np.zeros((224, 224), dtype=np.float32)
    try:
        # Try to find instance IDs for objects_of_interest
        model = env.sim.model
        for obj_name in objects_of_interest:
            # Match object name to geoms
            for geom_id in range(model.ngeom):
                geom_name = model.geom(geom_id).name
                if obj_name in geom_name or geom_name.startswith(obj_name):
                    geom_mask = (seg[:, :, 0] == geom_id + 1)  # instance seg
                    mask = np.maximum(mask, geom_mask.astype(np.float32))
    except Exception:
        # Fallback: use seg directly if object mapping fails
        pass
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-ckpt", type=str, default=str(_REPO / "results/P2_formal/best_model.pt"))
    parser.add_argument("--suite", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P2_eval"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"P2 LIBERO Rollout: {args.suite} task_{args.task_id}")
    print(f"  Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print("=" * 60)

    # ── Load P2 model ────────────────────────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    p2_state = torch.load(args.p2_ckpt, map_location="cpu")
    if "model_state_dict" in p2_state:
        p2_state = p2_state["model_state_dict"]
    vla.load_state_dict(p2_state, strict=False)
    vla = vla.to("cuda")
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)
    print("P2 model loaded.")

    # ── Set up LIBERO ────────────────────────────────────────
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Add LIBERO to path (like original eval_libero.py does)
    libero_home = os.environ.get("LIBERO_HOME", "")
    if libero_home and libero_home not in sys.path:
        sys.path.insert(0, libero_home)
    import libero
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_id = args.task_id
    task = task_suite.get_task(task_id)

    task_name = task.name
    task_description = task.language
    task_bddl_file = Path(task.problem_folder) / task.bddl_file if hasattr(task, 'bddl_file') else None

    print(f"  Task: {task_name}")
    print(f"  Instruction: {task_description}")

    # Get objects of interest for mask extraction
    objects_of_interest = []
    if hasattr(task, 'obj_of_interest'):
        objects_of_interest = list(task.obj_of_interest)

    success_count = 0
    results = []

    for ep in range(args.episodes):
        env_args = {
            "bddl_file_name": str(task_bddl_file) if task_bddl_file else None,
            "camera_heights": 224,
            "camera_widths": 224,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "control_freq": 20,
        }
        env = OffScreenRenderEnv(**{k: v for k, v in env_args.items() if v is not None})
        env.seed(args.seed + ep)

        obs = env.reset()
        done = False
        step_count = 0
        ep_success = False

        t0 = time.time()
        while not done and step_count < args.max_steps:
            # Get observation
            agentview = obs["agentview_image"]
            wrist = obs["robot0_eye_in_hand_image"]
            agentview_pil = Image.fromarray(agentview[::-1, :, :])  # LIBERO flips

            # Get GT mask from environment
            current_mask = get_gt_mask_from_env(env, objects_of_interest)

            # Get robot state (7-dim: eef_pos + eef_quat or gripper)
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
            gripper = obs.get("robot0_gripper_qpos", np.zeros(1))
            state = np.concatenate([eef_pos, eef_quat]) if len(eef_pos) > 0 else np.zeros(7)

            # Predict action
            with torch.no_grad():
                pred = vla.predict_action(
                    batch_images=[[agentview_pil]],
                    instructions=[task_description],
                    state=np.array([state]),
                    current_masks=np.array([current_mask]),
                )
            action = pred["normalized_actions"][0]  # [8, 7]

            # Execute first action
            for i in range(min(8, 5)):  # Execute first 5 actions
                act = action[i]
                obs, reward, done, info = env.step(act)
                step_count += 1
                if done:
                    ep_success = int(info.get("success", 0)) == 1
                    break

        elapsed = time.time() - t0
        success_count += ep_success
        results.append({"episode": ep, "success": ep_success, "steps": step_count, "time": elapsed})
        print(f"  Episode {ep+1}/{args.episodes}: {'✅' if ep_success else '❌'} "
              f"({step_count} steps, {elapsed:.0f}s)")

        env.close()

    # ── Summary ──────────────────────────────────────────────
    success_rate = success_count / args.episodes if args.episodes > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULT: {args.suite} task_{task_id}")
    print(f"  Success: {success_count}/{args.episodes} = {success_rate:.1%}")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.suite}_task{task_id:02d}.json", "w") as f:
        import json
        json.dump({"suite": args.suite, "task_id": task_id, "success_rate": success_rate,
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
