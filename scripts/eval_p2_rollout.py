#!/usr/bin/env python
"""
P2 LIBERO Rollout — RGB-only inference (no mask).
=================================================
Uses P2-trained action model. Same interface as LaRA-VLA: RGB + instruction → action.

Usage:
    python scripts/eval_p2_rollout.py --suite libero_goal --task-id 0 --episodes 10
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import imageio
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-ckpt", type=str, default=str(_REPO / "results/P2_formal/best_model.pt"))
    parser.add_argument("--suite", type=str, default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P2_eval"))
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = (_REPO / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"P2 LIBERO Rollout (RGB-only): {args.suite} task_{args.task_id}")
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
    libero_home = os.environ.get("LIBERO_HOME", "")
    if libero_home and libero_home not in sys.path:
        sys.path.insert(0, libero_home)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()
    task_suite = bm[args.suite]()
    task = task_suite.get_task(args.task_id)
    bddl_file = f"{task.problem_folder}/{task.bddl_file}"
    task_name = task.name
    task_description = task.language

    bddl_dir = get_libero_path("bddl_files")
    os.chdir(bddl_dir)

    print(f"  Task: {task_name}")
    print(f"  Instruction: {task_description}")

    viz_dir = output_dir / f"viz_{args.suite}_task{args.task_id:02d}"
    if not args.no_debug:
        viz_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    results = []

    for ep in range(args.episodes):
        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=224, camera_widths=224)
        env.seed(args.seed + ep)
        obs = env.reset()

        done = False
        step_count = 0
        ep_success = False
        ep_frames = []

        t0 = time.time()
        while not done and step_count < args.max_steps:
            agentview_pil = Image.fromarray(obs["agentview_image"][::-1, :, :])
            if not args.no_debug:
                ep_frames.append(np.array(agentview_pil))

            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
            state = np.concatenate([eef_pos, eef_quat])

            with torch.no_grad():
                pred = vla.predict_action(
                    batch_images=[[agentview_pil]],
                    instructions=[task_description],
                    state=np.array([state]),
                )
            action = pred["normalized_actions"][0]

            # Execute action chunk
            for i in range(len(action)):
                obs, reward, done, info = env.step(action[i])
                step_count += 1
                if done:
                    ep_success = True
                    break

        # Save GIF
        if ep_frames and not args.no_debug:
            gif_path = viz_dir / f"ep{ep:02d}_rollout.gif"
            imageio.mimsave(str(gif_path), ep_frames, fps=10, loop=0)

        elapsed = time.time() - t0
        success_count += ep_success
        results.append({"episode": ep, "success": ep_success, "steps": step_count, "time": elapsed})
        print(f"  Episode {ep+1}/{args.episodes}: {'✅' if ep_success else '❌'} "
              f"({step_count} steps, {elapsed:.0f}s)")

        env.close()

    success_rate = success_count / args.episodes if args.episodes > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULT: {args.suite} task_{args.task_id}")
    print(f"  Success: {success_count}/{args.episodes} = {success_rate:.1%}")
    print(f"{'='*60}")

    with open(output_dir / f"{args.suite}_task{args.task_id:02d}.json", "w") as f:
        json.dump({"suite": args.suite, "task_id": args.task_id,
                   "success_rate": success_rate, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
