#!/usr/bin/env python
"""
P2-New LIBERO Rollout — RGB-only inference (no mask).
=======================================================
Mask-supervised, mask-free-inference. Same interface as LaRA-VLA:
    RGB + instruction → action.

Loads P2-New (no-mask) checkpoint. No external segmentation/GT masks needed.

Usage:
    python scripts/eval_p2_nomask_rollout.py --suite libero_goal --task-id 0 --episodes 10
    python scripts/eval_p2_nomask_rollout.py --suite libero_10 --task-id 0 --episodes 20
    python scripts/eval_p2_nomask_rollout.py --suite all --episodes 10
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

# All LIBERO suites
ALL_SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]


def run_one_episode(env, vla, task_description, max_steps, no_debug, ep_frames):
    """Run a single episode. Returns (success, step_count, ep_frames)."""
    obs = env.reset()
    done = False
    step_count = 0
    ep_success = False

    while not done and step_count < max_steps:
        agentview_pil = Image.fromarray(obs["agentview_image"][::-1, :, :])
        if not no_debug:
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

    return ep_success, step_count, ep_frames


def evaluate_suite(args, vla, suite, task_ids=None):
    """Evaluate all tasks in a suite."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()
    task_suite = bm[suite]()

    if task_ids is None:
        task_ids = list(range(len(task_suite.tasks)))

    bddl_dir = get_libero_path("bddl_files")
    os.chdir(bddl_dir)

    results = {}
    for tid in task_ids:
        task = task_suite.get_task(tid)
        bddl_file = f"{task.problem_folder}/{task.bddl_file}"
        task_description = task.language

        output_dir = (_REPO / args.output_dir).resolve()
        viz_dir = output_dir / f"viz_{suite}_task{tid:02d}"
        if not args.no_debug:
            viz_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        ep_results = []

        for ep in range(args.episodes):
            env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=224, camera_widths=224)
            env.seed(args.seed + ep)

            ep_frames = []
            t0 = time.time()
            ep_success, step_count, ep_frames = run_one_episode(
                env, vla, task_description, args.max_steps, args.no_debug, ep_frames)
            elapsed = time.time() - t0

            # Save GIF
            if ep_frames and not args.no_debug:
                gif_path = viz_dir / f"ep{ep:02d}_rollout.gif"
                imageio.mimsave(str(gif_path), ep_frames, fps=10, loop=0)

            success_count += int(ep_success)
            ep_results.append({"episode": ep, "success": ep_success,
                               "steps": step_count, "time": elapsed})
            print(f"  [{suite} t{tid:02d}] Ep {ep+1}/{args.episodes}: "
                  f"{'✅' if ep_success else '❌'} ({step_count} steps, {elapsed:.0f}s)")

            env.close()

        sr = success_count / args.episodes if args.episodes > 0 else 0
        results[f"{suite}_task{tid:02d}"] = {"success_rate": sr, "results": ep_results}
        print(f"  [{suite} t{tid:02d}] SR: {success_count}/{args.episodes} = {sr:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-ckpt", type=str, default=str(_REPO / "results/P2_nomask/best_model.pt"),
                        help="Path to P2-New (no-mask) checkpoint")
    parser.add_argument("--suite", type=str, default="libero_goal",
                        help="Suite name or 'all' for all four suites")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P2_nomask_eval"))
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = (_REPO / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"P2-New LIBERO Rollout (RGB-only, no-mask)")
    print(f"  P2 checkpoint: {args.p2_ckpt}")
    print(f"  Suite: {args.suite}  Task: {args.task_id}")
    print(f"  Episodes: {args.episodes}  Max steps: {args.max_steps}")
    print("=" * 60)

    # ── Load P2-New model ────────────────────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05,
                         "current_mask": 0.05},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    # Load P2-New weights
    if not Path(args.p2_ckpt).exists():
        print(f"WARNING: P2 checkpoint not found: {args.p2_ckpt}")
        print("  Using base LaRA-VLA weights only. Results will not reflect P2 training.")
    else:
        p2_state = torch.load(args.p2_ckpt, map_location="cpu")
        if "model_state_dict" in p2_state:
            p2_state = p2_state["model_state_dict"]
        vla.load_state_dict(p2_state, strict=False)
        print("P2-New weights loaded.")

    vla = vla.to("cuda")
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)
    print("Model ready. Inference: RGB + instruction → action (no mask).")

    # ── Set up LIBERO ────────────────────────────────────────
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    libero_home = os.environ.get("LIBERO_HOME", "")
    if libero_home and libero_home not in sys.path:
        sys.path.insert(0, libero_home)

    # ── Evaluate ─────────────────────────────────────────────
    all_results = {}
    if args.suite == "all":
        for suite in ALL_SUITES:
            print(f"\n{'='*60}")
            print(f"Suite: {suite}")
            print(f"{'='*60}")
            suite_results = evaluate_suite(args, vla, suite)
            all_results.update(suite_results)
    else:
        suite_results = evaluate_suite(args, vla, args.suite, task_ids=[args.task_id])
        all_results.update(suite_results)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    srs = []
    for name, r in sorted(all_results.items()):
        sr = r["success_rate"]
        srs.append(sr)
        print(f"  {name}: {sr:.1%}")
    if srs:
        print(f"  Average: {np.mean(srs):.1%}")
    print(f"{'='*60}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results,
                   "average_sr": float(np.mean(srs)) if srs else 0}, f, indent=2)
    print(f"Results saved to {output_dir}/summary.json")


if __name__ == "__main__":
    main()
