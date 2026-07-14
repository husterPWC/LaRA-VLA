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


def get_held_object(env, objects_of_interest):
    """Determine which object the robot is holding from gripper state + positions."""
    try:
        obs = env._get_observations()
        gripper = obs.get("robot0_gripper_qpos", np.array([0]))
        # gripper_qpos[0] < 0 → closed (holding); > 0 → open (reaching)
        is_closed = float(gripper[0]) < 0
        if not is_closed or not objects_of_interest:
            return None

        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        best_obj = None
        best_dist = float("inf")
        for obj_name in objects_of_interest:
            pos_key = f"{obj_name}_pos"
            if pos_key in obs:
                dist = np.linalg.norm(obs[pos_key] - eef_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj_name
        return best_obj
    except Exception:
        return None


def get_gt_mask_from_env(env, objects_of_interest, suite="", debug=False):
    """Extract binary mask of objects_of_interest from environment instance segmentation.
    Supports dynamic filtering for libero_10 based on gripper state.
    """
    """Extract binary mask of objects_of_interest from environment instance segmentation.

    MuJoCo/robosuite segmentation is (H,W,2): ch0=object type, ch1=instance ID.
    We filter ch0==GEOM_TYPE (5) and match ch1 against geom IDs of target objects.
    """
    if not objects_of_interest:
        return np.zeros((224, 224), dtype=np.float32)
    try:
        seg = env.sim.render(camera_name="agentview", height=224, width=224,
                             mode="offscreen", segmentation=True)
        model = env.sim.model
        GEOM_TYPE = 5  # mjOBJ_GEOM

        # seg[...,0] = type, seg[...,1] = instance ID
        if seg.ndim == 3 and seg.shape[2] >= 2:
            seg_type = seg[:, :, 0]
            seg_id = seg[:, :, 1]
        else:
            seg_type = np.zeros_like(seg)
            seg_id = seg

        if debug:
            print(f"  [MASK] seg type unique: {sorted(np.unique(seg_type))}")
            print(f"  [MASK] seg id unique count: {len(np.unique(seg_id))}")
            # Print ALL geom names matching objects or containing keywords
            obj_keywords = set()
            for obj in objects_of_interest:
                for part in obj.lower().replace("_", " ").split():
                    if len(part) >= 3:
                        obj_keywords.add(part)
            for gid in range(model.ngeom):
                gname = (model.geom(gid).name or "").lower()
                bname = (model.body(model.geom_bodyid[gid]).name or "").lower()
                if any(kw in gname or kw in bname for kw in obj_keywords):
                    sid = gid + 1
                    print(f"  [MASK]   geom[{gid}] seg_id={sid} '{model.geom(gid).name}' body='{model.body(model.geom_bodyid[gid]).name}'")

        # Find target geom IDs by matching object names against geom/body names.
        # Object names like "wooden_cabinet_1_middle_region" are split into:
        #   base = "wooden_cabinet_1", region = "middle"
        # Match base against geom/body, and region against body suffix.
        target_ids = set()
        for obj_name in objects_of_interest:
            parts = obj_name.lower().split("_")
            # Find the numeric suffix (e.g., "_1") to split base from region
            base_end = 0
            for i, p in enumerate(parts):
                if p.isdigit():
                    base_end = i + 1
                    break
            base = "_".join(parts[:base_end])  # e.g., "wooden_cabinet_1"
            region = "_".join(parts[base_end:])  # e.g., "middle_region" or ""
            base_norm = base.replace("_", "")
            region_norm = region.replace("_", "")

            for gid in range(model.ngeom):
                gname = (model.geom(gid).name or "").lower().replace("_", "")
                bname = (model.body(model.geom_bodyid[gid]).name or "").lower().replace("_", "")
                matched = False
                if gname and base_norm in gname:
                    matched = True
                elif bname and base_norm in bname:
                    matched = True
                if matched and region_norm:
                    # Region filter: only match geoms whose body contains region keyword
                    region_parts = [p for p in parts[base_end:] if p not in ("region", "")]
                    if region_parts:
                        matched = any(rp in bname for rp in region_parts)
                if matched:
                    target_ids.add(gid + 1)

        if debug:
            print(f"  [MASK] objects: {objects_of_interest}")
            print(f"  [MASK] target seg inst ids: {sorted(target_ids)}")

        # Dynamic filtering for libero_10
        if suite == "libero_10":
            held = get_held_object(env, objects_of_interest)
            if held is not None:
                # Filter to only show held object
                held_ids = set()
                held_lower = held.lower().replace("_", "")
                for gid in range(model.ngeom):
                    gname = (model.geom(gid).name or "").lower().replace("_", "")
                    bname = (model.body(model.geom_bodyid[gid]).name or "").lower().replace("_", "")
                    if (gname and held_lower in gname) or (bname and held_lower in bname):
                        held_ids.add(gid + 1)
                if held_ids:
                    target_ids = held_ids
                    if debug:
                        print(f"  [MASK] dynamic: held={held}, ids={sorted(target_ids)}")

        # Build mask: geom type + target instance IDs
        is_geom = (seg_type == GEOM_TYPE)
        is_target = np.isin(seg_id, list(target_ids))
        mask = (is_geom & is_target).astype(np.float32)

        if debug:
            print(f"  [MASK] mask px: {int(mask.sum())} (geom px: {is_geom.sum()}, target px: {is_target.sum()})")

        # Flip like LIBERO observation: agentview[::-1, :, :]
        mask = mask[::-1, :].copy()
        return mask
    except Exception as e:
        import traceback
        traceback.print_exc()
        return np.zeros((224, 224), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-ckpt", type=str, default=str(_REPO / "results/P2_formal/best_model.pt"))
    parser.add_argument("--suite", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P2_eval"))
    parser.add_argument("--index-path", type=str,
                        default=str(_REPO / "output" / "spatial_lara_libero_no_noops" /
                                    "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl"))
    parser.add_argument("--no-debug", action="store_true", help="Skip saving debug images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"P2 LIBERO Rollout: {args.suite} task_{args.task_id}")
    print(f"  Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print("=" * 60)

    # ── Load P2 model ────────────────────────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, norm_stats_all = read_mode_config(Path(CKPT))
    # Extract action norm stats from first dataset
    first_ds = next(iter(norm_stats_all.values()))
    norm_stats = first_ds["action"]  # {q01, q99, mask}
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
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_id = args.task_id
    task = task_suite.get_task(task_id)

    task_name = task.name
    task_description = task.language
    # BDDL path: problem_folder/bddl_file relative to bddl_files dir
    task_bddl_file = f"{task.problem_folder}/{task.bddl_file}" if hasattr(task, 'bddl_file') else None

    # Change to bddl_files dir (env checks os.path.exists on bare bddl_file_name)
    bddl_dir = get_libero_path("bddl_files")
    os.chdir(bddl_dir)

    print(f"  Task: {task_name}")
    print(f"  Instruction: {task_description}")
    print(f"  BDDL: {task_bddl_file}")
    print(f"  Working dir: {bddl_dir}")

    # Get objects_of_interest from training index (most reliable)
    objects_of_interest = []
    index_path = Path(args.index_path)
    if Path(index_path).exists():
        with open(index_path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("suite") == args.suite and e.get("task_id") == task_id:
                    if "objects_of_interest" in e:
                        objects_of_interest = e["objects_of_interest"]
                        break
    if not objects_of_interest:
        raise RuntimeError(
            f"No objects_of_interest found for {args.suite} task_{task_id}. "
            f"Check index: {index_path}"
        )

    success_count = 0
    results = []

    # Make debug dir
    viz_dir = Path(os.path.abspath(args.output_dir)) / f"viz_{args.suite}_task{task_id:02d}"
    if not args.no_debug:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        env_args = {
            "bddl_file_name": task_bddl_file,
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
        ep_frames = []  # Collect all frames for GIF

        t0 = time.time()
        first_step = True
        while not done and step_count < args.max_steps:
            # Get observation
            agentview = obs["agentview_image"]
            wrist = obs["robot0_eye_in_hand_image"]
            agentview_pil = Image.fromarray(agentview[::-1, :, :])  # LIBERO flips

            # Get GT mask from environment
            current_mask = get_gt_mask_from_env(env, objects_of_interest, suite=args.suite, debug=first_step)
            if first_step:
                print(f"  [DEBUG] objects_of_interest: {objects_of_interest}")
                first_step = False

            # Save frame for GIF
            if not args.no_debug:
                overlay_rgb = np.array(agentview_pil).astype(np.float32) * 0.5
                overlay_rgb[:, :, 0] += current_mask * 128
                overlay_rgb = np.clip(overlay_rgb, 0, 255).astype(np.uint8)
                ep_frames.append(np.hstack([np.array(agentview_pil), overlay_rgb]))

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
            action = pred["normalized_actions"][0]  # [8, 7] — normalized [-1,1]

            # Actions from predict_action are already in LIBERO-compatible range
            if step_count < 3:
                print(f"  [ACT] step={step_count} action[0]={action[0][:3]}... min={action.min():.3f} max={action.max():.3f}")

            # Execute action chunk (up to 8 steps, stop early if done)
            for i in range(len(action)):
                obs, reward, done, info = env.step(action[i])
                step_count += 1
                if done:
                    ep_success = True
                    break

        # Save rollout GIF
        if ep_frames and not args.no_debug:
            gif_path = viz_dir / f"ep{ep:02d}_rollout.gif"
            imageio.mimsave(str(gif_path), ep_frames, fps=10, loop=0)
            print(f"  📹 GIF saved: {gif_path}")

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
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [PATH] output_dir={output_dir}")
    with open(output_dir / f"{args.suite}_task{task_id:02d}.json", "w") as f:
        json.dump({"suite": args.suite, "task_id": task_id, "success_rate": success_rate,
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
