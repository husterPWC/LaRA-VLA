#!/usr/bin/env python
"""
LIBERO Environment Inspection Tool
==================================
Purpose: Inspect the structure of each LIBERO task for building Spatial-LaRA dataset.

For each task, this script:
  1. Creates the LIBERO environment
  2. Lists all camera names
  3. Lists all object bodies (non-robot, non-fixture) with 6D poses
  4. Lists all geom names grouped by object
  5. Shows obj_of_interest and their segmentation IDs
  6. Saves RGB + segmentation visualization
  7. Shows observation structure

Usage:
    python tools/inspect_libero_env.py \
        --suite libero_spatial \
        --task-id 0 \
        --camera-names agentview robot0_eye_in_hand

    python tools/inspect_libero_env.py \
        --suite all \
        --task-id 0

Output:
    A detailed text report to stdout, plus optional image files.

Author: Spatial-LaRA project
"""

import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np

# ── Path setup ──────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_LARA_REPRO = _REPO_ROOT.parent  # lara_repro/

# Set LIBERO paths
_LIBERO_HOME = os.environ.get("LIBERO_HOME", str(_LARA_REPRO / "LIBERO"))
os.environ.setdefault("LIBERO_HOME", _LIBERO_HOME)
os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(_LIBERO_HOME) / "libero"))

if _LIBERO_HOME not in sys.path:
    sys.path.insert(0, _LIBERO_HOME)


# ── Imports (after path setup) ──────────────────────────────────
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv

# Silence robosuite warnings
import warnings
warnings.filterwarnings("ignore")


# ── Constants ───────────────────────────────────────────────────
FIXTURE_PREFIXES = [
    "world", "table", "robot", "gripper", "mount",
    "wall", "floor", "counter", "sink", "cabinet",
]

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


# ── Helpers ─────────────────────────────────────────────────────
def is_object_body(name: str) -> bool:
    """Check if a body name represents a manipulable object."""
    return not any(name.startswith(p) for p in FIXTURE_PREFIXES)


def is_object_body_strict(name: str, obj_prefixes: set) -> bool:
    """Check if body belongs to known object instances (using obj_of_interest)."""
    for prefix in obj_prefixes:
        if name.startswith(prefix):
            return True
    return False


def print_separator(title: str, char: str = "=") -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── Main inspection function ────────────────────────────────────
def inspect_task(
    suite_name: str,
    task_id: int,
    camera_names: list[str],
    resolution: int = 128,
    save_images: bool = False,
    output_dir: str = "",
) -> dict:
    """
    Inspect a single LIBERO task.

    Returns:
        dict with task metadata (objects, cameras, poses, etc.)
    """
    # ── Load task ───────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown suite: {suite_name}. Options: {list(benchmark_dict.keys())}")

    task_suite = benchmark_dict[suite_name]()
    if task_id >= task_suite.n_tasks:
        raise ValueError(f"Task ID {task_id} out of range (0-{task_suite.n_tasks - 1})")

    task = task_suite.get_task(task_id)

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = task_suite.get_task_init_states(task_id)

    print_separator(f"Task [{task_id}]: {task.name}")
    print(f"  Language: {task.language}")
    print(f"  Problem:  {task.problem_folder}")
    print(f"  BDDL:     {task.bddl_file}")

    # ── Create env ──────────────────────────────────────────────
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "camera_segmentations": "element",
    }
    env = SegmentationRenderEnv(**env_args)

    # Use first init state
    if len(init_states) > 0:
        obs = env.reset()
        obs = env.set_init_state(init_states[0])
    else:
        obs = env.reset()

    sim = env.sim
    model = sim.model
    data = sim.data

    # ── 1. Camera names ────────────────────────────────────────
    print_separator("1. Camera Names")
    for i, name in enumerate(model.camera_names):
        marker = " ← used" if name in camera_names else ""
        print(f"  [{i}] {name}{marker}")

    # ── 2. Object bodies & 6D poses ────────────────────────────
    print_separator("2. Object Bodies & 6D Poses")
    obj_bodies = []
    for i, name in enumerate(model.body_names):
        if is_object_body(name):
            pos = data.body_xpos[i].copy()
            quat = data.body_xquat[i].copy()
            geom_adr = model.body_geomadr[i]
            num_geoms = model.body_geomnum[i]
            geom_names = []
            if geom_adr >= 0:
                for g in range(num_geoms):
                    gid = geom_adr + g
                    if gid < len(model.geom_names):
                        geom_names.append(model.geom_names[gid])

            obj_bodies.append({
                "body_name": name,
                "body_id": i,
                "pos": pos.tolist(),
                "quat": quat.tolist(),
                "geom_names": geom_names,
            })
            print(f"  [{i:3d}] {name}")
            print(f"         pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"         quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
            if geom_names:
                print(f"         geoms: {geom_names}")

    # ── 3. All geom names (grouped) ─────────────────────────────
    print_separator("3. All Geom Names (first 40)")
    for i, name in enumerate(model.geom_names[:40]):
        body_id = model.geom_bodyid[i]
        body_name = model.body_names[body_id] if body_id < len(model.body_names) else "??"
        obj_tag = " [OBJ]" if is_object_body(body_name) else ""
        print(f"  [{i:3d}] {name:45s} → body[{body_id}]={body_name}{obj_tag}")

    if len(model.geom_names) > 40:
        print(f"  ... ({len(model.geom_names) - 40} more geoms)")

    # ── 4. obj_of_interest ──────────────────────────────────────
    print_separator("4. Objects of Interest")
    print(f"  {env.obj_of_interest}")

    # ── 5. Segmentation mapping ─────────────────────────────────
    print_separator("5. Segmentation ID Mapping (instance_to_id)")
    # Sort by seg_id, handling None gracefully
    valid_items = [(k, v) for k, v in env.instance_to_id.items() if v is not None]
    invalid_items = [(k, v) for k, v in env.instance_to_id.items() if v is None]
    for name, seg_id in sorted(valid_items, key=lambda x: x[1]):
        is_interest = " ★" if name in env.obj_of_interest else ""
        print(f"  seg_id={seg_id:3d}: {name}{is_interest}")
    for name, seg_id in invalid_items:
        print(f"  seg_id=None: {name}")

    # ── 6. Observation structure ────────────────────────────────
    print_separator("6. Observation Structure")
    for k, v in sorted(obs.items()):
        if isinstance(v, np.ndarray):
            extra = f", range=[{v.min():.3f}, {v.max():.3f}]" if v.size > 0 and v.dtype.kind in 'fi' else ""
            print(f"  {k:45s}: shape={str(v.shape):20s}, dtype={str(v.dtype):8s}{extra}")
        else:
            print(f"  {k:45s}: {v}")

    # ── 7. EEF state ───────────────────────────────────────────
    print_separator("7. Robot State")
    print(f"  EEF pos:       {obs['robot0_eef_pos']}")
    print(f"  EEF quat:      {obs['robot0_eef_quat']}")
    print(f"  Gripper qpos:  {obs['robot0_gripper_qpos']}")

    # ── 8. Per-object pose in observations ──────────────────────
    print_separator("8. Per-Object Pose (from LIBERO obs)")
    for obj_name in env.obj_of_interest:
        pos_key = f"{obj_name}_pos"
        quat_key = f"{obj_name}_quat"
        if pos_key in obs:
            print(f"  {pos_key}: {obs[pos_key]}")
        if quat_key in obs:
            print(f"  {quat_key}: {obs[quat_key]}")
        # Object-to-EEF relative pose
        rel_pos_key = f"{obj_name}_to_robot0_eef_pos"
        rel_quat_key = f"{obj_name}_to_robot0_eef_quat"
        if rel_pos_key in obs:
            print(f"  {rel_pos_key}: {obs[rel_pos_key]}")
        if rel_quat_key in obs:
            print(f"  {rel_quat_key}: {obs[rel_quat_key]}")

    # ── 9. Save images ─────────────────────────────────────────
    if save_images:
        out_dir = Path(output_dir) / suite_name / f"task_{task_id:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        import imageio
        for cam_name in camera_names:
            rgb_key = f"{cam_name}_image"
            seg_key = f"{cam_name}_segmentation_element"

            if rgb_key in obs:
                rgb_path = out_dir / f"{cam_name}_rgb.png"
                # LIBERO returns images in OpenGL convention — rotate 180° for standard view
                rgb_img = obs[rgb_key][::-1, ::-1]
                imageio.imwrite(str(rgb_path), rgb_img)
                print(f"\n  Saved RGB: {rgb_path}")

            if seg_key in obs:
                seg_path = out_dir / f"{cam_name}_seg.png"
                # Apply same 180° rotation as RGB to keep alignment
                seg_img = obs[seg_key][::-1, ::-1].squeeze()
                # Normalize to 0-255 for visibility (element seg values are small ints)
                seg_unique = np.unique(seg_img)
                if len(seg_unique) > 1:
                    seg_viz = (seg_img.astype(np.float32) * (255.0 / max(seg_unique.max(), 1))).astype(np.uint8)
                else:
                    seg_viz = seg_img.astype(np.uint8)
                imageio.imwrite(str(seg_path), seg_viz)
                n_objects = len(env.instance_to_id)
                print(f"  Saved SEG: {seg_path} ({n_objects} instances, ids={sorted(seg_unique.tolist())[:12]}...)")

    # ── Build metadata dict (BEFORE close to avoid EGL cleanup issues) ─
    metadata = {
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "problem_folder": task.problem_folder,
        "bddl_file": task.bddl_file,
        "cameras": list(model.camera_names),
        "obj_of_interest": list(env.obj_of_interest),
        "instance_to_id": dict(env.instance_to_id),
        "object_bodies": obj_bodies,
    }

    # Close env (EGL cleanup may fail — safe to ignore)
    try:
        env.close()
    except Exception:
        pass

    return metadata


# ── CLI ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LIBERO Environment Inspection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite", type=str, default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "all"],
        help="Task suite name (default: libero_spatial). Use 'all' for all suites.",
    )
    parser.add_argument(
        "--task-id", type=int, default=0,
        help="Task index within the suite (0-9 for standard suites)",
    )
    parser.add_argument(
        "--camera-names", type=str, nargs="+",
        default=["agentview", "robot0_eye_in_hand"],
        help="Camera names to use (default: agentview robot0_eye_in_hand)",
    )
    parser.add_argument(
        "--resolution", type=int, default=128,
        help="Camera resolution (default: 128)",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save RGB and segmentation images to disk",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_REPO_ROOT / "output" / "libero_inspect"),
        help="Output directory for saved images",
    )
    parser.add_argument(
        "--json", type=str, default="",
        help="Save full metadata as JSON to this path",
    )

    args = parser.parse_args()

    # ── Handle "all" suites ─────────────────────────────────────
    if args.suite == "all":
        suites = ALL_SUITES
    else:
        suites = [args.suite]

    all_results = {}

    for suite_name in suites:
        if len(suites) > 1:
            print(f"\n{'#' * 72}")
            print(f"#  SUITE: {suite_name}")
            print(f"{'#' * 72}")

        # Get number of tasks
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()
        n_tasks = task_suite.n_tasks

        if args.task_id >= 0:
            task_ids = [args.task_id]
        else:
            task_ids = list(range(n_tasks))

        suite_results = {}
        for tid in task_ids:
            print(f"\n{'▶' * 36}")
            print(f"▶  SUITE={suite_name}  TASK={tid}/{n_tasks - 1}")
            print(f"{'▶' * 36}")
            try:
                result = inspect_task(
                    suite_name=suite_name,
                    task_id=tid,
                    camera_names=args.camera_names,
                    resolution=args.resolution,
                    save_images=args.save_images,
                    output_dir=args.output_dir,
                )
                suite_results[str(tid)] = result
            except Exception as e:
                print(f"\n  ❌ ERROR: {e}")
                suite_results[str(tid)] = {"error": str(e)}

        all_results[suite_name] = suite_results

    # ── Save JSON ───────────────────────────────────────────────
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Metadata saved to: {json_path}")

    print("\n✅ Inspection complete.")


if __name__ == "__main__":
    main()
