"""Generate machine-local LIBERO paths without modifying benchmark source."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--libero-home", type=Path, required=True)
parser.add_argument("--datasets", type=Path, required=True,
                    help="LIBERO HDF5 root, not the LaRA LeRobot training root")
args = parser.parse_args()
root = args.libero_home.resolve() / "libero" / "libero"
paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
         "init_states": root / "init_files", "datasets": args.datasets.resolve(),
         "assets": root / "assets"}
for path in paths.values():
    if not path.is_dir():
        raise FileNotFoundError(path)
target = args.libero_home.resolve() / "libero" / "config.yaml"
# JSON is a YAML subset; stdlib keeps bootstrap independent of either environment.
target.write_text(json.dumps({k: str(v) for k, v in paths.items()}, indent=2) + "\n")
print(target)
