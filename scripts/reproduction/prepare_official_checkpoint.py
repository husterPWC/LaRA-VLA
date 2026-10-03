"""Create a machine-local view of the released checkpoint for R1/R2.

The released run is kept untouched. Only framework.qwenvl.base_vlm changes in
the active YAML; the checkpoint is linked rather than copied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changed_fields(before, after, prefix="") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            changes.extend(changed_fields(before.get(key), after.get(key), child))
        return changes
    return [] if before == after else [prefix]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="steps_25000_pytorch_model.pt")
    args = parser.parse_args()

    source = args.source_run.resolve()
    backbone = args.backbone.resolve()
    output = args.output_run.resolve()
    source_config = source / "config.yaml"
    source_checkpoint = source / "checkpoints" / args.checkpoint_name
    required = [
        source_config,
        source / "config.json",
        source / "dataset_statistics.json",
        source_checkpoint,
        backbone / "config.json",
        backbone / "model.safetensors.index.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    original_config = yaml.safe_load(source_config.read_text())
    config = yaml.safe_load(source_config.read_text())
    old_backbone = config["framework"]["qwenvl"]["base_vlm"]
    config["framework"]["qwenvl"]["base_vlm"] = str(backbone)
    changes = changed_fields(original_config, config)
    if changes != ["framework.qwenvl.base_vlm"]:
        raise RuntimeError(f"Unexpected semantic config changes: {changes}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    shutil.copy2(source_config, output / "config.official.yaml")
    shutil.copy2(source / "config.json", output / "config.official.json")
    shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
    for optional in ("README.md", "summary.jsonl"):
        if (source / optional).is_file():
            shutil.copy2(source / optional, output / optional)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )

    checkpoint_link = checkpoints / args.checkpoint_name
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != source_checkpoint:
            raise RuntimeError(f"Existing link points elsewhere: {checkpoint_link}")
    elif checkpoint_link.exists():
        raise FileExistsError(checkpoint_link)
    else:
        checkpoint_link.symlink_to(source_checkpoint)

    manifest = {
        "source_run": str(source),
        "source_config_sha256": sha256(source_config),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_size": source_checkpoint.stat().st_size,
        "backbone": str(backbone),
        "base_vlm_before": old_backbone,
        "base_vlm_after": str(backbone),
        "changed_config_fields": changes,
        "prepared_checkpoint": str(checkpoint_link),
    }
    (output / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
