"""严格验证 R4 数据集与 FAST tokenizer 的官方文件。

只对 Hugging Face ``.metadata`` 对应的发布文件生成指纹；后续生成的
steps cache、lock 和其他额外文件不影响官方资产验收。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "dataset": {
        "revision": "fbe4f71c2fd5a6e4f9171c78e51e2f6567277fff",
        "file_count": 5163,
        "total_bytes": 2011668877,
        "manifest_sha256": (
            "99e134b7fda131b0d33b83baaec820131909be190c76a3cecf613ae30248d719"
        ),
    },
    "fast_tokenizer": {
        "revision": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
        "file_count": 7,
        "total_bytes": 698459,
        "manifest_sha256": (
            "7cd45fc4ea68c30f3ec2ecee636feb0bb145d0d21414c222a0d3069ee7d389e2"
        ),
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(root: Path) -> dict:
    root = root.resolve(strict=True)
    metadata_root = root / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir():
        raise FileNotFoundError(
            f"缺少 Hugging Face metadata 目录: {metadata_root}"
        )

    metadata_files = sorted(
        metadata_root.rglob("*.metadata"),
        key=lambda path: path.relative_to(metadata_root).as_posix(),
    )
    if not metadata_files:
        raise RuntimeError(f"没有找到 metadata 文件: {metadata_root}")

    revisions = set()
    records = []
    missing = []
    for metadata_path in metadata_files:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise RuntimeError(f"metadata 为空: {metadata_path}")
        revisions.add(lines[0])

        metadata_relative = metadata_path.relative_to(metadata_root).as_posix()
        relative = metadata_relative[: -len(".metadata")]
        asset_path = root / relative
        if not asset_path.is_file():
            missing.append(relative)
            continue
        records.append(
            (relative, asset_path.stat().st_size, file_sha256(asset_path))
        )

    manifest = hashlib.sha256()
    for relative, size, digest in records:
        manifest.update(f"{digest}  {size}  {relative}\n".encode("utf-8"))

    return {
        "root": str(root),
        "revisions": sorted(revisions),
        "metadata_count": len(metadata_files),
        "file_count": len(records),
        "total_bytes": sum(record[1] for record in records),
        "manifest_sha256": manifest.hexdigest(),
        "missing_files": missing,
    }


def validate(name: str, actual: dict) -> None:
    expected = EXPECTED[name]
    checks = {
        "revisions": actual["revisions"] == [expected["revision"]],
        "metadata_count": actual["metadata_count"] == expected["file_count"],
        "file_count": actual["file_count"] == expected["file_count"],
        "total_bytes": actual["total_bytes"] == expected["total_bytes"],
        "manifest_sha256": (
            actual["manifest_sha256"] == expected["manifest_sha256"]
        ),
        "missing_files": not actual["missing_files"],
    }
    actual["checks"] = checks
    actual["status"] = "pass" if all(checks.values()) else "fail"
    if actual["status"] != "pass":
        raise RuntimeError(
            f"{name} 指纹不一致: "
            + json.dumps(
                {"expected": expected, "actual": actual},
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fast-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "dataset": fingerprint(args.dataset_root),
        "fast_tokenizer": fingerprint(args.fast_tokenizer),
    }
    validate("dataset", report["dataset"])
    validate("fast_tokenizer", report["fast_tokenizer"])
    report["status"] = "pass"

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("R4 ASSET VERIFICATION PASS")


if __name__ == "__main__":
    main()
