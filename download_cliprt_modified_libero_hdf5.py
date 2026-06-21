import os
import time
import argparse
from pathlib import Path

# ============================================================
# 0. 必须放在 import huggingface_hub 之前
# ============================================================

# 使用 hf-mirror；如果服务器可以直连 huggingface.co，可以注释掉
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 禁用 xet，避免部分网络环境访问 xet 后端出问题
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 如果服务器上有残留代理但不可用，就清掉
# 如果你需要代理，把这段注释掉，然后在 shell 里 export HTTP_PROXY/HTTPS_PROXY
for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(key, None)

from huggingface_hub import snapshot_download, list_repo_files
from huggingface_hub import constants


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local-dir",
        type=str,
        default="/home/robot/codePWC/lara_repro/datasets/clip-rt/modified_libero_hdf5",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/home/robot/codePWC/lara_repro/datasets/clip-rt/hf_cache",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_10"],
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "all"],
        help="建议先只下 libero_10 验证，不要一上来下全量 110GB。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="hf-mirror 容易 429，建议 1 或 2。",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只列出匹配到的文件，不下载。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=120,
        help="遇到 429 / 网络错误后等待秒数。",
    )

    return parser.parse_args()


def normalize_suite_names(suites):
    if "all" in suites:
        return ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    return suites


def file_matches_suite(path: str, suite: str) -> bool:
    """
    尽量兼容不同命名：
    libero_10 / libero10 / libero-long / long 等。
    """
    p = path.lower()

    if suite == "libero_spatial":
        keys = ["libero_spatial", "libero-spatial", "spatial"]
    elif suite == "libero_object":
        keys = ["libero_object", "libero-object", "object"]
    elif suite == "libero_goal":
        keys = ["libero_goal", "libero-goal", "goal"]
    elif suite == "libero_10":
        keys = ["libero_10", "libero-10", "libero10", "libero_long", "libero-long", "long"]
    else:
        keys = [suite]

    return any(k in p for k in keys)


def is_hdf5_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(".hdf5") or p.endswith(".h5")


def main():
    args = parse_args()

    repo_id = "clip-rt/modified_libero_hdf5"
    local_dir = Path(args.local_dir)
    cache_dir = Path(args.cache_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    suites = normalize_suite_names(args.suites)

    print("=" * 100)
    print("Downloading CLIP-RT / OpenVLA-style modified LIBERO HDF5")
    print("=" * 100)
    print(f"Repo ID:        {repo_id}")
    print(f"HF_ENDPOINT:    {os.environ.get('HF_ENDPOINT')}")
    print(f"Hub ENDPOINT:   {constants.ENDPOINT}")
    print(f"Disable XET:    {os.environ.get('HF_HUB_DISABLE_XET')}")
    print(f"Local dir:      {local_dir}")
    print(f"Cache dir:      {cache_dir}")
    print(f"Suites:         {suites}")
    print("=" * 100)

    print("\nListing repo files...")
    files = list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )

    roots = sorted(set(f.split("/")[0] for f in files))
    print("\nTop-level entries:")
    for r in roots:
        print(f"  - {r}")

    selected_files = []
    for f in files:
        if not is_hdf5_file(f):
            continue
        if any(file_matches_suite(f, s) for s in suites):
            selected_files.append(f)

    selected_files = sorted(set(selected_files))

    print("\nMatched HDF5 files:")
    for f in selected_files[:200]:
        print(f"  - {f}")
    if len(selected_files) > 200:
        print(f"  ... and {len(selected_files) - 200} more")

    print(f"\nTotal matched files: {len(selected_files)}")

    if len(selected_files) == 0:
        print("\n❌ No matched HDF5 files found.")
        print("请先用 --list-only 看文件结构，然后根据实际目录名修改 file_matches_suite()。")
        return

    if args.list_only:
        print("\nList-only mode. No download performed.")
        return

    print("\nStart downloading...")
    print("注意：如果遇到 429 Too Many Requests，脚本会等待后重试。")
    print("=" * 100)

    last_err = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"\nDownload attempt {attempt}/{args.retries}")

            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                cache_dir=str(cache_dir),
                allow_patterns=selected_files,
                max_workers=args.max_workers,
                force_download=False,
            )

            print("\n✅ Download finished.")
            print(f"Local path: {path}")
            break

        except Exception as e:
            last_err = e
            print(f"\n⚠️ Download failed on attempt {attempt}/{args.retries}")
            print(str(e)[:2000])

            if attempt == args.retries:
                print("\n❌ Reached max retries.")
                raise

            print(f"\nSleeping {args.sleep} seconds before retry...")
            time.sleep(args.sleep)

    print("\nChecking downloaded files:")
    missing = []
    for rel in selected_files:
        p = local_dir / rel
        if p.exists():
            print(f"  ✅ {p}")
        else:
            print(f"  ❌ Missing: {p}")
            missing.append(str(p))

    print("\nSummary:")
    print(f"  selected files: {len(selected_files)}")
    print(f"  missing files:  {len(missing)}")
    print(f"  local dir:      {local_dir}")

    print("\nSet this later if needed:")
    print(f"export CLIPRT_MODIFIED_LIBERO_HDF5_ROOT={local_dir}")

    if len(missing) == 0:
        print("\n✅ All selected files exist.")
    else:
        print("\n⚠️ Some files are missing. Re-run the same command; downloaded files will be reused.")


if __name__ == "__main__":
    main()
