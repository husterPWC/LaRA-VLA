import os
from pathlib import Path

# ============================================================
# 0. 必须放在 import huggingface_hub 之前
# ============================================================

# 使用 hf-mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 禁用 xet，避免有些网络环境访问 xet 后端出问题
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 如果服务器上有残留代理，但代理不可用，可以先清掉
# 如果你需要代理，就把这几行注释掉，然后在 shell 里正确 export 代理
for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(key, None)

# 注意：huggingface_hub 必须在上面环境变量设置之后再 import
from huggingface_hub import snapshot_download
from huggingface_hub import constants


def main():
    repo_id = "lovejuly/libero_lerobot_all"

    local_dir = Path("/home/robot/codePWC/lara_repro/datasets/lovejuly/libero_lerobot_all")
    cache_dir = Path("/home/robot/codePWC/lara_repro/datasets/lovejuly/hf_cache")

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "libero_spatial_no_noops_1.0.0_lerobot/**",
        "libero_object_no_noops_1.0.0_lerobot/**",
        "libero_goal_no_noops_1.0.0_lerobot/**",
        "libero_10_no_noops_1.0.0_lerobot/**",
    ]

    print("=" * 80)
    print("Downloading LaRA-VLA LIBERO LeRobot dataset")
    print("=" * 80)
    print(f"Repo ID:        {repo_id}")
    print(f"HF_ENDPOINT:    {os.environ.get('HF_ENDPOINT')}")
    print(f"Hub ENDPOINT:   {constants.ENDPOINT}")
    print(f"Disable XET:    {os.environ.get('HF_HUB_DISABLE_XET')}")
    print(f"Local dir:      {local_dir}")
    print(f"Cache dir:      {cache_dir}")
    print("Suites:")
    for p in allow_patterns:
        print(f"  - {p}")
    print("=" * 80)

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
        max_workers=8,
        force_download=False,
    )

    print("\n✅ LaRA-VLA LIBERO dataset downloaded successfully!")
    print(f"Local path: {path}")

    expected_dirs = [
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    ]

    print("\nChecking expected folders:")
    all_ok = True
    for name in expected_dirs:
        folder = local_dir / name
        if folder.exists():
            print(f"  ✅ {folder}")
        else:
            print(f"  ❌ Missing: {folder}")
            all_ok = False

    print("\nSet this later for training:")
    print(f"export LIBERO_LEROBOT_ROOT={local_dir}")

    if all_ok:
        print("\n✅ All expected LIBERO folders exist.")
    else:
        print("\n⚠️ Some folders are missing. Check download logs.")


if __name__ == "__main__":
    main()