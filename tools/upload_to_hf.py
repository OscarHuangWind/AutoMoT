"""
Upload AutoMoT checkpoints to HuggingFace.

Usage:
    HF_TOKEN=YOUR_HF_WRITE_TOKEN python tools/upload_to_hf.py --username YOUR_HF_USERNAME

What gets uploaded (repo: YOUR_HF_USERNAME/automot_ckpt):
  model.safetensors     <- Combined AutoMoT + BEV encoder weights (~13G)
  base_config/          <- Qwen3VL base config + tokenizer (~12M, no model weights)

Use --delete-old to remove split checkpoint folders from the HF repo.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ROOT     = Path(__file__).parent.parent
COMBINED_CKPT = REPO_ROOT / "Automot" / "checkpoints" / "model.safetensors"
BASE_CONFIG   = REPO_ROOT / "Automot" / "checkpoints"
SKIP_PATTERNS = {"optimizer.", "scheduler.pt", "data_status.pt", ".cache", ".venv"}


def should_skip(path: Path) -> bool:
    name = path.name
    return any(name.startswith(p) or p in name for p in SKIP_PATTERNS)


def upload_file(api: HfApi, local_path: Path, repo_id: str, path_in_repo: str):
    size_gb = local_path.stat().st_size / 1e9
    size_str = f" ({size_gb:.2f} GB)" if size_gb > 0.1 else ""
    print(f"  Uploading {path_in_repo}{size_str}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def upload_directory(api: HfApi, local_dir: Path, repo_id: str, repo_prefix: str):
    if not local_dir.exists():
        print(f"  WARNING: {local_dir} does not exist, skipping.")
        return
    files = [f for f in local_dir.rglob("*") if f.is_file() and not should_skip(f)]
    print(f"\nUploading {len(files)} files from {local_dir} -> {repo_id}/{repo_prefix}/")
    for i, fpath in enumerate(files, 1):
        rel = fpath.relative_to(local_dir)
        print(f"  [{i}/{len(files)}] ", end="")
        upload_file(api, fpath, repo_id, f"{repo_prefix}/{rel}")


def delete_repo_folder(api: HfApi, repo_id: str, folder: str):
    """Delete all files under folder/ in the HF repo."""
    try:
        all_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        to_delete = [f for f in all_files if f.startswith(folder + "/") or f == folder]
        if not to_delete:
            print(f"  (nothing found under {folder}/)")
            return
        for f in to_delete:
            print(f"  Deleting {f}")
            api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"  WARNING: could not delete {folder}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",      default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace write token. Defaults to HF_TOKEN.")
    parser.add_argument("--username",   required=True, help="HuggingFace username")
    parser.add_argument("--repo-name",  default="automot_ckpt", help="HF repo name")
    parser.add_argument("--private",    action="store_true", default=False)
    parser.add_argument("--delete-old", action="store_true", default=False,
                        help="Delete split checkpoint folders from the HF repo")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or HF_TOKEN is required")

    repo_id = f"{args.username}/{args.repo_name}"
    api = HfApi(token=args.token)

    print(f"Creating/updating repo: {repo_id} (private={args.private})")
    create_repo(repo_id=repo_id, repo_type="model", private=args.private,
                token=args.token, exist_ok=True)

    # Optionally clean up separate checkpoint folders.
    if args.delete_old:
        print("\n[Deleting split checkpoint folders from HF repo...]")
        delete_repo_folder(api, repo_id, "automot")
        delete_repo_folder(api, repo_id, "bev_encoder")

    # Upload combined checkpoint
    print(f"\n[1/2] Combined checkpoint: {COMBINED_CKPT}")
    upload_file(api, COMBINED_CKPT, repo_id, "model.safetensors")

    # Upload config files (all non-safetensors files in checkpoints/)
    # Excludes model.safetensors and large binary weights
    config_files = [f for f in BASE_CONFIG.rglob("*") if f.is_file()
                    and not should_skip(f)
                    and not f.suffix == ".safetensors"
                    and not f.name.endswith(".safetensors.index.json")]
    print(f"\n[2/2] Config files ({len(config_files)}) from {BASE_CONFIG}")
    for i, fpath in enumerate(config_files, 1):
        rel = fpath.relative_to(BASE_CONFIG)
        print(f"  [{i}/{len(config_files)}] ", end="")
        upload_file(api, fpath, repo_id, str(rel))

    print(f"\nDone! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
