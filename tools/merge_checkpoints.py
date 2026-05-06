"""
Merge AutoMoT + TransFuser checkpoints into a single safetensors file.

Usage:
    python tools/merge_checkpoints.py

Output:
    Automot/checkpoints/combined/model.safetensors  (~13.5G)

Key convention in combined file:
    - AutoMoT weights:     original keys (e.g. language_model.XXX)
    - TransFuser weights:  transfuser_backbone.<key>  (backbone.XXX in pth -> transfuser_backbone.XXX)
"""

import os
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).parent.parent
MOT_CKPT   = REPO_ROOT / "Automot" / "checkpoints" / "mot" / "0025000" / "model.safetensors"
TRANSFUSER_CKPT = REPO_ROOT / "Automot" / "checkpoints" / "transfuser" / "all_towns" / "model_0030_1.pth"
OUT_DIR    = REPO_ROOT / "Automot" / "checkpoints"
OUT_FILE   = OUT_DIR / "model.safetensors"

BEV_ENCODER_PREFIX = "bev_encoder."


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load AutoMoT weights
    print(f"Loading AutoMoT weights from: {MOT_CKPT}")
    mot_state = load_file(str(MOT_CKPT))
    print(f"  {len(mot_state)} keys loaded.")

    # 2. Load TransFuser weights (torch .pth)
    print(f"Loading TransFuser weights from: {TRANSFUSER_CKPT}")
    pth_state = torch.load(str(TRANSFUSER_CKPT), map_location="cpu")
    # Only keep backbone.* keys, strip "backbone." prefix, add BEV_ENCODER_PREFIX
    transfuser_state = {}
    for k, v in pth_state.items():
        if k.startswith("backbone."):
            new_key = BEV_ENCODER_PREFIX + k[len("backbone."):]
            transfuser_state[new_key] = v
    print(f"  {len(transfuser_state)} backbone keys kept (from {len(pth_state)} total).")

    # 3. Check for key conflicts
    overlap = set(mot_state.keys()) & set(transfuser_state.keys())
    if overlap:
        raise RuntimeError(f"Key conflict between AutoMoT and TransFuser: {overlap}")

    # 4. Merge and save
    combined = {**mot_state, **transfuser_state}
    print(f"Total combined keys: {len(combined)}")
    print(f"Saving to: {OUT_FILE}  (this may take a while for ~13.5G file)...")
    save_file(combined, str(OUT_FILE))
    size_gb = OUT_FILE.stat().st_size / 1024**3
    print(f"Done. File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
