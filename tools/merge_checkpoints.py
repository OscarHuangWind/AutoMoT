"""
Merge AutoMoT and BEV encoder checkpoints into one safetensors file.

Example:
    python tools/merge_checkpoints.py \
      --automot-checkpoint /path/to/automot/model.safetensors \
      --bev-encoder-checkpoint /path/to/bev_encoder/model.pth \
      --output Automot/checkpoints/model.safetensors

Only ``backbone.*`` entries from the BEV encoder checkpoint are copied. They are
stored under the ``bev_encoder.`` prefix expected by AutoMoT.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BEV_ENCODER_PREFIX = "bev_encoder."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automot-checkpoint", required=True, type=Path)
    parser.add_argument("--bev-encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading AutoMoT weights from: {args.automot_checkpoint}")
    automot_state = load_file(str(args.automot_checkpoint))
    print(f"Loaded {len(automot_state)} AutoMoT tensors.")

    print(f"Loading BEV encoder weights from: {args.bev_encoder_checkpoint}")
    bev_state = torch.load(str(args.bev_encoder_checkpoint), map_location="cpu")
    bev_encoder_state = {}
    for key, value in bev_state.items():
        if key.startswith("backbone."):
            bev_encoder_state[BEV_ENCODER_PREFIX + key[len("backbone."):]] = value
    print(f"Kept {len(bev_encoder_state)} BEV encoder backbone tensors.")

    overlap = set(automot_state) & set(bev_encoder_state)
    if overlap:
        sample = sorted(overlap)[:10]
        raise RuntimeError(f"Key conflict while merging checkpoints: {sample}")

    combined = {**automot_state, **bev_encoder_state}
    print(f"Saving {len(combined)} tensors to: {args.output}")
    save_file(combined, str(args.output))
    size_gb = args.output.stat().st_size / 1024**3
    print(f"Done. File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
