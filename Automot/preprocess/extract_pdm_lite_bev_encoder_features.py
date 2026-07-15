#!/usr/bin/env python3
"""Extract BEV encoder features for PDM-Lite routes."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

AUTOMOT_ROOT = Path(__file__).resolve().parents[1]
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))


def _first_path(value):
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, str) and value:
        return value
    return None


def _route_and_frame_from_sample(row):
    frame_path = _first_path(row.get("front"))
    if frame_path is None:
        images = row.get("image")
        if isinstance(images, list) and images:
            frame_path = images[-1]
        elif isinstance(images, str):
            frame_path = images
    if frame_path is None:
        return None, None

    frame_path = Path(frame_path)
    if len(frame_path.parts) < 3:
        return None, None
    route = Path(*frame_path.parts[:-2])
    frame = frame_path.stem
    return route.as_posix(), frame


def collect_frames_from_jsonl(jsonl_paths):
    route_to_frames = defaultdict(set)
    total_rows = 0
    skipped_rows = 0

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                total_rows += 1
                row = json.loads(line)
                route, frame = _route_and_frame_from_sample(row)
                if route is None or frame is None:
                    skipped_rows += 1
                    continue
                route_to_frames[route].add(frame)

    return route_to_frames, total_rows, skipped_rows


def collect_all_route_frames(pdm_root):
    route_to_frames = defaultdict(set)
    for rgb_dir in pdm_root.glob("*/*/rgb"):
        route = rgb_dir.parent.relative_to(pdm_root).as_posix()
        for image_path in rgb_dir.glob("*.jpg"):
            route_to_frames[route].add(image_path.stem)
    return route_to_frames


def load_lidar_bev(route_dir, frame, extractor, allow_laz_fallback):
    bev_path = route_dir / "bev_encoder_lidar_bev" / f"{frame}.npy"
    if bev_path.exists():
        bev = np.load(bev_path)
        if bev.ndim == 3 and bev.shape[0] not in (1, 2) and bev.shape[-1] in (1, 2):
            bev = np.transpose(bev, (2, 0, 1))
        expected_channels = (2 if extractor.config.use_ground_plane else 1) * extractor.config.lidar_seq_len
        if bev.ndim != 3:
            raise ValueError(f"Expected lidar BEV with shape [C,H,W], got {bev.shape}")
        if bev.shape[0] != expected_channels:
            if bev.shape[0] == 2 and expected_channels == 1:
                bev = bev[1:2]
            else:
                raise ValueError(
                    f"Expected {expected_channels} lidar BEV channels, got {bev.shape[0]} at {bev_path}"
                )
        return torch.from_numpy(bev.astype(np.float32, copy=False)).unsqueeze(0)

    if allow_laz_fallback:
        lidar_path = route_dir / "lidar" / f"{frame}.laz"
        if not lidar_path.exists():
            lidar_path = route_dir / "lidar" / f"{frame}.las"
        if lidar_path.exists():
            return extractor.preprocess_lidar(str(lidar_path))

    raise FileNotFoundError(f"Missing lidar BEV for frame {frame} in {route_dir}")


def extract_route_features(
    extractor,
    pdm_root,
    route,
    frames,
    batch_size,
    output_name,
    overwrite,
    allow_laz_fallback,
    save_upsample,
):
    route_dir = pdm_root / route
    output_dir = route_dir / "bev_encoder_feature"
    output_path = output_dir / output_name
    if output_path.exists() and not overwrite:
        return "skipped", output_path

    frame_nums = []
    bev_features = []
    bev_upsamples = []

    frames = sorted(str(frame).zfill(4) for frame in frames)
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        rgb_batch = []
        lidar_batch = []
        valid_frames = []

        for frame in batch_frames:
            rgb_path = route_dir / "rgb" / f"{frame}.jpg"
            if not rgb_path.exists():
                print(f"[skip] missing RGB: {rgb_path}")
                continue
            try:
                rgb_batch.append(extractor.preprocess_rgb(str(rgb_path)))
                lidar_batch.append(load_lidar_bev(route_dir, frame, extractor, allow_laz_fallback))
                valid_frames.append(frame)
            except Exception as exc:
                print(f"[skip] {route}/{frame}: {exc}")

        if not valid_frames:
            continue

        rgb = torch.cat(rgb_batch, dim=0)
        lidar = torch.cat(lidar_batch, dim=0)
        outputs = extractor(rgb, lidar)

        bev = outputs["bev_feature"]
        if bev is None:
            raise RuntimeError("BEV encoder backbone did not return bev_feature")

        frame_nums.extend(valid_frames)
        bev_features.append(bev.detach().float().cpu())

        upsample = outputs.get("bev_feature_upscale")
        if save_upsample and upsample is not None:
            bev_upsamples.append(upsample.detach().float().cpu())

    if not frame_nums:
        return "empty", output_path

    payload = {
        "frame_nums": frame_nums,
        "bev_features": torch.cat(bev_features, dim=0),
    }
    if save_upsample and bev_upsamples:
        payload["bev_upsamples"] = torch.cat(bev_upsamples, dim=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)
    return "written", output_path


def parse_args():
    automot_model_path = Path(
        os.environ.get("AUTOMOT_MODEL_PATH", str(AUTOMOT_ROOT / "checkpoints"))
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdm-root", default=os.environ.get("PDM_DATA_DIR", ""), help="PDM-Lite data root")
    parser.add_argument("--jsonl", action="append", default=[], help="JSONL index to scan; can be passed more than once")
    parser.add_argument("--all-route-frames", action="store_true", help="Extract every rgb/*.jpg frame in each route")
    parser.add_argument(
        "--config-dir",
        default=os.environ.get("BEV_ENCODER_CONFIG_DIR", str(automot_model_path)),
        help="Directory containing bev_config.json; defaults to AUTOMOT_MODEL_PATH",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "BEV_ENCODER_CHECKPOINT",
            str(automot_model_path / "model.safetensors"),
        ),
        help="BEV encoder weights; defaults to AUTOMOT_MODEL_PATH/model.safetensors",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-routes", type=int, default=-1)
    parser.add_argument("--limit-frames", type=int, default=-1)
    parser.add_argument("--output-name", default="route_features.pt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-laz-fallback", action="store_true")
    parser.add_argument("--no-upsample", action="store_true", help="Do not store bev_upsamples in route_features.pt")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.pdm_root:
        raise ValueError("Set --pdm-root or PDM_DATA_DIR")

    pdm_root = Path(args.pdm_root)
    if args.all_route_frames:
        route_to_frames = collect_all_route_frames(pdm_root)
        total_rows = skipped_rows = 0
    else:
        if not args.jsonl:
            raise ValueError("Pass --jsonl or use --all-route-frames")
        route_to_frames, total_rows, skipped_rows = collect_frames_from_jsonl(args.jsonl)

    routes = sorted(route_to_frames)
    if args.limit_routes >= 0:
        routes = routes[:args.limit_routes]

    print(f"Routes: {len(routes)}")
    if not args.all_route_frames:
        print(f"Scanned rows: {total_rows}, skipped rows: {skipped_rows}")

    if args.dry_run:
        for route in routes[:10]:
            frames = sorted(route_to_frames[route])
            print(f"{route}: {len(frames)} frames")
        return

    from mot.modeling.bev_encoder.backbone_extractor import BEVEncoderBackboneExtractor

    extractor = BEVEncoderBackboneExtractor(
        config_path=args.config_dir,
        model_path=args.checkpoint or None,
        device=args.device,
    )

    written = skipped = empty = 0
    for index, route in enumerate(routes, start=1):
        frames = sorted(route_to_frames[route])
        if args.limit_frames >= 0:
            frames = frames[:args.limit_frames]

        status, output_path = extract_route_features(
            extractor=extractor,
            pdm_root=pdm_root,
            route=route,
            frames=frames,
            batch_size=args.batch_size,
            output_name=args.output_name,
            overwrite=args.overwrite,
            allow_laz_fallback=args.allow_laz_fallback,
            save_upsample=not args.no_upsample,
        )
        if status == "written":
            written += 1
        elif status == "skipped":
            skipped += 1
        else:
            empty += 1
        print(f"[{index}/{len(routes)}] {status}: {output_path}")

    print(f"Done. written={written}, skipped={skipped}, empty={empty}")


if __name__ == "__main__":
    main()
