import argparse
import json
import os
from pathlib import Path


def _first_path(value):
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, str):
        return value
    return None


def convert_row(row):
    front_path = _first_path(row.get("front"))
    if not front_path:
        return None

    scene_dir = os.path.dirname(os.path.dirname(front_path))
    frame = os.path.splitext(os.path.basename(front_path))[0]
    row["bev_encoder_feature"] = os.path.join(scene_dir, "bev_encoder_feature", "route_features.pt")
    row["bev_encoder_feature_frame"] = frame

    conversations = row.get("conversations", [])
    for turn in conversations:
        if turn.get("from") == "human":
            turn["value"] = turn.get("value", "").replace("<front>", "<bev>")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if args.limit >= 0 and line_idx >= args.limit:
                break
            row = json.loads(line)
            converted = convert_row(row)
            if converted is None:
                skipped += 1
                continue
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} rows to {output_path}")
    if skipped:
        print(f"skipped {skipped} rows without front path")


if __name__ == "__main__":
    main()
