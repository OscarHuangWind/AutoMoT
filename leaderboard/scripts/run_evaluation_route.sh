#!/bin/bash
# ============================================================
# Bench2Drive Route-by-Route Evaluation Script
# Runs evaluation for all 220 routes one by one, skipping already completed ones.
# ============================================================

# Auto-detect paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEADERBOARD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${LEADERBOARD_DIR}/.." && pwd)"

# Auto-detect CARLA_ROOT if not set
if [ -z "${CARLA_ROOT}" ]; then
    if [ -d "$(dirname "${PROJECT_ROOT}")/carla" ]; then
        export CARLA_ROOT="$(dirname "${PROJECT_ROOT}")/carla"
    elif [ -d "${HOME}/carla" ]; then
        export CARLA_ROOT="${HOME}/carla"
    else
        echo "ERROR: CARLA_ROOT is not set and could not be auto-detected."
        echo "Please export CARLA_ROOT=/path/to/carla"
        exit 1
    fi
    echo "Auto-detected CARLA_ROOT=${CARLA_ROOT}"
fi

BASE_PORT=2000
BASE_TM_PORT=2001
IS_BENCH2DRIVE=True
BASE_ROUTES=${LEADERBOARD_DIR}/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/mot_b2d_agent.py
TEAM_CONFIG=${PROJECT_ROOT}/MoT-DP/checkpoints/carla_dit_best/carla_policy_best
BASE_CHECKPOINT_ENDPOINT=eval
SAVE_PATH=./eval_v1/
PLANNER_TYPE=only_traj
GPU_RANK=0

EVAL_OUTPUT_DIR="${SCRIPT_DIR}/v_2json_open"
mkdir -p "$EVAL_OUTPUT_DIR"
# export USE_EMA_WEIGHTS=1  # Disabled: EMA weights cause decoding crash (empty answer_ids)
EVAL_JSON_DIR="${PROJECT_ROOT}/eval_json"
SPLIT_FILE_1="${EVAL_JSON_DIR}/b2d_all_routes_split1.json"
SPLIT_FILE_2="${EVAL_JSON_DIR}/b2d_all_routes_split2.json"
MERGED_FILE="${EVAL_JSON_DIR}/b2d_all_routes_merged.json"
TM_SEED="${TM_SEED:-3407}"

# Check split files exist
if [ ! -f "$SPLIT_FILE_1" ] || [ ! -f "$SPLIT_FILE_2" ]; then
    echo "Error: split files not found:"
    echo "  - $SPLIT_FILE_1"
    echo "  - $SPLIT_FILE_2"
    exit 1
fi

# Merge split files
python3 - "$SPLIT_FILE_1" "$SPLIT_FILE_2" "$MERGED_FILE" << 'PY'
import json, sys
path1, path2, out_path = sys.argv[1:4]

def load_ids(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('ids') or [int(r['id']) for r in data.get('routes', [])]

ids = load_ids(path1) + load_ids(path2)
missing_ids = [1711, 1773]
for mid in missing_ids:
    if mid not in ids:
        ids.append(mid)
with open(out_path, 'w') as f:
    json.dump({'ids': ids}, f, indent=2)
print(f"Merged {len(ids)} route ids -> {out_path}")
PY

# Extract route IDs
mapfile -t ROUTE_IDS < <(python3 - "$MERGED_FILE" << 'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r') as f:
    data = json.load(f)
ids = data.get('ids') or [int(r['id']) for r in data.get('routes', [])]
for i in ids:
    print(i)
PY
)

TOTAL=${#ROUTE_IDS[@]}
CURRENT=0
FAILED_ROUTES=()

echo "=========================================="
echo "Starting evaluation for $TOTAL routes"
echo "Using TrafficManager seed: ${TM_SEED}"
echo "=========================================="

if [ "$TOTAL" -eq 0 ]; then
    echo "No route IDs found. Check split files."
    exit 1
fi

for ROUTE_ID in "${ROUTE_IDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    EXISTING_EVAL="${EVAL_OUTPUT_DIR}/eval_${ROUTE_ID}.json"
    if [ -f "$EXISTING_EVAL" ]; then
        echo ""
        echo "=========================================="
        echo "[$CURRENT/$TOTAL] Skipping route: $ROUTE_ID (already exists)"
        echo "=========================================="
        continue
    fi
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Running route: $ROUTE_ID"
    echo "=========================================="
    
    PORT=$BASE_PORT
    TM_PORT=$BASE_TM_PORT
    ROUTES="${BASE_ROUTES}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
    
    bash "${SCRIPT_DIR}/run_evaluation.sh" \
        $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG \
        $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK "$ROUTE_ID" "$TM_SEED"
    
    EVAL_EXIT_CODE=$?
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "WARNING: run_evaluation.sh failed for route $ROUTE_ID (exit code: $EVAL_EXIT_CODE)"
        FAILED_ROUTES+=("$ROUTE_ID")
    fi
    
    # Copy eval.json to target directory (same as original project)
    if [ -f "$CHECKPOINT_ENDPOINT" ]; then
        NEW_NAME="${EVAL_OUTPUT_DIR}/eval_${ROUTE_ID}.json"
        cp "$CHECKPOINT_ENDPOINT" "$NEW_NAME"
        echo "Saved: $NEW_NAME"
    else
        echo "Warning: $CHECKPOINT_ENDPOINT not found, skipping save"
    fi
    
    echo "[$CURRENT/$TOTAL] Route $ROUTE_ID completed"
done

echo ""
echo "=========================================="
echo "All $TOTAL routes completed!"
echo "Results saved in: ${EVAL_OUTPUT_DIR}/"
if [ ${#FAILED_ROUTES[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED_ROUTES[@]} route(s) failed: ${FAILED_ROUTES[*]}"
    echo "You can re-run the script to retry failed routes (existing results will be skipped)."
else
    echo "All routes succeeded."
fi
echo "=========================================="
