#!/bin/bash
# Usage: bash run_evaluation.sh <PORT> <TM_PORT> <IS_BENCH2DRIVE> <ROUTES> <TEAM_AGENT> <TEAM_CONFIG> <CHECKPOINT_ENDPOINT> <SAVE_PATH> <PLANNER_TYPE> <GPU_RANK> [ROUTES_SUBSET] [TM_SEED]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEADERBOARD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${LEADERBOARD_DIR}/.." && pwd)"

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
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_DIR}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_DIR}/team_code
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/scenario_runner
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/Automot
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/Automot/mot

export SCENARIO_RUNNER_ROOT=${PROJECT_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${LEADERBOARD_DIR}
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}
export ROUTES_SUBSET=${11:-""}
export TM_SEED=${12:-0}
export RECORD_PATH="${RECORD_PATH:-}"
export RESUME="${RESUME:-False}"

export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

AUTOMOT_CONDA_ENV="${AUTOMOT_CONDA_ENV:-automot}"
if [ -f "${CONDA_PREFIX}/../../etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/../../etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "${AUTOMOT_CONDA_ENV}"
else
    echo "WARNING: conda was not found; using the current Python environment."
fi

export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export AUTOMOT_COMPILE_BLOCK_MASK="${AUTOMOT_COMPILE_BLOCK_MASK:-0}"

export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using TrafficManager seed: ${TM_SEED}"

if [ "${DEBUG_CHALLENGE}" != "0" ] || [ "${FORCE_CUDA_LAUNCH_BLOCKING}" = "1" ]; then
        echo "Running with CUDA_LAUNCH_BLOCKING=1 (debug mode)"
        CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
                --routes=${ROUTES} \
                --repetitions=${REPETITIONS} \
                --track=${CHALLENGE_TRACK_CODENAME} \
                --checkpoint=${CHECKPOINT_ENDPOINT} \
                --agent=${TEAM_AGENT} \
                --agent-config=${TEAM_CONFIG} \
                --debug=${DEBUG_CHALLENGE} \
                --record=${RECORD_PATH} \
                --resume=${RESUME} \
                --port=${PORT} \
                --traffic-manager-port=${TM_PORT} \
                --traffic-manager-seed=${TM_SEED} \
                --gpu-rank=${GPU_RANK} \
                $([ -n "$ROUTES_SUBSET" ] && echo "--routes-subset=$ROUTES_SUBSET")
else
        echo "Running without CUDA_LAUNCH_BLOCKING (recommended for performance)"
        CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
                --routes=${ROUTES} \
                --repetitions=${REPETITIONS} \
                --track=${CHALLENGE_TRACK_CODENAME} \
                --checkpoint=${CHECKPOINT_ENDPOINT} \
                --agent=${TEAM_AGENT} \
                --agent-config=${TEAM_CONFIG} \
                --debug=${DEBUG_CHALLENGE} \
                --record=${RECORD_PATH} \
                --resume=${RESUME} \
                --port=${PORT} \
                --traffic-manager-port=${TM_PORT} \
                --traffic-manager-seed=${TM_SEED} \
                --gpu-rank=${GPU_RANK} \
                $([ -n "$ROUTES_SUBSET" ] && echo "--routes-subset=$ROUTES_SUBSET")
fi
