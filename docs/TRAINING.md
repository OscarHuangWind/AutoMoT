# AutoMoT Training

This guide assumes the repository is cloned as `automot`, but all commands use
paths relative to the repository root.

## Local Paths

The default training configuration targets a single 80GB GPU. For 40GB GPUs,
use 2+ cards with multi-GPU FSDP rather than full training on a single card.

Set the required paths:

```bash
export AUTOMOT_MODEL_PATH="/path/to/automot/checkpoint"
export PDM_DATA_DIR="/path/to/pdm_lite"
```

`AUTOMOT_MODEL_PATH` should point to the downloaded [Oscar-Huang/AutoMoT](https://huggingface.co/Oscar-Huang/AutoMoT) checkpoint directory.

Optional overrides:

```bash
export QWEN3VL_PATH="/path/to/qwen3vl/config-and-tokenizer"
export CHECKPOINT_DIR="/path/to/checkpoints/traj_meta"
export RESULTS_DIR="/path/to/training_logs"
```

If omitted, `QWEN3VL_PATH` defaults to `AUTOMOT_MODEL_PATH`.
`CHECKPOINT_DIR` stores model checkpoints and defaults to
`Automot/checkpoints/traj_meta`. `RESULTS_DIR` stores training logs such as
`log.txt` and defaults to `Automot/results`.

## PDM-Lite Inputs

Download the PDM-Lite training JSONL indexes from [HqH1111/AutoMoT-PDM-Lite-BEV-Encoder-Indexes](https://huggingface.co/datasets/HqH1111/AutoMoT-PDM-Lite-BEV-Encoder-Indexes) into a local JSONL directory:

```bash
export PDM_JSONL_DIR="/path/to/pdm_lite_jsonl"
mkdir -p "$PDM_JSONL_DIR"
hf download HqH1111/AutoMoT-PDM-Lite-BEV-Encoder-Indexes \
  pdm_lite_2hz_2tp_train_bev_encoder.jsonl \
  pdm_lite_2hz_2tp_val_bev_encoder.jsonl \
  --repo-type dataset \
  --local-dir "$PDM_JSONL_DIR"
```

Each downloaded row retains four historical front-camera frames for Qwen3-VL reasoning and uses `<bev>` for the action branch. The action branch supports two BEV data paths:

| Mode | Required inputs | Behavior |
|------|-----------------|----------|
| Precomputed BEV features, recommended | `bev_encoder_feature`, `bev_encoder_feature_frame`, and `<PDM_DATA_DIR>/<bev_encoder_feature>` | The dataset reads `route_features.pt`, selects `bev_encoder_feature_frame`, and passes 64 BEV tokens (`8 x 8`) to AutoMoT. The BEV encoder backbone is not instantiated in the step. |
| Online BEV fallback | `front` or the latest `image` frame, plus `bev_encoder_lidar_bev/<frame>.npy` under the same route | If the feature field is missing or the file cannot be loaded, the training loop initializes a frozen BEV encoder from `AUTOMOT_MODEL_PATH/model.safetensors`, extracts the feature online, and passes the same 64 BEV tokens to AutoMoT. |

Rows with `<bev>` are skipped only when both paths are unavailable. In either
mode, the current RGB frame is used only by the BEV encoder and is not inserted
as an additional AutoMoT image token. The four historical images remain part of
the Qwen3-VL reasoning context.

## BEV Encoder Feature Cache

If `route_features.pt` is not already present in your PDM-Lite root, you can
still train with the online BEV fallback. For faster normal training,
precompute one feature cache per PDM-Lite route with
`Automot/preprocess/extract_pdm_lite_bev_encoder_features.py`:

```bash
PYTHONPATH=Automot:Automot/mot python Automot/preprocess/extract_pdm_lite_bev_encoder_features.py \
  --pdm-root "$PDM_DATA_DIR" \
  --jsonl "$PDM_JSONL_DIR/pdm_lite_2hz_2tp_train_bev_encoder.jsonl" \
  --jsonl "$PDM_JSONL_DIR/pdm_lite_2hz_2tp_val_bev_encoder.jsonl" \
  --device cuda:0 \
  --batch-size 16
```

The extractor loads `bev_config.json` and the `bev_encoder.*` tensors from
`AUTOMOT_MODEL_PATH/model.safetensors`, so cached features use the same BEV
encoder weights as the released AutoMoT checkpoint. `--config-dir` and
`--checkpoint` can override these defaults for a compatible custom encoder.

The script writes:

```text
<PDM_DATA_DIR>/<scenario>/<route>/bev_encoder_feature/route_features.pt
```

Each `route_features.pt` contains:

- `frame_nums`: frame ids such as `0003`
- `bev_features`: `[num_frames, 1512, 8, 8]`
- `bev_upsamples`: `[num_frames, 64, 64, 64]` when the backbone returns it

The extractor reads `rgb/<frame>.jpg` and `bev_encoder_lidar_bev/<frame>.npy`.

## Training

Start training on one 80GB GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash Automot/scripts/train_automot.sh
```

For multi-GPU training, set `CUDA_VISIBLE_DEVICES` and `NUM_GPUS` consistently:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
bash Automot/scripts/train_automot.sh
```

For a short train/eval check:

```bash
CUDA_VISIBLE_DEVICES=0 \
TOTAL_STEPS=20 \
LOG_EVERY=10 \
DO_EVAL=True \
EVAL_EVERY=10 \
EVAL_MAX_STEPS=4 \
NUM_WORKERS=1 \
bash Automot/scripts/train_automot.sh
```

Successful runs print finite training loss. With `DO_EVAL=True`, validation
trajectory metrics are printed during evaluation.
