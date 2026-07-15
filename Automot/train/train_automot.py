import os
import sys
automot_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mot_root = os.path.join(automot_root, "mot")
for path in (automot_root, mot_root):
    if path not in sys.path:
        sys.path.insert(0, path)
transformers_fork = os.environ.get("TRANSFORMERS_FORK", "")
if transformers_fork:
    sys.path.insert(0, transformers_fork)
from transformers import AutoProcessor
import functools
import json
import yaml
import glob
import socket
import datetime
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
os.environ["SAFETENSORS_FAST"] = "0"
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

try:
    import wandb
except ImportError:
    class _NoOpWandB:
        class _Config:
            @staticmethod
            def update(*args, **kwargs):
                return None

        config = _Config()

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def define_metric(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def finish(*args, **kwargs):
            return None

    wandb = _NoOpWandB()

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from data.automot.dataset_base import DataConfig, PackedDataset, collate_wrapper
from mot.modeling.automot import (
    AutoMoTConfig, AutoMoT,
    Qwen3VLTextConfig, Qwen3VLForConditionalGenerationMoT
)
from mot.modeling.automot.checkpoint_utils import normalize_automot_state_dict

from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils_qwen3vl import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper,
    fsdp_ema_setup, fsdp_ema_update,
)


@dataclass
class ModelArguments:
    model_path: str = field(
        default=os.environ.get("AUTOMOT_MODEL_PATH", ""),
        metadata={"help": "Path to the AutoMoT checkpoint or HuggingFace cache."}
    )
    qwen3vl_path: str = field(
        default=os.environ.get("QWEN3VL_PATH", os.environ.get("AUTOMOT_MODEL_PATH", "Qwen/Qwen3-VL-4B-Instruct")),
        metadata={"help": "Path to the Qwen3VL base model for config loading"}
    )
    llm_path: str = field(
        default="Qwen/Qwen3-4B-Thinking-2507",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen3-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    mot_num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads for MoT attention components. Defaults to half of regular attention heads if not specified."}
    )
    mot_num_key_value_heads: int = field(
        default=4,
        metadata={"help": "Number of key-value heads for MoT attention components. Defaults to half of regular KV heads if not specified."}
    )
    mot_intermediate_size: int = field(
        default=4864,
        metadata={"help": "Intermediate size for MoT MLP components. Defaults to same as regular intermediate_size if not specified."}
    )
    tie_word_embeddings: bool = field(
        default=True,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen3VLDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vit_patch_size: int = field(
        default=16,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in connector MLPs."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    reasoning_query_dim: int = field(
        default=588,
        metadata={"help": "Dimension of the reasoning query embedding."}
    )
    reasoning_query_max_num_tokens: int = field(
        default=8,
        metadata={"help": "Maximum number of tokens in the reasoning query."}
    )
    action_query_dim: int = field(
        default=588,
        metadata={"help": "Dimension of the action query embedding."}
    )
    action_query_max_num_tokens: int = field(
        default=26,
        metadata={"help": "Maximum number of tokens in the action query."}
    )

@dataclass
class DataArguments:
    train_config_file: str = field(
        default="mot/data/automot/configs/automot_traj_train.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    eval_config_file: str = field(
        default="mot/data/automot/configs/automot_traj_eval.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )
    reasoning_expert_max_num_tokens: int = field(
        default=8,
        metadata={"help": "Maximum number of tokens in the reasoning query."}
    )

@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )
    traj_gen: bool = field(
        default=False,
        metadata={"help": "Train trajectory generation branch."}
    )
    fast_reasoning: bool = field(
        default=False,
        metadata={"help": "Train fast reasoning branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="automot",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="automot_train",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether do evaluation."}
    )
    eval_every: int = field(
        default=1000,
        metadata={"help": "Run evaluation every N training steps."}
    )
    eval_max_steps: int = field(
        default=167,
        metadata={"help": "Maximum validation batches per evaluation; non-positive values evaluate all batches."}
    )
    eval_use_ema: bool = field(
        default=True,
        metadata={"help": "Evaluate EMA weights when available."}
    )
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=1000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=10000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=1150,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=2e-5,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW beta1 coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW beta2 coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW epsilon for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    ce_weight: float = field(
        default=0.4,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    aux_ce_h_weight: float = field(
        default=0.4,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    aux_ce_v_weight: float = field(
        default=0.4,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    traj_weight: float = field(
        default=2.0,
        metadata={"help": "Scaling factor for the trajectory loss."}
    )
    velocity_weight: float = field(
        default=0.5,
        metadata={"help": "Scaling factor for the velocity loss."}
    )
    route_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the route loss."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=4,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=True,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    freeze_llm_gen: bool = field(
        default=False,
        metadata={"help": "Freeze LLM (Qwen) weights when training generation (trajectory/image) head only; keep new heads trainable."}
    )
    freeze_llm_und: bool = field(
        default=True,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    freeze_lidar: bool = field(
        default=True,
        metadata={"help": "freeze LiDAR branch (backbone + patch_embed)"}
    )
    train_waypoints_head_only: bool = field(
        default=False,
        metadata={"help": "Train only the waypoint prediction head."}
    )
    copy_init_mot: bool = field(
        default=False,
        metadata={"help": "Duplicate initial mot experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

def run_evaluation(
    fsdp_model,
    ema_model,
    val_loader,
    device,
    training_args,
    model_path,
    logger=None,
    curr_step=0,
):
    if val_loader is None:
        return {}

    model_to_eval = fsdp_model
    model_to_eval.eval()
    lm = model_to_eval.language_model
    lm.reasoning = True
    lm.model.set_reasoning_mode_all(True)

    ce_sum           = torch.zeros((), device=device, dtype=torch.float32)
    traj_sum         = torch.zeros((), device=device, dtype=torch.float32)
    route_sum        = torch.zeros((), device=device, dtype=torch.float32)
    velocity_sum     = torch.zeros((), device=device, dtype=torch.float32)
    l2_1s_sum        = torch.zeros((), device=device, dtype=torch.float32)
    l2_2s_sum        = torch.zeros((), device=device, dtype=torch.float32)
    l2_3s_sum        = torch.zeros((), device=device, dtype=torch.float32)
    cnt_sum          = torch.zeros((), device=device, dtype=torch.float32)

    def _to_scalar(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().to(torch.float32)
        if x.numel() != 1:
            x = x.mean()
        return x

    online_bev_encoder = None
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if training_args.eval_max_steps > 0 and i >= training_args.eval_max_steps:
                break

            data = data.cuda(device).to_dict()

            sample_lens = data.get("sample_lens", None)
            if sample_lens is not None:
                if torch.is_tensor(sample_lens):
                    num_samples = int(sample_lens.numel())
                else:
                    num_samples = int(len(sample_lens))
                w = torch.tensor(float(max(num_samples, 1)), device=device, dtype=torch.float32)
            else:
                w = torch.tensor(1.0, device=device, dtype=torch.float32)

            data.pop("batch_data_indexes", None)
            data.pop("ce_loss_weights", None)
            online_bev_encoder, data = prepare_bev_encoder_batch(
                data,
                online_bev_encoder,
                model_path,
                device,
                logger=logger,
            )

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                out = model_to_eval(**data)

            ce            = out.get("ce", None)
            traj          = out.get("traj_loss", None)
            route_loss    = out.get("route_loss", None)
            velocity_loss = out.get("velocity_loss", None)
            l2_1s         = out.get("l2_1s", None)
            l2_2s         = out.get("l2_2s", None)
            l2_3s         = out.get("l2_3s", None)

            if traj is None:
                continue

            if ce is not None:
                ce_sum += _to_scalar(ce) * w

            traj_sum += _to_scalar(traj) * w

            if route_loss is not None:
                route_sum += _to_scalar(route_loss) * w
            if velocity_loss is not None:
                velocity_sum += _to_scalar(velocity_loss) * w

            if l2_1s is not None:
                l2_1s_sum += _to_scalar(l2_1s) * w
            if l2_2s is not None:
                l2_2s_sum += _to_scalar(l2_2s) * w
            if l2_3s is not None:
                l2_3s_sum += _to_scalar(l2_3s) * w

            cnt_sum += w

    for t in [
        ce_sum,
        traj_sum,
        route_sum,
        velocity_sum,
        l2_1s_sum,
        l2_2s_sum,
        l2_3s_sum,
        cnt_sum,
    ]:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    metrics = {}
    if cnt_sum.item() > 0:
        if ce_sum.item() != 0:
            metrics["val/ce"] = (ce_sum / cnt_sum).item()

        metrics["val/traj_loss"]      = (traj_sum / cnt_sum).item()
        metrics["val/route_loss"]     = (route_sum / cnt_sum).item()
        metrics["val/velocity_loss"]  = (velocity_sum / cnt_sum).item()
        metrics["val/l2_1s"]          = (l2_1s_sum / cnt_sum).item()
        metrics["val/l2_2s"]          = (l2_2s_sum / cnt_sum).item()
        metrics["val/l2_3s"]          = (l2_3s_sum / cnt_sum).item()
        metrics["val/total_traj_weighted"] = metrics["val/traj_loss"] * training_args.traj_weight

    if dist.get_rank() == 0:
        msg = f"[EVAL step={curr_step}] " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        if logger:
            logger.info(msg)
        else:
            print(msg)

    return metrics


class ProgressTracker:
    """Track training progress and estimate remaining time"""
    def __init__(self, total_steps, log_every=10):
        self.total_steps = total_steps
        self.log_every = log_every
        self.start_time = time()
        self.last_update_time = time()
        self.step_times = []
        self.recent_step_times = []
        self.max_recent_times = 50  # Keep last 50 step times for better ETA estimation

    def update(self, current_step):
        """Update progress tracking"""
        current_time = time()

        if current_step > 0:
            # Calculate time since last update
            step_duration = (current_time - self.last_update_time) / self.log_every
            self.step_times.append(step_duration)
            self.recent_step_times.append(step_duration)

            # Keep only recent times for better ETA
            if len(self.recent_step_times) > self.max_recent_times:
                self.recent_step_times.pop(0)

        self.last_update_time = current_time

    def get_progress_info(self, current_step):
        """Get comprehensive progress information"""
        elapsed_time = time() - self.start_time
        progress_pct = (current_step / self.total_steps) * 100

        # Estimate remaining time using recent step times
        if len(self.recent_step_times) > 0:
            avg_step_time = sum(self.recent_step_times) / len(self.recent_step_times)
            remaining_steps = self.total_steps - current_step
            estimated_remaining_seconds = remaining_steps * avg_step_time

            # Format remaining time
            remaining_time_str = self._format_duration(estimated_remaining_seconds)

            # Estimate total time
            estimated_total_seconds = self.total_steps * avg_step_time
            total_time_str = self._format_duration(estimated_total_seconds)

            # Calculate current speed
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
        else:
            remaining_time_str = "Calculating..."
            total_time_str = "Calculating..."
            steps_per_sec = 0

        return {
            'current_step': current_step,
            'total_steps': self.total_steps,
            'progress_pct': progress_pct,
            'elapsed_time': self._format_duration(elapsed_time),
            'remaining_time': remaining_time_str,
            'estimated_total_time': total_time_str,
            'steps_per_sec': steps_per_sec,
            'eta': self._get_eta_timestamp(estimated_remaining_seconds if len(self.recent_step_times) > 0 else None)
        }

    def _format_duration(self, seconds):
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h{int(minutes)}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{int(days)}d{int(hours)}h"

    def _get_eta_timestamp(self, remaining_seconds):
        """Get estimated completion timestamp"""
        if remaining_seconds is None:
            return "Calculating..."

        eta_timestamp = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)
        return eta_timestamp.strftime("%Y-%m-%d %H:%M:%S")

def load_safetensors_weights(model_path):
    """Load weights from single or multiple safetensors files."""
    # Try single file first (like AutoMoT 2B)
    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        print(f"Loading from single file: {single_file}")
        return normalize_automot_state_dict(load_file(single_file))

    # Try multiple files (like Qwen3VL-4B)
    pattern = os.path.join(model_path, "model-*.safetensors")
    safetensor_files = sorted(glob.glob(pattern))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    print(f"Loading from multiple files: {safetensor_files}")
    combined_state_dict = {}

    for file_path in safetensor_files:
        file_state_dict = load_file(file_path)
        combined_state_dict.update(file_state_dict)
        print(f"Loaded {len(file_state_dict)} parameters from {os.path.basename(file_path)}")

    print(f"Total loaded parameters: {len(combined_state_dict)}")
    return normalize_automot_state_dict(combined_state_dict)


def _load_lidar_bev_npy(path, expected_channels):
    bev = np.load(path)
    if bev.ndim == 3 and bev.shape[0] not in (1, 2) and bev.shape[-1] in (1, 2):
        bev = np.transpose(bev, (2, 0, 1))
    if bev.ndim != 3:
        raise ValueError(f"Expected lidar BEV with shape [C,H,W], got {bev.shape} at {path}")
    if bev.shape[0] != expected_channels:
        if bev.shape[0] == 2 and expected_channels == 1:
            bev = bev[1:2]
        else:
            raise ValueError(f"Expected {expected_channels} lidar BEV channels, got {bev.shape[0]} at {path}")
    return torch.from_numpy(bev.astype(np.float32, copy=False)).unsqueeze(0)


@torch.no_grad()
def materialize_bev_encoder_features(bev_encoder, feature_inputs, device):
    features = []
    expected_channels = None
    for item in feature_inputs:
        if torch.is_tensor(item):
            feature = item
        elif isinstance(item, dict) and torch.is_tensor(item.get("feature")):
            feature = item["feature"]
        elif isinstance(item, dict):
            if bev_encoder is None:
                raise RuntimeError("Online BEV encoder input requires an initialized BEV encoder")
            if expected_channels is None:
                expected_channels = (
                    2 if bev_encoder.config.use_ground_plane else 1
                ) * bev_encoder.config.lidar_seq_len
            rgb = bev_encoder.preprocess_rgb(item["rgb_path"])
            lidar = _load_lidar_bev_npy(item["lidar_bev_path"], expected_channels)
            outputs = bev_encoder(rgb, lidar)
            feature = outputs["bev_feature"]
            if feature is None:
                raise RuntimeError("BEV encoder did not return bev_feature")
        else:
            raise TypeError(f"Unsupported BEV encoder feature input: {type(item)!r}")

        if feature.dim() == 4 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        elif feature.dim() != 3:
            raise ValueError(f"Expected BEV encoder feature [1512,8,8], got {tuple(feature.shape)}")
        features.append(feature.to(device=device, dtype=torch.bfloat16))
    return torch.stack(features, dim=0)


def prepare_bev_encoder_batch(data, online_bev_encoder, model_path, device, logger=None):
    feature_inputs = data.pop("bev_encoder_feature_inputs", None)
    if not feature_inputs:
        return online_bev_encoder, data

    needs_online_bev_encoder = any(
        isinstance(item, dict) and not torch.is_tensor(item.get("feature"))
        for item in feature_inputs
    )
    if needs_online_bev_encoder and online_bev_encoder is None:
        from mot.modeling.bev_encoder.backbone_extractor import BEVEncoderBackboneExtractor

        online_bev_encoder = BEVEncoderBackboneExtractor(
            config_path=model_path,
            model_path=os.path.join(model_path, "model.safetensors"),
            device=f"cuda:{device}" if isinstance(device, int) else str(device),
        )
        online_bev_encoder.eval()
        for param in online_bev_encoder.parameters():
            param.requires_grad = False
        if logger is not None and dist.get_rank() == 0:
            logger.info("Online BEV encoder fallback initialized from model_path.")

    data["bev_encoder_feature"] = materialize_bev_encoder_features(
        online_bev_encoder,
        feature_inputs,
        device,
    )
    return online_bev_encoder, data

def count_component_parameters(model):
    llm_params = 0
    mot_params = 0
    vit_params = 0
    other_params = 0

    llm_trainable = 0
    mot_trainable = 0
    vit_trainable = 0
    other_trainable = 0

    print("\n" + "="*100)
    print("PARAMETER COUNT BY COMPONENT")
    print("="*100)
    print(f"{'Component':<15} {'Parameter name':<45} {'Shape':<20} {'Trainable':<10} {'Count':<12}")
    print("-" * 102)

    for name, param in model.named_parameters():
        n_param = param.numel()
        is_trainable = param.requires_grad

        if 'language_model' in name:
            if any(mot_keyword in name for mot_keyword in ['mot_gen', '_gen']):
                mot_params += n_param
                if is_trainable:
                    mot_trainable += n_param
            else:
                llm_params += n_param
                if is_trainable:
                    llm_trainable += n_param
        elif 'vision_model' in name or 'vit' in name.lower() or 'visual' in name:
            vit_params += n_param
            if is_trainable:
                vit_trainable += n_param
        else:
            other_params += n_param
            if is_trainable:
                other_trainable += n_param

    total_params = llm_params + mot_params + vit_params + other_params
    total_trainable = llm_trainable + mot_trainable + vit_trainable + other_trainable

    print("\nCOMPONENT PARAMETER SUMMARY:")
    print(f"LLM Parameters:   {llm_params:>12,} ({llm_params/1e9:.3f}B) | Trainable: {llm_trainable:>12,} ({llm_trainable/1e9:.3f}B)")
    print(f"mot Parameters:   {mot_params:>12,} ({mot_params/1e9:.3f}B) | Trainable: {mot_trainable:>12,} ({mot_trainable/1e9:.3f}B)")
    print(f"ViT Parameters:   {vit_params:>12,} ({vit_params/1e9:.3f}B) | Trainable: {vit_trainable:>12,} ({vit_trainable/1e9:.3f}B)")
    print(f"Other Parameters: {other_params:>12,} ({other_params/1e9:.3f}B) | Trainable: {other_trainable:>12,} ({other_trainable/1e9:.3f}B)")
    print(f"Total Parameters: {total_params:>12,} ({total_params/1e9:.3f}B) | Trainable: {total_trainable:>12,} ({total_trainable/1e9:.3f}B)")

    return {
        'llm': {'total': llm_params, 'trainable': llm_trainable},
        'mot': {'total': mot_params, 'trainable': mot_trainable},
        'vit': {'total': vit_params, 'trainable': vit_trainable},
        'other': {'total': other_params, 'trainable': other_trainable},
        'total': {'total': total_params, 'trainable': total_trainable}
    }

def convert_model_dtype_with_exceptions(model, target_dtype, exclude_buffer_patterns=None):

    if exclude_buffer_patterns is None:
        exclude_buffer_patterns = []

    for name, param in model.named_parameters():
        param.data = param.data.to(target_dtype)

    for name, buffer in model.named_buffers():
        should_exclude = any(pattern in name for pattern in exclude_buffer_patterns)

        if should_exclude:
            print(f"Skipped buffer: {name} (kept as {buffer.dtype})")
        else:
            buffer.data = buffer.data.to(target_dtype)
            print(f"Converted buffer: {name} to {target_dtype}")

    return model

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert torch.cuda.is_available()
    hostname = socket.gethostname()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    print(f"[{hostname}] Starting process with LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}")
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online"
        )

        wandb.define_metric("train/step")
        wandb.define_metric("val/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="val/step")

        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
        # Log multi-node setup info
        logger.info("Multi-node training setup:")
        logger.info(f"  Total nodes: {world_size // torch.cuda.device_count()}")
        logger.info(f"  Total processes: {world_size}")
        logger.info(f"  GPUs per node: {torch.cuda.device_count()}")

        # Initialize progress tracker
        progress_tracker = ProgressTracker(training_args.total_steps, training_args.log_every)
        logger.info(f"Progress tracking initialized for {training_args.total_steps} steps")
    else:
        logger = create_logger(None, dist.get_rank())
        progress_tracker = None
    dist.barrier()

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    qwen3vl_config_path = model_args.qwen3vl_path
    with open(f"{qwen3vl_config_path}/config.json", "r") as f:
        full_config = json.load(f)
    text_config_dict = full_config["text_config"]
    vision_config_dict = full_config["vision_config"]
    llm_config = Qwen3VLTextConfig(**text_config_dict)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und

    # MoT parameter resize configuration
    if model_args.mot_num_attention_heads is not None:
        llm_config.mot_num_attention_heads = model_args.mot_num_attention_heads
    if model_args.mot_num_key_value_heads is not None:
        llm_config.mot_num_key_value_heads = model_args.mot_num_key_value_heads
    if model_args.mot_intermediate_size is not None:
        llm_config.mot_intermediate_size = model_args.mot_intermediate_size

    if training_args.finetune_from_hf:
        language_model = Qwen3VLForConditionalGenerationMoT(llm_config)
    else:
        language_model = Qwen3VLForConditionalGenerationMoT.from_pretrained(
        model_args.llm_path,
        config=llm_config,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        )

    if training_args.copy_init_mot:
        language_model.init_mot()

    vit_config = Qwen3VLVisionConfig(**vision_config_dict)
    vision_model = None
    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vision_model = Qwen3VLVisionModel(vit_config)
        else:
            vision_model = Qwen3VLVisionModel.from_pretrained(model_args.vit_path, config=vit_config)
        vision_model.eval()

    config = AutoMoTConfig(
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vision_config=vit_config,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        reasoning_query_dim=model_args.reasoning_query_dim,
        reasoning_query_tokens=model_args.reasoning_query_max_num_tokens,
        action_query_dim=model_args.action_query_dim,
        action_query_tokens=model_args.action_query_max_num_tokens,
    )

    model = AutoMoT(
        language_model,
        vision_model,
        config
    )

    state_dict = load_safetensors_weights(model_args.model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    actual_missing_keys = [k for k in missing_keys if k != 'language_model.lm_head.weight']
    expected_bev_encoder_keys = [k for k in unexpected_keys if k.startswith("bev_encoder.")]
    actual_unexpected_keys = [k for k in unexpected_keys if not k.startswith("bev_encoder.")]
    actual_unexpected_keys = [k for k in actual_unexpected_keys if k != 'language_model.lm_head.weight']
    print(
        f"Loaded AutoMoT weights: {len(actual_missing_keys)} missing, "
        f"{len(actual_unexpected_keys)} unexpected"
    )

    if actual_missing_keys:
        print(f"Missing keys: {actual_missing_keys[:10]}")
    if actual_unexpected_keys:
        print(f"Unexpected keys: {actual_unexpected_keys[:5]}")
    if expected_bev_encoder_keys:
        print(
            "Skipped BEV encoder backbone weights at main-model load. "
            "They are not needed when training from precomputed BEV encoder features; "
            "the online fallback loads them separately when raw RGB/lidar BEV inputs are used."
        )

    processor = AutoProcessor.from_pretrained(model_args.qwen3vl_path)
    tokenizer = processor.tokenizer
    model.tokenizer = tokenizer

    model = convert_model_dtype_with_exceptions(
        model,
        torch.bfloat16,
        exclude_buffer_patterns=['inv_freq']
    )

    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit:
        model.vision_model.eval()
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if training_args.freeze_llm:
        for name, param in model.language_model.named_parameters():
            if "mot" in name:
                param.requires_grad = not training_args.freeze_llm_gen
            else:
                param.requires_grad = False

        for name, module in model.language_model.named_modules():
            if "mot" in name:
                if training_args.freeze_llm_gen:
                    module.eval()
                else:
                    module.train()
            else:
                module.eval()
    else:
        if training_args.freeze_llm_gen:
            for name, param in model.language_model.named_parameters():
                if "mot" in name:
                    param.requires_grad = False

    if training_args.freeze_llm_und:
        llm = model.language_model

        SHARED_KEYS = ("embed_tokens", "rotary_emb", "lm_head")

        BASE_KEYS = (
            "mlp", "input_layernorm", "post_attention_layernorm",
            "q_proj", "k_proj", "v_proj", "o_proj",
            "q_norm", "k_norm", "norm",
        )

        for name, param in llm.named_parameters():
            if any(k in name for k in SHARED_KEYS):
                param.requires_grad = False
                continue
            if any(f".{k}." in name or name.endswith(f".{k}.weight") or name.endswith(f".{k}.bias")
                for k in BASE_KEYS):
                param.requires_grad = False
            if "_mot_gen" in name:
                param.requires_grad = True
                continue
            else:
                param.requires_grad = False
    if training_args.train_waypoints_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.waypoints_head.named_parameters():
            param.requires_grad = True

        print("[INFO] Training waypoints head only.")

    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    if training_args.copy_init_mot:
        print("Initializing MoT weights from the first transformer block.")
        model.language_model.init_mot()

    print("\n" + "="*80)
    print("PARAMETER COUNTING (BEFORE FSDP WRAPPING)")
    print("="*80)
    count_component_parameters(model)

    fsdp_ignored_modules = []
    ema_fsdp_ignored_modules = []
    if training_args.freeze_vit and getattr(model, "vision_model", None) is not None:
        ignored_module_device = torch.device("cuda", device)
        model.vision_model.to(device=ignored_module_device)
        ema_model.vision_model.to(device=ignored_module_device)
        fsdp_ignored_modules.append(model.vision_model)
        ema_fsdp_ignored_modules.append(ema_model.vision_model)

    ema_model = fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=ema_fsdp_ignored_modules)
    fsdp_model = fsdp_wrapper(model, fsdp_config, ignored_modules=fsdp_ignored_modules)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn
    )

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # Restore optimizer and scheduler state when resuming a full checkpoint.
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config,
        )

    # Setup packed dataloader
    with open(data_args.train_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

    dataset_config = DataConfig(grouped_datasets=dataset_meta)

    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=None,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        reasoning_text_max_num_tokens=model_args.reasoning_query_max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )

    train_dataset.set_epoch(data_args.data_seed)
    dataloader_kwargs = {}
    if data_args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = data_args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        **dataloader_kwargs,
    )

    # Prepare models for training:
    fsdp_model.train()
    ema_model.eval()

    val_loader = None
    if training_args.do_eval:
        with open(data_args.eval_config_file, "r") as stream:
            dataset_meta = yaml.safe_load(stream)

        val_dataset_config = DataConfig(grouped_datasets=dataset_meta)

        if training_args.visual_und:
            val_dataset_config.vit_patch_size = model_args.vit_patch_size
            val_dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side

        val_dataset = PackedDataset(
            val_dataset_config,
            tokenizer=tokenizer,
            special_tokens=None,
            local_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            num_workers=data_args.num_workers,
            expected_num_tokens=training_args.expected_num_tokens,
            max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
            max_num_tokens=data_args.max_num_tokens,
            reasoning_text_max_num_tokens=model_args.reasoning_query_max_num_tokens,
            max_buffer_size=data_args.max_buffer_size,
            prefer_buffer_before=data_args.prefer_buffer_before,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
            data_status=None,
        )
        val_dataset.set_epoch(0)

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=data_args.num_workers,
            pin_memory=True,
            collate_fn=collate_wrapper(),
            drop_last=False,
            **dataloader_kwargs,
        )

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    online_bev_encoder = None
    for curr_step, data in enumerate(train_loader, start=train_step):
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        online_bev_encoder, data = prepare_bev_encoder_batch(
            data,
            online_bev_encoder,
            model_args.model_path,
            device,
            logger=logger,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**data)


        loss = 0
        ce = loss_dict.get("ce", None)
        traj_loss = loss_dict.get("traj_loss", None)
        route_loss = loss_dict.get("route_loss", None)
        velocity_loss= loss_dict.get("velocity_loss", None)
        if ce is not None:
            labels = data['packed_label_ids']
            labels = labels.to(device) if torch.is_tensor(labels) else torch.as_tensor(labels, device=device)
            total_ce_tokens = (labels != -100).sum()
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)
        if traj_loss is not None:
            loss = loss + traj_loss * training_args.traj_weight
        if route_loss is not None:
            loss = loss + route_loss * training_args.route_weight
        if velocity_loss is not None:
            loss = loss + velocity_loss * training_args.velocity_weight
        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        # Log loss values:
        if curr_step % training_args.log_every == 0:

            if dist.get_rank() == 0 and progress_tracker is not None:
                progress_tracker.update(curr_step)
                progress_info = progress_tracker.get_progress_info(curr_step)
            else:
                progress_info = {'progress_pct': (curr_step / training_args.total_steps) * 100,
                               'elapsed_time': 'N/A', 'remaining_time': 'N/A', 'eta': 'N/A', 'steps_per_sec': 0}
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = ""
            if dist.get_rank() == 0:
                message = f"(step={curr_step:07d}/{training_args.total_steps}) [{progress_info['progress_pct']:.1f}%] "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            if dist.get_rank() == 0:
                message += f"Steps/Sec: {steps_per_sec:.2f}, "
                message += f"Elapsed: {progress_info['elapsed_time']}, "
                message += f"Remaining: {progress_info['remaining_time']}, "
                message += f"ETA: {progress_info['eta']}"
                logger.info(message)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['progress_pct'] = progress_info['progress_pct']
            wandb_log['steps_per_sec'] = progress_info['steps_per_sec']
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
        if training_args.do_eval and (curr_step > 0) and (curr_step % training_args.eval_every == 0):
            dist.barrier()
            eval_metrics = run_evaluation(
                fsdp_model=fsdp_model,
                ema_model=ema_model,
                val_loader=val_loader,
                device=device,
                training_args=training_args,
                model_path=model_args.model_path,
                logger=logger if dist.get_rank() == 0 else None,
                curr_step=curr_step,
            )

            if dist.get_rank() == 0 and eval_metrics:
                eval_metrics = { (k if k.startswith("val/") else f"val/{k}") : v
                                for k, v in eval_metrics.items() }
                eval_metrics["val/step"] = curr_step
                wandb.log(eval_metrics, commit=True)

            dist.barrier()
            fsdp_model.train()
            ema_model.eval()

        if curr_step + 1 >= training_args.total_steps:
            break

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
