import copy
from typing import List, Tuple, Optional, Dict, Any
import re
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask
from data.reasoning.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    add_special_tokens,
    prepare_attention_mask_per_sample,
)
from .qwen3vl_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

import sys
# sys.path.insert(0, ...)  # Removed: use pip-installed transformers
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer as Qwen3Tokenizer
import os

# Qwen3VL model paths - centralized configuration
# Tokenizer path: contains tokenizer files (tokenizer.json, vocab.json, etc.)
QWEN3VL_TOKENIZER_PATH = None  # Set via ModelArguments.model_path
# Processor path: contains proper Qwen3VL config with model_type
QWEN3VL_PROCESSOR_PATH = None  # Set via ModelArguments.qwen3vl_path
# For backward compatibility
QWEN3VL_MODEL_PATH = QWEN3VL_TOKENIZER_PATH

# Auto-detect paths if not explicitly set
_automot_dir = os.path.dirname(os.path.abspath(__file__))
# automot.py is at Automot/mot/modeling/automotive/automot.py
# Automot root is 3 levels up
_mot_dp_root = os.path.dirname(os.path.dirname(os.path.dirname(_automot_dir)))

if QWEN3VL_TOKENIZER_PATH is None:
    _default_tokenizer_path = os.path.join(_mot_dp_root, "checkpoints", "mot", "0025000")
    if os.path.isdir(_default_tokenizer_path):
        QWEN3VL_TOKENIZER_PATH = _default_tokenizer_path

if QWEN3VL_PROCESSOR_PATH is None:
    _default_processor_path = os.path.join(_mot_dp_root, "checkpoints")
    if os.path.isdir(_default_processor_path) and os.path.isfile(os.path.join(_default_processor_path, "preprocessor_config.json")):
        QWEN3VL_PROCESSOR_PATH = _default_processor_path

# Use local Qwen3VL tokenizer
# Workaround for HuggingFace validation error with local paths
# Temporarily disable repo_id validation by monkey-patching
import huggingface_hub.utils._validators as validators
original_validate_repo_id = validators.validate_repo_id

def patched_validate_repo_id(repo_id):
    # Skip validation if it looks like a local path
    if repo_id and (repo_id.startswith('/') or repo_id.startswith('./')):
        return
    return original_validate_repo_id(repo_id)

validators.validate_repo_id = patched_validate_repo_id

if QWEN3VL_TOKENIZER_PATH is not None:
    try:
        tokenizer = Qwen3Tokenizer.from_pretrained(QWEN3VL_TOKENIZER_PATH, local_files_only=True, trust_remote_code=True)
    finally:
        # Restore original validation
        validators.validate_repo_id = original_validate_repo_id
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
else:
    validators.validate_repo_id = original_validate_repo_id
    tokenizer = None
    new_token_ids = None
    num_new_tokens = 0

class AutoMoTConfig(PretrainedConfig):
    """
    AutoMoT Configuration for Qwen3VL integration.
    
    This configuration adapts the original AutoMoTive config to work with 
    Qwen3VL's vision model and qwen3vl_navit text processing.
    """
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vision_config=None,  # Changed from vit_config to vision_config
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        # Qwen3VL specific vision parameters
        vision_spatial_merge_size=2,
        vision_max_num_patches=4096,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        num_waypoints=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vision_config = vision_config  # Qwen3VL vision config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        # self._attn_implementation = "sdpa"  # default attn implementation
        # Qwen3VL vision specific
        self.vision_spatial_merge_size = vision_spatial_merge_size
        self.vision_max_num_patches = vision_max_num_patches
        
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.num_waypoints = num_waypoints
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """
    
    def __init__(
        self, 
        token_size: int = 2560,
        hidden_size: int = 256,
        hidden_size2: int = 512,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer
        
        # MLP: 2 -> 256 -> 512 -> 2560
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),            # 2 -> 256
            nn.ReLU(True), 
            nn.Linear(hidden_size, hidden_size2), # 256 -> 512
            nn.ReLU(True), 
            nn.Linear(hidden_size2, token_size)   # 512 -> 2560
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        Returns:
            Output with dims [B, N, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)
        return x

class RouteHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int = 256,
        future_waypoints: int = 20,
    ):
        super().__init__()
        self.future_waypoints = future_waypoints

        # learnable queries
        self.query = nn.Parameter(
            0.02 * torch.randn(1, future_waypoints, hidden_size)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim * 2),
            nn.SiLU(True),
            nn.Linear(mlp_dim * 2, mlp_dim),
            nn.SiLU(True),
            nn.Linear(mlp_dim, 2, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, W, hidden_size)
        Returns:
            route: (B, W, 2)
        """
        # delta -> absolute via cumsum
        route = self.mlp(features).cumsum(dim=1)
        return route

    def build_queries(self, batch_size: int) -> torch.Tensor:
        """
        Returns:
            (B, W, hidden_size)
        """
        return self.query.expand(batch_size, -1, -1)


class WaypointsHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int = 512,
        mlp_hidden: int = 256,
        num_waypoints: int = 6,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints

        self.query = nn.Parameter(
            0.02 * torch.randn(1, num_waypoints, hidden_size)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.SiLU(True),
            nn.Linear(mlp_dim, mlp_hidden),
            nn.SiLU(True),
            nn.Linear(mlp_hidden, 2, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, hidden_size)
        Returns:
            waypoints: (B, T, 2)
        """
        waypoints = self.mlp(features)
        return waypoints

    def build_queries(self, batch_size: int) -> torch.Tensor:
        return self.query.expand(batch_size, -1, -1)


#class AutoMoT(PreTrainedModel):
class AutoMoT(Qwen3VLPreTrainedModel):
    """AutoMoT model using Qwen3VL's pretrained vision-text alignment."""
    config_class = AutoMoTConfig
    base_model_prefix = 'automot'

    def __init__(self, language_model, vision_model, config: AutoMoTConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_mot = "MoT" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self.route_head = RouteHead(hidden_size=self.hidden_size)
        self.target_point_encoder = WaypointInputAdaptor(token_size=self.hidden_size)
        
        # TransFuser projector: 1512 -> hidden_size (2560)
        self.bev_encoder_proj = nn.Linear(1512, self.hidden_size, bias=True)
        
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, self.hidden_size),
        )
        self.waypoints_head = WaypointsHead(hidden_size=self.hidden_size)
        self.velocity_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_size, 3),
        )
        self.start_id = int(tokenizer.encode('<|im_start|>', add_special_tokens=False)[-1])
        self.end_id = int(tokenizer.encode('<|im_end|>', add_special_tokens=False)[-1])
        self.comma_id = int(tokenizer.encode(',', add_special_tokens=False)[-1])
        print("start id, end id, comma id are:", self.start_id, self.end_id, self.comma_id)
        self.user_prompt = "<|im_start|>user\n"
        self.assistant_prompt = "<|im_end|>\n<|im_start|>assistant"
        self.generation_start_token = "\n"

        if config.visual_gen:
            self.vision_model = vision_model
            self.reasoning_query_dim = config.reasoning_query_dim
            self.reasoning_query_tokens = config.reasoning_query_tokens
            self.reasoning_queries = nn.Embedding(
                num_embeddings=self.reasoning_query_tokens,
                embedding_dim=self.reasoning_query_dim,
            )
            self.reasoning_projector = MLPconnector(self.reasoning_query_dim, self.hidden_size, config.connector_act)
            self.action_query_dim = config.action_query_dim
            self.action_query_tokens = config.action_query_tokens
            self.route_queries = nn.Embedding(
                num_embeddings=20,
                embedding_dim=self.action_query_dim,
            )
            self.route_projector = MLPconnector(self.action_query_dim, self.hidden_size, config.connector_act)
            self.waypoint_queries = nn.Embedding(
                num_embeddings=6,
                embedding_dim=self.action_query_dim,
            )
            self.waypoint_projector = MLPconnector(self.action_query_dim, self.hidden_size, config.connector_act)
        if config.visual_und:
            # Qwen3VL vision model with pretrained alignment
            # No connector or position embedding needed - already handled internally!
            self.vision_model = vision_model  # Qwen3VLVisionModel
            
            # Create official Qwen3VL processor using AutoProcessor
            from transformers import AutoProcessor
            self.vision_processor = AutoProcessor.from_pretrained(QWEN3VL_PROCESSOR_PATH, local_files_only=True, trust_remote_code=True) if QWEN3VL_PROCESSOR_PATH else None


        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def ce_from_dict(self, logits: torch.Tensor, prob_dict: dict, order=("accelerate", "constant", "slow")):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        device, dtype = logits.device, logits.dtype
        t = torch.tensor([float(prob_dict.get(k.strip().lower(), 0.0)) for k in order],
                        device=device, dtype=dtype)
        t = torch.clamp(t, min=1e-8)
        t = t / t.sum()
        log_q = F.log_softmax(logits, dim=-1)
        return -(t * log_q).sum(dim=-1).mean() 

    def scores_from_start(self, logp, t_start, BANK, alpha=1.0, ce_quota: int = 10):
        if t_start is None:
            return None
        hard_cap = logp.size(0)
        if ce_quota is not None and ce_quota > 0:
            hard_cap = min(hard_cap, t_start + ce_quota)

        outs = []
        for ids in BANK:
            Lk = len(ids)
            if t_start + Lk > hard_cap:
                outs.append(logp.new_tensor(-1e9))
                continue
            lp = 0.0
            for i, tid in enumerate(ids):
                lp += logp[t_start + i, tid]
            outs.append(lp / (Lk ** alpha)) 
        return torch.stack(outs, dim=0)

    def vad_traj_loss(self, pred_offset, gt_offset, lam_pos_1s: float = 0.2):
        """
        pred_offset, gt_offset: (B, 6, 2)
        weight scheme: 332211 over the 6 timesteps
        """
        assert pred_offset.shape == gt_offset.shape
        assert gt_offset.dim() == 3 and gt_offset.size(1) == 6 and gt_offset.size(2) == 2

        # (6,) -> (1,6,1) -> broadcast to (B,6,2)
        w = gt_offset.new_tensor([3, 3, 2, 2, 1, 1]).view(1, 6, 1)
        
        l1 = (pred_offset - gt_offset).abs() * w
        avg_factor = w.sum() * gt_offset.size(0) * gt_offset.size(2)  # B * sum(w) * 2
        loss = l1.sum() / avg_factor.clamp(min=1.0)
        return loss

    def ade_fde_loss(
        self,
        pred,              # (B, T, 2)
        gt,                # (B, T, 2)
        mask=None,
        n_1s=2,
        n_2s=4,
        n_3s=6,
    ):
        """
        Trajectory loss with per-second L2 metrics.
        
        Args:
            pred: (B, T, 2) predicted trajectory
            gt: (B, T, 2) ground truth trajectory
            mask: optional (B, T) validity mask
            n_1s, n_2s, n_3s: number of points for 1s, 2s, 3s
        
        Returns:
            loss: scalar (ADE)
            l2_1s: L2 error for first 1s
            l2_2s: L2 error for first 2s
            l2_3s: L2 error for full 3s
        """
        disp = torch.norm(pred - gt, dim=-1)  # (B, T)
        B, T = disp.shape
        
        n_1s = min(n_1s, T)
        n_2s = min(n_2s, T)
        n_3s = min(n_3s, T)
        
        disp_1s = disp[:, :n_1s]  # (B, 4)
        disp_2s = disp[:, :n_2s]  # (B, 8)
        disp_3s = disp[:, :n_3s]  # (B, 12)
        
        if mask is not None:
            mask = mask.float()
            denom = mask.sum().clamp(min=1.0)
            ade = (disp * mask).sum() / denom
            
            m1 = mask[:, :n_1s]
            m2 = mask[:, :n_2s]
            m3 = mask[:, :n_3s]
            l2_1s = (disp_1s * m1).sum() / m1.sum().clamp(min=1.0)
            l2_2s = (disp_2s * m2).sum() / m2.sum().clamp(min=1.0)
            l2_3s = (disp_3s * m3).sum() / m3.sum().clamp(min=1.0)
        else:
            ade = disp.mean()
            l2_1s = disp_1s.mean()
            l2_2s = disp_2s.mean()
            l2_3s = disp_3s.mean()
        return l2_1s, l2_2s, l2_3s

    def position_to_offset(self, traj_gt: torch.Tensor):
        offset = traj_gt.clone()
        offset[..., 1:, :] = traj_gt[..., 1:, :] - traj_gt[..., :-1, :]
        offset[..., 0, :] = traj_gt[..., 0, :]
        return offset

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        ce_loss_weights: Optional[torch.BoolTensor] = None,
        traj_loss_indexes: Optional[torch.LongTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_und_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_und_text_indexes: Optional[torch.LongTensor] = None,
        packed_gen_text_indexes: Optional[torch.LongTensor] = None,
        packed_reasoning_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        traj_gt: Optional[torch.Tensor] = None,
        route_gt: Optional[torch.Tensor] = None,
        # for visual generation
        packed_action_token_indexes: Optional[torch.LongTensor] = None,
        route_loss_indexes: Optional[torch.LongTensor] = None,
        v_indexes: Optional[torch.LongTensor] = None,
        future_speeds_tensors: Optional[torch.Tensor] = None,
        target_point_indexes: Optional[torch.LongTensor] = None,
        action_query_token_seqlens: List[int] = None,
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        image_tensor_list: Optional[torch.Tensor] = None,
        image_grid_thw_list: Optional[torch.Tensor] = None,
        v_target_point: Optional[torch.Tensor] = None,
        probs: Optional[List[Dict[str, Any]]] = None,
        ### for bev encoder tokens
        bev_feature: Optional[torch.Tensor] = None,
        packed_bev_indexes: Optional[torch.LongTensor] = None,

    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
            
            bev_feature: [B, 1512, 8, 8] BEV encoder feature
            packed_bev_indexes: [B*64] indexes for 64 spatial tokens per sample
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed , deepstack_image_embeds = self.get_image_features(image_tensor_list, image_grid_thw_list)
            packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
            deepstack_visual_embeds = deepstack_image_embeds
            visual_pos_masks = torch.zeros(
                packed_sequence.shape[0], 
                dtype=torch.bool, 
                device=packed_sequence.device
            )
            visual_pos_masks[packed_vit_token_indexes] = True

        # TransFuser feature processing
        if self.config.visual_und:
            if bev_feature is not None and packed_bev_indexes is not None:
                
                x = bev_feature
                print("TransFuser feature shape:", x.shape)
                if x.dim() == 4:
                    B, C, H, W = x.shape
                    assert C == 1512, f"expect 1512 channels, got {C}"
                    assert H * W == 64, f"expect 8x8=64 spatial positions, got {H}x{W}={H*W}"
                    
                    # [B, 1512, 8, 8] -> [B, 1512, 64] -> [B, 64, 1512]
                    x = x.flatten(2).transpose(1, 2)  # [B, 64, 1512]
                    
                    # [B, 64, 1512] -> [B*64, 1512]
                    bev_tok = x.reshape(-1, 1512)  # [B*64, 1512]
                    
                    bev_tok = self.bev_encoder_proj(bev_tok)  # [B*64, 2560]

                elif x.dim() == 3:
                    B, N, C = x.shape
                    assert N == 64 and C == 1512, f"expect [B, 64, 1512], got [{B}, {N}, {C}]"
                    bev_tok = x.reshape(-1, 1512)  # [B*64, 1512]
                    bev_tok = self.bev_encoder_proj(bev_tok)  # [B*64, 2560]

                elif x.dim() == 2:
                    N, C = x.shape
                    assert C == 1512, f"expect 1512 dim, got {C}"
                    bev_tok = self.bev_encoder_proj(x)  # [N, 2560]

                else:
                    raise ValueError(f"bev_feature dim must be 2/3/4, got {x.dim()}")

                bev_tok = bev_tok.to(device=packed_sequence.device, dtype=packed_sequence.dtype)
                packed_bev_indexes = packed_bev_indexes.to(device=packed_sequence.device)

                assert packed_bev_indexes.ndim == 1
                assert bev_tok.shape[0] == packed_bev_indexes.shape[0], \
                    f"len mismatch: bev_tok={bev_tok.shape[0]} vs packed_bev_indexes={packed_bev_indexes.shape[0]}"
                assert bev_tok.shape[1] == packed_sequence.shape[1], \
                    f"hidden mismatch: bev_tok={bev_tok.shape[1]} vs hidden={packed_sequence.shape[1]}"

                packed_sequence[packed_bev_indexes] = bev_tok

        if target_point_indexes is not None and v_target_point is not None:
            # v_target_point: (B, 5) = [v, tp1_x, tp1_y, tp2_x, tp2_y]
            assert v_target_point.dim() == 2 and v_target_point.size(1) >= 5, \
                f"v_target_point shape invalid: {tuple(v_target_point.shape)}"

            assert target_point_indexes.numel() % 2 == 0, \
                f"target_point_indexes must be even, got {target_point_indexes.numel()}"

            B_v = v_target_point.size(0)
            B_tp = target_point_indexes.numel() // 2

            assert B_v == B_tp, \
                f"Mismatch: v_target_point B={B_v} vs target_point_indexes B={B_tp} (len={target_point_indexes.numel()})"

            # (B, 4) -> (B, 2, 2)
            target_points = v_target_point[:, 1:5].reshape(B_v, 2, 2)
            # (B, 2, C)
            target_point_embed = self.target_point_encoder(target_points)
            # (B*2, C)
            packed_target_point_embed = target_point_embed.reshape(-1, target_point_embed.size(-1))
            packed_sequence[target_point_indexes] = packed_target_point_embed


        # ----- velocity: 1 token per sample -----
        if v_indexes is not None and v_target_point is not None:
            assert v_target_point.dim() == 2 and v_target_point.size(1) >= 1, \
                f"v_target_point shape invalid: {tuple(v_target_point.shape)}"

            B_v = v_target_point.size(0)
            B_idx = v_indexes.numel()

            assert B_v == B_idx, \
                f"Mismatch: v_target_point B={B_v} vs v_indexes B={B_idx} (len={v_indexes.numel()})"

            velocity = v_target_point[:, 0:1]  # (B, 1)
            velocity_embed = self.velocity_encoder(velocity)  # (B, C)

            packed_sequence[v_indexes] = velocity_embed


        if self.config.visual_gen:
            if packed_reasoning_token_indexes is not None:
                batch_query_count = packed_reasoning_token_indexes.shape[0]
                batch_size = batch_query_count // self.reasoning_query_tokens
                reasoning_tokens = self.reasoning_queries(torch.arange(self.reasoning_query_tokens, device=packed_sequence.device))
                packed_reasoning_tokens = reasoning_tokens.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, reasoning_tokens.shape[-1])
                packed_reasoning_query_embed = self.reasoning_projector(packed_reasoning_tokens)
                packed_sequence[packed_reasoning_token_indexes] = packed_reasoning_query_embed

        if route_loss_indexes is not None:
            batch_query_count = route_loss_indexes.shape[0]
            batch_size = batch_query_count // 20
            
            query_ids = torch.arange(20, device=packed_sequence.device)
            route_tokens = self.route_queries(query_ids)
            packed_route_tokens = route_tokens.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, route_tokens.shape[-1])
            packed_route_query_embed = self.route_projector(packed_route_tokens)
            packed_sequence[route_loss_indexes] = packed_route_query_embed   
        if traj_loss_indexes is not None:
            batch_query_count = traj_loss_indexes.shape[0]
            batch_size = batch_query_count // 6
            
            query_ids = torch.arange(6, device=packed_sequence.device)
            waypoint_tokens = self.waypoint_queries(query_ids)
            packed_waypoint_tokens = waypoint_tokens.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, waypoint_tokens.shape[-1])
            packed_waypoint_query_embed = self.waypoint_projector(packed_waypoint_tokens)
            packed_sequence[traj_loss_indexes] = packed_waypoint_query_embed         
        ### We should divide packed_vit_token_indexes into to two parts. 5 images for 1st transformer and 1 image and 1 lidar for 2nd transformer
        extra_inputs = {}
        if self.use_mot:
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
                packed_gen_token_indexes=torch.cat([packed_bev_indexes, target_point_indexes, v_indexes, packed_reasoning_token_indexes, route_loss_indexes, traj_loss_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        # mse = None
        # if self.config.visual_gen:
        #     packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
        #     target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
        #     has_mse = packed_timesteps > 0
        #     mse = (packed_mse_preds - target[has_mse]) ** 2
        ce = None
        if ce_loss_indexes is not None and len(ce_loss_indexes) > 0:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            predicted_token_ids = torch.argmax(packed_ce_preds, dim=-1)
            tokenizer = getattr(self, 'tokenizer', None)
            
            if tokenizer is None and hasattr(self.config, 'tokenizer'):
                tokenizer = self.config.tokenizer
            
            try:
                lab = packed_label_ids
                gt_ids = [t for t in lab if t != -100]
                N = len(gt_ids)
                # gt_text = tokenizer.decode(gt_ids, skip_special_tokens=False)
                pred = predicted_token_ids
                if torch.is_tensor(pred):
                    pred = pred.detach().cpu().tolist()
                pred = pred[:N] 
                predicted_text = tokenizer.decode(pred, skip_special_tokens=False)
                segments = re.findall(r"<\|im_start\|>.*?<\|im_end\|>", predicted_text, flags=re.DOTALL)

                with open("output_updatedv2_debug_speed.log", "a", encoding="utf-8") as f:
                    if segments:
                        for seg in segments:
                            f.write(seg.replace("\n", "\\n") + "\n")
                    else:
                        f.write("Predicted text: " + predicted_text.replace("\n", "\\n") + "\n")
            except Exception as e:
                print("Tokenizer decode failed:", e)

            ce = F.cross_entropy(
                packed_ce_preds, 
                packed_label_ids,
                reduction="none",
            )
        ### velocity loss
        if traj_gt is not None and v_indexes is not None:
            velocity_feats = last_hidden_state[v_indexes]  # (B, C)  one token per sample
            velocity = self.velocity_head(velocity_feats)
            velocity_loss = F.smooth_l1_loss(velocity, future_speeds_tensors)           # (B, 2)

        route_loss = None
        if route_gt is not None and route_loss_indexes is not None:
            route_feats = last_hidden_state[route_loss_indexes]  # (B*20, C) = (220, C)
            T_route = 20
            B = route_feats.shape[0] // T_route  # 220 // 20 = 11
            
            # pred -> (B*20, 2) -> (B, 20, 2)
            pred_routes = self.route_head(route_feats)  # (220, 2)
            pred_routes = pred_routes.view(B, T_route, 2)  # (11, 20, 2)
            
            # gt -> (B, 20, 2)
            if route_gt.dim() == 2 and route_gt.size(-1) == 2:
                route_gt_valid = route_gt.view(B, T_route, 2)
            elif route_gt.dim() == 3:
                route_gt_valid = route_gt[:B]
            else:
                raise ValueError(f"Unexpected route_gt shape: {route_gt.shape}")
            
            route_loss = F.l1_loss(pred_routes, route_gt_valid)

        traj_loss = None
        if traj_gt is not None and traj_loss_indexes is not None:
            traj_feats = last_hidden_state[traj_loss_indexes]  # (B*12, C) = (132, C)
            T_traj = 6
            B = traj_feats.shape[0] // T_traj  # 132 // 12 = 11
            
            # pred -> (B*12, 2) -> (B, 12, 2)
            pred_trajs = self.waypoints_head(traj_feats)  # (132, 2)
            pred_trajs = pred_trajs.view(B, T_traj, 2)  # (11, 12, 2)
            
            # gt -> (B, 12, 2)
            if traj_gt.dim() == 2 and traj_gt.size(-1) == 2:
                traj_gt_valid = traj_gt.view(B, T_traj, 2)
            elif traj_gt.dim() == 3:
                traj_gt_valid = traj_gt[:B]
            else:
                raise ValueError(f"Unexpected traj_gt shape: {traj_gt.shape}")
            traj_offset = self.position_to_offset(traj_gt_valid)
            traj_loss = self.vad_traj_loss(pred_trajs,traj_offset)
            pred_trajs = pred_trajs.cumsum(dim=1)
            l2_1s, l2_2s, l2_3s = self.ade_fde_loss(pred_trajs, traj_gt_valid)
            l2_avg = ( l2_1s + l2_2s + l2_3s ) / 3
        result = dict(
            ce=ce,
            traj_loss=traj_loss,
            route_loss=route_loss,
            velocity_loss=velocity_loss,
            l2_1s=l2_1s,
            l2_2s=l2_2s,
            l2_3s=l2_3s,
            l2_avg=l2_avg,
        )
        return {k: v for k, v in result.items() if v is not None}
    
    def extract_all_eos_spans(self, valid_ids, eos_token_id):
        outputs = []
        temp = []
        for tid in valid_ids:
            temp.append(tid)
            if tid == eos_token_id:
                outputs.append(temp)
                temp = []
        return outputs

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        """Tokenize prompts and prepare packed sequences for inference."""
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            if '<|im_start|>' not in prompt:
                # prompt = "You are a smart autonomous agent and driving an self-driving car. Keep the necessary contents only in the answer. " + prompt + self.assistant_prompt
                prompt += self.assistant_prompt

            # print(repr(prompt))

            text_ids = tokenizer.encode(prompt)
            # text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)
        
        text_position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0),
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=attention_mask
        )
        
        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int, device=device),
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_position_ids": text_position_ids_3d,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    def _cached_prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        """Tokenize prompts and prepare packed sequences for inference."""
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        device = self.language_model.model.embed_tokens.weight.device
        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int, device=device),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        """Update cache with text tokens for incremental generation."""
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_mot:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            # special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        # n_video_tokens = special_video_mask.sum()
        # special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        # if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        #     raise ValueError(
        #         f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
        #     )

        return special_image_mask, None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.vision_model.dtype)
        image_embeds, deepstack_image_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def prepare_vit_images_qwen3vl(self, curr_kvlens, curr_rope, images, new_token_ids):
        """Prepare Qwen3VL vision images using official processor.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            im_start_ids = tokenizer.encode(self.user_prompt, add_special_tokens=False)
            packed_text_ids.extend(im_start_ids)
            packed_text_indexes.extend(range(_curr, _curr + len(im_start_ids)))
            packed_indexes.extend(range(curr, curr + len(im_start_ids)))
            curr += len(im_start_ids)
            _curr += len(im_start_ids)

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_vit_token_indexes.extend(vit_positions)

            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            total_seq_len = len(im_start_ids) + 1 + num_vision_tokens + 1  # +1 for start_of_image, +1 for end_of_image
            packed_seqlens.append(total_seq_len)
            newlens.append(curr_kvlen + total_seq_len)
            new_rope.append(curr_position_id + 1)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)
        
        position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0), 
            image_grid_thw=packed_vit_position_ids_tensor,
            video_grid_thw=None, 
            attention_mask=attention_mask
        )
        
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_position_ids": position_ids_3d, 
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    def prepare_kv_cache(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, tokenizer):
        """Prepare generation with 3D position_ids for vision and text processing.
        
        This function combines vision processing with proper 3D position_ids calculation
        following official Qwen3VL implementation.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            tokenizer: The tokenizer for encoding special tokens
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()

        curr_position_id = 0
        # print(f"DEBUG: curr_kvlens = {curr_kvlens}")
        # print(f"DEBUG: curr_rope = {curr_rope}")
        if curr_kvlens and curr_rope:
            curr_position_id = curr_rope[0]
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        split_lens.append(len(user_prompt_ids))
        attn_modes.append('causal')

        for image in images:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)
            attn_modes.append('full')

        instruction_prompt = self.clean_instruction_prompt(instruction_prompt)
        full_instruction_prompt = instruction_prompt + "<|im_end|>"
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)
        split_lens.append(len(instruction_prompt_ids))
        attn_modes.append('causal')

        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        curr_position_start = curr_rope[0] if curr_rope else 0
        newlens.append(total_curr_kvlen + total_seq_len)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)

        position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0),  
            image_grid_thw=packed_vit_position_ids_tensor, 
            video_grid_thw=None, 
            attention_mask=attention_mask,
            tokenizer=tokenizer 
        )
        new_rope.append(position_ids_3d[0].max().item() + 1)
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )
        key_values_lens_tensor = torch.tensor(curr_kvlens, dtype=torch.int, device=device)
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "nested_attention_masks": nested_attention_masks,
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_position_ids": position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(key_values_lens_tensor, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    # def prepare_fast_kvcache(self, curr_kvlens, curr_rope, images, new_token_ids, tokenizer, reasoning_learnable_tokens, action_learnable_tokens, target_point_max_num_tokens, v_num_token):
    #     """Prepare generation with 3D position_ids for vision and text processing.
        
    #     This function combines vision processing with proper 3D position_ids calculation
    #     following official Qwen3VL implementation.
        
    #     Args:
    #         curr_kvlens: Current KV cache lengths  
    #         curr_rope: Current RoPE positions
    #         images: List of PIL Images
    #         new_token_ids: Special token IDs dictionary
    #         tokenizer: The tokenizer for encoding special tokens
            
    #     Returns:
    #         generation_input: Processed tensors for model forward
    #         newlens: Updated KV lengths
    #         new_rope: Updated RoPE positions
    #     """
    #     packed_vit_token_indexes = list()
    #     vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
    #     packed_text_ids, packed_text_indexes = list(), list()
    #     packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
    #     packed_key_value_indexes, packed_reasoning_token_indexes, packed_action_token_indexes = list(), list(), list()
    #     target_point_indexes, velocity_indexes = list(), list()
    #     _curr = curr = 0
    #     newlens, new_rope = list(), list()
    #     split_lens, attn_modes, nested_attention_masks = list(), list(), list()
    #     if curr_kvlens and curr_rope:
    #         for curr_kvlen in curr_kvlens:
    #             packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
    #             curr += curr_kvlen
        
    #     # process image + lidar
    #     for image in images:
    #         packed_text_ids.append(new_token_ids['start_of_image'])
    #         packed_text_indexes.append(_curr)
    #         packed_indexes.append(curr)
    #         curr += 1
    #         _curr += 1

    #         # Use official Qwen3VL processor to process image
    #         processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
    #         pixel_values = processed["pixel_values"]
    #         grid_thw = processed["image_grid_thw"]
            
    #         # Get the processed tokens from processor (includes complete vision token sequence)
    #         vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
    #         num_vision_tokens = len(vision_token_ids)
            
    #         # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
    #         packed_text_ids.extend(vision_token_ids.tolist())
    #         packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
    #         packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
    #         # All positions need vision embeddings (no special tokens to skip)
    #         vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
    #         # Store vision data
    #         packed_vit_tokens.append(pixel_values)
    #         packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
    #         vit_token_seqlens.append(num_vision_tokens)
    #         # All positions need vision embeddings
    #         packed_vit_token_indexes.extend(vit_positions)
            
    #         # Update position counters
    #         curr += num_vision_tokens
    #         _curr += num_vision_tokens

    #         packed_text_ids.append(new_token_ids['end_of_image'])
    #         packed_text_indexes.append(_curr)
    #         packed_indexes.append(curr)
    #         curr += 1
    #         _curr += 1
    #         split_lens.append(num_vision_tokens+2)
    #         attn_modes.append('full')

    #     target_point_indexes.extend(range(curr, curr + 1))
    #     curr += target_point_max_num_tokens  
    #     v_indexes.append(curr)
    #     curr += v_num_token
    #     curr_split_len += target_point_max_num_tokens + v_num_token
    #     attn_modes.append("full")
    #     split_lens.append(target_point_max_num_tokens + v_num_token)        
    #     # add N reasoning learnable tokens to the end
    #     packed_reasoning_token_indexes.extend(range(_curr, _curr + reasoning_learnable_tokens))
    #     packed_indexes.extend(range(curr, curr + reasoning_learnable_tokens))
    #     curr += reasoning_learnable_tokens
    #     _curr += reasoning_learnable_tokens

    #     total_seq_len = _curr 
    #     packed_seqlens.append(total_seq_len)
    #     split_lens.append(reasoning_learnable_tokens)
    #     attn_modes.append('full')

    #     # add M action learnable tokens to the end
    #     packed_action_token_indexes.extend(range(_curr, _curr + action_learnable_tokens))
    #     packed_indexes.extend(range(curr, curr + action_learnable_tokens))
    #     curr += action_learnable_tokens
    #     _curr += action_learnable_tokens

    #     total_seq_len = _curr 
    #     packed_seqlens.append(total_seq_len)
    #     split_lens.append(action_learnable_tokens)
    #     attn_modes.append('full')

    #     device = self.language_model.model.embed_tokens.weight.device
    #     nested_attention_masks.append(
    #         prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
    #     )       

    #     packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
    #     packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
    #     packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
    #     attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)

    #     position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
    #         input_ids=packed_text_ids_tensor.unsqueeze(0),
    #         image_grid_thw=packed_vit_position_ids_tensor,
    #         video_grid_thw=None, 
    #         attention_mask=attention_mask,
    #         num_learnable_tokens=reasoning_learnable_tokens + action_learnable_tokens + target_point_max_num_tokens + v_num_token,
    #         tokenizer=tokenizer
    #     )
    #     new_rope.append(position_ids_3d[0].max().item() + 1)

    #     generation_input = {
    #         "packed_text_ids": packed_text_ids_tensor,
    #         "nested_attention_masks": nested_attention_masks,
    #         "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
    #         "target_point_indexes": torch.tensor(target_point_indexes, dtype=torch.long, device=device),
    #         "v_indexes": torch.tensor(v_indexes, dtype=torch.long, device=device),
    #         "packed_reasoning_token_indexes": torch.tensor(packed_reasoning_token_indexes, dtype=torch.long, device=device),
    #         "packed_action_token_indexes": torch.tensor(packed_action_token_indexes, dtype=torch.long, device=device),
    #         "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
    #         "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
    #         "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
    #         "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
    #         "packed_position_ids": position_ids_3d, 
    #         "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
    #         "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
    #         "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
    #     }

    #     return generation_input, newlens, new_rope

    def prepare_fast_kvcache(self, curr_kvlens, curr_rope, trans_feat, new_token_ids, tokenizer, reasoning_learnable_tokens, action_learnable_tokens, target_point_max_num_tokens, v_num_token, num_route_tokens, num_traj_tokens):
        """Prepare generation with 3D position_ids for vision and text processing.
        
        This function combines vision processing with proper 3D position_ids calculation
        following official Qwen3VL implementation.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            tokenizer: The tokenizer for encoding special tokens
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        """
        packed_bev_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes, packed_reasoning_token_indexes, packed_action_token_indexes = list(), list(), list()
        target_point_indexes, v_indexes = list(), list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()
        if curr_kvlens and curr_rope:
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        bev_token_max_num_tokens = trans_feat.shape[-1] * trans_feat.shape[-2]
        packed_bev_token_indexes.extend(range(_curr, _curr + bev_token_max_num_tokens))
        curr = curr + bev_token_max_num_tokens
        _curr = _curr + bev_token_max_num_tokens
        split_lens.append(bev_token_max_num_tokens)
        attn_modes.append('full')
        # process image + lidar
        # for image in images:
        #     packed_text_ids.append(new_token_ids['start_of_image'])
        #     packed_text_indexes.append(_curr)
        #     packed_indexes.append(curr)
        #     curr += 1
        #     _curr += 1

        #     # Use official Qwen3VL processor to process image
        #     processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
        #     pixel_values = processed["pixel_values"]
        #     grid_thw = processed["image_grid_thw"]
            
        #     # Get the processed tokens from processor (includes complete vision token sequence)
        #     vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
        #     num_vision_tokens = len(vision_token_ids)
            
        #     # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
        #     packed_text_ids.extend(vision_token_ids.tolist())
        #     packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
        #     packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
        #     # All positions need vision embeddings (no special tokens to skip)
        #     vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
        #     # Store vision data
        #     packed_vit_tokens.append(pixel_values)
        #     packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
        #     vit_token_seqlens.append(num_vision_tokens)
        #     # All positions need vision embeddings
        #     packed_vit_token_indexes.extend(vit_positions)
            
        #     # Update position counters
        #     curr += num_vision_tokens
        #     _curr += num_vision_tokens

        #     packed_text_ids.append(new_token_ids['end_of_image'])
        #     packed_text_indexes.append(_curr)
        #     packed_indexes.append(curr)
        #     curr += 1
        #     _curr += 1
        #     split_lens.append(num_vision_tokens+2)
        #     attn_modes.append('full')
        target_point_indexes.extend(range(_curr, _curr + 2))
        curr += target_point_max_num_tokens
        _curr += target_point_max_num_tokens  
        v_indexes.append(_curr)
        curr += v_num_token
        _curr += v_num_token
        attn_modes.append("full")
        split_lens.append(target_point_max_num_tokens + v_num_token)        
        # add N reasoning learnable tokens to the end
        packed_reasoning_token_indexes.extend(range(_curr, _curr + reasoning_learnable_tokens))
        packed_indexes.extend(range(curr, curr + reasoning_learnable_tokens))
        curr += reasoning_learnable_tokens
        _curr += reasoning_learnable_tokens

        total_seq_len = _curr 
        packed_seqlens.append(total_seq_len)
        split_lens.append(reasoning_learnable_tokens)
        attn_modes.append('full')

        # add M action learnable tokens to the end (route + traj)
        packed_action_token_indexes.extend(range(_curr, _curr + num_route_tokens + num_traj_tokens))
        packed_indexes.extend(range(curr, curr + num_route_tokens + num_traj_tokens))

        # route tokens
        curr += num_route_tokens
        _curr += num_route_tokens
        split_lens.append(num_route_tokens)
        attn_modes.append('full')

        # traj tokens  
        curr += num_traj_tokens
        _curr += num_traj_tokens
        split_lens.append(num_traj_tokens)
        attn_modes.append('full')

        device = self.language_model.model.embed_tokens.weight.device
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )       

        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_bev_token_indexes_tensor = torch.tensor(packed_bev_token_indexes, dtype=torch.long, device=device)
        # packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        # attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)

        # position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
        #     input_ids=packed_text_ids_tensor.unsqueeze(0),
        #     image_grid_thw=packed_vit_position_ids_tensor,
        #     video_grid_thw=None, 
        #     attention_mask=attention_mask,
        #     num_learnable_tokens=reasoning_learnable_tokens + action_learnable_tokens + target_point_max_num_tokens + v_num_token,
        #     tokenizer=tokenizer
        # )
        total_len = bev_token_max_num_tokens + reasoning_learnable_tokens + action_learnable_tokens + target_point_max_num_tokens + v_num_token
        position_ids_1d = torch.arange(total_len, device=packed_bev_token_indexes_tensor.device).unsqueeze(0).expand(1, -1)
        position_ids_3d = position_ids_1d.unsqueeze(0).expand(3, -1, -1)
        rope_deltas = torch.zeros(1, 1, device=packed_bev_token_indexes_tensor.device)
        new_rope.append(position_ids_3d[0].max().item() + 1)
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "nested_attention_masks": nested_attention_masks,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "target_point_indexes": torch.tensor(target_point_indexes, dtype=torch.long, device=device),
            "v_indexes": torch.tensor(v_indexes, dtype=torch.long, device=device),
            "packed_reasoning_token_indexes": torch.tensor(packed_reasoning_token_indexes, dtype=torch.long, device=device),
            "packed_action_token_indexes": torch.tensor(packed_action_token_indexes, dtype=torch.long, device=device),
            # "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            # "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            # "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_bev_token_indexes": packed_bev_token_indexes_tensor,
            "packed_position_ids": position_ids_3d, 
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }

        return generation_input, newlens, new_rope

    def clean_instruction_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            return prompt
        return re.sub(
            r'^(?:\s*(?:<image>|<lidar>|<front>|<trans>))+',
            '',
            prompt
        ).lstrip()

    def prepare_generation(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, tokenizer):
        """Prepare generation with 3D position_ids for vision and text processing.
        
        This function combines vision processing with proper 3D position_ids calculation
        following official Qwen3VL implementation.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            tokenizer: The tokenizer for encoding special tokens
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        if curr_kvlens and curr_rope:
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        
        for image in images:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

        full_instruction_prompt = instruction_prompt + self.assistant_prompt
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)

        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        newlens.append(total_curr_kvlen + total_seq_len)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)
        
        position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0), 
            image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
            video_grid_thw=None, 
            attention_mask=attention_mask,
            tokenizer=tokenizer 
        )
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # Add indexes for the current sequence being processed
        current_sequence_length = len(packed_text_ids)
        # For the current sequence, use the starting KV cache position, not the accumulated curr
        kv_cache_start = sum(curr_kvlens) if curr_kvlens else 0
        packed_key_value_indexes.extend(range(kv_cache_start, kv_cache_start + current_sequence_length))
        
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_position_ids": position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(newlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    def prepare_fast_generation(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, num_learnable_tokens, tokenizer):
        """Prepare generation with 3D position_ids for vision and text processing.
        
        This function combines vision processing with proper 3D position_ids calculation
        following official Qwen3VL implementation.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            tokenizer: The tokenizer for encoding special tokens
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        Common image input:[image,image,image,image,image(current view),image(copy current view),image(lidar)]
        image(copy current view) and image(lidar) are used in 2nd transformer
        """

        packed_vit_token_indexes, packed_und_vit_token_indexes, packed_gen_vit_token_indexes = list(), list(), list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes, packed_und_text_indexes, packed_gen_text_indexes = list(), list(), list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        packed_learnable_token_indexes = list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        if curr_kvlens and curr_rope:
            curr_position_id = curr_rope[0]
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        split_lens.append(len(user_prompt_ids))
        attn_modes.append('causal')
        images_und = images[:-2]  # All images except the last two for und transformer
        images_gen = images[-2:]  # The last two images for gen transformer
        
        # 2. process und images
        for image in images_und:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            #packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_und_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')

        # 3. add instruction_prompt + assistant_prompt after images_und
        instruction_prompt = self.clean_instruction_prompt(instruction_prompt)
        full_instruction_prompt = instruction_prompt + "<|im_end|>"
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)
        split_lens.append(len(instruction_prompt_ids))  # +2 for start and end tokens
        attn_modes.append('causal')
        
        # 4. process gen images
        for image in images_gen:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            #packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_gen_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')
        
        # 5. add 8 learnable tokens to the end
        packed_learnable_token_indexes.extend(range(_curr, _curr + num_learnable_tokens))
        packed_indexes.extend(range(curr, curr + num_learnable_tokens))
        curr += num_learnable_tokens
        _curr += num_learnable_tokens
        split_lens.append(num_learnable_tokens)
        attn_modes.append('full')
        
        # 6. pad to max_num_tokens
        und_idxs = [
            idx for idx in packed_text_indexes
            if idx not in packed_gen_text_indexes
        ]
        packed_und_text_indexes = und_idxs
        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        curr_position_start = curr_rope[0] if curr_rope else 0
        newlens.append(total_curr_kvlen + total_seq_len)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_und_vit_token_indexes_tensor = torch.tensor(packed_und_vit_token_indexes, dtype=torch.long, device=device)
        packed_gen_vit_token_indexes_tensor = torch.tensor(packed_gen_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones_like(packed_text_ids_tensor).unsqueeze(0)
        position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
            input_ids=packed_text_ids_tensor.unsqueeze(0),  # [1, seq_len]
            image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
            video_grid_thw=None,  
            attention_mask=attention_mask,
            num_learnable_tokens=num_learnable_tokens,
            tokenizer=tokenizer 
        )
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # Add indexes for the current sequence being processed
        current_sequence_length = len(packed_text_ids)
        # For the current sequence, use the starting KV cache position, not the accumulated curr
        kv_cache_start = sum(curr_kvlens) if curr_kvlens else 0
        packed_key_value_indexes.extend(range(kv_cache_start, kv_cache_start + current_sequence_length))
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )
        packed_text_ids_tensor = packed_text_ids_tensor[packed_text_ids_tensor != 151655]  # remove all <|image_pad|> tokens for fast generation
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor.to(device=device, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_gen_text_indexes": torch.tensor(packed_gen_text_indexes, dtype=torch.long, device=device),
            "packed_und_text_indexes": torch.tensor(packed_und_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "nested_attention_masks": nested_attention_masks,
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_und_vit_token_indexes": packed_und_vit_token_indexes_tensor,
            "packed_gen_vit_token_indexes": packed_gen_vit_token_indexes_tensor,
            "packed_learnable_token_indexes": torch.tensor(packed_learnable_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": position_ids_3d,  
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(newlens, dtype=torch.int, device=device),
            "curr": int(curr),
        }

        return generation_input, newlens, new_rope

    def _cached_prepare_vit_images_qwen3vl(self, curr_kvlens, curr_rope, images, new_token_ids):
        """Prepare Qwen3VL vision images using official processor.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_vit_token_indexes.extend(vit_positions)

            # Position and sequence length tracking  
            packed_position_ids.extend([curr_position_id] * num_vision_tokens)
            packed_seqlens.append(num_vision_tokens)
            newlens.append(curr_kvlen + num_vision_tokens)
            new_rope.append(curr_position_id + 1)

        device = self.language_model.model.embed_tokens.weight.device
        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": torch.stack(packed_vit_position_ids, dim=0).to(device),  # Stack grid_thw
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        """Update cache with vision tokens for incremental generation using Qwen3VL."""
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        # packed_sequence[packed_text_indexes] = packed_text_embedding

        # Qwen3VL vision processing - packed_vit_tokens contains pixel_values, packed_vit_position_ids contains grid_thw
        image_embeds, deepstack_image_embeds = self.get_image_features(packed_vit_tokens, packed_vit_position_ids)
        image_embeds = torch.cat(image_embeds, dim=0).to(packed_text_embedding.device, packed_text_embedding.dtype)
        image_mask, _ = self.get_placeholder_mask(
            packed_text_ids, inputs_embeds=packed_text_embedding, image_features=image_embeds
        )
        packed_text_embedding = packed_text_embedding.masked_scatter(image_mask, image_embeds)
        packed_sequence[packed_text_indexes] = packed_text_embedding

        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds

        # packed_vit_token_embed, packed_deepstack_image_embed = self.vision_model(
        #     hidden_states=packed_vit_tokens,
        #     grid_thw=packed_vit_position_ids,
        # )
        # visual_pos_masks = None
        # deepstack_visual_embeds = None
        
        # No connector or manual position embedding needed for Qwen3VL
        # Ensure dtype compatibility
        # if packed_vit_token_embed.dtype != packed_sequence.dtype:
        #     packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        # packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_mot:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids, 
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            # args for deepstack
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    @torch.no_grad
    def forward_cache_update_generation(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        nested_attention_masks: list = None,
    ):
        """Update cache with generation tokens using 3D position_ids for Qwen3VL."""
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        
        # Qwen3VL vision processing - packed_vit_tokens contains pixel_values, packed_vit_position_ids contains grid_thw
        image_embeds, deepstack_image_embeds = self.get_image_features(packed_vit_tokens, packed_vit_position_ids)
        image_embeds = torch.cat(image_embeds, dim=0).to(packed_text_embedding.device, packed_text_embedding.dtype)
        image_mask, _ = self.get_placeholder_mask(
            packed_text_ids, inputs_embeds=packed_text_embedding, image_features=image_embeds
        )
        packed_text_embedding = packed_text_embedding.masked_scatter(image_mask, image_embeds)
        packed_sequence[packed_text_indexes] = packed_text_embedding
        
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
        
        extra_inputs = {}
        if self.use_mot:
            extra_inputs = {"mode": "und"}
            
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids, 
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            attention_mask = nested_attention_masks,
            is_causal=False,
            # args for deepstack
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **extra_inputs,
        )
        past_key_values = output.past_key_values
        return past_key_values, packed_position_ids

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        """Prepare start tokens for text generation."""
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            gen_start_id = tokenizer.encode('\n', add_special_tokens=False)
            packed_start_tokens.extend(gen_start_id)
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        device = self.language_model.model.embed_tokens.weight.device
        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long, device=device),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }

        return generation_input
    def prepare_fast_generation_senna(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, num_learnable_tokens, action_learnable_tokens, tokenizer):
        """Prepare generation with 3D position_ids for vision and text processing.
        
        This function combines vision processing with proper 3D position_ids calculation
        following official Qwen3VL implementation.
        
        Args:
            curr_kvlens: Current KV cache lengths  
            curr_rope: Current RoPE positions
            images: List of PIL Images
            new_token_ids: Special token IDs dictionary
            tokenizer: The tokenizer for encoding special tokens
            
        Returns:
            generation_input: Processed tensors for model forward
            newlens: Updated KV lengths
            new_rope: Updated RoPE positions
        Common image input:[image,image,image,image,image(current view),image(copy current view),image(lidar)]
        image(copy current view) and image(lidar) are used in 2nd transformer
        """

        packed_vit_token_indexes, packed_und_vit_token_indexes, packed_gen_vit_token_indexes = list(), list(), list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes, packed_und_text_indexes, packed_gen_text_indexes = list(), list(), list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        packed_learnable_token_indexes, packed_action_token_indexes = list(), list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        curr_position_id = 0
        # print(f"DEBUG: curr_kvlens = {curr_kvlens}")
        # print(f"DEBUG: curr_rope = {curr_rope}")
        if curr_kvlens and curr_rope:
            curr_position_id = curr_rope[0]
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        # print(f"DEBUG: initial packed_key_value_indexes = {packed_key_value_indexes}")
        # print(f"DEBUG: curr after existing KV = {curr}")
        
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        split_lens.append(len(user_prompt_ids))
        attn_modes.append('causal')
        images_und = images[:-2]  # All images except the last two for und transformer
        images_gen = images[-2:]  # The last two images for gen transformer
        # 2. process und images
        for image in images_und:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            #packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_und_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')

        # 3. add instruction_prompt + assistant_prompt after images_und
        assistant_prompt = "<|im_end|>"
        instruction_prompt = self.clean_instruction_prompt(instruction_prompt)
        print("Cleaned instruction prompt:", instruction_prompt)
        full_instruction_prompt = instruction_prompt + assistant_prompt
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)
        split_lens.append(len(instruction_prompt_ids))  # +2 for start and end tokens
        attn_modes.append('causal')
        # 4. process gen images
        for image in images_gen:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            #packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_gen_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')
        # 5. add 8 learnable tokens to the end
        packed_learnable_token_indexes.extend(range(_curr, _curr + num_learnable_tokens))
        packed_indexes.extend(range(curr, curr + num_learnable_tokens))
        curr += num_learnable_tokens
        _curr += num_learnable_tokens
        split_lens.append(num_learnable_tokens)
        attn_modes.append('full')
        # 5. add 1 action learnable tokroot/qihang_projects/AutoMoTive_qihang_action/evaluation/eval_automot_fast_thinking_senna.shens to the end
        packed_action_token_indexes.extend(range(_curr, _curr + action_learnable_tokens))
        packed_indexes.extend(range(curr, curr + action_learnable_tokens))
        curr += action_learnable_tokens
        _curr += action_learnable_tokens
        split_lens.append(action_learnable_tokens)
        attn_modes.append('full')

        und_idxs = [
            idx for idx in packed_text_indexes
            if idx not in packed_gen_text_indexes
        ]
        packed_und_text_indexes = und_idxs
        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        curr_position_start = curr_rope[0] if curr_rope else 0
        newlens.append(total_curr_kvlen + total_seq_len)
        # new_rope.append(curr_position_start + total_seq_len)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_und_vit_token_indexes_tensor = torch.tensor(packed_und_vit_token_indexes, dtype=torch.long, device=device)
        packed_gen_vit_token_indexes_tensor = torch.tensor(packed_gen_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones_like(packed_text_ids_tensor).unsqueeze(0)
        position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
            input_ids=packed_text_ids_tensor.unsqueeze(0),  # [1, seq_len]
            image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
            video_grid_thw=None,  
            attention_mask=attention_mask,
            num_learnable_tokens = action_learnable_tokens + num_learnable_tokens,
            tokenizer=tokenizer 
        )
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # Add indexes for the current sequence being processed
        current_sequence_length = len(packed_text_ids)
        # For the current sequence, use the starting KV cache position, not the accumulated curr
        kv_cache_start = sum(curr_kvlens) if curr_kvlens else 0
        packed_key_value_indexes.extend(range(kv_cache_start, kv_cache_start + current_sequence_length))
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )
        packed_text_ids_tensor = packed_text_ids_tensor[packed_text_ids_tensor != 151655]  # remove all <|image_pad|> tokens for fast generation
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor.to(device=device, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_gen_text_indexes": torch.tensor(packed_gen_text_indexes, dtype=torch.long, device=device),
            "packed_und_text_indexes": torch.tensor(packed_und_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "nested_attention_masks": nested_attention_masks,
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_action_token_indexes": torch.tensor(packed_action_token_indexes, dtype=torch.long, device=device),
            "packed_und_vit_token_indexes": packed_und_vit_token_indexes_tensor,
            "packed_gen_vit_token_indexes": packed_gen_vit_token_indexes_tensor,
            "packed_learnable_token_indexes": torch.tensor(packed_learnable_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(newlens, dtype=torch.int, device=device),
            "curr": int(curr),
        }

        return generation_input, newlens, new_rope
    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        """Generate text tokens using cached key-values for incremental generation."""
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_mot:
                extra_inputs = {"mode": "und"}

            # For text generation, convert 1D position_ids to 3D format if needed
            if packed_query_position_ids.dim() == 1:
                position_ids_3d = packed_query_position_ids.unsqueeze(0).expand(3, -1)
            else:
                position_ids_3d = packed_query_position_ids

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=position_ids_3d,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)