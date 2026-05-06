from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from flash_attn import flash_attn_varlen_func
from transformers import AutoTokenizer as Qwen3Tokenizer
import sys
# sys.path.insert(0, ...)  # Removed: use pip-installed transformers

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig as _Qwen3VLTextConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig as _Qwen3VLConfig,
    Qwen3VLVisionConfig
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLVisionAttention,
    Qwen3VLTextAttention,
    Qwen3VLVisionMLP,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb,
)


# Global compilation optimization (following qwen3_navit pattern)
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)


class Qwen3VLTextConfig(_Qwen3VLTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3VLTextModel`] with NavIT optimizations. 
    It is used to instantiate a Qwen3-VL text model according to the specified arguments, defining the model architecture 
    with NavIT enhancements for efficient packed sequence processing.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3VLModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        
        # NavIT specific parameters
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply QK normalization in attention layers.
        layer_module (`str`, *optional*, defaults to `"Qwen3VLDecoderLayer"`):
            The decoder layer module to use. Options: "Qwen3VLDecoderLayer", "Qwen3VLMoTDecoderLayer", "Qwen3VLMoEDecoderLayer".
        freeze_und (`bool`, *optional*, defaults to `False`):
            Whether to freeze understanding tokens during training.
        
        # MoT (Mixture of Tokens) specific parameters
        mot_num_attention_heads (`int`, *optional*, defaults to half of num_attention_heads):
            Number of attention heads for MoT.
        mot_num_key_value_heads (`int`, *optional*, defaults to half of num_key_value_heads):
            Number of key-value heads for MoT.
        mot_intermediate_size (`int`, *optional*, defaults to intermediate_size):
            Intermediate size for MoT MLP.

    Example:
    ```python
    >>> from transformers import Qwen3VLTextModel, Qwen3VLTextConfig
    >>> # Initializing a Qwen3-VL NavIT style configuration
    >>> configuration = Qwen3VLTextConfig()
    >>> # Initializing a model from the Qwen3-VL NavIT style configuration
    >>> model = Qwen3VLTextModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl_text_navit"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        
        # NavIT specific parameters (following qwen3_navit pattern)
        qk_norm=True,
        layer_module="Qwen3VLDecoderLayer",
        freeze_und=False,
        
        # MoT-specific parameters (configurable attention heads) - following qwen3_navit defaults
        mot_num_attention_heads=16,
        mot_num_key_value_heads=4,
        # MoT-specific MLP size - following qwen3_navit default (half of regular)
        mot_intermediate_size=11008,  # 22016 // 2, following qwen3_navit pattern of reducing MLP size
        
        **kwargs,
    ):
        # Call parent constructor with all standard parameters
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        
        # NavIT specific parameters (following qwen3_navit pattern)
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und
        
        # Set default MoT head counts if not specified (following qwen3_navit pattern)
        self.mot_num_attention_heads = mot_num_attention_heads if mot_num_attention_heads is not None else num_attention_heads // 2
        self.mot_num_key_value_heads = mot_num_key_value_heads if mot_num_key_value_heads is not None else num_key_value_heads // 2
        
        # Set default MoT intermediate size if not specified (following qwen3_navit pattern - half of regular)
        self.mot_intermediate_size = mot_intermediate_size if mot_intermediate_size is not None else intermediate_size // 2
        
        # Ensure MoT head configuration is valid (following qwen3_navit pattern)
        assert self.mot_num_attention_heads % self.mot_num_key_value_heads == 0, \
            f"mot_num_attention_heads ({self.mot_num_attention_heads}) must be divisible by mot_num_key_value_heads ({self.mot_num_key_value_heads})"
        
        self.mot_num_key_value_groups = self.mot_num_attention_heads // self.mot_num_key_value_heads
        self.mot_head_dim = hidden_size // num_attention_heads  # Keep same head dimension


class NaiveCache:
    """
    Simple cache implementation for NavIT models following qwen3_navit pattern.
    
    This cache stores key and value tensors for each layer to enable efficient
    inference with past key values in packed sequence processing.
    """
    
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    """
    Base class for model outputs with past key values for NavIT models.
    
    Args:
        packed_query_sequence (`torch.FloatTensor` of shape `(total_tokens, hidden_size)`):
            Packed query sequence from the model.
        past_key_values (`Optional[NaiveCache]`):
            Contains the cached key and value states for efficient generation.
    """
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def pad_sequence(tensor, pad_size):
    """
    Utility function to pad sequences for NavIT processing.
    
    Args:
        tensor: Input tensor of shape (H, L, D)
        pad_size: Number of padding tokens to add
    
    Returns:
        Padded tensor of shape (H, L + pad_size, D)
    """
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttention(Qwen3VLTextAttention):
    """
    Qwen3VL Text Packed Attention with NavIT optimizations.
    
    This class extends Qwen3VLTextAttention to support packed sequence processing
    for efficient batch processing of variable-length sequences. Follows the exact
    pattern from qwen3_navit.py but adapted for Qwen3VL text components.
    
    Args:
        config: Qwen3VLTextConfig with NavIT parameters
        layer_idx: Layer index for this attention module
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # Note: Qwen3VLTextAttention already has q_norm and k_norm built-in
        # Unlike qwen3_navit.py which conditionally adds them, Qwen3VL always uses QK norm
        # So we don't need to override them here - they're inherited from parent class
        # 
        # Parent class already sets:
        # self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.config.num_attention_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states = packed_key_states.reshape(-1, self.config.num_attention_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.config.num_attention_heads, self.head_dim)

            unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0), 
                packed_key_states.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.config.num_attention_heads * self.head_dim)
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.config.num_attention_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        # packed_query_states = packed_query_states.to(torch.bfloat16)
        # packed_key_states = packed_key_states.to(torch.bfloat16)
        # packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.config.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.config.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        # cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        # cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        # packed_attn_output = flash_attn_varlen_func(
        #     q=packed_query_states,
        #     k=merged_key_states,
        #     v=merged_value_states,
        #     cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        #     cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        #     max_seqlen_q=max(query_lens).item(),
        #     max_seqlen_k=max(key_values_lens).item(),
        #     causal=is_causal,
        # )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        packed_attn_output, attn_weights = attention_interface(
            self,
            packed_query_states.transpose(1,0).unsqueeze(0),
            merged_key_states.transpose(1,0).unsqueeze(0),
            merged_value_states.transpose(1,0).unsqueeze(0),
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        # input_shape = packed_attn_output.shape[:-1]
        # packed_attn_output = packed_attn_output.reshape(*input_shape, -1).contiguous()

        packed_attn_output = packed_attn_output.reshape(-1, self.config.num_attention_heads * self.head_dim).contiguous()
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoT(Qwen3VLTextAttention):
    """
    Qwen3VL Text Packed Attention with MoT (Mixture of Tokens) support.
    
    This class extends PackedAttention to support MoT where different tokens
    can use different attention head configurations. Follows the exact pattern 
    from qwen3_navit.py but adapted for Qwen3VL text components.
    
    Args:
        config: Qwen3VLTextConfig with NavIT and MoT parameters
        layer_idx: Layer index for this attention module
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # MoT-specific head configuration (from config)
        self.mot_num_heads = config.mot_num_attention_heads
        self.mot_num_key_value_heads = config.mot_num_key_value_heads
        self.mot_num_key_value_groups = config.mot_num_key_value_groups
        self.mot_head_dim = config.mot_head_dim
        
        # Note: Qwen3VLTextAttention already has q_norm and k_norm built-in
        # We need additional normalization layers for MoT generation tokens
        self.q_norm_mot_gen = Qwen3VLTextRMSNorm(self.mot_head_dim, eps=config.rms_norm_eps)
        self.k_norm_mot_gen = Qwen3VLTextRMSNorm(self.mot_head_dim, eps=config.rms_norm_eps)

        # MoT-specific projection layers for generation tokens
        self.q_proj_mot_gen = nn.Linear(self.config.hidden_size, self.mot_num_heads * self.mot_head_dim, bias=self.config.attention_bias)
        self.k_proj_mot_gen = nn.Linear(self.config.hidden_size, self.mot_num_key_value_heads * self.mot_head_dim, bias=self.config.attention_bias)
        self.v_proj_mot_gen = nn.Linear(self.config.hidden_size, self.mot_num_key_value_heads * self.mot_head_dim, bias=self.config.attention_bias)
        self.o_proj_mot_gen = nn.Linear(self.mot_num_heads * self.mot_head_dim, self.config.hidden_size, bias=self.config.attention_bias)
        self.reasoning = False

    def forward(self, *args, **kwargs):
        """Forward method that dispatches to training or inference based on model state."""
        if self.training or self.reasoning:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        # Calculate maximum dimensions for padding
        max_heads_q = max(self.config.num_attention_heads, self.mot_num_heads)
        max_heads_kv = max(self.config.num_key_value_heads, self.mot_num_key_value_heads)
        max_head_dim = max(self.head_dim, self.mot_head_dim)
        
        # Padding with zero as it will not affect the attention calculation
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], max_heads_q, max_head_dim))
        packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], max_heads_kv, max_head_dim))
        packed_value_states = packed_sequence.new_zeros((packed_sequence.shape[0], max_heads_kv, max_head_dim))

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        # Process und tokens and fill in their positions
        if len(packed_und_token_indexes) > 0:
            und_q = self.q_proj(packed_sequence_und).view(-1, self.config.num_attention_heads, self.head_dim)
            und_k = self.k_proj(packed_sequence_und).view(-1, self.config.num_key_value_heads, self.head_dim)
            und_v = self.v_proj(packed_sequence_und).view(-1, self.config.num_key_value_heads, self.head_dim)
            
            packed_query_states[packed_und_token_indexes] = und_q
            packed_key_states[packed_und_token_indexes] = und_k
            packed_value_states[packed_und_token_indexes] = und_v

        # Process gen tokens and fill in their positions  
        if len(packed_gen_token_indexes) > 0:
            gen_q = self.q_proj_mot_gen(packed_sequence_gen).view(-1, self.mot_num_heads, self.mot_head_dim)
            gen_k = self.k_proj_mot_gen(packed_sequence_gen).view(-1, self.mot_num_key_value_heads, self.mot_head_dim)
            gen_v = self.v_proj_mot_gen(packed_sequence_gen).view(-1, self.mot_num_key_value_heads, self.mot_head_dim)
            
            packed_query_states[packed_gen_token_indexes, :self.mot_num_heads, :self.mot_head_dim] = gen_q
            packed_key_states[packed_gen_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim] = gen_k
            packed_value_states[packed_gen_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim] = gen_v

        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        # Apply normalization to und tokens
        if len(packed_und_token_indexes) > 0:
            packed_query_states_[packed_und_token_indexes] = self.q_norm(
                packed_query_states[packed_und_token_indexes]
            )
            packed_key_states_[packed_und_token_indexes] = self.k_norm(
                packed_key_states[packed_und_token_indexes]
            )
            if self.config.freeze_und:
                packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
                packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()

        # Apply normalization to gen tokens
        if len(packed_gen_token_indexes) > 0:
            packed_query_states_[packed_gen_token_indexes, :self.mot_num_heads, :self.mot_head_dim] = self.q_norm_mot_gen(
                packed_query_states[packed_gen_token_indexes, :self.mot_num_heads, :self.mot_head_dim]
            )
            packed_key_states_[packed_gen_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim] = self.k_norm_mot_gen(
                packed_key_states[packed_gen_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim]
            )

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            # Handle GQA expansion for mixed head configurations
            max_heads_total = max(self.config.num_attention_heads, self.mot_num_heads)
            
            packed_key_states_expanded = packed_sequence.new_zeros((packed_sequence.shape[0], max_heads_total, max_head_dim))
            packed_value_states_expanded = packed_sequence.new_zeros((packed_sequence.shape[0], max_heads_total, max_head_dim))
            
            # Expand und tokens using regular GQA
            if len(packed_und_token_indexes) > 0:
                und_key_expanded = packed_key_states_[packed_und_token_indexes][:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
                packed_key_states_expanded[packed_und_token_indexes] = und_key_expanded.reshape(-1, self.config.num_attention_heads, self.head_dim)
                
                und_value_expanded = packed_value_states[packed_und_token_indexes][:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
                packed_value_states_expanded[packed_und_token_indexes] = und_value_expanded.reshape(-1, self.config.num_attention_heads, self.head_dim)
            
            # Expand gen tokens using MoT GQA
            if len(packed_gen_token_indexes) > 0:
                gen_key_expanded = packed_key_states_[packed_gen_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim][:, :, None, :].repeat(1, 1, self.mot_num_key_value_groups, 1)
                packed_key_states_expanded[packed_gen_token_indexes, :self.mot_num_heads, :self.mot_head_dim] = gen_key_expanded.reshape(-1, self.mot_num_heads, self.mot_head_dim)
                
                gen_value_expanded = packed_value_states[packed_gen_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim][:, :, None, :].repeat(1, 1, self.mot_num_key_value_groups, 1)
                packed_value_states_expanded[packed_gen_token_indexes, :self.mot_num_heads, :self.mot_head_dim] = gen_value_expanded.reshape(-1, self.mot_num_heads, self.mot_head_dim)
            
            packed_key_states_ = packed_key_states_expanded
            packed_value_states = packed_value_states_expanded

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), 
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        # Extract attention output and apply appropriate projections
        packed_attn_output = packed_attn_output.transpose(0, 1)  # [seq_len, heads, head_dim]
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape[0], self.config.hidden_size)
        
        # Project und tokens using regular projection
        if len(packed_und_token_indexes) > 0:
            und_output = packed_attn_output[packed_und_token_indexes].reshape(-1, self.config.num_attention_heads * self.head_dim)
            packed_attn_output_[packed_und_token_indexes] = self.o_proj(und_output)
        
        # Project gen tokens using MoT projection
        if len(packed_gen_token_indexes) > 0:
            gen_output = packed_attn_output[packed_gen_token_indexes, :self.mot_num_heads, :self.mot_head_dim].reshape(-1, self.mot_num_heads * self.mot_head_dim)
            packed_attn_output_[packed_gen_token_indexes] = self.o_proj_mot_gen(gen_output)

        return packed_attn_output_

    def forward_inference(
        self,
        attention_mask,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        mode="und",
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == 'und':
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.config.num_attention_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'gen':
            # packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            # Calculate maximum dimensions for unified tensor shape
            max_heads_q = max(self.config.num_attention_heads, self.mot_num_heads)
            max_heads_kv = max(self.config.num_key_value_heads, self.mot_num_key_value_heads)
            max_head_dim = max(self.head_dim, self.mot_head_dim)

            # Directly create 3D tensors with final shape
            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], max_heads_q, max_head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], max_heads_kv, max_head_dim)
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], max_heads_kv, max_head_dim)
            )

            # Apply projections and fill text tokens
            if len(packed_text_indexes) > 0:
                packed_query_states[packed_text_indexes, :self.config.num_attention_heads, :self.head_dim] = \
                    self.q_proj(packed_text_query_sequence).view(-1, self.config.num_attention_heads, self.head_dim)
                packed_key_states[packed_text_indexes, :self.config.num_key_value_heads, :self.head_dim] = \
                    self.k_proj(packed_text_query_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)
                packed_value_states[packed_text_indexes, :self.config.num_key_value_heads, :self.head_dim] = \
                    self.v_proj(packed_text_query_sequence).view(-1, self.config.num_key_value_heads, self.head_dim)

            # Apply projections and fill vae tokens
            if len(packed_vae_token_indexes) > 0:
                packed_query_states[packed_vae_token_indexes, :self.mot_num_heads, :self.mot_head_dim] = \
                    self.q_proj_mot_gen(packed_vae_query_sequence).view(-1, self.mot_num_heads, self.mot_head_dim)
                packed_key_states[packed_vae_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim] = \
                    self.k_proj_mot_gen(packed_vae_query_sequence).view(-1, self.mot_num_key_value_heads, self.mot_head_dim)
                packed_value_states[packed_vae_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim] = \
                    self.v_proj_mot_gen(packed_vae_query_sequence).view(-1, self.mot_num_key_value_heads, self.mot_head_dim)

            # Apply normalization
            # packed_query_states = packed_query_states.to(torch.float32)
            if len(packed_text_indexes) > 0:
                packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
                packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            if len(packed_vae_token_indexes) > 0:
                packed_query_states[packed_vae_token_indexes, :self.mot_num_heads, :self.mot_head_dim] = self.q_norm_mot_gen(packed_query_states[packed_vae_token_indexes, :self.mot_num_heads, :self.mot_head_dim])
                packed_key_states[packed_vae_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim] = self.k_norm_mot_gen(packed_key_states[packed_vae_token_indexes, :self.mot_num_key_value_heads, :self.mot_head_dim])

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )
        if mode == 'gen' and len(packed_vae_token_indexes) > 0:
            packed_key_states[packed_vae_token_indexes, :, self.mot_head_dim:] = 0
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            max_heads_kv = max(self.config.num_key_value_heads, self.mot_num_key_value_heads)
            max_head_dim = max(self.head_dim, self.mot_head_dim)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, max_heads_kv, max_head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, max_heads_kv, max_head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens
        if query_lens.dim() > 1:
            query_lens = query_lens.view(-1)
        if key_values_lens.dim() > 1:
            key_values_lens = key_values_lens.view(-1)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if packed_query_states.shape[0] != merged_key_states.shape[0] and mode == "gen":
            T_q = packed_query_states.shape[0]          
            T_k = merged_key_states.shape[0]      
            prefix_len = T_k - T_q                
            attn_curr = attention_mask[0]         
            attention_mask = torch.zeros((T_q, T_k), device=attn_curr.device, dtype=attn_curr.dtype)
            attention_mask[:, prefix_len:] = attn_curr
            attention_mask = attention_mask.unsqueeze(0)
            packed_attn_output, attn_weights = attention_interface(
                self,
                packed_query_states.transpose(1,0).unsqueeze(0),
                merged_key_states.transpose(1,0).unsqueeze(0),
                merged_value_states.transpose(1,0).unsqueeze(0),
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
        else: 
            packed_attn_output, attn_weights = attention_interface(
                self,
                packed_query_states.transpose(1,0).unsqueeze(0),
                merged_key_states.transpose(1,0).unsqueeze(0),
                merged_value_states.transpose(1,0).unsqueeze(0),
                attention_mask=None if attention_mask is None else attention_mask[0].unsqueeze(0),
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
        if mode == 'und':
            packed_attn_output = packed_attn_output.reshape(-1, self.config.num_attention_heads * self.head_dim).contiguous()
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen':
            # Handle mixed head dimensions for generation mode
            packed_attn_output = packed_attn_output.squeeze(0)  # [720, 32, 128]
            
            packed_attn_output_proj = packed_attn_output.new_zeros(packed_attn_output.shape[0], self.config.hidden_size)
            
            if len(packed_text_indexes) > 0:
                text_output = packed_attn_output[packed_text_indexes, :self.config.num_attention_heads, :self.head_dim]
                text_output = text_output.reshape(-1, self.config.num_attention_heads * self.head_dim)
                packed_attn_output_proj[packed_text_indexes] = self.o_proj(text_output)
            
            if len(packed_vae_token_indexes) > 0:
                vae_output = packed_attn_output[packed_vae_token_indexes, :self.mot_num_heads, :self.mot_head_dim]
                vae_output = vae_output.reshape(-1, self.mot_num_heads * self.mot_head_dim)
                packed_attn_output_proj[packed_vae_token_indexes] = self.o_proj_mot_gen(vae_output)
            
            packed_attn_output = packed_attn_output_proj

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen3VLDecoderLayer(nn.Module):
    """
    Qwen3VL Decoder Layer with NavIT packed sequence support.
    
    This class extends the standard Qwen3VL decoder layer to support packed sequence
    processing for efficient batch handling. Follows the exact pattern from qwen3_navit.py
    but adapted for Qwen3VL components.
    
    Args:
        config: Qwen3VLTextConfig with NavIT parameters
        layer_idx: Layer index for this decoder layer
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reasoning = False

    def forward(self, *args, **kwargs):
        """Forward method that dispatches to training or inference based on model state."""
        if self.training or self.reasoning:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Training forward pass with packed sequence processing."""
        
        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence 

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        attention_mask,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        """Inference forward pass with packed sequence processing and KV caching."""

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            #attention_mask=attention_mask,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen3VLMoTDecoderLayer(nn.Module):
    """
    Qwen3VL MoT (Mixture of Tokens) Decoder Layer with NavIT packed sequence support.
    
    This class extends the standard decoder layer to support MoT where different token types
    use different attention and MLP configurations. Understanding tokens use standard config
    while generation tokens use optimized MoT config.
    
    Args:
        config: Qwen3VLTextConfig with NavIT and MoT parameters
        layer_idx: Layer index for this decoder layer
        attn_module: Attention module class (default: PackedAttentionMoT)
    """
    
    def __init__(
        self, 
        config, 
        layer_idx: Optional[int] = None, 
        attn_module: Optional[type] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        #self.freeze_und = config.freeze_und
        self.freeze_und = False

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        
        # Create modified config for MoT MLP with custom intermediate_size
        mot_mlp_config = type(config)(**config.__dict__)
        mot_mlp_config.intermediate_size = config.mot_intermediate_size # e.g.: from 22016 -> 11008
        self.mlp_mot_gen = Qwen3VLTextMLP(mot_mlp_config)
        
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reasoning = False

    def forward(self, *args, **kwargs):
        """Forward method that dispatches to training or inference based on model state."""
        if self.training or self.reasoning:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        """Training forward pass with MoT token processing."""

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_mot_gen(packed_sequence[packed_gen_token_indexes])

        # Self Attention
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
    
        packed_sequence_[packed_gen_token_indexes] = self.mlp_mot_gen(
            self.post_attention_layernorm_mot_gen(packed_sequence[packed_gen_token_indexes])
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def forward_inference(
        self,
        attention_mask,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        """Inference forward pass with MoT token processing."""
        
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            # packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            # packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_ = self.input_layernorm_mot_gen(packed_query_sequence)
            #packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_mot_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            attention_mask = attention_mask,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_mot_gen(packed_vae_query_sequence).to(torch.bfloat16)

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_mot_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


# ================================================================================
# PackedAttention completed - adapting qwen3_navit.py pattern to Qwen3VL
# Key differences from qwen3_navit.py implementation:
# 1. Inherits from Qwen3VLTextAttention (vs Qwen3Attention in qwen3_navit)
# 2. QK normalization is ALWAYS enabled in Qwen3VL (vs conditional in qwen3_navit)
# 3. No need to add q_norm/k_norm - already provided by parent class
# 4. Parent class provides: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
# 5. PackedAttentionMoT adds MoT-specific projections and normalization
# 6. Qwen3VLDecoderLayer uses Qwen3VLTextMLP and Qwen3VLTextRMSNorm
# ================================================================================


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """
    Qwen3VL Text Model with NavIT packed sequence support.
    
    This model handles the text processing portion of Qwen3VL with optimizations
    for packed sequences and Flash Attention.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_mot = 'MoT' in config.layer_module
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_mot:
            self.norm_mot_gen = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.reasoning = False
        
        # Initialize weights and apply final processing
        self.post_init()

    def set_reasoning_mode_all(self, enable: bool = True):

        for module in self.modules():
            if hasattr(module, "reasoning"):
                module.reasoning = enable

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(self, *args, **kwargs):
        if self.training or self.reasoning:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)
    
    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        if self.config.freeze_und:#57896
            packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()
        
        # create position embeddings to be shared across the decoder layers
        # Handle both 1D and 3D position_ids formats
        if packed_position_ids.dim() == 1:
            # Old 1D format: add batch dimension
            cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        elif packed_position_ids.dim() == 2 and packed_position_ids.shape[0] == 3:
            # New 3D format: shape [3, seq_len] needs batch dimension -> [3, 1, seq_len]
            cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(1))
        elif packed_position_ids.dim() == 3 and packed_position_ids.shape[0] == 3:
            # Official format: shape [3, batch_size, seq_len] - use directly
            cos, sin = self.rotary_emb(packed_sequence, packed_position_ids)
        else:
            # Fallback: assume batch dimension already included
            cos, sin = self.rotary_emb(packed_sequence, packed_position_ids)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)
          
        extra_inputs = {}
        if self.use_mot:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                packed_sequence = self._deepstack_process(
                    packed_sequence,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )      
        if self.use_mot:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_mot_gen(packed_sequence[packed_gen_token_indexes])
            return packed_sequence_
        else:
            return self.norm(packed_sequence)
    
    def forward_inference(
        self,
        attention_mask,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> BaseNavitOutputWithPast:
        
        # create position embeddings to be shared across the decoder layers
        # Handle both 1D and 3D position_ids formats
        if packed_query_position_ids.dim() == 1:
            # Old 1D format: add batch dimension
            cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        elif packed_query_position_ids.dim() == 2 and packed_query_position_ids.shape[0] == 3:
            # New 3D format: shape [3, seq_len] needs batch dimension -> [3, 1, seq_len]
            cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(1))
        elif packed_query_position_ids.dim() == 3 and packed_query_position_ids.shape[0] == 3:
            # Official format: shape [3, batch_size, seq_len] - use directly
            cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids)
        else:
            # Fallback: assume batch dimension already included
            cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)
        
        extra_inputs = {}
        if self.use_mot:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_vae_token_indexes is not None
                #assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )        
        for layer_idx, decoder_layer in enumerate(self.layers):
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                attention_mask=attention_mask,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                packed_query_sequence = self._deepstack_process(
                    packed_query_sequence,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        
        if self.use_mot:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_mot_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen3VLForConditionalGenerationMoT(Qwen3VLPreTrainedModel):
    """
    Qwen3VL For Conditional Generation with MoT (Mixture of Tokens) support.
    
    This model combines the text model with generation capabilities, supporting both
    standard and MoT (Mixture of Tokens) training/inference modes.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = None
        self.reasoning = False

        # Initialize weights and apply final processing
        self.post_init()

    def init_mot(self):
        """Initialize MoT parameters by copying from understanding parameters."""
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                original_name = name.replace("_mot_gen", "")
                if original_name in self.state_dict():
                    param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training or self.reasoning:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Training forward pass with packed sequence support."""

        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        attention_mask: list=None,
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> BaseNavitOutputWithPast:
        """Inference forward pass with KV caching and MoT support."""

        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            attention_mask=attention_mask,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return outputs

    def get_rope_index_fast_thinking(
        self, 
        input_ids: torch.Tensor, 
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer = None,
        spatial_merge_size = 2,
        num_learnable_tokens = 8
    ):
        """
        Calculate 3D position_ids for multi-dimensional RoPE (mRoPE) following official Qwen3VL implementation.
        """
        # Get spatial_merge_size from config and token IDs from tokenizer
        v_start_id = tokenizer.encode("<|vision_start|>", add_special_tokens=False)[0]
        img_pad_id = tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
        if tokenizer is None:
            raise ValueError("Tokenizer is required for get_rope_index")
            
        mrope_position_deltas = []
        
        if input_ids is not None and image_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0], 
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            
            for i, input_ids_seq in enumerate(total_input_ids):
                input_ids_seq = input_ids_seq[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids_seq == v_start_id).squeeze(1)
                vision_tokens = input_ids_seq[vision_start_indices + 1] if len(vision_start_indices) > 0 else torch.tensor([], device=input_ids_seq.device)
                image_nums = (vision_tokens == img_pad_id).sum() if len(vision_tokens) > 0 else 0
                input_tokens = input_ids_seq.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums
                
                for _ in range(image_nums):
                    if img_pad_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(img_pad_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    
                    if ed_image < len(input_tokens):
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1], 
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                        
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        
                        # Add text positions before vision
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        
                        # Add vision positions with official 3D layout
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                
                # Add remaining text positions
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            ## add learnable token positions at the end
            if num_learnable_tokens > 0:
                max_pos = position_ids.max(dim=-1).values[0]   # [B]
                start = max_pos + 1                            # [B]
                offset = torch.arange(
                    num_learnable_tokens,
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                )
                extra_1d = start.view(-1, 1) + offset.view(1, -1)
                extra = extra_1d.view(1, -1, num_learnable_tokens).expand(3, -1, -1)
                position_ids = torch.cat([position_ids, extra], dim=-1)

            # Return with batch dimension preserved (shape [3, batch_size, seq_len])
            return position_ids, mrope_position_deltas
        else:
            # Fallback for no images
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                seq_len = input_ids.shape[1]
                batch_size = input_ids.shape[0]
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(3, batch_size, -1)
            
            rope_deltas = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
            return position_ids, rope_deltas

    def get_rope_index(
        self, 
        input_ids: torch.Tensor, 
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer = None
    ):
        """
        Calculate 3D position_ids for multi-dimensional RoPE (mRoPE) following official Qwen3VL implementation.
        """
        # Get spatial_merge_size from config and token IDs from tokenizer
        spatial_merge_size = getattr(self.config, 'vision_config', {}).get('spatial_merge_size', 2)
        
        if tokenizer is None:
            raise ValueError("Tokenizer is required for get_rope_index")
            
        image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        video_token_id = tokenizer.convert_tokens_to_ids('<|video_pad|>')
        vision_start_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        mrope_position_deltas = []
        
        if input_ids is not None and image_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0], 
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            
            for i, input_ids_seq in enumerate(total_input_ids):
                input_ids_seq = input_ids_seq[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids_seq == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_seq[vision_start_indices + 1] if len(vision_start_indices) > 0 else torch.tensor([], device=input_ids_seq.device)
                image_nums = (vision_tokens == image_token_id).sum() if len(vision_tokens) > 0 else 0
                input_tokens = input_ids_seq.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums
                
                for _ in range(image_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    
                    if ed_image < len(input_tokens):
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1], 
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                        
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        
                        # Add text positions before vision
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        
                        # Add vision positions with official 3D layout
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                
                # Add remaining text positions
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            
            # Return with batch dimension preserved (shape [3, batch_size, seq_len])
            return position_ids, mrope_position_deltas
        else:
            # Fallback for no images
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                seq_len = input_ids.shape[1]
                batch_size = input_ids.shape[0]
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(3, batch_size, -1)
            
            rope_deltas = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
            return position_ids, rope_deltas


# Decoder layer mappings
Decoder_layer_dict = {
    "Qwen3VLDecoderLayer": Qwen3VLDecoderLayer,
    "Qwen3VLMoTDecoderLayer": Qwen3VLMoTDecoderLayer,
}