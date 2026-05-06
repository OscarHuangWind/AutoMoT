
from copy import deepcopy
from typing import List, Dict, Optional, Union, Any
from PIL import Image
import torch
from data.reasoning.data_utils import pil_img2rgb, create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask
from modeling.automot.qwen3_navit import NaiveCache
import os
import numpy as np

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

USER_PROMPT = '''<|im_start|>system\nYou are a mature and professional driver.<|im_end|>\n<|im_start|>user\n'''
# USER_PROMPT = '''<|im_start|>user\n'''

class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, max_num_tokens=2816, max_num_reasoning_token=64, lidar_reasoning_token=197, visual_gen=False, visual_und=True):
        torch.set_num_threads(1)
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.max_num_tokens = max_num_tokens
        if visual_gen:
            self.query_tokens = self.model.reasoning_queries(torch.arange(self.model.reasoning_query_tokens, device=self.model.device))
            self.reasoning_query_tokens = self.model.reasoning_queries(torch.arange(self.model.reasoning_query_tokens, device=self.model.device))
            self.route_query_tokens = self.model.route_queries(torch.arange(20, device=self.model.device))
            self.waypoint_query_tokens = self.model.waypoint_queries(torch.arange(6, device=self.model.device))
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        

        return gen_context

    @torch.no_grad()
    def reasoning_update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes        

        return generation_input, gen_context

    @torch.no_grad()
    def update_context_qwen3vl(self, user_prompt, instruction_prompt, image_list, gen_context):

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_generation(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            user_prompt=user_prompt,
            instruction_prompt=instruction_prompt, 
            images=image_list,
            new_token_ids=self.new_token_ids,
            tokenizer=self.tokenizer,
        )
        past_key_values, packed_position_ids = self.model.forward_cache_update_generation(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        gen_context['packed_position_ids'] = packed_position_ids
        
        return gen_context

    @torch.no_grad()
    def gen_fast_reasoning_trajectory(
        self,
        last_hidden_state: torch.FloatTensor,
        v_target_point: torch.Tensor,
        packed_action_token_indexes: torch.LongTensor,# [num_reasoning]
    ):
        action_hidden_states = last_hidden_state[packed_action_token_indexes]   # [num_reasoning, hidden_dim]
        action = self.model.waypoints_head(action_hidden_states)  # [num_reasoning, num_points*2]
        
        pred_traj = action.reshape(-1, 6, 2)  # [num_reasoning, num_points, 2]

        return pred_traj

    @torch.no_grad()
    def traj_metric(
        self,
        last_hidden_state: torch.FloatTensor,
        packed_action_token_indexes: torch.LongTensor,  # [num_reasoning]
        gt: torch.LongTensor,
    ):
        action_hidden_states = last_hidden_state[packed_action_token_indexes]   # [num_reasoning, hidden_dim]
        action = self.model.traj_head(action_hidden_states)  # [num_reasoning, num_points*2]
        
        if isinstance(gt, list):
            gt = torch.tensor(gt, dtype=torch.float32, device=action.device)
        
        pred_traj = action.reshape(-1, 6, 2)  # [num_reasoning, num_points, 2]
        gt_traj = gt.reshape(-1, 6, 2)  # [num_reasoning, num_points, 2]
        print('pred_traj', pred_traj)
        print('gt_traj', gt_traj)
        l2_distances = torch.norm(pred_traj - gt_traj, dim=-1)  # [num_reasoning, num_points]
        
        l2_1s = l2_distances[:, :2].mean().item() if l2_distances.shape[1] >= 2 else 0.0
        l2_2s = l2_distances[:, :4].mean().item() if l2_distances.shape[1] >= 4 else 0.0
        ade = l2_distances.mean().item()
        
        # Final displacement error (FDE) - last timestep
        fde = l2_distances[:, -1].mean().item()
        
        return {
            'l2_1s': l2_1s,
            'l2_2s': l2_2s,
            'ade': ade,
            'fde': fde,
            'pred_traj':pred_traj,
        }
    
    @torch.no_grad()
    def gen_fast_reasoning_decision(
        self,
        last_hidden_state: torch.FloatTensor,
        packed_reasoning_token_indexes: torch.LongTensor,# [num_reasoning]
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        reasoning_hidden_states = last_hidden_state[packed_reasoning_token_indexes]   # [num_reasoning, vocab]
        reasoning_logits = self.model.language_model.lm_head(reasoning_hidden_states)
        num_tokens_per_sample = self.model.reasoning_query_tokens
        B_ce = reasoning_logits.shape[0] // num_tokens_per_sample
        
        tokenizer = getattr(self, 'tokenizer', None) or getattr(self.config, 'tokenizer', None)
        action_words = ["stop", "keep"]
        action_token_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in action_words]
        
        logits_per_sample = reasoning_logits.view(B_ce, num_tokens_per_sample, -1)
        second_token_logits = logits_per_sample[:, 1, :]
        action_logits = second_token_logits[:, action_token_ids]
        pred_indices = action_logits.argmax(dim=-1)
        # import pdb;pdb.set_trace()
        if do_sample:
            probs = torch.softmax(reasoning_logits / temperature, dim=-1)
            pred_token_ids = torch.multinomial(probs, 1).squeeze(-1)
        else:
            pred_token_ids = reasoning_logits.argmax(dim=-1)
        
        # print('pred_token_ids',pred_token_ids)
        answer_ids = self.extract_all_bos_eos_with_special(pred_token_ids.tolist(), self.new_token_ids['bos_token_id'], self.new_token_ids['eos_token_id'])
        text = self.tokenizer.decode(answer_ids[0], skip_special_tokens=False)
        print('text',text)

        return text, reasoning_hidden_states

    @torch.no_grad()
    def based_kv_cache_context_fast_qwen3vl(self, image_list, gen_context, reasoning_tokens, action_tokens, v_target_point):

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        
        generation_input, kv_lens, ropes = self.model.prepare_fast_kvcache(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=image_list,
            new_token_ids=self.new_token_ids,
            tokenizer=self.tokenizer,
            reasoning_learnable_tokens=reasoning_tokens,
            action_learnable_tokens=action_tokens,
            target_point_max_num_tokens=target_point_max_num_tokens,
            v_num_token=v_num_token,
            num_route_tokens=num_route_tokens,
            num_traj_tokens=num_traj_tokens,
        )

        old_pos = gen_context['packed_position_ids']           # [3, 1, L_old]
        new_pos = generation_input['packed_position_ids']      # [3, 1, L_new]
        attention_mask = generation_input['nested_attention_masks']
        offset = old_pos.max(dim=-1, keepdim=True).values + 1  # [3, 1, 1]
        new_pos = new_pos + offset
        query_lens_fast = new_pos.shape[2] 
        key_values_lens = torch.as_tensor(gen_context['kv_lens'], dtype=torch.int, device=new_pos.device)
        generation_input['packed_position_ids'] = new_pos
        gen_context['packed_position_ids'] = torch.cat([old_pos, new_pos], dim=-1)
        packed_key_value_indexes = generation_input['packed_key_value_indexes']
        # key_values_lens = generation_input['key_values_lens']
        # image_tensor_list = generation_input['packed_vit_tokens']
        # image_grid_thw_list = generation_input['packed_vit_position_ids']
        # packed_vit_token_indexes = generation_input['packed_vit_token_indexes']
        # vit_token_seqlens = generation_input['vit_token_seqlens']
        packed_bev_token_indexes = generation_input['packed_bev_token_indexes']
        packed_reasoning_token_indexes = generation_input['packed_reasoning_token_indexes']
        packed_action_token_indexes = generation_input['packed_action_token_indexes']
        packed_position_ids = generation_input['packed_position_ids']
        packed_text_indexes = generation_input['packed_text_indexes']
        packed_text_ids = generation_input['packed_text_ids']
        #### QH:add text embedding ####
        packed_text_embedding = self.model.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence_fast = packed_text_embedding.new_zeros(size=(query_lens_fast, self.model.hidden_size))
        packed_sequence_fast[packed_text_indexes] = packed_text_embedding
        #### QH:add vit token embedding ####
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed , deepstack_image_embeds = self.model.get_image_features(image_tensor_list, image_grid_thw_list)
        packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
        deepstack_visual_embeds = deepstack_image_embeds
        packed_sequence_fast[packed_vit_token_indexes] = packed_vit_token_embed
        visual_pos_masks = torch.zeros(
            packed_sequence_fast.shape[0], 
            dtype=torch.bool, 
            device=packed_sequence_fast.device
        )
        visual_pos_masks[packed_vit_token_indexes] = True
        #### QH:add learnable token embedding ####
        packed_sequence_fast[packed_reasoning_token_indexes] = self.model.reasoning_projector(self.reasoning_query_tokens)
        packed_sequence_fast[packed_action_token_indexes] = self.model.action_projector(self.action_query_tokens)
        packed_query_token_indexes = torch.cat([packed_reasoning_token_indexes, packed_action_token_indexes], dim=0)
        packed_position_ids = torch.tensor(packed_position_ids, dtype=torch.int64, device=self.model.device)
        packed_query_indexes_fast = torch.cat([packed_text_indexes, packed_query_token_indexes], dim=0)
        packed_query_indexes = packed_query_indexes_fast + key_values_lens.sum().long()
        extra_inputs = {"mode": "gen"}
        
        last_hidden_state = self.model.language_model.forward_inference(
            packed_query_sequence=packed_sequence_fast,
            query_lens=torch.tensor([query_lens_fast], device=packed_key_value_indexes.device),
            attention_mask = attention_mask,
            packed_query_position_ids=new_pos,  
            packed_query_indexes=packed_query_indexes,
            packed_vae_token_indexes=packed_query_indexes_fast,  
            past_key_values=past_key_values, 
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            packed_text_indexes = torch.empty(0, dtype=torch.long, device=packed_key_value_indexes.device),
            update_past_key_values=False,
            is_causal=False,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **extra_inputs,
        )
        v_target_point = v_target_point.to(packed_sequence_fast.device)
        packed_anchor_token_indexes = packed_action_token_indexes[0]
        packed_action_token_indexes = packed_action_token_indexes[-1]
        pred_anchor = self.model.anchor_head(last_hidden_state[0][packed_anchor_token_indexes])
        v = v_target_point[:1]
        gen_text = self.gen_fast_reasoning_decision(last_hidden_state[0], packed_reasoning_token_indexes, do_sample=False, temperature=0.0)
        # gen_traj = self.gen_fast_reasoning_trajectory(last_hidden_state[0], v_target_point, packed_action_token_indexes)
        gen_traj = self.gen_fast_reasoning_trajectory(last_hidden_state = last_hidden_state[0], v = v, pred_anchor = pred_anchor, packed_action_token_indexes = packed_action_token_indexes)
        return gen_text, gen_traj

    @torch.no_grad()
    def based_kv_cache_context_fast_qwen3vl_dp(self, trans_feat, gen_context, reasoning_tokens, action_tokens, v_target_point, target_point_max_num_tokens=2, v_num_token=1, num_route_tokens=20, num_traj_tokens=6):

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        
        generation_input, kv_lens, ropes = self.model.prepare_fast_kvcache(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            trans_feat=trans_feat,
            new_token_ids=self.new_token_ids,
            tokenizer=self.tokenizer,
            reasoning_learnable_tokens=reasoning_tokens,
            action_learnable_tokens=action_tokens,
            target_point_max_num_tokens=target_point_max_num_tokens, 
            v_num_token=v_num_token,
            num_route_tokens=num_route_tokens,
            num_traj_tokens=num_traj_tokens,
        )

        old_pos = gen_context['packed_position_ids']           # [3, 1, L_old]
        new_pos = generation_input['packed_position_ids']      # [3, 1, L_new]
        attention_mask = generation_input['nested_attention_masks']
        offset = old_pos.max(dim=-1, keepdim=True).values + 1  # [3, 1, 1]
        new_pos = new_pos + offset
        query_lens_fast = new_pos.shape[2] 
        key_values_lens = torch.as_tensor(gen_context['kv_lens'], dtype=torch.int, device=new_pos.device)
        generation_input['packed_position_ids'] = new_pos
        gen_context['packed_position_ids'] = torch.cat([old_pos, new_pos], dim=-1)
        packed_key_value_indexes = generation_input['packed_key_value_indexes']

        # key_values_lens = generation_input['key_values_lens']
        # image_tensor_list = generation_input['packed_vit_tokens']
        # image_grid_thw_list = generation_input['packed_vit_position_ids']
        # packed_vit_token_indexes = generation_input['packed_vit_token_indexes']
        # vit_token_seqlens = generation_input['vit_token_seqlens']
        packed_bev_token_indexes = generation_input['packed_bev_token_indexes']
        packed_reasoning_token_indexes = generation_input['packed_reasoning_token_indexes']
        packed_action_token_indexes = generation_input['packed_action_token_indexes']
        packed_position_ids = generation_input['packed_position_ids']
        # packed_text_indexes = generation_input['packed_text_indexes']
        packed_text_ids = generation_input['packed_text_ids']
        packed_target_point_indexes = generation_input['target_point_indexes']
        packed_v_indexes = generation_input['v_indexes']
        #### QH:add text embedding ####
        packed_text_embedding = self.model.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence_fast = packed_text_embedding.new_zeros(size=(query_lens_fast, self.model.hidden_size))
        # packed_sequence_fast[packed_text_indexes] = packed_text_embedding
        #### QH:add vit token embedding ####
        # cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        # cu_seqlens = cu_seqlens.to(torch.int32)
        # max_seqlen = torch.max(vit_token_seqlens).item()
        # packed_vit_token_embed , deepstack_image_embeds = self.model.get_image_features(image_tensor_list, image_grid_thw_list)
        # packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
        # deepstack_visual_embeds = deepstack_image_embeds
        # packed_sequence_fast[packed_vit_token_indexes] = packed_vit_token_embed
        # visual_pos_masks = torch.zeros(
        #     packed_sequence_fast.shape[0], 
        #     dtype=torch.bool, 
        #     device=packed_sequence_fast.device
        # )
        # visual_pos_masks[packed_vit_token_indexes] = True
        trans_feat = trans_feat.to(self.model.language_model.model.embed_tokens.weight.device)
        x = trans_feat
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == 1512, f"expect 1512 channels, got {C}"
            assert H * W == 64, f"expect 8x8=64 spatial positions, got {H}x{W}={H*W}"
            
            # [B, 1512, 8, 8] -> [B, 1512, 64] -> [B, 64, 1512]
            x = x.flatten(2).transpose(1, 2)  # [B, 64, 1512]
            
            # [B, 64, 1512] -> [B*64, 1512]
            bev_tok = x.reshape(-1, 1512)  # [B*64, 1512]
            
            bev_tok = self.model.bev_encoder_proj(bev_tok)  # [B*64, 2560]

        elif x.dim() == 3:
            B, N, C = x.shape
            assert N == 64 and C == 1512, f"expect [B, 64, 1512], got [{B}, {N}, {C}]"
            bev_tok = x.reshape(-1, 1512)  # [B*64, 1512]
            bev_tok = self.model.bev_encoder_proj(bev_tok)  # [B*64, 2560]

        elif x.dim() == 2:
            N, C = x.shape
            assert C == 1512, f"expect 1512 dim, got {C}"
            bev_tok = self.model.bev_encoder_proj(x)  # [N, 2560]

        else:
            raise ValueError(f"bev_feature dim must be 2/3/4, got {x.dim()}")
        packed_sequence_fast[packed_bev_token_indexes] = bev_tok
        #### QH:add learnable token embedding ####
        v_target_point = v_target_point.to(packed_sequence_fast.device)
        # Handle batch dimension: squeeze if (1, 5) -> (5,)
        if v_target_point.dim() == 2:
            v_target_point = v_target_point.squeeze(0)  # (5,)
        packed_route_token_indexes = packed_action_token_indexes[:20]
        packed_wp_token_indexes = packed_action_token_indexes[20:]
        v = v_target_point[0].unsqueeze(0).to(packed_sequence_fast.device)  # (1,)
        target_points = v_target_point[1:5].reshape(1, 2, 2).to(packed_sequence_fast.device) 
        target_point_embed = self.model.target_point_encoder(target_points)  
        packed_target_point_embed = target_point_embed.reshape(-1, target_point_embed.size(-1))
        packed_sequence_fast[packed_target_point_indexes] = packed_target_point_embed  
        velocity_embed = self.model.velocity_encoder(v)  # (B, C)
        packed_sequence_fast[packed_v_indexes] = velocity_embed
        packed_sequence_fast[packed_reasoning_token_indexes] = self.model.reasoning_projector(self.reasoning_query_tokens)
        packed_sequence_fast[packed_route_token_indexes] = self.model.route_projector(self.route_query_tokens)
        packed_sequence_fast[packed_wp_token_indexes] = self.model.waypoint_projector(self.waypoint_query_tokens)
        packed_query_token_indexes = torch.cat([packed_bev_token_indexes, packed_target_point_indexes, packed_v_indexes, packed_reasoning_token_indexes, packed_route_token_indexes, packed_wp_token_indexes], dim=0)
        packed_position_ids = torch.tensor(packed_position_ids, dtype=torch.int64, device=self.model.device)
        packed_query_indexes_fast = packed_query_token_indexes
        packed_query_indexes = packed_query_indexes_fast + key_values_lens.sum().long()
        extra_inputs = {"mode": "gen"}
        
        last_hidden_state = self.model.language_model.forward_inference(
            packed_query_sequence=packed_sequence_fast,
            query_lens=torch.tensor([query_lens_fast], device=packed_key_value_indexes.device),
            attention_mask = attention_mask,
            packed_query_position_ids=new_pos,  
            packed_query_indexes=packed_query_indexes,
            packed_vae_token_indexes=packed_query_indexes_fast,  
            past_key_values=past_key_values, 
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            packed_text_indexes = torch.empty(0, dtype=torch.long, device=packed_key_value_indexes.device),
            update_past_key_values=False,
            is_causal=False,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **extra_inputs,
        )
        v_target_point = v_target_point.to(packed_sequence_fast.device)
        route = self.model.route_head(last_hidden_state[0][packed_route_token_indexes])
        route = route.view(-1, 20, 2)  # (11, 20, 2)
        gen_text, reasoning_hidden_states = self.gen_fast_reasoning_decision(last_hidden_state[0], packed_reasoning_token_indexes, do_sample=False, temperature=0.0)
        gen_traj = self.gen_fast_reasoning_trajectory(last_hidden_state[0], v_target_point, packed_wp_token_indexes)
        gen_traj = gen_traj.cumsum(dim=1)
        print("gen_traj",gen_traj)
        # gen_traj = self.gen_fast_reasoning_trajectory(last_hidden_state = last_hidden_state[0], v = v, pred_anchor = pred_anchor, packed_action_token_indexes = packed_action_token_indexes)
        return gen_text, gen_traj, route, reasoning_hidden_states

    @torch.no_grad()
    def update_kv_cache_context_qwen3vl(self, user_prompt, instruction_prompt, image_list, gen_context):

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_kv_cache(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            user_prompt=user_prompt,
            instruction_prompt=instruction_prompt, 
            images=image_list,
            new_token_ids=self.new_token_ids,
            tokenizer=self.tokenizer,
        )
        past_key_values, packed_position_ids = self.model.forward_cache_update_generation(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        gen_context['packed_position_ids'] = packed_position_ids
        
        return gen_context

    @torch.no_grad()
    def update_context_qwen3vl_fast_thinking(self, user_prompt, instruction_prompt, image_list, gen_context):

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_fast_generation(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            user_prompt=user_prompt,
            instruction_prompt=instruction_prompt, 
            images=image_list,
            new_token_ids=self.new_token_ids,
            num_learnable_tokens=9,
            tokenizer=self.tokenizer,
        )
        #past_key_values = self.model.forward_cache_update_generation(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        #gen_context['past_key_values'] = past_key_values
        
        return generation_input, gen_context
    
    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=False, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images_qwen3vl(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def reasoning_update_context_image(self, image, gen_context, vae=True, vit=True, lidar=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
    
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_gen_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        
        return generation_input, gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image
    
    @torch.no_grad()
    def gen_fast_reasoning(
        self,
        last_hidden_state: torch.FloatTensor,
        packed_reasoning_token_indexes: torch.LongTensor,# [num_reasoning]
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        reasoning_hidden_states = last_hidden_state[packed_reasoning_token_indexes]   # [num_reasoning, vocab]
        reasoning_logits = self.model.language_model.lm_head(reasoning_hidden_states)
        # import pdb;pdb.set_trace()
        if do_sample:
            probs = torch.softmax(reasoning_logits / temperature, dim=-1)
            pred_token_ids = torch.multinomial(probs, 1).squeeze(-1)
        else:
            pred_token_ids = reasoning_logits.argmax(dim=-1)
        
        # print('pred_token_ids',pred_token_ids)
        answer_ids = self.extract_all_bos_eos_with_special(pred_token_ids.tolist(), self.new_token_ids['bos_token_id'], self.new_token_ids['eos_token_id'])
        text = self.tokenizer.decode(answer_ids[0], skip_special_tokens=False)
        print('text',text)

        return text
    
    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 32, do_sample: bool = False, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        
        output = self.tokenizer.decode(unpacked_latent[:,0])
        # print("raw output", output)
        
        # Handle different generation formats
        if '<|im_end|>' in output:
            output = output.split('<|im_end|>')[0]
        if '<|im_start|>' in output:
            output = output.split('<|im_start|>')[-1]
        
        # Clean up newlines at the beginning
        output = output.lstrip('\n ')
        
        return output

    def extract_all_bos_eos_with_special(self, valid_ids, bos_token_id, eos_token_id):
        outputs = []
        temp = []
        in_span = False
        for tid in valid_ids:
            if tid == bos_token_id:
                if in_span and temp:
                    outputs.append(temp)
                in_span = True
                temp = [bos_token_id]  
            elif tid == eos_token_id and in_span:
                temp.append(eos_token_id) 
                outputs.append(temp)
                in_span = False
            elif in_span:
                temp.append(tid)
        if in_span and temp:
            outputs.append(temp)
        return outputs

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        reasoning_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.0,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    
                    gen_context = self.update_context_text(input_term, gen_context)
                    # cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    # For Qwen3VL, keep original PIL image for vit processing
                    original_image = input_term
                    if not understanding_output:  # Only transform for VAE if needed
                        input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                        image_shapes = input_term.size[::-1]
                    else:
                        image_shapes = original_image.size[::-1]
                    
                    gen_context = self.update_context_image(original_image, gen_context, vae=not understanding_output)
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    #for 2nd transformer
    @torch.no_grad()
    def Sequence_Reasoning_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        reasoning_output=True,
        max_num_reasoning_token=8,
        lidar_reasoning_token=197,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.0,
        timestep_shift=3.0,
        num_timesteps=50,
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        if reasoning_output:
            self.model.language_model.reasoning = True
            self.model.language_model.model.set_reasoning_mode_all(True)

        output_list = {}
        gen_context = self.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                
                if reasoning_output:
                    gen_context = self.reasoning_update_context_text(system_prompt, gen_context)
            
            generation_text_input_dict_list = []
            image_list = []
            user_prompt = USER_PROMPT
            for input_term in input_lists:
                if isinstance(input_term, str):
                    instruction_prompt = input_term
                elif isinstance(input_term, Image.Image):
                    image_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")   
            generation_input, gen_context = self.update_context_qwen3vl_fast_thinking(user_prompt, instruction_prompt, image_list, gen_context)

            # image vit + learnable tokens + lidar bev
            if reasoning_output:
                #### QH:create sequence and config needed ####
                curr_rope_id = 0
                packed_text_ids = generation_input['packed_text_ids']
                packed_text_indexes = generation_input['packed_text_indexes']
                image_tensor_list = generation_input['packed_vit_tokens']
                image_grid_thw_list = generation_input['packed_vit_position_ids']
                packed_vit_token_indexes = generation_input['packed_vit_token_indexes']
                vit_token_seqlens = generation_input['vit_token_seqlens']
                packed_learnable_token_indexes = generation_input['packed_learnable_token_indexes']
                packed_position_ids = generation_input['packed_position_ids']
                packed_und_text_indexes = generation_input['packed_und_text_indexes']
                packed_gen_text_indexes = generation_input['packed_gen_text_indexes']
                packed_und_vit_token_indexes = generation_input['packed_und_vit_token_indexes']
                packed_gen_vit_token_indexes = generation_input['packed_gen_vit_token_indexes']
                sample_lens = generation_input['curr']
                #### QH:add text embedding ####
                packed_text_embedding = self.model.language_model.model.embed_tokens(packed_text_ids)
                packed_sequence = packed_text_embedding.new_zeros(size=(sample_lens, self.model.hidden_size))
                packed_sequence[packed_text_indexes] = packed_text_embedding
                #### QH:add vit token embedding ####
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(vit_token_seqlens).item()
                packed_vit_token_embed , deepstack_image_embeds = self.model.get_image_features(image_tensor_list, image_grid_thw_list)
                packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
                deepstack_visual_embeds = deepstack_image_embeds
                packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
                visual_pos_masks = torch.zeros(
                    packed_sequence.shape[0], 
                    dtype=torch.bool, 
                    device=packed_sequence.device
                )
                visual_pos_masks[packed_vit_token_indexes] = True
                #### QH:add learnable token embedding ####
                packed_sequence[packed_learnable_token_indexes] = self.model.reasoning_projector(self.query_tokens)
                packed_position_ids = packed_position_ids.to(self.model.device, dtype=torch.long)
                #### QH:prepare attention mask ####
                attention_mask = generation_input['nested_attention_masks']
                ## for debug inference time 
                packed_und_token_indexes=torch.cat([packed_und_text_indexes, packed_und_vit_token_indexes], dim=0)
                packed_gen_token_indexes=torch.cat([packed_gen_text_indexes, packed_gen_vit_token_indexes, packed_learnable_token_indexes], dim=0)

                ## for debug inference time
                extra_inputs = {}
                extra_inputs.update(
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )
                last_hidden_state = self.model.language_model(
                    packed_sequence=packed_sequence,
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    visual_pos_masks=visual_pos_masks,
                    packed_position_ids=packed_position_ids,
                    **extra_inputs,
                )

                gen_text = self.gen_fast_reasoning(last_hidden_state, packed_learnable_token_indexes, do_sample=False, temperature=0.0)
                output_list["text"]=gen_text

                with open("output_inference.log", "a", encoding="utf-8") as f:
                    f.write("\n\ninference text: " + str(gen_text) + "\n")

        return output_list
    #2nd transformer save hidden_states for dp
    @torch.no_grad()
    def Sequence_Reasoning_inference_dp(
        self,
        input_lists: List[Union[str, Image.Image]],
        resolved_lidar_paths: List[str],
        think=False,
        understanding_output=False,
        reasoning_output=True,
        max_num_reasoning_token=8,
        lidar_reasoning_token=197,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.0,
        timestep_shift=3.0,
        num_timesteps=50,
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        if reasoning_output:
            self.model.language_model.reasoning = True
            self.model.language_model.model.set_reasoning_mode_all(True)

        output_list = {}
        gen_context = self.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                
                if reasoning_output:
                    gen_context = self.reasoning_update_context_text(system_prompt, gen_context)
            generation_text_input_dict_list = []
            image_list = []
            user_prompt = USER_PROMPT
            for input_term in input_lists:
                print('input term is:', input_term)
                if isinstance(input_term, str):
                    instruction_prompt = input_term
                elif isinstance(input_term, Image.Image):
                    image_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")   
            generation_input, gen_context = self.update_context_qwen3vl_fast_thinking(user_prompt, instruction_prompt, image_list, gen_context)

            # image vit + learnable tokens + lidar bev 
            if reasoning_output:
                #### QH:create sequence and config needed ####
                curr_rope_id = 0
                packed_text_ids = generation_input['packed_text_ids']
                packed_text_indexes = generation_input['packed_text_indexes']
                image_tensor_list = generation_input['packed_vit_tokens']
                image_grid_thw_list = generation_input['packed_vit_position_ids']
                packed_vit_token_indexes = generation_input['packed_vit_token_indexes']
                vit_token_seqlens = generation_input['vit_token_seqlens']
                packed_learnable_token_indexes = generation_input['packed_learnable_token_indexes']
                packed_position_ids = generation_input['packed_position_ids']
                packed_und_text_indexes = generation_input['packed_und_text_indexes']
                packed_gen_text_indexes = generation_input['packed_gen_text_indexes']
                packed_und_vit_token_indexes = generation_input['packed_und_vit_token_indexes']
                packed_gen_vit_token_indexes = generation_input['packed_gen_vit_token_indexes']
                sample_lens = generation_input['curr']
                #### QH:add text embedding ####
                packed_text_embedding = self.model.language_model.model.embed_tokens(packed_text_ids)
                packed_sequence = packed_text_embedding.new_zeros(size=(sample_lens, self.model.hidden_size))
                packed_sequence[packed_text_indexes] = packed_text_embedding
                #### QH:add vit token embedding ####
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(vit_token_seqlens).item()
                packed_vit_token_embed , deepstack_image_embeds = self.model.get_image_features(image_tensor_list, image_grid_thw_list)
                packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
                deepstack_visual_embeds = deepstack_image_embeds
                packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
                visual_pos_masks = torch.zeros(
                    packed_sequence.shape[0], 
                    dtype=torch.bool, 
                    device=packed_sequence.device
                )
                visual_pos_masks[packed_vit_token_indexes] = True
                #### QH:add learnable token embedding ####
                packed_sequence[packed_learnable_token_indexes] = self.model.reasoning_projector(self.query_tokens)
                packed_position_ids = torch.tensor(packed_position_ids, dtype=torch.int64, device=self.model.device)
                #### QH:prepare attention mask ####
                attention_mask = generation_input['nested_attention_masks']
                packed_und_token_indexes=torch.cat([packed_und_text_indexes, packed_und_vit_token_indexes], dim=0)
                packed_gen_token_indexes=torch.cat([packed_gen_text_indexes, packed_gen_vit_token_indexes, packed_learnable_token_indexes], dim=0)
                extra_inputs = {}
                extra_inputs.update(
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )
                last_hidden_state = self.model.language_model(
                    packed_sequence=packed_sequence,
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    visual_pos_masks=visual_pos_masks,
                    packed_position_ids=packed_position_ids,
                    **extra_inputs,
                )
                packed_dp_token_indexes=torch.cat([packed_gen_vit_token_indexes, packed_learnable_token_indexes], dim=0)
                save_for_dp = last_hidden_state[packed_dp_token_indexes]
                gen_text, pred_token_ids = self.gen_fast_reasoning(last_hidden_state, packed_learnable_token_indexes, do_sample=False, temperature=0.0)
                output_list["text"]=gen_text
                if resolved_lidar_paths and len(resolved_lidar_paths) > 0:
                    lidar_path = resolved_lidar_paths[0]
                    lidar_dir = os.path.dirname(lidar_path)
                    scene_root = os.path.dirname(lidar_dir)

                    dp_dir = os.path.join(scene_root, "dp_vl_feature")
                    os.makedirs(dp_dir, exist_ok=True)

                    lidar_filename = os.path.basename(lidar_path) 
                    stem, _ = os.path.splitext(lidar_filename)
                    save_path = os.path.join(dp_dir, f"{stem}.pt")
                    packed_gen_image_vit_token_indexes = packed_gen_vit_token_indexes[:512]  # only save image vit tokens
                    gen_vit_feat = last_hidden_state[packed_gen_image_vit_token_indexes]
                    query_feat   = last_hidden_state[packed_learnable_token_indexes]
                    num_q = query_feat.shape[0]

                    answer_token_indexes = torch.arange(
                        num_q, dtype=torch.long, device=query_feat.device
                    )

                    try:
                        ids_list = pred_token_ids.tolist()
                        IM_START_ID = 151644
                        IM_END_ID = 151645

                        if IM_START_ID in ids_list and IM_END_ID in ids_list:
                            s = ids_list.index(IM_START_ID)
                            e = ids_list.index(IM_END_ID, s + 1)

                            span_len = max(1, e - s + 1)
                            span_len = min(span_len, num_q)
                            answer_token_indexes = torch.arange(
                                span_len, dtype=torch.long, device=query_feat.device
                            )
                    except Exception as _:
                        pass

                    torch.save(
                        {
                            "gen_vit_tokens": gen_vit_feat.detach().cpu(),
                            "reasoning_query_tokens": query_feat.detach().cpu(),
                            "answer_token_indexes": answer_token_indexes.detach().cpu(),
                        },
                        save_path,
                    )
                    print(f"answer_token_indexes: {answer_token_indexes}")
                    print(f"[DP] saved feature dict to {save_path}, "
                          f"gen_vit_tokens: {gen_vit_feat.shape}, "
                          f"reasoning_query_tokens: {query_feat.shape}")
                with open("output_inference.log", "a", encoding="utf-8") as f:
                    f.write("\n\ninference text: " + str(gen_text) + "\n")

        return output_list
    
    # for 2nd transformer kv cache
    @torch.no_grad()
    def kv_cache_fixed_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=True,  
        max_think_token_n=1000,
        do_sample=False,
        reasoning_output=False,
        text_temperature=0.0,
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            user_prompt = USER_PROMPT
            image_list = []
            for input_term in input_lists:
                print('input term is:', input_term)
                if isinstance(input_term, str):
                    instruction_prompt = input_term
                elif isinstance(input_term, Image.Image):
                    image_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
                
            gen_context = self.update_kv_cache_context_qwen3vl(user_prompt, instruction_prompt, image_list, gen_context)

        return gen_context

    # for 2nd transformer kv cache
    @torch.no_grad()
    def kv_cache_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=True,  
        max_think_token_n=1000,
        do_sample=False,
        reasoning_output=False,
        reasoning_tokens=8,
        action_tokens=1,
        text_temperature=0.0,
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        slow_input_lists = input_lists[:-3] + input_lists[-1:]
        fast_input_lists = input_lists[-3:-1]# just current image and lidar
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            gen_context = self.kv_cache_fixed_inference(slow_input_lists)
            gen_text = self.based_kv_cache_context_fast_qwen3vl(fast_input_lists, gen_context, reasoning_tokens, action_tokens)
            output_list.append(gen_text)
        return output_list

    #for 2nd transformer kv cache
    @torch.no_grad()
    def kv_cache_inference_slow_fast(
        self,
        input_lists: List[Union[str, Image.Image]],
        v_target_point: Optional[torch.Tensor] = None,
        think=False,
        understanding_output=True,  
        max_think_token_n=1000,
        do_sample=False,
        reasoning_output=False,
        reasoning_tokens=8,
        action_tokens=1,
        text_temperature=0.0,
        image_shapes=(1024, 1024),
        frame_idx: int = 0,  
        slow_update_interval: int = 4, 
    ) -> List[Union[str, Image.Image]]:
        output_list = []
        # print('frame_idx',frame_idx)
        slow_input_lists = input_lists[:-3] + input_lists[-1:]
        fast_input_lists = input_lists[-3:-1]  # just current image and lidar

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # if frame_idx % slow_update_interval == 0 or not hasattr(self, '_cached_gen_context'):
            self._cached_gen_context = self.kv_cache_fixed_inference(slow_input_lists)
            
            gen_text, gen_traj = self.based_kv_cache_context_fast_qwen3vl(fast_input_lists, self._cached_gen_context, reasoning_tokens, action_tokens, v_target_point)
            
            output_dict = {
                'text': gen_text,
                'traj': gen_traj,
            }

            output_list.append(output_dict)

        return output_list     

    #for 2nd transformer kv cache
    @torch.no_grad()
    def kv_cache_inference_slow_fast_dp(
        self,
        input_lists: List[Union[str, Image.Image]],
        resolved_lidar_paths: List[str],
        v_target_point: Optional[torch.Tensor] = None,
        think=False,
        understanding_output=True,  
        max_think_token_n=1000,
        do_sample=False,
        reasoning_output=False,
        reasoning_tokens=8,
        trans_feat=None,
        action_tokens=26,
        text_temperature=0.0,
        image_shapes=(1024, 1024),
        frame_idx: int = 0,  
        slow_update_interval: int = 2, 
    ) -> List[Union[str, Image.Image]]:
        output_list = []
        # print('frame_idx',frame_idx)
        # slow_input_lists = input_lists[:-3] + input_lists[-1:]
        # fast_input_lists = input_lists[-3:-1]  # just current image and lidar

        slow_input_lists = input_lists
        fast_input_lists = trans_feat

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if frame_idx % slow_update_interval == 0 or not hasattr(self, '_cached_gen_context'):
                self._cached_gen_context = self.kv_cache_fixed_inference(slow_input_lists)

            gen_text, gen_traj, route, reasoning_hidden_states = self.based_kv_cache_context_fast_qwen3vl_dp(fast_input_lists, self._cached_gen_context, reasoning_tokens, action_tokens, v_target_point)

            output_dict = {
                'text': gen_text,
                'traj': gen_traj,
                'route': route,
                'dp_vl_feature': reasoning_hidden_states,
            }

            output_list.append(output_dict)

        return output_list             

    @torch.no_grad()
    def qwen3vl_template_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=True,  
        max_think_token_n=1000,
        do_sample=False,
        reasoning_output=False,
        text_temperature=0.0,
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            user_prompt = USER_PROMPT
            image_list = []
            for input_term in input_lists:
                print('input term is:', input_term)
                if isinstance(input_term, str):
                    instruction_prompt = input_term
                elif isinstance(input_term, Image.Image):
                    image_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
                
            gen_context = self.update_context_qwen3vl(user_prompt, instruction_prompt, image_list, gen_context)

            gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
            output_list.append(gen_text)

        return output_list

    # for 1st transformer
    @torch.no_grad()
    def slow_reasoning(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=True, 
        max_think_token_n=1000,
        do_sample=False,
        reasoning_output=False,
        text_temperature=0.0,
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        # print('input_lists', input_lists)
        gen_context = self.init_gen_context()

        tok_dev = self.model.language_model.model.embed_tokens.weight.device

        def _to_device(x, device):
            import torch
            if torch.is_tensor(x):
                return x.to(device)
            if isinstance(x, (list, tuple)):
                return type(x)(_to_device(y, device) for y in x)
            if isinstance(x, dict):
                return {k: _to_device(v, device) for k, v in x.items()}
            return x

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    print("Question:", input_term)
                    gen_context = self.update_context_text(input_term, gen_context)
                    gen_context = _to_device(gen_context, tok_dev)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                    gen_context = _to_device(gen_context, tok_dev)
                    image_shapes = input_term.size[::-1]
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_context = _to_device(gen_context, tok_dev)
                print('gen_context', gen_context)
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n
                )
                output_list.append(gen_text)

        return output_list
    
    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        width = max(1, int(width))
        height = max(1, int(height))
        return image.resize((width, height), Image.Resampling.LANCZOS)
        
    def __call__(
        self, 
        image: Optional[list] = None,
        front: Optional[Image.Image] = None, 
        lidar: Optional[Image.Image] = None,
        ego_status_tensor: Optional[torch.Tensor] = None,
        nav_command_tensor: Optional[torch.Tensor] = None,
        hist_ego_status_tensor: Optional[torch.Tensor] = None,
        hist_waypoints_tensor: Optional[torch.Tensor] = None,
        v_target_point: Optional[torch.Tensor] = None,
        trans_feat: Optional[torch.Tensor] = None,
        text: Optional[str] = None, 
        class_pre: Optional[str] = None,
        visual_gen: bool = False,
        reasoning_output: bool = True,
        resolved_lidar_paths: Optional[list] = None,
        resolved_front_paths: Optional[list] = None,
        gt_traj: Optional[list] = None,
        frame_idx: Optional[list] = None,
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None, 'front': None, 'class_pre': None}

        if image is None and text is None and front is None:
            print('Please provide at least one input: either an image, lidar, or text.')
            return output_dict

        input_list = []
        if image is not None:
            for i in range(len(image)):
                image[i] = self.resize_image(image[i], width=512, height=256)
                input_list.append(image[i])
        # if reasoning_output:
        #     input_list.append(image[-1]) # add the current image for 2nd transformer reasoning
        if class_pre is not None:
            input_list.append(class_pre)
        # if lidar is not None:
        #     for i in range(len(lidar)):
        #         input_list.append(lidar[i])
        # if front is not None:
        #     for i in range(len(front)):
        #         front[i] = self.resize_image(front[i], width=512, height=256)
        #         input_list.append(front[i])
        if text is not None:
            input_list.append(text)

        # output_list = self.kv_cache_inference_slow_fast(input_list, reasoning_output=reasoning_output, v_target_point=v_target_point,
        #                                                 reasoning_tokens=self.model.config.reasoning_query_tokens,
        #                                                 action_tokens=self.model.config.action_query_tokens,                                              
        #                                                 **kargs)
        output_list = self.kv_cache_inference_slow_fast_dp(input_list, reasoning_output=reasoning_output, resolved_lidar_paths=resolved_lidar_paths, v_target_point=v_target_point,
                                                        reasoning_tokens=self.model.config.reasoning_query_tokens, trans_feat=trans_feat,
                                                        action_tokens=self.model.config.action_query_tokens, frame_idx=frame_idx,                                              
                                                        **kargs)
        # output_list = self.kv_cache_inference(input_list, reasoning_output=reasoning_output, 
        #                                       reasoning_tokens=self.model.config.reasoning_query_tokens,
        #                                       action_tokens=self.model.config.action_query_tokens, 
        #                                       **kargs)
        
        ##### Cached ####
        #output_list = self.qwen3vl_template_inference(input_list, **kargs)
        # output_list = self.Sequence_Reasoning_inference(input_list, reasoning_output=reasoning_output,**kargs)
        #output_list = self.Sequence_Reasoning_inference_dp(input_list, reasoning_output=reasoning_output, resolved_lidar_paths=resolved_lidar_paths,**kargs)
        #output_dict = output_list["text"]
        
        output_dict = output_list[0]
        return output_dict