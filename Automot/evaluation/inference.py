from typing import Any, Dict, Optional

from PIL import Image
import torch

from modeling.automot.qwen3vl_navit import NaiveCache


USER_PROMPT = "<|im_start|>system\nYou are a mature and professional driver.<|im_end|>\n<|im_start|>user\n"


class InterleaveInferencer:
    def __init__(
        self,
        model,
        tokenizer,
        vit_transform,
        new_token_ids,
        max_num_tokens=2816,
        max_num_reasoning_token=64,
        lidar_reasoning_token=197,
    ):
        torch.set_num_threads(1)
        self.model = model
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.max_num_tokens = max_num_tokens
        self.reasoning_query_tokens = self.model.reasoning_queries(
            torch.arange(self.model.reasoning_query_tokens, device=self.model.device)
        )
        self.route_query_tokens = self.model.route_queries(torch.arange(20, device=self.model.device))
        self.waypoint_query_tokens = self.model.waypoint_queries(torch.arange(6, device=self.model.device))

    def init_gen_context(self):
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

    def _autocast(self):
        return torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    @torch.no_grad()
    def _update_text_image_cache(self, input_lists, gen_context):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        image_list = []
        instruction_prompt = ""
        for input_term in input_lists:
            if isinstance(input_term, str):
                instruction_prompt = input_term
            elif isinstance(input_term, Image.Image):
                image_list.append(input_term)
            else:
                raise ValueError(f"Unsupported input type: {type(input_term)}")

        generation_input, kv_lens, ropes = self.model.prepare_kv_cache(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            user_prompt=USER_PROMPT,
            instruction_prompt=instruction_prompt,
            images=image_list,
            new_token_ids=self.new_token_ids,
            tokenizer=self.tokenizer,
        )
        past_key_values, packed_position_ids = self.model.forward_cache_update_generation(
            past_key_values, **generation_input
        )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        gen_context["packed_position_ids"] = packed_position_ids
        return gen_context

    @torch.no_grad()
    def _decode_reasoning_tokens(
        self,
        last_hidden_state: torch.FloatTensor,
        packed_reasoning_token_indexes: torch.LongTensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        reasoning_hidden_states = last_hidden_state[packed_reasoning_token_indexes]
        reasoning_logits = self.model.language_model.lm_head(reasoning_hidden_states)
        if do_sample:
            probs = torch.softmax(reasoning_logits / temperature, dim=-1)
            pred_token_ids = torch.multinomial(probs, 1).squeeze(-1)
        else:
            pred_token_ids = reasoning_logits.argmax(dim=-1)

        answer_ids = self.extract_all_bos_eos_with_special(
            pred_token_ids.tolist(),
            self.new_token_ids["bos_token_id"],
            self.new_token_ids["eos_token_id"],
        )
        if answer_ids:
            return self.tokenizer.decode(answer_ids[0], skip_special_tokens=False)
        return self.tokenizer.decode(pred_token_ids.tolist(), skip_special_tokens=False)

    @torch.no_grad()
    def _predict_trajectory(
        self,
        last_hidden_state: torch.FloatTensor,
        packed_action_token_indexes: torch.LongTensor,
    ):
        action_hidden_states = last_hidden_state[packed_action_token_indexes]
        action = self.model.waypoints_head(action_hidden_states)
        return action.reshape(-1, 6, 2)

    def _project_bev_encoder_feature(self, bev_encoder_feature):
        x = bev_encoder_feature.to(self.model.language_model.model.embed_tokens.weight.device)
        if x.dim() == 4:
            batch_size, channels, height, width = x.shape
            if channels != 1512:
                raise ValueError(f"Expected BEV encoder feature with 1512 channels, got {channels}")
            if height * width != 64:
                raise ValueError(f"Expected 8x8 BEV encoder feature, got {height}x{width}")
            x = x.flatten(2).transpose(1, 2).reshape(batch_size * 64, 1512)
        elif x.dim() == 3:
            batch_size, num_tokens, channels = x.shape
            if num_tokens != 64 or channels != 1512:
                raise ValueError(f"Expected BEV encoder feature [B, 64, 1512], got {tuple(x.shape)}")
            x = x.reshape(batch_size * 64, 1512)
        elif x.dim() == 2:
            if x.shape[-1] != 1512:
                raise ValueError(f"Expected BEV encoder feature dim 1512, got {x.shape[-1]}")
        else:
            raise ValueError(f"BEV encoder feature must be 2D, 3D, or 4D, got {x.dim()}D")
        return self.model.bev_encoder_proj(x)

    @torch.no_grad()
    def kv_cache_inference(
        self,
        bev_encoder_feature,
        gen_context,
        reasoning_tokens,
        action_tokens,
        v_target_point,
        target_point_max_num_tokens=2,
        v_num_token=1,
        num_route_tokens=20,
        num_traj_tokens=6,
    ):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, kv_lens, ropes = self.model.prepare_fast_kvcache(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            bev_encoder_feature=bev_encoder_feature,
            new_token_ids=self.new_token_ids,
            tokenizer=self.tokenizer,
            reasoning_learnable_tokens=reasoning_tokens,
            action_learnable_tokens=action_tokens,
            target_point_max_num_tokens=target_point_max_num_tokens,
            v_num_token=v_num_token,
            num_route_tokens=num_route_tokens,
            num_traj_tokens=num_traj_tokens,
        )

        old_pos = gen_context["packed_position_ids"]
        new_pos = generation_input["packed_position_ids"]
        attention_mask = generation_input["nested_attention_masks"]
        new_pos = new_pos + old_pos.max(dim=-1, keepdim=True).values + 1
        query_lens_fast = new_pos.shape[2]
        key_values_lens = torch.as_tensor(gen_context["kv_lens"], dtype=torch.int, device=new_pos.device)
        gen_context["packed_position_ids"] = torch.cat([old_pos, new_pos], dim=-1)

        packed_key_value_indexes = generation_input["packed_key_value_indexes"]
        packed_bev_token_indexes = generation_input["packed_bev_token_indexes"]
        packed_reasoning_token_indexes = generation_input["packed_reasoning_token_indexes"]
        packed_action_token_indexes = generation_input["packed_action_token_indexes"]
        packed_text_ids = generation_input["packed_text_ids"]
        packed_target_point_indexes = generation_input["target_point_indexes"]
        packed_v_indexes = generation_input["v_indexes"]

        packed_text_embedding = self.model.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence_fast = packed_text_embedding.new_zeros(size=(query_lens_fast, self.model.hidden_size))
        packed_sequence_fast[packed_bev_token_indexes] = self._project_bev_encoder_feature(bev_encoder_feature)

        v_target_point = v_target_point.to(packed_sequence_fast.device)
        if v_target_point.dim() == 2:
            v_target_point = v_target_point.squeeze(0)

        packed_route_token_indexes = packed_action_token_indexes[:20]
        packed_wp_token_indexes = packed_action_token_indexes[20:]
        velocity = v_target_point[0].unsqueeze(0).to(packed_sequence_fast.device)
        target_points = v_target_point[1:5].reshape(1, 2, 2).to(packed_sequence_fast.device)

        packed_sequence_fast[packed_target_point_indexes] = self.model.target_point_encoder(target_points).reshape(
            -1, self.model.hidden_size
        )
        packed_sequence_fast[packed_v_indexes] = self.model.velocity_encoder(velocity)
        packed_sequence_fast[packed_reasoning_token_indexes] = self.model.reasoning_projector(
            self.reasoning_query_tokens
        )
        packed_sequence_fast[packed_route_token_indexes] = self.model.route_projector(self.route_query_tokens)
        packed_sequence_fast[packed_wp_token_indexes] = self.model.waypoint_projector(self.waypoint_query_tokens)

        packed_query_token_indexes = torch.cat(
            [
                packed_bev_token_indexes,
                packed_target_point_indexes,
                packed_v_indexes,
                packed_reasoning_token_indexes,
                packed_route_token_indexes,
                packed_wp_token_indexes,
            ],
            dim=0,
        )
        packed_query_indexes = packed_query_token_indexes + key_values_lens.sum().long()

        last_hidden_state = self.model.language_model.forward_inference(
            packed_query_sequence=packed_sequence_fast,
            query_lens=torch.tensor([query_lens_fast], device=packed_key_value_indexes.device),
            attention_mask=attention_mask,
            packed_query_position_ids=new_pos,
            packed_query_indexes=packed_query_indexes,
            packed_mot_token_indexes=packed_query_token_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            packed_text_indexes=torch.empty(0, dtype=torch.long, device=packed_key_value_indexes.device),
            update_past_key_values=False,
            is_causal=False,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            mode="gen",
        )

        route = self.model.route_head(last_hidden_state[0][packed_route_token_indexes]).view(-1, 20, 2)
        text = self._decode_reasoning_tokens(
            last_hidden_state[0],
            packed_reasoning_token_indexes,
            do_sample=False,
            temperature=0.0,
        )
        traj = self._predict_trajectory(last_hidden_state[0], packed_wp_token_indexes).cumsum(dim=1)
        return {
            "text": text,
            "traj": traj,
            "route": route,
        }

    @torch.no_grad()
    def _run_kv_cache_inference(
        self,
        input_lists,
        v_target_point: Optional[torch.Tensor] = None,
        bev_encoder_feature: Optional[torch.Tensor] = None,
        reasoning_tokens=8,
        action_tokens=26,
        frame_idx: Optional[int] = 0,
        slow_update_interval: int = 2,
        **unused_kwargs,
    ) -> Dict[str, Any]:
        if bev_encoder_feature is None:
            raise ValueError("bev_encoder_feature is required for AutoMoT inference.")
        if v_target_point is None:
            raise ValueError("v_target_point is required for AutoMoT inference.")

        frame_idx = int(frame_idx or 0)
        with self._autocast():
            if frame_idx % slow_update_interval == 0 or not hasattr(self, "_cached_gen_context"):
                gen_context = self.init_gen_context()
                self._cached_gen_context = self._update_text_image_cache(input_lists, gen_context)

            return self.kv_cache_inference(
                bev_encoder_feature=bev_encoder_feature,
                gen_context=self._cached_gen_context,
                reasoning_tokens=reasoning_tokens,
                action_tokens=action_tokens,
                v_target_point=v_target_point,
            )

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

    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        width = max(1, int(width))
        height = max(1, int(height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def __call__(
        self,
        image: Optional[list] = None,
        front: Optional[list] = None,
        lidar: Optional[list] = None,
        ego_status_tensor: Optional[torch.Tensor] = None,
        nav_command_tensor: Optional[torch.Tensor] = None,
        hist_ego_status_tensor: Optional[torch.Tensor] = None,
        hist_waypoints_tensor: Optional[torch.Tensor] = None,
        v_target_point: Optional[torch.Tensor] = None,
        bev_encoder_feature: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        class_pre: Optional[str] = None,
        reasoning_output: bool = True,
        gt_traj: Optional[list] = None,
        frame_idx: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if image is None and text is None and front is None:
            return {"image": None, "text": None, "front": None, "class_pre": None}

        input_list = []
        if image is not None:
            for item in image:
                input_list.append(self.resize_image(item, width=512, height=256))
        if class_pre is not None:
            input_list.append(class_pre)
        if text is not None:
            input_list.append(text)

        return self._run_kv_cache_inference(
            input_list,
            reasoning_output=reasoning_output,
            v_target_point=v_target_point,
            reasoning_tokens=self.model.config.reasoning_query_tokens,
            bev_encoder_feature=bev_encoder_feature,
            action_tokens=self.model.config.action_query_tokens,
            frame_idx=frame_idx,
            **kwargs,
        )
