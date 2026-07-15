
import random
import json
import math
import torch
from typing import Optional
from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate,
    len2weight,
    prepare_attention_mask_per_sample,
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .video_utils import FrameSampler

class DataConfig:
    def __init__(
        self,
        grouped_datasets,
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_config,
        tokenizer,
        special_tokens,
        local_rank,
        world_size,
        num_workers,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        reasoning_text_max_num_tokens=8,
        action_expert_max_num_tokens=26,
        target_point_max_num_tokens=2,
        bev_encoder_token_max_num_tokens=64,
        v_num_token=1,
        num_route_tokens=20,
        num_traj_tokens=6,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self.bos_eos = True
        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]

        for dataset in grouped_datasets:
            if hasattr(dataset, 'reasoning_text_max_num_tokens'):
                dataset.reasoning_text_max_num_tokens = reasoning_text_max_num_tokens
            if hasattr(dataset, 'action_expert_max_num_tokens'):
                dataset.action_expert_max_num_tokens = action_expert_max_num_tokens
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        # General template tokens.
        self.im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        self.im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        self.v_start_id = tokenizer.encode("<|vision_start|>", add_special_tokens=False)[0]
        self.v_end_id = tokenizer.encode("<|vision_end|>", add_special_tokens=False)[0]
        self.img_pad_id = tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]

        # Qwen3-VL role tokens.
        self.assistant_id = tokenizer.encode("assistant", add_special_tokens=False)
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.user_id = tokenizer.encode("user", add_special_tokens=False)
        self.system_id = tokenizer.encode("system", add_special_tokens=False)

        self.reasoning_text_max_num_tokens = reasoning_text_max_num_tokens
        self.bev_encoder_token_max_num_tokens = bev_encoder_token_max_num_tokens
        self.action_expert_max_num_tokens = action_expert_max_num_tokens
        self.target_point_max_num_tokens = target_point_max_num_tokens
        self.v_num_token = v_num_token
        self.num_route_tokens = num_route_tokens
        self.num_traj_tokens = num_traj_tokens
    def build_vision_prompt_fast_thinking(
            self,
            question_ids: list,
            image_token_counts,
            number_front_tokens: list,
            number_lidar_vit_tokens: list,
            tokenizer,
            counts_unit: str = "pads",
            merge_size: int = 4,
            add_generation_prompt: bool = False,
            use_system_prompt: bool = True,
            system_text: str = "You are a mature and professional driver.",
        ) -> list:
            # Encode system text
            system_text_ids = tokenizer.encode(system_text, add_special_tokens=False)

            # Compute pads for the left-side historical images
            pads_per_image = []
            for c in image_token_counts:
                if counts_unit == "pads":
                    pads = int(c)
                else:
                    pads = int(math.ceil(int(c) / merge_size))
                pads_per_image.append(max(0, pads))

            # Compute pads for front images
            pads_per_front = []
            for c in number_front_tokens:
                if counts_unit == "pads":
                    pads = int(c)
                else:
                    pads = int(math.ceil(int(c) / merge_size))
                pads_per_front.append(max(0, pads))

            # Compute pads for LiDAR images
            pads_per_lidar = []
            for c in number_lidar_vit_tokens:
                if counts_unit == "pads":
                    pads = int(c)
                else:
                    pads = int(math.ceil(int(c) / merge_size))
                pads_per_lidar.append(max(0, pads))

            prompt_ids = []

            # System prompt block
            if use_system_prompt:
                # <|im_start|>system\n system_text<|im_end|>\n
                prompt_ids.append(self.im_start_id)
                prompt_ids.extend(self.system_id)
                prompt_ids.append(self.newline_id)
                prompt_ids.extend(system_text_ids)
                prompt_ids.extend([self.im_end_id, self.newline_id])

            # User block header <|im_start|>user\n
            prompt_ids.append(self.im_start_id)
            prompt_ids.extend(self.user_id)
            prompt_ids.append(self.newline_id)

            # Left-side: Historical image vision blocks
            for pads in pads_per_image:
                prompt_ids.append(self.v_start_id)
                if pads > 0:
                    prompt_ids.extend([self.img_pad_id] * pads)
                prompt_ids.append(self.v_end_id)

            # Append question text
            prompt_ids.extend(question_ids)

            # End of question
            prompt_ids.append(self.im_end_id)

            # Append LiDAR blocks
            if number_lidar_vit_tokens is not None:
                for pads in pads_per_lidar:
                    prompt_ids.append(self.v_start_id)
                    if pads > 0:
                        prompt_ids.extend([self.img_pad_id] * pads)
                    prompt_ids.append(self.v_end_id)

            # Append front image block
            for pads in pads_per_front:
                prompt_ids.append(self.v_start_id)
                if pads > 0:
                    prompt_ids.extend([self.img_pad_id] * pads)
                prompt_ids.append(self.v_end_id)

            # Optional generation prompt (only used during inference)
            if add_generation_prompt:
                prompt_ids.append(self.im_start_id)
                prompt_ids.extend(self.assistant_id)
                prompt_ids.append(self.newline_id)

            return prompt_ids

    def build_bev_encoder_prompt(
        self,
        question_ids: list,
        image_token_counts,
        tokenizer,
        counts_unit: str = "pads",
        merge_size: int = 4,
        add_generation_prompt: bool = False,
        use_system_prompt: bool = True,
        system_text: str = "You are a mature and professional driver.",
    ) -> list:
        # Encode system text
        system_text_ids = tokenizer.encode(system_text, add_special_tokens=False)

        # Compute pads for the left-side historical images
        pads_per_image = []
        for c in image_token_counts:
            if counts_unit == "pads":
                pads = int(c)
            else:
                pads = int(math.ceil(int(c) / merge_size))
            pads_per_image.append(max(0, pads))

        prompt_ids = []

        # System prompt block: <|im_start|>system\n ... <|im_end|>\n
        if use_system_prompt:
            prompt_ids.append(self.im_start_id)
            prompt_ids.extend(self.system_id)
            prompt_ids.append(self.newline_id)
            prompt_ids.extend(system_text_ids)
            prompt_ids.extend([self.im_end_id, self.newline_id])

        # User block header <|im_start|>user\n
        prompt_ids.append(self.im_start_id)
        prompt_ids.extend(self.user_id)
        prompt_ids.append(self.newline_id)

        # Left-side: Historical image vision blocks
        for pads in pads_per_image:
            prompt_ids.append(self.v_start_id)
            if pads > 0:
                prompt_ids.extend([self.img_pad_id] * pads)
            prompt_ids.append(self.v_end_id)

        # Append question text
        prompt_ids.extend(question_ids)

        # End of user message
        prompt_ids.append(self.im_end_id)
        # Optional generation prompt (only used during inference)
        if add_generation_prompt:
            prompt_ids.append(self.im_start_id)
            prompt_ids.extend(self.assistant_id)
            prompt_ids.append(self.newline_id)

        return prompt_ids


    def build_vision_prompt(
        self,
        question_ids: list,
        gt_ids: list,
        image_token_counts,
        tokenizer,
        counts_unit: str = "pads",
        merge_size: int = 4,
        add_generation_prompt: bool = False,
        use_system_prompt: bool = True,
        system_text: str = "You are a mature and professional driver.",
    ) -> list:

        system_text_ids = tokenizer.encode(system_text, add_special_tokens=False)

        pads_per_image = []
        for c in image_token_counts:
            if counts_unit == "pads":
                pads = int(c)
            else:
                pads = int(math.ceil(int(c) / merge_size))
            pads_per_image.append(max(0, pads))

        prompt_ids = []

        if use_system_prompt:
            prompt_ids.append(self.im_start_id)
            prompt_ids.extend(self.system_id)
            prompt_ids.append(self.newline_id)
            prompt_ids.extend(system_text_ids)
            prompt_ids.extend([self.im_end_id, self.newline_id])

        prompt_ids.append(self.im_start_id)
        prompt_ids.extend(self.user_id)
        prompt_ids.append(self.newline_id)

        # vision blocks
        for pads in pads_per_image:
            prompt_ids.append(self.v_start_id)
            prompt_ids.extend([self.img_pad_id] * pads)
            prompt_ids.append(self.v_end_id)

        # question
        prompt_ids.extend(question_ids)
        prompt_ids.extend([self.im_end_id, self.newline_id, self.im_start_id])
        prompt_ids.extend(self.assistant_id)
        prompt_ids.append(self.newline_id)

        # ground truth answer
        prompt_ids.extend(gt_ids)
        prompt_ids.extend([self.im_end_id])

        # Optional: generation prompt (to be tested)
        if add_generation_prompt:
            prompt_ids.append(self.im_start_id)
            prompt_ids.extend(self.assistant_id)
            prompt_ids.append(self.newline_id)

        return prompt_ids


    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                           = 0,
            sample_lens                    = list(),
            packed_position_ids            = list(),
            nested_attention_masks         = list(),
            split_lens                     = list(),
            attn_modes                     = list(),
            packed_text_ids                = list(),
            packed_text_indexes            = list(),
            packed_label_ids               = list(),
            ce_loss_indexes                = list(),
            ce_loss_weights                = list(),
            traj_loss_indexes              = list(),
            route_loss_indexes             = list(),
            target_point_indexes           = list(),
            v_indexes                      = list(),
            packed_vit_tokens              = list(),
            vit_token_seqlens              = list(),
            packed_action_token_indexes    = list(),
            action_query_token_seqlens     = list(),
            packed_vit_token_indexes       = list(),
            packed_und_vit_token_indexes   = list(),
            packed_gen_vit_token_indexes   = list(),
            packed_und_text_indexes        = list(),
            packed_gen_text_indexes        = list(),
            image_tensor_list              = list(),
            image_grid_thw_list            = list(),
            packed_reasoning_token_indexes = list(),
            probs                          = list(),
            traj_gt                        = list(),
            route_gt                       = list(),
            future_speeds_tensors          = list(),
            v_target_point                 = list(),
            bev_encoder_feature             = list(),
            bev_encoder_feature_inputs      = list(),
            packed_bev_indexes              = list(),
        )
        return sequence_status

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer = None,
        spatial_merge_size = 2,
    ):
        """
        Calculate 3D position_ids for multi-dimensional RoPE (mRoPE) following official Qwen3VL implementation.
        """
        # Get spatial_merge_size from config and token IDs from tokenizer

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
                vision_start_indices = torch.argwhere(input_ids_seq == self.v_start_id).squeeze(1)
                vision_tokens = input_ids_seq[vision_start_indices + 1] if len(vision_start_indices) > 0 else torch.tensor([], device=input_ids_seq.device)
                image_nums = (vision_tokens == self.img_pad_id).sum() if len(vision_tokens) > 0 else 0
                input_tokens = input_ids_seq.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums

                for _ in range(image_nums):
                    if self.img_pad_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(self.img_pad_id, st)
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

    def get_rope_index_fast_thinking(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer = None,
        spatial_merge_size = 2,
        num_learnable_tokens = 11, # 10 learnable tokens 1 action token
    ):
        """
        Calculate 3D position_ids for multi-dimensional RoPE (mRoPE) following official Qwen3VL implementation.
        """
        # Get spatial_merge_size from config and token IDs from tokenizer

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
                vision_start_indices = torch.argwhere(input_ids_seq == self.v_start_id).squeeze(1)
                vision_tokens = input_ids_seq[vision_start_indices + 1] if len(vision_start_indices) > 0 else torch.tensor([], device=input_ids_seq.device)
                image_nums = (vision_tokens == self.img_pad_id).sum() if len(vision_tokens) > 0 else 0
                input_tokens = input_ids_seq.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums

                for _ in range(image_nums):
                    if self.img_pad_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(self.img_pad_id, st)
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
            # Add learnable token positions at the end.
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

            if num_learnable_tokens > 0:
                max_pos = position_ids.max(dim=-1).values[0]
                start = max_pos + 1
                offset = torch.arange(
                    num_learnable_tokens,
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                )
                extra_1d = start.view(-1, 1) + offset.view(1, -1)
                extra = extra_1d.view(1, -1, num_learnable_tokens).expand(3, -1, -1)
                position_ids = torch.cat([position_ids, extra], dim=-1)

            rope_deltas = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
            return position_ids, rope_deltas

    def to_tensor(self, sequence_status):
        sequence_status['packed_position_ids'] = torch.cat(sequence_status['packed_position_ids'], dim=2).tolist()
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # Visual tokenizer payload.
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        if len(sequence_status['packed_reasoning_token_indexes']) > 0:
            data['packed_und_vit_token_indexes'] = torch.tensor(sequence_status['packed_und_vit_token_indexes'])
            data['packed_gen_vit_token_indexes'] = torch.tensor(sequence_status['packed_gen_vit_token_indexes'])
            data['packed_reasoning_token_indexes'] = torch.tensor(sequence_status['packed_reasoning_token_indexes'])
            data['packed_und_text_indexes'] = torch.tensor(sequence_status['packed_und_text_indexes'])
            data['packed_gen_text_indexes'] = torch.tensor(sequence_status['packed_gen_text_indexes'])

        if sequence_status['probs'] is not None:
            data['probs'] = sequence_status['probs']

        if len(sequence_status['packed_action_token_indexes']) > 0:
            data['packed_action_token_indexes'] = torch.tensor(sequence_status['packed_action_token_indexes'])
            data['action_query_token_seqlens'] = torch.tensor(sequence_status['action_query_token_seqlens'])
            data['packed_bev_indexes'] = torch.tensor(sequence_status['packed_bev_indexes'])
            data['target_point_indexes'] = torch.tensor(sequence_status['target_point_indexes'])
            data['v_indexes'] = torch.tensor(sequence_status['v_indexes'])

        if len(sequence_status['traj_gt']) > 0:
            data['traj_gt'] = torch.cat(sequence_status['traj_gt'], dim=0)
            data['route_gt'] = torch.cat(sequence_status['route_gt'], dim=0)
            data['future_speeds_tensors'] = torch.stack(sequence_status['future_speeds_tensors'], dim=0)
            data['v_target_point'] = torch.stack(sequence_status['v_target_point'], dim=0)  # (N, 3)
            if len(sequence_status['bev_encoder_feature_inputs']) > 0:
                data['bev_encoder_feature_inputs'] = sequence_status['bev_encoder_feature_inputs']
            elif len(sequence_status['bev_encoder_feature']) > 0:
                data['bev_encoder_feature'] = torch.stack(sequence_status['bev_encoder_feature'], dim=0)  # (N, 1512, 8, 8)
            data['traj_loss_indexes'] = torch.tensor(sequence_status['traj_loss_indexes'])
            data['route_loss_indexes'] = torch.tensor(sequence_status['route_loss_indexes'])

        # Text generation labels.
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])
        if len(sequence_status['image_tensor_list']) > 0:
            data['image_tensor_list'] = torch.cat(sequence_status['image_tensor_list'], dim=0)
        if len(sequence_status['image_grid_thw_list']) > 0:
            data['image_grid_thw_list'] = torch.cat(sequence_status['image_grid_thw_list'], dim=0)
        return data


    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # Skip samples that exceed the token budget.
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                continue

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # Skip samples that exceed the token budget.
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def move_text_to_front(self, sequence_plan):
        types = [item.get('type') for item in sequence_plan]

        has_text = any(t == 'text' for t in types)
        has_fast_text = any(t == 'fast_text' for t in types)

        if not has_text and not has_fast_text:
            return sequence_plan

        assert not (has_text and has_fast_text), \
            "sequence_plan cannot contain both 'text' and 'fast_text' items."

        front_type = 'text' if has_text else 'fast_text'

        front_items = [item for item in sequence_plan if item.get('type') == front_type]
        other_items = [item for item in sequence_plan if item.get('type') != front_type]

        return front_items + other_items


    def pack_sequence(self, sample, sequence_status):
        front_tensor_list = sample['front_tensor_list'] if 'front_tensor_list' in sample else []
        traj_gt = sample.get('ego_fut_trajs_tensor', None)
        sequence_status['v_target_point'].append(sample['v_target_point']) if 'v_target_point' in sample else []
        if sample.get('bev_encoder_feature') is not None:
            sequence_status['bev_encoder_feature_inputs'].append({
                'feature': sample['bev_encoder_feature'].squeeze(0),
            })
        elif sample.get('online_bev_encoder_input') is not None:
            sequence_status['bev_encoder_feature_inputs'].append(sample['online_bev_encoder_input'])
        self.has_traj = traj_gt is not None
        sequence_status['traj_gt'].append(sample['ego_fut_trajs_tensor']) if 'ego_fut_trajs_tensor' in sample else []
        sequence_status['route_gt'].append(sample['route_future_tensor']) if 'route_future_tensor' in sample else []
        sequence_status['future_speeds_tensors'].append(sample['future_speeds_tensor']) if 'future_speeds_tensor' in sample else []
        number_front_vit_tokens = sample['number_front_vit_tokens'].copy()if 'number_front_vit_tokens' in sample else []
        image_tensor_list = sample.get('image_tensor_list', [])
        image_grid_thw_list_tensor = None
        if len(sample.get('image_tensor_list_concat', [])) > 0:
            sample_image_tensor = torch.cat(sample['image_tensor_list_concat'], dim=0)
            sequence_status['image_tensor_list'].append(sample_image_tensor)
        if len(sample.get('image_grid_thw_list', [])) > 0:
            image_grid_thw_list_tensor = torch.stack(sample['image_grid_thw_list'], dim=0)
            sequence_status['image_grid_thw_list'].append(image_grid_thw_list_tensor)
        number_lidar_vit_tokens = sample['number_lidar_vit_tokens'].copy() if 'number_lidar_vit_tokens' in sample else []
        lidar_tensor_list = sample['lidar_tensor_list'] if 'lidar_tensor_list' in sample else []
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']
        sequence_status['probs'].append(sample.get('prob_dict', None))
        number_vit_tokens = sample['number_vit_tokens'].copy()
        number_vit_tokens_fast_thinking = sample['number_vit_tokens'].copy()
        if number_vit_tokens:
            number_vit_tokens_fast_thinking.append(number_vit_tokens[-1])
        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        sequence_plan_text_first = self.move_text_to_front(sequence_plan)

        all_image_pad_positions = []
        for item in sequence_plan_text_first:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text' and item['loss'] != 1:
                text_ids_question = text_ids_list.pop(0)
                text_ids_answer = text_ids_list.pop(0)

                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                input_token_list = self.build_vision_prompt(
                    question_ids=text_ids_question,
                    gt_ids=text_ids_answer,
                    image_token_counts=number_vit_tokens,
                    tokenizer=self.tokenizer,
                )
                input_token_tensor = torch.tensor(input_token_list)
                tokenizer = self.tokenizer
                attention_mask = torch.ones_like(input_token_tensor).unsqueeze(0)
                vision_pairs = []
                vision_start_indices = []
                vision_end_indices = []
                for i, token_id in enumerate(input_token_list):
                    if token_id == self.v_start_id:
                        vision_start_indices.append(i)
                    elif token_id == self.v_end_id:
                        vision_end_indices.append(i)

                for start_idx, end_idx in zip(vision_start_indices, vision_end_indices):
                    vision_pairs.append((start_idx, end_idx + 1))
                if len(vision_pairs) > 0:
                    first_vision_start = vision_pairs[0][0]
                    last_vision_end = vision_pairs[-1][1]

                    first_segment_len = first_vision_start
                    if first_segment_len > 0:
                        split_lens.append(first_segment_len)
                        curr_rope_id += first_segment_len
                        attn_modes.append('causal')

                    prev_end = first_vision_start
                    for start_idx, end_idx in vision_pairs:
                        if start_idx > prev_end:
                            text_len = start_idx - prev_end
                            split_lens.append(text_len)
                            curr_rope_id += text_len
                            attn_modes.append('causal')

                        vision_len = end_idx - start_idx
                        split_lens.append(vision_len)
                        curr_rope_id += 1
                        attn_modes.append('full')

                        prev_end = end_idx

                    remaining_len = len(input_token_list) - last_vision_end
                    if remaining_len > 0:
                        split_lens.append(remaining_len)
                        curr_rope_id += remaining_len
                        attn_modes.append('causal')
                else:
                    split_lens.append(len(input_token_list))
                    curr_rope_id += len(input_token_list)
                    attn_modes.append('causal')
                for local_idx, tok in enumerate(input_token_list):
                    global_idx = curr + local_idx

                    if tok == self.img_pad_id:
                        all_image_pad_positions.append(global_idx)
                    else:
                        sequence_status['packed_text_ids'].append(tok)
                        sequence_status['packed_text_indexes'].append(global_idx)

                answer_prefix = [self.im_start_id, self.assistant_id[0], self.newline_id]
                for i in range(0, len(input_token_list) - len(answer_prefix) + 1):
                    if input_token_list[i:i + len(answer_prefix)] == answer_prefix:
                        answer_content_start = i + len(answer_prefix)
                        for j in range(answer_content_start, len(input_token_list)):
                            if input_token_list[j] == self.im_end_id:
                                answer_content_end = j
                                answer_length = answer_content_end - answer_content_start
                                loss_start_global = curr + answer_content_start - 1
                                loss_end_global   = curr + answer_content_end
                                sequence_status['ce_loss_indexes'].extend(
                                    range(loss_start_global, loss_end_global)
                                )

                                sequence_status['ce_loss_weights'].extend(
                                    [len2weight(answer_length + 1)] * (answer_length + 1)
                                )
                                actual_answer_tokens = input_token_list[answer_content_start:answer_content_end]
                                sequence_status['packed_label_ids'].extend(
                                    actual_answer_tokens + [self.im_end_id]
                                )

                                break
                        break
                curr += len(input_token_list)

                text_split_len = len([t for t in input_token_list if t != self.img_pad_id])
                curr_split_len += text_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)

                if 'lidar_grid_thw_list' in sample:
                    num_vit_tokens = number_vit_tokens_fast_thinking.pop(0)
                else:
                    num_vit_tokens = number_vit_tokens.pop(0)

                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                num_pads = int(math.ceil(num_vit_tokens))

                this_image_pad_positions = all_image_pad_positions[:num_pads]
                all_image_pad_positions = all_image_pad_positions[num_pads:]

                sequence_status['packed_vit_token_indexes'].extend(this_image_pad_positions)
                sequence_status['packed_vit_tokens'].append(image_tensor)
                sequence_status['vit_token_seqlens'].append(num_pads)
                sequence_status['packed_und_vit_token_indexes'].extend(this_image_pad_positions)
                curr_split_len += num_pads

            elif item['type'] == 'lidar_bev':
                lidar_tensor = lidar_tensor_list.pop(0)
                number_lidar_tokens = number_lidar_vit_tokens.pop(0)

                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                num_pads = int(math.ceil(number_lidar_tokens))
                lidar_image_pad_positions = all_image_pad_positions[:num_pads]
                all_image_pad_positions = all_image_pad_positions[num_pads:]

                sequence_status['packed_vit_token_indexes'].extend(lidar_image_pad_positions)
                sequence_status['packed_gen_vit_token_indexes'].extend(lidar_image_pad_positions)
                sequence_status['packed_vit_tokens'].append(lidar_tensor)
                sequence_status['vit_token_seqlens'].append(num_pads)

                curr_split_len += num_pads

            elif item['type'] == 'bev_encoder_feature':
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                sequence_status['packed_bev_indexes'].extend(range(curr, curr + self.bev_encoder_token_max_num_tokens))
                curr += self.bev_encoder_token_max_num_tokens
                curr_split_len += self.bev_encoder_token_max_num_tokens
                attn_modes.append("full")
                split_lens.append(self.bev_encoder_token_max_num_tokens)

            elif item['type'] == 'front_vit':
                front_tensor = front_tensor_list.pop(0)
                num_front_tokens = number_front_vit_tokens.pop(0)

                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                num_pads = int(math.ceil(num_front_tokens))
                front_image_pad_positions = all_image_pad_positions[:num_pads]
                all_image_pad_positions = all_image_pad_positions[num_pads:]

                sequence_status['packed_vit_token_indexes'].extend(front_image_pad_positions)
                sequence_status['packed_gen_vit_token_indexes'].extend(front_image_pad_positions)
                sequence_status['packed_vit_tokens'].append(front_tensor)
                sequence_status['vit_token_seqlens'].append(num_pads)

                curr_split_len += num_pads

            elif item['type'] == 'reasoning_text':
                reasoning_text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue
                tp_start = curr
                sequence_status['target_point_indexes'].extend([tp_start, tp_start + 1])
                curr += self.target_point_max_num_tokens  # == 2
                sequence_status['v_indexes'].append(curr)
                curr += self.v_num_token
                curr_split_len += self.target_point_max_num_tokens + self.v_num_token
                attn_modes.append("full")
                split_lens.append(self.target_point_max_num_tokens + self.v_num_token)

                sequence_status['packed_reasoning_token_indexes'].extend(range(curr, curr + self.reasoning_text_max_num_tokens))
                sequence_status['ce_loss_indexes'].extend(range(curr, curr + self.reasoning_text_max_num_tokens))
                sequence_status['ce_loss_weights'].extend(
                    [len2weight(self.reasoning_text_max_num_tokens)] * self.reasoning_text_max_num_tokens
                )
                if self.bos_eos:
                    sequence_status['packed_label_ids'].extend([self.im_start_id])
                    num_label_tokens = len(reasoning_text_ids)
                    pred_len = min(self.reasoning_text_max_num_tokens - 1, num_label_tokens)
                    sequence_status['packed_label_ids'].extend(reasoning_text_ids[:pred_len])
                    pred_len += 1
                else:
                    sequence_status['packed_label_ids'].extend(reasoning_text_ids)
                    num_label_tokens = len(reasoning_text_ids)
                    pred_len = min(self.reasoning_text_max_num_tokens, num_label_tokens)

                if pred_len < self.reasoning_text_max_num_tokens:
                    sequence_status['packed_label_ids'].extend([self.im_end_id])
                    pred_len += 1
                    sequence_status['packed_label_ids'].extend([-100] * (self.reasoning_text_max_num_tokens - pred_len))

                curr += self.reasoning_text_max_num_tokens
                curr_split_len += self.reasoning_text_max_num_tokens
                attn_modes.append("full")
                split_lens.append(self.reasoning_text_max_num_tokens)
                curr_rope_id += curr_split_len

                # Add action expert tokens.
                sequence_status['action_query_token_seqlens'].append(self.action_expert_max_num_tokens)
                sequence_status['packed_action_token_indexes'].extend(range(curr, curr + self.action_expert_max_num_tokens))
                sequence_status['route_loss_indexes'].extend(range(curr, curr + self.num_route_tokens))
                curr += self.num_route_tokens
                curr_split_len += self.num_route_tokens
                attn_modes.append("full")
                split_lens.append(self.num_route_tokens)
                sequence_status['traj_loss_indexes'].extend(range(curr, curr + self.num_traj_tokens))
                curr += self.num_traj_tokens
                curr_split_len += self.num_traj_tokens
                attn_modes.append("full")
                split_lens.append(self.num_traj_tokens)
                curr_rope_id += curr_split_len

            elif item['type'] == 'fast_text':
                text_ids_question = text_ids_list.pop(0)

                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                input_token_list = self.build_bev_encoder_prompt(
                    question_ids=text_ids_question,
                    image_token_counts=number_vit_tokens,
                    tokenizer=self.tokenizer,
                )
                input_token_tensor = torch.tensor(input_token_list)
                tokenizer = self.tokenizer
                attention_mask = torch.ones_like(input_token_tensor).unsqueeze(0)
                vision_pairs = []
                vision_start_indices = []
                vision_end_indices = []
                for i, token_id in enumerate(input_token_list):
                    if token_id == self.v_start_id:
                        vision_start_indices.append(i)
                    elif token_id == self.v_end_id:
                        vision_end_indices.append(i)

                for start_idx, end_idx in zip(vision_start_indices, vision_end_indices):
                    vision_pairs.append((start_idx, end_idx + 1))
                if len(vision_pairs) > 0:
                    first_vision_start = vision_pairs[0][0]
                    last_vision_end = vision_pairs[-1][1]

                    first_segment_len = first_vision_start
                    if first_segment_len > 0:
                        split_lens.append(first_segment_len)
                        curr_rope_id += first_segment_len
                        attn_modes.append('causal')

                    prev_end = first_vision_start
                    for start_idx, end_idx in vision_pairs:
                        if start_idx > prev_end:
                            text_len = start_idx - prev_end
                            split_lens.append(text_len)
                            curr_rope_id += text_len
                            attn_modes.append('causal')

                        vision_len = end_idx - start_idx
                        split_lens.append(vision_len)
                        curr_rope_id += 1
                        attn_modes.append('full')

                        prev_end = end_idx
                    remaining_len = len(input_token_list) - last_vision_end
                    if remaining_len > 0:
                        split_lens.append(remaining_len)
                        curr_rope_id += remaining_len
                        attn_modes.append('full')
                else:
                    split_lens.append(len(input_token_list))
                    curr_rope_id += len(input_token_list)
                    attn_modes.append('causal')
                for local_idx, tok in enumerate(input_token_list):
                    global_idx = curr + local_idx

                    if tok == self.img_pad_id:
                        all_image_pad_positions.append(global_idx)
                    else:
                        sequence_status['packed_text_ids'].append(tok)
                        sequence_status['packed_text_indexes'].append(global_idx)
                split = max(0, len(sequence_status['packed_text_indexes']))
                gen_idxs = sequence_status['packed_text_indexes'][split:]
                sequence_status['packed_gen_text_indexes'].extend(gen_idxs)
                gen_set = set(sequence_status['packed_gen_text_indexes'])
                und_idxs = [
                    idx for idx in sequence_status['packed_text_indexes']
                    if idx not in gen_set
                ]
                sequence_status['packed_und_text_indexes'] = und_idxs
                curr += len(input_token_list)
                text_split_len = len([t for t in input_token_list if t != self.img_pad_id])
                curr_split_len += text_split_len

            else:
                raise ValueError(f"Unsupported sequence item type: {item['type']}")

            if item.get('split_end', True):
                sample_lens += curr_split_len
        packed_vit_position_ids_tensor = image_grid_thw_list_tensor

        if 'lidar_grid_thw_list' in sample:
            num_learnable_tokens = (
                self.reasoning_text_max_num_tokens
                + self.action_expert_max_num_tokens
                + self.target_point_max_num_tokens
                + self.v_num_token
            )
            if any(item['type'] == 'bev_encoder_feature' for item in sequence_plan):
                num_learnable_tokens += self.bev_encoder_token_max_num_tokens
            position_ids_3d, rope_deltas = self.get_rope_index_fast_thinking(
                input_ids=input_token_tensor.unsqueeze(0),  # [1, seq_len]
                image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
                video_grid_thw=None,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                num_learnable_tokens=num_learnable_tokens,
            )
        else:
            position_ids_3d, rope_deltas = self.get_rope_index(
                input_ids=input_token_tensor.unsqueeze(0),  # [1, seq_len]
                image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
                video_grid_thw=None,
                attention_mask=attention_mask,
                tokenizer=tokenizer
            )

        sequence_status['packed_position_ids'].append(position_ids_3d)
        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)

        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]
        if "packed_gen_vit_token_indexes" in data.keys():
            self.packed_und_vit_token_indexes = data['packed_und_vit_token_indexes']
            self.packed_gen_vit_token_indexes = data['packed_gen_vit_token_indexes']
            self.packed_reasoning_token_indexes = data['packed_reasoning_token_indexes']
            self.packed_und_text_indexes = data['packed_und_text_indexes']
            self.packed_gen_text_indexes = data['packed_gen_text_indexes']

        if "packed_action_token_indexes" in data.keys():
            self.packed_action_token_indexes = data['packed_action_token_indexes']
            self.action_query_token_seqlens = data['action_query_token_seqlens']
            self.target_point_indexes = data['target_point_indexes']
            self.packed_bev_indexes = data['packed_bev_indexes']
            self.v_indexes = data['v_indexes']
        if "traj_gt" in data.keys():
            self.traj_gt = data['traj_gt']
            self.route_gt = data['route_gt']
            self.future_speeds_tensors = data['future_speeds_tensors']
            self.v_target_point = data['v_target_point']
            self.bev_encoder_feature = data.get('bev_encoder_feature', None)
            self.bev_encoder_feature_inputs = data.get('bev_encoder_feature_inputs', None)
            self.traj_loss_indexes = data['traj_loss_indexes']
            self.route_loss_indexes = data['route_loss_indexes']

        if "probs" in data.keys():
            self.probs =data["probs"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

        self.image_tensor_list = data.get("image_tensor_list", None)
        self.image_grid_thw_list = data.get("image_grid_thw_list", None)

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'packed_gen_vit_token_indexes'):
            self.packed_und_vit_token_indexes = self.packed_und_vit_token_indexes.pin_memory()
            self.packed_gen_vit_token_indexes = self.packed_gen_vit_token_indexes.pin_memory()
            self.packed_reasoning_token_indexes = self.packed_reasoning_token_indexes.pin_memory()
            self.packed_und_text_indexes = self.packed_und_text_indexes.pin_memory()
            self.packed_gen_text_indexes = self.packed_gen_text_indexes.pin_memory()
        if hasattr(self, 'packed_action_token_indexes'):
            self.packed_action_token_indexes = self.packed_action_token_indexes.pin_memory()
            self.action_query_token_seqlens = self.action_query_token_seqlens.pin_memory()
            self.target_point_indexes = self.target_point_indexes.pin_memory()
            self.packed_bev_indexes = self.packed_bev_indexes.pin_memory()
            self.v_indexes = self.v_indexes.pin_memory()
        if hasattr(self, 'traj_gt'):
            self.traj_gt = self.traj_gt.pin_memory()
            self.route_gt = self.route_gt.pin_memory()
            self.future_speeds_tensors = self.future_speeds_tensors.pin_memory()
            self.v_target_point = self.v_target_point.pin_memory()
            if self.bev_encoder_feature is not None:
                self.bev_encoder_feature = self.bev_encoder_feature.pin_memory()
            self.traj_loss_indexes = self.traj_loss_indexes.pin_memory()
            self.route_loss_indexes = self.route_loss_indexes.pin_memory()
        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()
            if self.image_tensor_list is not None:
                self.image_tensor_list = self.image_tensor_list.pin_memory()
            if self.image_grid_thw_list is not None:
                self.image_grid_thw_list = self.image_grid_thw_list.pin_memory()
        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()
        return self

    def cuda(self, device):
        if not isinstance(device, torch.device):
            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            else:
                device = torch.device(device)
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)
        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'packed_gen_vit_token_indexes'):
            self.packed_und_vit_token_indexes = self.packed_und_vit_token_indexes.to(device)
            self.packed_gen_vit_token_indexes = self.packed_gen_vit_token_indexes.to(device)
            self.packed_und_text_indexes = self.packed_und_text_indexes.to(device)
            self.packed_gen_text_indexes = self.packed_gen_text_indexes.to(device)
            self.packed_reasoning_token_indexes = self.packed_reasoning_token_indexes.to(device)

        if hasattr(self, 'packed_action_token_indexes'):
            self.packed_action_token_indexes = self.packed_action_token_indexes.to(device)
            self.action_query_token_seqlens = self.action_query_token_seqlens.to(device)
            self.target_point_indexes = self.target_point_indexes.to(device)
            self.packed_bev_indexes = self.packed_bev_indexes.to(device)
            self.v_indexes = self.v_indexes.to(device)
        if hasattr(self, 'traj_gt'):
            self.traj_gt = self.traj_gt.to(device)
            self.route_gt = self.route_gt.to(device)
            self.future_speeds_tensors = self.future_speeds_tensors.to(device)
            self.v_target_point = self.v_target_point.to(device)
            if self.bev_encoder_feature is not None:
                self.bev_encoder_feature = self.bev_encoder_feature.to(device)
            self.traj_loss_indexes = self.traj_loss_indexes.to(device)
            self.route_loss_indexes = self.route_loss_indexes.to(device)
        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)
            if self.image_tensor_list is not None:
                self.image_tensor_list = self.image_tensor_list.to(device)
            if self.image_grid_thw_list is not None:
                self.image_grid_thw_list = self.image_grid_thw_list.to(device)


        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens
            if self.image_tensor_list is not None:
                data['image_tensor_list'] = self.image_tensor_list
            if self.image_grid_thw_list is not None:
                data['image_grid_thw_list'] = self.image_grid_thw_list

        if hasattr(self, 'packed_gen_vit_token_indexes'):
            data['packed_und_vit_token_indexes'] = self.packed_und_vit_token_indexes
            data['packed_gen_vit_token_indexes'] = self.packed_gen_vit_token_indexes
            data['packed_reasoning_token_indexes'] = self.packed_reasoning_token_indexes
            data['packed_und_text_indexes'] = self.packed_und_text_indexes
            data['packed_gen_text_indexes'] = self.packed_gen_text_indexes

        if hasattr(self, 'packed_action_token_indexes'):
            data['packed_action_token_indexes'] = self.packed_action_token_indexes
            data['action_query_token_seqlens'] = self.action_query_token_seqlens
            data['target_point_indexes'] = self.target_point_indexes
            data['packed_bev_indexes'] = self.packed_bev_indexes
            data['v_indexes'] = self.v_indexes
        if hasattr(self, 'traj_gt'):
            data['traj_gt'] = self.traj_gt
            data['route_gt'] = self.route_gt
            data['future_speeds_tensors'] = self.future_speeds_tensors
            data['v_target_point'] = self.v_target_point
            data['bev_encoder_feature'] = self.bev_encoder_feature
            data['bev_encoder_feature_inputs'] = self.bev_encoder_feature_inputs
            data['traj_loss_indexes'] = self.traj_loss_indexes
            data['route_loss_indexes'] = self.route_loss_indexes

        if hasattr(self, 'probs'):
            data['probs'] = self.probs

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
