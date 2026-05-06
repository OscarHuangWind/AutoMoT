
import json
import os
import traceback
import re

from PIL import Image, ImageFile, PngImagePlugin
from transformers import AutoProcessor
import torch

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SftJSONLIterableDatasetpdmtraj(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, frame_sampler,
        jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer

        model_id = "Qwen/Qwen3-VL-4B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.frame_sampler = frame_sampler
        self.data_status = data_status

        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

        self.reasoning_text_max_num_tokens = 9

    @staticmethod
    def _parse_prompt_fields(value: str):
        # target point: (x, y)
        m_tp = re.search(
            r"target point is\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
            value, re.I
        )
        
        # velocity: v m/s
        m_v = re.search(
            r"velocity is\s*([-+]?\d*\.?\d+)\s*m/s",
            value, re.I
        )
        
        if m_tp and m_v:
            return [float(m_v.group(1)), float(m_tp.group(1)), float(m_tp.group(2))]
        return None

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_lines,
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def _trajectory_to_coords4(self, traj):
        """
        traj: [x1,y1,x2,y2,...]  -> coords[:4] (4,2) or None
        """
        if not isinstance(traj, list):
            return None
        if len(traj) < 2 or (len(traj) % 2 != 0):
            return None

        coords = []
        for i in range(0, len(traj), 2):
            try:
                x = float(traj[i])
                y = float(traj[i + 1])
            except Exception:
                return None
            coords.append([x, y])

        return coords

    def change_format(self, data, num_images, num_fronts, num_lidars):
        """
        """
        elements = []

        ego_fut_trajs = self._trajectory_to_coords4(data.get("trajectory", None))
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                value = conversation['value']
                parts = re.split(r'(<image>|<front>|<lidar>)', value)
                v_target_point = self._parse_prompt_fields(value)
                
                parts = re.split(r'(<image>|<front>|<lidar>)', value)
                img_i = 0
                front_i = 0
                lidar_i = 0

                for part in parts:
                    if part == '<image>':
                        if img_i < num_images:
                            elements.append({'type': 'image'})
                            img_i += 1
                    elif part == '<front>':
                        if front_i < num_fronts:
                            elements.append({'type': 'front'})
                            front_i += 1
                    elif part == '<lidar>':
                        if lidar_i < num_lidars:
                            elements.append({'type': 'lidar'})
                            lidar_i += 1
                    else:
                        text = part.strip()
                        if text:
                            elements.append({
                                'type': 'fast_text',
                                'has_loss': 0,
                                'text': text,
                            })

            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'reasoning_text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })

        return elements, ego_fut_trajs, v_target_point

    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        width = max(1, int(width))
        height = max(1, int(height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                image_grid_thw_list = []
                front_tensor_list = []
                front_grid_thw_list = []
                future_tensor_list = []
                future_grid_thw_list = []
                text_list = []
                text_ids_list = []
                sequence_plan = []
                number_vit_tokens = []
                number_front_vit_tokens = []
                ego_fut_trajs_tensor = None
                lidar_tensor_list = []
                lidar_grid_thw_list = []
                number_lidar_vit_tokens = []

                try:
                    data_item = json.loads(data)

                    raw_images = None
                    raw_fronts = None
                    raw_lidars = None

                    if 'image' in data_item:
                        if isinstance(data_item['image'], list):
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, image)))
                                for image in data_item['image']
                            ]
                        else:
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, data_item['image'])))
                            ]
                    elif 'video' in data_item:
                        raw_images = self.frame_sampler(os.path.join(image_dir, data_item['video']))
                        special_tokens = '<image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if '<video>' in item['value']:
                                item['value'] = item['value'].replace('<video>', special_tokens)
                                break
                        else:
                            raise ValueError("Cannot find <video> in the conversation!")

                    if 'front' in data_item:
                        if isinstance(data_item['front'], list):
                            raw_fronts = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, front)))
                                for front in data_item['front']
                            ]
                        else:
                            raw_fronts = [pil_img2rgb(Image.open(os.path.join(image_dir, data_item['front'])))]

                    if 'lidar' in data_item:
                        if isinstance(data_item['lidar'], list):
                            raw_lidars = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, lidar)))
                                for lidar in data_item['lidar']
                            ]
                        else:
                            raw_lidars = [pil_img2rgb(Image.open(os.path.join(image_dir, data_item['lidar'])))]

                    # optional future images
                    future_val = data_item.get("future", None)
                    raw_futures = []
                    if isinstance(future_val, list):
                        for p in future_val:
                            if isinstance(p, str) and p.strip() != "":
                                raw_futures.append(p.strip())
                    elif isinstance(future_val, str):
                        if future_val.strip() != "":
                            raw_futures.append(future_val.strip())

                    prob_dict = None
                    p = data_item.get("probs", None)
                    if isinstance(p, list):
                        if p and isinstance(p[0], dict):
                            d = p[0]
                            prob_dict = {
                                "stop":       float(d.get("stop", 0.0)),
                                "accelerate": float(d.get("accelerate", 0.0)),
                                "decelerate": float(d.get("decelerate", d.get("slow", 0.0))),
                                "keep":       float(d.get("keep", d.get("constant", 0.0))),
                            }
                        elif len(p) >= 4:
                            prob_dict = {
                                "stop":       float(p[0]),
                                "accelerate": float(p[1]),
                                "decelerate": float(p[2]),
                                "keep":       float(p[3]),
                            }
                        else:
                            prob_dict = None

                except Exception:
                    traceback.print_exc()
                    continue

                # --- images ---
                if raw_images:
                    for raw_image in raw_images:
                        raw_image = self.resize_image(raw_image, width=512, height=256)
                        input_dic = self.processor(text='', images=raw_image, return_tensors="pt")
                        image_tensor = input_dic['pixel_values']
                        image_grid_thw = input_dic['image_grid_thw']
                        number_of_patches = image_tensor.shape[0]
                        image_tensor_list.append(image_tensor)
                        image_grid_thw_list.append(image_grid_thw[0])
                        number_of_tokens_per_sample = int(number_of_patches / 4)
                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)
                        number_vit_tokens.append(number_of_tokens_per_sample)

                image_num = len(image_tensor_list)
                image_tensor_list_concat = list(image_tensor_list)

                # --- lidar ---
                if raw_lidars:
                    for raw_lidar in raw_lidars:
                        lidar_dic = self.processor(text='', images=raw_lidar, return_tensors="pt")
                        lidar_tensor = lidar_dic['pixel_values']
                        lidar_grid_thw = lidar_dic['image_grid_thw']
                        number_of_patches = lidar_tensor.shape[0]
                        lidar_tensor_list.append(lidar_tensor)
                        lidar_grid_thw_list.append(lidar_grid_thw[0])
                        number_of_lidar_tokens_per_sample = int(number_of_patches / 4)
                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)
                        number_lidar_vit_tokens.append(number_of_lidar_tokens_per_sample)

                # for fast thinking, add the bev map
                if len(lidar_tensor_list) > 0:
                    image_tensor_list_concat.append(lidar_tensor_list[0])

                # --- front ---
                if raw_fronts:
                    for raw_front in raw_fronts:
                        raw_front = self.resize_image(raw_front, width=512, height=256)
                        front_dic = self.processor(text='', images=raw_front, return_tensors="pt")
                        front_tensor = front_dic['pixel_values']
                        front_grid_thw = front_dic['image_grid_thw']
                        number_of_patches = front_tensor.shape[0]
                        front_tensor_list.append(front_tensor)
                        front_grid_thw_list.append(front_grid_thw[0])
                        number_of_front_tokens_per_sample = int(number_of_patches / 4)
                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)
                        number_front_vit_tokens.append(number_of_front_tokens_per_sample)

                # for fast thinking, add the front at end
                if len(front_tensor_list) > 0:
                    image_tensor_list_concat.append(front_tensor_list[0])

                # --- build sequence plan + trajectory label ---
                elements, ego_fut_trajs, v_target_point = self.change_format(
                    data_item,
                    image_num,
                    len(front_tensor_list),
                    len(lidar_tensor_list),
                )

                # trajectory tensor label
                ego_fut_trajs_tensor = None
                if ego_fut_trajs is not None and len(ego_fut_trajs) > 0:
                    ego_fut_trajs_tensor = torch.tensor(ego_fut_trajs, dtype=torch.float32)

                # optional future images
                if len(raw_futures) > 0:
                    try:
                        for f_path in raw_futures:
                            img = pil_img2rgb(Image.open(os.path.join(image_dir, f_path)))
                            img = self.resize_image(img, width=512, height=256)
                            fdic = self.processor(text='', images=img, return_tensors="pt")
                            future_tensor_list.append(fdic["pixel_values"])
                            future_grid_thw_list.append(fdic["image_grid_thw"][0])
                    except Exception as e:
                        print("Future image load failed:", raw_futures, e)
                        future_tensor_list = []
                        future_grid_thw_list = []

                # --- convert elements to plan + tokenize text ---
                for item in elements:
                    if item['type'] == 'fast_text':
                        text_data = item['text']
                        text_list.append(text_data)
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                'type': 'fast_text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)

                    elif item['type'] == 'image':
                        sequence_plan.append({
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })

                    elif item['type'] == 'front':
                        sequence_plan.append({
                            'type': 'front_bev',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })

                    elif item['type'] == 'lidar':
                        sequence_plan.append({
                            'type': 'lidar_bev',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })

                    elif item['type'] == 'reasoning_text':
                        text_data = item['text']
                        text_list.append(text_data)
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += self.reasoning_text_max_num_tokens
                            current_plan = {
                                'type': 'reasoning_text',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)

                has_loss = [x['loss'] for x in sequence_plan]
                if sum(has_loss) == 0 and ego_fut_trajs_tensor is None:
                    print('No loss defined and no ego_fut_trajs, skipped.')
                    continue
                v_target_point_tensor = None
                if v_target_point is not None:
                    v_target_point_tensor = torch.tensor(v_target_point, dtype=torch.float32)
                yield dict(
                    image_tensor_list=image_tensor_list,
                    image_tensor_list_concat=image_tensor_list_concat,
                    image_grid_thw_list=image_grid_thw_list,
                    front_tensor_list=front_tensor_list,
                    front_grid_thw_list=front_grid_thw_list,
                    lidar_tensor_list=lidar_tensor_list,
                    lidar_grid_thw_list=lidar_grid_thw_list,
                    number_lidar_vit_tokens=number_lidar_vit_tokens,
                    future_tensor_list=future_tensor_list,
                    future_grid_thw_list=future_grid_thw_list,
                    number_vit_tokens=number_vit_tokens,
                    number_front_vit_tokens=number_front_vit_tokens,
                    ego_fut_trajs_tensor=ego_fut_trajs_tensor,
                    v_target_point=v_target_point_tensor,
                    text_ids_list=text_ids_list,
                    prob_dict=prob_dict,
                    text_list=text_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
