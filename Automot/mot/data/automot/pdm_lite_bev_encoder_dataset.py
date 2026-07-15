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


class PdmLiteBEVEncoderDataset(DistributedIterableDataset):
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

        model_id = os.environ.get("QWEN3VL_PATH", "Qwen/Qwen3-VL-4B-Instruct")
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
        self.reasoning_text_max_num_tokens = 40

    @staticmethod
    def _parse_prompt_fields(value: str):
        # velocity: v m/s
        m_v = re.search(
            r"velocity is\s*([-+]?\d*\.?\d+)\s*m/s",
            value, re.I
        )

        # two target points: (x1, y1), (x2, y2)
        m_tp = re.search(
            r"target point is\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*,\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
            value, re.I
        )

        if m_tp and m_v:
            return [
                float(m_v.group(1)),      # velocity
                float(m_tp.group(1)),     # tp1_x
                float(m_tp.group(2)),     # tp1_y
                float(m_tp.group(3)),     # tp2_x
                float(m_tp.group(4)),     # tp2_y
            ]
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
        traj: [x1,y1,x2,y2,...] -> coords (N,2) or None
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
        elements = []

        ego_fut_trajs = self._trajectory_to_coords4(data.get("trajectory", None))
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                value = conversation['value']

                parts = re.split(r'(<image>|<front>|<lidar>|<bev>)', value)
                v_target_point = self._parse_prompt_fields(value)

                img_i = 0
                front_i = 0
                lidar_i = 0
                bev_i = 0

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
                    elif part == '<bev>':
                        elements.append({'type': 'bev_encoder'})
                        bev_i += 1
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

    @staticmethod
    def _has_bev_encoder_marker(data_item: dict) -> bool:
        """Return True if any human prompt contains '<bev>'."""
        return PdmLiteBEVEncoderDataset._has_prompt_marker(data_item, "<bev>")

    @staticmethod
    def _has_prompt_marker(data_item: dict, marker: str) -> bool:
        convs = data_item.get("conversations", [])
        for c in convs:
            if c.get("from") == "human" and marker in c.get("value", ""):
                return True
        return False

    @staticmethod
    def _first_path(value):
        if isinstance(value, list) and value:
            return value[0]
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _resolve_online_bev_encoder_input(self, data_item: dict, image_dir: str):
        rgb_rel = self._first_path(data_item.get("front"))
        if rgb_rel is None:
            image_value = data_item.get("image")
            if isinstance(image_value, list) and image_value:
                rgb_rel = image_value[-1]
            elif isinstance(image_value, str) and image_value.strip():
                rgb_rel = image_value
        if rgb_rel is None:
            return None

        rgb_path = os.path.join(image_dir, rgb_rel)
        if not os.path.exists(rgb_path):
            return None

        frame = os.path.splitext(os.path.basename(rgb_rel))[0]
        route_dir = os.path.dirname(os.path.dirname(rgb_rel))
        lidar_bev_rel = os.path.join(route_dir, "bev_encoder_lidar_bev", f"{frame}.npy")
        lidar_bev_path = os.path.join(image_dir, lidar_bev_rel)
        if not os.path.exists(lidar_bev_path):
            return None

        return {
            "rgb_path": rgb_path,
            "lidar_bev_path": lidar_bev_path,
        }

    @staticmethod
    def _num_bev_encoder_tokens(feature: torch.Tensor) -> int:
        if feature.dim() == 4:
            return int(feature.shape[-2] * feature.shape[-1])
        if feature.dim() == 3:
            return int(feature.shape[1])
        if feature.dim() == 2:
            return int(feature.shape[0])
        raise ValueError(f"Unsupported BEV encoder feature shape: {tuple(feature.shape)}")

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

                bev_encoder_feature_path = None
                bev_encoder_feature = None
                online_bev_encoder_input = None
                number_bev_encoder_tokens = 0

                try:
                    data_item = json.loads(data)

                    raw_images = None
                    raw_fronts = None
                    raw_lidars = None

                    needs_bev_encoder = self._has_bev_encoder_marker(data_item)
                    needs_front_tokens = self._has_prompt_marker(data_item, "<front>")
                    needs_lidar_tokens = self._has_prompt_marker(data_item, "<lidar>")

                    bev_encoder_feature_path = data_item.get("bev_encoder_feature", None)

                    def _load_bev_encoder_feature(feature_full):
                        try:
                            feature_obj = torch.load(feature_full, map_location="cpu", weights_only=True)
                        except TypeError:
                            feature_obj = torch.load(feature_full, map_location="cpu")
                        if isinstance(feature_obj, dict) and "bev_features" in feature_obj:
                            frame = data_item.get("bev_encoder_feature_frame", None)
                            if frame is None:
                                front_value = data_item.get("front", None)
                                if isinstance(front_value, list) and len(front_value) > 0:
                                    frame = os.path.splitext(os.path.basename(front_value[0]))[0]
                                elif isinstance(front_value, str):
                                    frame = os.path.splitext(os.path.basename(front_value))[0]
                            if frame is None:
                                frame = str(data_item.get("id", "")).rsplit("_", 1)[-1]
                            frame = str(frame).zfill(4)

                            frame_nums = [str(x).zfill(4) for x in feature_obj.get("frame_nums", [])]
                            if frame not in frame_nums:
                                raise KeyError(f"frame {frame} not found in {feature_full}")
                            frame_idx = frame_nums.index(frame)
                            return feature_obj["bev_features"][frame_idx:frame_idx + 1]
                        return feature_obj

                    if needs_bev_encoder:
                        if isinstance(bev_encoder_feature_path, str) and bev_encoder_feature_path.strip() != "":
                            feature_full = os.path.join(image_dir, bev_encoder_feature_path)
                            if os.path.exists(feature_full):
                                try:
                                    bev_encoder_feature = _load_bev_encoder_feature(feature_full)
                                    number_bev_encoder_tokens = self._num_bev_encoder_tokens(bev_encoder_feature)
                                except Exception as e:
                                    print(f"[WARN] load bev_encoder_feature failed, trying online BEV encoder: {feature_full} {e}")
                            else:
                                print(f"[WARN] bev_encoder_feature missing, trying online BEV encoder: {feature_full}")

                        if bev_encoder_feature is None:
                            online_bev_encoder_input = self._resolve_online_bev_encoder_input(data_item, image_dir)
                            if online_bev_encoder_input is None:
                                print(f"[SKIP] <bev> present but no bev_encoder_feature or online BEV encoder input at row {row_idx}")
                                continue
                            number_bev_encoder_tokens = 64
                    else:
                        if isinstance(bev_encoder_feature_path, str) and bev_encoder_feature_path.strip() != "":
                            feature_full = os.path.join(image_dir, bev_encoder_feature_path)
                            if os.path.exists(feature_full):
                                try:
                                    bev_encoder_feature = _load_bev_encoder_feature(feature_full)
                                    number_bev_encoder_tokens = self._num_bev_encoder_tokens(bev_encoder_feature)
                                except Exception:
                                    bev_encoder_feature = None
                                    number_bev_encoder_tokens = 0

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

                    if needs_front_tokens and 'front' in data_item:
                        if isinstance(data_item['front'], list):
                            raw_fronts = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, front)))
                                for front in data_item['front']
                            ]
                        else:
                            raw_fronts = [pil_img2rgb(Image.open(os.path.join(image_dir, data_item['front'])))]

                    if needs_lidar_tokens and 'lidar' in data_item:
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

                    # future_speeds: [speed_1s, speed_2s, speed_3s]
                    future_speeds = None
                    fs = data_item.get("future_speeds", None)
                    if isinstance(fs, list) and len(fs) >= 3:
                        try:
                            future_speeds = [float(fs[0]), float(fs[1]), float(fs[2])]
                        except Exception:
                            future_speeds = None

                    # route_future: 20 waypoints -> (20,2)
                    route_future = None
                    route_data = data_item.get("route", None)
                    if isinstance(route_data, list) and len(route_data) >= 40:
                        try:
                            route_future = [
                                [float(route_data[i]), float(route_data[i + 1])]
                                for i in range(0, 40, 2)
                            ]
                        except Exception:
                            route_future = None

                except Exception:
                    traceback.print_exc()
                    continue

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

                elements, ego_fut_trajs, v_target_point = self.change_format(
                    data_item,
                    image_num,
                    len(front_tensor_list),
                    len(lidar_tensor_list),
                )

                ego_fut_trajs_tensor = None
                if ego_fut_trajs is not None and len(ego_fut_trajs) > 0:
                    ego_fut_trajs_tensor = torch.tensor(ego_fut_trajs, dtype=torch.float32)

                future_speeds_tensor = None
                if future_speeds is not None:
                    future_speeds_tensor = torch.tensor(future_speeds, dtype=torch.float32)

                route_future_tensor = None
                if route_future is not None:
                    route_future_tensor = torch.tensor(route_future, dtype=torch.float32)

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
                            sequence_plan.append({
                                'type': 'fast_text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

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
                            'type': 'front_vit',
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

                    elif item['type'] == 'bev_encoder':
                        sequence_plan.append({
                            'type': 'bev_encoder_feature',
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
                            sequence_plan.append({
                                'type': 'reasoning_text',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

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
                    future_speeds_tensor=future_speeds_tensor,
                    route_future_tensor=route_future_tensor,
                    v_target_point=v_target_point_tensor,
                    text_ids_list=text_ids_list,
                    prob_dict=prob_dict,
                    text_list=text_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,

                    number_bev_encoder_tokens=number_bev_encoder_tokens,
                    bev_encoder_feature=bev_encoder_feature,
                    online_bev_encoder_input=online_bev_encoder_input,
                    bev_encoder_feature_path=bev_encoder_feature_path,

                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
