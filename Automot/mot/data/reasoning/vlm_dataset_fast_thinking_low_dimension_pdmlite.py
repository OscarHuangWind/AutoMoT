
import json
import os
import traceback
import pickle  # kept for compatibility, not used here
from PIL import Image, ImageFile, PngImagePlugin
import re
from transformers import AutoProcessor
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
import torch

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SftJSONLIterableDatasetpdmtraj(DistributedIterableDataset):
    """
    PDM traj dataset:
      - LLM side: keep prompt text (fast_text + reasoning_text) as usual
      - MLP side: parse numeric condition from human prompt (target_point, velocity, acceleration)
      - No cond_valid_mask (assume all provided)
    """

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        frame_sampler,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer

        model_id = "Qwen/Qwen3-VL-4B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.frame_sampler = frame_sampler
        self.data_status = data_status

        self.data_paths = self.get_data_paths(
            jsonl_path_list=jsonl_path_list,
            data_dir_list=data_dir_list,
            num_used_data=num_used_data,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )
        self.set_epoch()

        # cap reasoning token count (your original design)
        self.reasoning_text_max_num_tokens = 9

    def _parse_prompt_fields(self, value: str):
        """
        Parse from human prompt:
          - target point is (x, y)
          - current velocity is v m/s
          - acceleration is a m/s^2
        Return:
          (target_point_tensor[2], velocity_tensor[1], acceleration_tensor[1])
        If parsing fails -> None
        """
        m_tp = re.search(
            r"target point is\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)",
            value,
            re.I,
        )
        m_v = re.search(
            r"current velocity is\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*m/s",
            value,
            re.I,
        )
        m_a = re.search(
            r"acceleration is\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*m/s\^2",
            value,
            re.I,
        )

        if (m_tp is None) or (m_v is None) or (m_a is None):
            return None

        target_point = torch.tensor(
            [float(m_tp.group(1)), float(m_tp.group(2))], dtype=torch.float32
        )
        velocity = torch.tensor([float(m_v.group(1))], dtype=torch.float32)
        acceleration = torch.tensor([float(m_a.group(1))], dtype=torch.float32)
        return target_point, velocity, acceleration


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
            with open(jsonl_path, "r") as f:
                raw_data = f.readlines()

            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)

            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])

        return data_paths

    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        width = max(1, int(width))
        height = max(1, int(height))
        return image.resize((width, height), Image.Resampling.LANCZOS)


    def change_format(self, data, num_images, num_fronts, num_lidars):
        elements = []
        ego_fut_trajs = None  # parsed from gpt output (coords)

        for conversation in data["conversations"]:
            if conversation["from"] == "human":
                value = conversation["value"]
                parts = re.split(r"(<image>|<front>|<lidar>)", value)
                img_i, front_i, lidar_i = 0, 0, 0

                for part in parts:
                    if part == "<image>":
                        if img_i < num_images:
                            elements.append({"type": "image"})
                            img_i += 1
                    elif part == "<front>":
                        if front_i < num_fronts:
                            elements.append({"type": "front"})
                            front_i += 1
                    elif part == "<lidar>":
                        if lidar_i < num_lidars:
                            elements.append({"type": "lidar"})
                            lidar_i += 1
                    else:
                        text = part.strip()
                        if text:
                            elements.append(
                                {
                                    "type": "fast_text",
                                    "has_loss": 0,
                                    "text": text,
                                }
                            )

            elif conversation["from"] == "gpt":
                traj_str = conversation["value"]
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", traj_str)

                ego_fut_trajs = None
                if len(nums) >= 2 and (len(nums) % 2 == 0):
                    coords = []
                    for i in range(0, len(nums), 2):
                        coords.append([float(nums[i]), float(nums[i + 1])])

                    # keep first 4 waypoints (your current choice)
                    if len(coords) >= 4:
                        ego_fut_trajs = coords[:4]

                elements.append(
                    {
                        "type": "reasoning_text",
                        "has_loss": 1,
                        "text": conversation["value"],
                    }
                )

        return elements, ego_fut_trajs


    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = (self.data_status[worker_id] + 1) if (self.data_status is not None) else 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]

            for row_idx, (data, image_dir) in enumerate(
                data_paths_per_worker_, start=row_start_id
            ):
                num_tokens = 0

                image_tensor_list = []
                image_grid_thw_list = []

                front_tensor_list = []
                front_grid_thw_list = []

                lidar_tensor_list = []
                lidar_grid_thw_list = []
                number_lidar_vit_tokens = []

                future_tensor_list = []
                future_grid_thw_list = []

                text_list = []
                text_ids_list = []
                sequence_plan = []

                number_vit_tokens = []
                number_front_vit_tokens = []

                ego_fut_trajs_tensor = None

                target_point = None
                velocity = None
                acceleration = None
                cond_numeric = None  # [x, y, v, a]

                try:
                    data_item = json.loads(data)

                    # -------- parse numeric conditions from human prompt (MLP side) --------
                    parsed_ok = False
                    for conv in data_item.get("conversations", []):
                        if conv.get("from") == "human":
                            parsed = self._parse_prompt_fields(conv.get("value", ""))
                            if parsed is not None:
                                target_point, velocity, acceleration = parsed
                                cond_numeric = torch.cat(
                                    [target_point, velocity, acceleration], dim=0
                                )  # shape [4]
                                parsed_ok = True
                            break
                    if not parsed_ok:
                        # you said "all provided"; so if failed, treat as bad sample
                        continue
                    # ---------------------------------------------------------------------

                    raw_images = None
                    raw_fronts = None
                    raw_lidars = None

                    # images / video
                    if "image" in data_item:
                        if isinstance(data_item["image"], list):
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, p)))
                                for p in data_item["image"]
                            ]
                        else:
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, data_item["image"])))
                            ]
                    elif "video" in data_item:
                        raw_images = self.frame_sampler(os.path.join(image_dir, data_item["video"]))
                        special_tokens = "<image>" * len(raw_images)
                        replaced = False
                        for item in data_item["conversations"]:
                            if "<video>" in item.get("value", ""):
                                item["value"] = item["value"].replace("<video>", special_tokens)
                                replaced = True
                                break
                        if not replaced:
                            raise ValueError("Cannot find <video> in the conversation!")

                    # front
                    if "front" in data_item:
                        if isinstance(data_item["front"], list):
                            raw_fronts = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, p)))
                                for p in data_item["front"]
                            ]
                        else:
                            raw_fronts = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, data_item["front"])))
                            ]

                    # lidar
                    if "lidar" in data_item:
                        if isinstance(data_item["lidar"], list):
                            raw_lidars = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, p)))
                                for p in data_item["lidar"]
                            ]
                        else:
                            raw_lidars = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, data_item["lidar"])))
                            ]

                    # future image paths (optional)
                    future_val = data_item.get("future", None)
                    raw_futures = []
                    if isinstance(future_val, list):
                        raw_futures = [p.strip() for p in future_val if isinstance(p, str) and p.strip()]
                    elif isinstance(future_val, str):
                        if future_val.strip():
                            raw_futures = [future_val.strip()]

                    # probs (optional)
                    prob_dict = None
                    p = data_item.get("probs", None)
                    if isinstance(p, list):
                        if p and isinstance(p[0], dict):
                            d = p[0]
                            prob_dict = {
                                "stop": float(d.get("stop", 0.0)),
                                "accelerate": float(d.get("accelerate", 0.0)),
                                "decelerate": float(d.get("decelerate", d.get("slow", 0.0))),
                                "keep": float(d.get("keep", d.get("constant", 0.0))),
                            }
                        elif len(p) >= 4:
                            prob_dict = {
                                "stop": float(p[0]),
                                "accelerate": float(p[1]),
                                "decelerate": float(p[2]),
                                "keep": float(p[3]),
                            }

                except Exception:
                    traceback.print_exc()
                    continue

                # -------- encode images --------
                if raw_images:
                    for raw_image in raw_images:
                        raw_image = self.resize_image(raw_image, width=512, height=256)
                        input_dic = self.processor(text="", images=raw_image, return_tensors="pt")
                        image_tensor = input_dic["pixel_values"]
                        image_grid_thw = input_dic["image_grid_thw"]

                        number_of_patches = image_tensor.shape[0]
                        image_tensor_list.append(image_tensor)
                        image_grid_thw_list.append(image_grid_thw[0])

                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)

                        number_vit_tokens.append(int(number_of_patches / 4))

                image_num = len(image_tensor_list)
                image_tensor_list_concat = list(image_tensor_list)

                # -------- encode lidar --------
                if raw_lidars:
                    for raw_lidar in raw_lidars:
                        lidar_dic = self.processor(text="", images=raw_lidar, return_tensors="pt")
                        lidar_tensor = lidar_dic["pixel_values"]
                        lidar_grid_thw = lidar_dic["image_grid_thw"]

                        number_of_patches = lidar_tensor.shape[0]
                        lidar_tensor_list.append(lidar_tensor)
                        lidar_grid_thw_list.append(lidar_grid_thw[0])

                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)

                        number_lidar_vit_tokens.append(int(number_of_patches / 4))

                # for fast thinking: append bev map (lidar) once
                if len(lidar_tensor_list) > 0:
                    image_tensor_list_concat.append(lidar_tensor_list[0])

                # -------- encode front --------
                if raw_fronts:
                    for raw_front in raw_fronts:
                        raw_front = self.resize_image(raw_front, width=512, height=256)
                        front_dic = self.processor(text="", images=raw_front, return_tensors="pt")
                        front_tensor = front_dic["pixel_values"]
                        front_grid_thw = front_dic["image_grid_thw"]

                        number_of_patches = front_tensor.shape[0]
                        front_tensor_list.append(front_tensor)
                        front_grid_thw_list.append(front_grid_thw[0])

                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)

                        number_front_vit_tokens.append(int(number_of_patches / 4))

                # for fast thinking: append front once
                if len(front_tensor_list) > 0:
                    image_tensor_list_concat.append(front_tensor_list[0])

                # -------- build sequence plan & parse gpt traj --------
                elements, ego_fut_trajs = self.change_format(
                    data_item,
                    num_images=image_num,
                    num_fronts=len(front_tensor_list),
                    num_lidars=len(lidar_tensor_list),
                )

                if ego_fut_trajs is not None and len(ego_fut_trajs) > 0:
                    ego_fut_trajs_tensor = torch.tensor(ego_fut_trajs, dtype=torch.float32)

                # -------- encode future images (optional) --------
                if len(raw_futures) > 0:
                    try:
                        for f_path in raw_futures:
                            img = pil_img2rgb(Image.open(os.path.join(image_dir, f_path)))
                            img = self.resize_image(img, width=512, height=256)
                            fdic = self.processor(text="", images=img, return_tensors="pt")
                            future_tensor_list.append(fdic["pixel_values"])
                            future_grid_thw_list.append(fdic["image_grid_thw"][0])
                    except Exception as e:
                        print("Future image load failed:", raw_futures, e)
                        future_tensor_list = []
                        future_grid_thw_list = []

                # -------- tokenize text & finalize plan --------
                for item in elements:
                    if item["type"] == "fast_text":
                        text_data = item["text"]
                        text_list.append(text_data)
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            sequence_plan.append(
                                {
                                    "type": "fast_text",
                                    "enable_cfg": 0,
                                    "loss": item["has_loss"],
                                    "special_token_loss": 0,
                                    "special_token_label": None,
                                }
                            )

                    elif item["type"] == "image":
                        sequence_plan.append(
                            {
                                "type": "vit_image",
                                "enable_cfg": 0,
                                "loss": 0,
                                "special_token_loss": 0,
                                "special_token_label": None,
                            }
                        )

                    elif item["type"] == "front":
                        sequence_plan.append(
                            {
                                "type": "front_bev",
                                "enable_cfg": 0,
                                "loss": 0,
                                "special_token_loss": 0,
                                "special_token_label": None,
                            }
                        )

                    elif item["type"] == "lidar":
                        sequence_plan.append(
                            {
                                "type": "lidar_bev",
                                "enable_cfg": 0,
                                "loss": 0,
                                "special_token_loss": 0,
                                "special_token_label": None,
                            }
                        )

                    elif item["type"] == "reasoning_text":
                        text_data = item["text"]
                        text_list.append(text_data)
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += self.reasoning_text_max_num_tokens
                            sequence_plan.append(
                                {
                                    "type": "reasoning_text",
                                    "enable_cfg": 0,
                                    "loss": 1,
                                    "special_token_loss": 0,
                                    "special_token_label": None,
                                }
                            )

                has_loss = [x["loss"] for x in sequence_plan]
                if sum(has_loss) == 0 and ego_fut_trajs_tensor is None:
                    print("No loss defined and no ego_fut_trajs, skipped.")
                    continue

                yield dict(
                    # vision
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
                    # traj label (optional)
                    ego_fut_trajs_tensor=ego_fut_trajs_tensor,
                    # text (LLM side)
                    text_ids_list=text_ids_list,
                    text_list=text_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    # probs (optional)
                    prob_dict=prob_dict,
                    # numeric cond (MLP side) 
                    target_point=target_point,        # [2]
                    velocity=velocity,                # [1]
                    acceleration=acceleration,        # [1]
                    cond_numeric=cond_numeric,        # [4] = [x, y, v, a]
                    # meta
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
