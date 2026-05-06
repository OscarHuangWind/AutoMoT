
import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin
import re
from transformers import AutoProcessor
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SftJSONLIterableDataset(DistributedIterableDataset):
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
        processor = AutoProcessor.from_pretrained(model_id)
        self.processor = processor
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
        self.reasoning_text_max_num_tokens = 8 #this parameter will be set from outside

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

    def change_format(self, data, num_images, num_lidars):
        elements = []
        ## --jh-- change the format to include lidar
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                # value = conversation['from']
                value = conversation['value']
                # Split while keeping the markers
                parts = re.split(r'(<image>|<lidar>)', value)
                img_i = 0
                lidar_i = 0
                ### for fast thinking, we add the last frame again ###
                for part in parts:
                    if part == '<image>':
                        if img_i < num_images:
                            elements.append({'type': 'image'})
                            img_i += 1
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
                inf_type = data.get('inference_type') or conversation.get('inference_type') # --jh-- inf type is in data not in conversation
                if inf_type == 'fast reasoning':                    
                    elements.append({
                        'type': 'reasoning_text',
                        'has_loss': 1,
                        'text': conversation['value'],
                    })
                else:
                    elements.append({
                        'type': 'text',
                        'has_loss': 1,
                        'text': conversation['value'],
                    })
                break
        return elements
    
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
        transform_stride = self.transform.stride

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
                lidar_tensor_list = [] # --jh-- add lidar
                lidar_grid_thw_list = []
                text_list = []
                text_ids_list = []
                sequence_plan = []
                number_vit_tokens = []
                number_lidar_vit_tokens = []
                
                try:
                    data_item = json.loads(data)
                    raw_images = None
                    raw_lidars = None # --jh-- add lidar
                    if 'image' in data_item:
                        if type(data_item['image']) == list:
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
                    if 'lidar' in data_item:
                        if type(data_item['lidar']) == list:
                            raw_lidars = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, lidar)))
                                for lidar in data_item['lidar']
                            ]
                        else:
                            raw_lidars = [pil_img2rgb(Image.open(os.path.join(image_dir, data_item['lidar'])))]
                    prob_dict = None
                    if 'probs' in data_item:
                        p = data_item['probs']
                    else:
                        p = None

                    if isinstance(p, list):
                        p = p[0] if p and isinstance(p[0], dict) else None

                    if isinstance(p, dict):
                        prob_dict = {
                            'accelerate': float(p.get('accelerate', 0.0)),
                            'constant' : float(p.get('constant' , 0.0)),
                            'slow'     : float(p.get('slow'     , 0.0)),
                        }
                    else:
                        prob_dict = None
                except:
                    traceback.print_exc()
                    continue

                if raw_images:
                    for raw_image in raw_images: 
                        h, w = raw_image.size[1], raw_image.size[0] #h=512 w=1024
                        #raw_image = self.resize_image(raw_image, width=w/2, height=h/2) 
                        input_dic = self.processor(
                        text='',
                        images=raw_image, 
                        return_tensors="pt"
                        )
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
                ### for fast thinking, repeat the last frame ###
                image_tensor_list_concat.append(image_tensor_list_concat[-1])
                image_tensor_list.append(image_tensor_list[-1])

                if raw_lidars:
                    for raw_lidar in raw_lidars:
                        h, w = raw_lidar.size[1], raw_lidar.size[0]
                        lidar_dic = self.processor(
                        text='',
                        images=raw_lidar, 
                        return_tensors="pt"
                        )
                        lidar_tensor = lidar_dic['pixel_values']
                        lidar_grid_thw = lidar_dic['image_grid_thw']
                        number_of_patches = lidar_tensor.shape[0]
                        lidar_tensor_list.append(lidar_tensor)
                        lidar_grid_thw_list.append(lidar_grid_thw[0])
                        number_of_lidar_tokens_per_sample = int(number_of_patches / 4)
                        num_tokens += number_of_patches / 4
                        num_tokens = int(num_tokens)
                        number_lidar_vit_tokens.append(number_of_lidar_tokens_per_sample)
                ### for fast thinking, add the bev map ###
                image_tensor_list_concat.append(lidar_tensor_list[0])
                elements = self.change_format(
                    data_item, 
                    image_num,
                    len(lidar_tensor_list), # --jh-- add lidar
                )

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
                        current_plan = {
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                    ### jh: add lidar queries to the current plan ###
                    elif item['type'] == 'lidar':
                        current_plan = {
                            'type': 'lidar_bev',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                    ### Oscar: add reasoning queries to the current plan ###
                    elif item['type'] == 'reasoning_text':
                        text_data = item['text']
                        text_list.append(text_data)
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            # num_tokens += len(text_ids)                        
                            num_tokens += self.reasoning_text_max_num_tokens
                            current_plan = {
                                'type': 'reasoning_text',
                                'enable_cfg': 0,
                                'loss': 1, # need to be trained
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                        
                        sequence_plan.append(current_plan)

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    image_tensor_list_concat=image_tensor_list_concat,
                    image_grid_thw_list=image_grid_thw_list,
                    lidar_tensor_list=lidar_tensor_list,
                    lidar_grid_thw_list=lidar_grid_thw_list, 
                    number_vit_tokens=number_vit_tokens,
                    number_lidar_vit_tokens=number_lidar_vit_tokens,
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