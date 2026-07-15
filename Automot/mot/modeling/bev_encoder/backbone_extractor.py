"""BEV encoder backbone feature extractor."""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from safetensors import safe_open

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()

from mot.modeling.bev_encoder.config import GlobalConfig
from mot.modeling.bev_encoder.bev_encoder import BEVEncoderBackbone
import mot.modeling.bev_encoder.bev_encoder_utils as t_u


class BEVEncoderBackboneExtractor(nn.Module):
    """Load the driving backbone and expose frozen BEV feature extraction."""
    
    def __init__(self, config_path: str, model_path: str = None, device: str = 'cuda:0',
                 state_dict: dict = None):
        super().__init__()
        
        self.device = torch.device(device)
        self.config_path = config_path
        
        self.config = self._load_config(config_path)
        
        self.backbone = BEVEncoderBackbone(self.config)
        
        self._load_weights(config_path, model_path, state_dict=state_dict)
        
        self._freeze_parameters()
        
        self.backbone.to(self.device)
        self.backbone.eval()
        
        print(f"BEV Encoder Backbone initialized on {device}")
        print(f"  - Config: {config_path}")
        print("  - All parameters frozen")
        
    def _load_config(self, config_path: str) -> GlobalConfig:
        # Try bev_config.json first (merged config dir), fall back to config.json
        bev_cfg = os.path.join(config_path, 'bev_config.json')
        config_file = bev_cfg if os.path.isfile(bev_cfg) else os.path.join(config_path, 'config.json')
        
        with open(config_file, 'rt', encoding='utf-8') as f:
            json_config = f.read()
        
        loaded_config = jsonpickle.decode(json_config)
        
        config = GlobalConfig()
        if isinstance(loaded_config, dict):
            config.__dict__.update(loaded_config)
        else:
            config.__dict__.update(vars(loaded_config))
        
        return config
    
    def _load_weights(self, config_path: str, model_path: str = None, state_dict: dict = None):
        if state_dict is not None:
            print("Loading BEV encoder weights from pre-loaded state_dict.")
            backbone_state_dict = self._extract_backbone_state_dict(state_dict)
        else:
            if model_path is None:
                safetensors_path = os.path.join(config_path, 'model.safetensors')
                if os.path.isfile(safetensors_path):
                    model_path = safetensors_path
                else:
                    pth_candidates = sorted(
                        os.path.join(config_path, file)
                        for file in os.listdir(config_path)
                        if file.startswith('model') and file.endswith('.pth')
                    )
                    if len(pth_candidates) == 1:
                        model_path = pth_candidates[0]
                    elif len(pth_candidates) > 1:
                        raise ValueError(
                            "Multiple BEV encoder checkpoints found; pass model_path explicitly"
                        )
            
            if model_path is None:
                raise FileNotFoundError(f"No model weights found in {config_path}")
            
            print(f"Loading weights from: {model_path}")
            
            if model_path.endswith('.safetensors'):
                backbone_state_dict = self._load_safetensors_backbone_state_dict(model_path)
            else:
                full_state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(full_state_dict, dict):
                    for key in ("state_dict", "model", "module"):
                        if key in full_state_dict and isinstance(full_state_dict[key], dict):
                            full_state_dict = full_state_dict[key]
                            break
                backbone_state_dict = self._extract_backbone_state_dict(full_state_dict)
        
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

    @staticmethod
    def _extract_backbone_state_dict(full_state_dict):
        backbone_state_dict = {}
        has_prefixed_keys = any(
            key.startswith('bev_encoder.') or key.startswith('backbone.')
            for key in full_state_dict
        )
        for key, value in full_state_dict.items():
            if key.startswith('bev_encoder.'):
                backbone_state_dict[key[len('bev_encoder.'):]] = value
            elif key.startswith('backbone.'):
                backbone_state_dict[key[len('backbone.'):]] = value
            elif not has_prefixed_keys:
                backbone_state_dict[key] = value
        if not backbone_state_dict:
            raise KeyError("No BEV encoder weights found in checkpoint state dict")
        return backbone_state_dict

    @staticmethod
    def _load_safetensors_backbone_state_dict(model_path):
        with safe_open(model_path, framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
            has_prefixed_keys = any(
                key.startswith('bev_encoder.') or key.startswith('backbone.')
                for key in keys
            )
            backbone_state_dict = {}
            for key in keys:
                if key.startswith('bev_encoder.'):
                    backbone_state_dict[key[len('bev_encoder.'):]] = handle.get_tensor(key)
                elif key.startswith('backbone.'):
                    backbone_state_dict[key[len('backbone.'):]] = handle.get_tensor(key)
                elif not has_prefixed_keys:
                    backbone_state_dict[key] = handle.get_tensor(key)
        if not backbone_state_dict:
            raise KeyError(f"No BEV encoder weights found in {model_path}")
        return backbone_state_dict
            
    def _freeze_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.backbone.parameters())
        frozen_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
    
    @torch.no_grad()
    def forward(self, rgb: torch.Tensor, lidar_bev: torch.Tensor):
        """
        
        Args:
            
        Returns:
        """
        rgb = rgb.to(self.device)
        lidar_bev = lidar_bev.to(self.device)
        
        
        bev_feature, bev_feature_upscale, fused_features, image_feature_grid = \
            self._forward_with_intermediate(rgb, lidar_bev)
        
        return {
            'bev_feature': bev_feature,
            'bev_feature_upscale': bev_feature_upscale,
            'fused_features': fused_features,
            'image_feature_grid': image_feature_grid
        }
    
    def _forward_with_intermediate(self, image, lidar):
        """
        """
        if self.config.normalize_imagenet:
            image_features = t_u.normalize_imagenet(image)
        else:
            image_features = image
        
        if self.backbone.lidar_video:
            batch_size = lidar.shape[0]
            lidar_features = lidar.view(batch_size, -1, self.config.lidar_seq_len, 
                                        self.config.lidar_resolution_height,
                                        self.config.lidar_resolution_width)
        else:
            lidar_features = lidar
        
        image_layers = iter(self.backbone.image_encoder.items())
        lidar_layers = iter(self.backbone.lidar_encoder.items())
        
        # Stem layer
        if len(self.backbone.image_encoder.return_layers) > 4:
            image_features = self.backbone.forward_layer_block(
                image_layers, self.backbone.image_encoder.return_layers, image_features)
        if len(self.backbone.lidar_encoder.return_layers) > 4:
            lidar_features = self.backbone.forward_layer_block(
                lidar_layers, self.backbone.lidar_encoder.return_layers, lidar_features)
        
        for i in range(4):
            image_features = self.backbone.forward_layer_block(
                image_layers, self.backbone.image_encoder.return_layers, image_features)
            lidar_features = self.backbone.forward_layer_block(
                lidar_layers, self.backbone.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self.backbone.fuse_features(image_features, lidar_features, i)
        
        if self.config.detect_boxes or self.config.use_bev_semantic:
            if self.backbone.lidar_video:
                lidar_features_for_bev = torch.mean(lidar_features, dim=2)
            else:
                lidar_features_for_bev = lidar_features
            x4 = lidar_features_for_bev
        else:
            x4 = None
        
        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features
        
        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features_pooled = self.backbone.global_pool_img(image_features)
            image_features_pooled = torch.flatten(image_features_pooled, 1)
            lidar_features_pooled = self.backbone.global_pool_lidar(lidar_features)
            lidar_features_pooled = torch.flatten(lidar_features_pooled, 1)
            
            if self.config.add_features:
                lidar_features_pooled = self.backbone.lidar_to_img_features_end(lidar_features_pooled)
                fused_features = image_features_pooled + lidar_features_pooled
            else:
                fused_features = torch.cat((image_features_pooled, lidar_features_pooled), dim=1)
        
        if self.config.detect_boxes or self.config.use_bev_semantic:
            bev_feature_upscale = self.backbone.top_down(x4)
        else:
            bev_feature_upscale = None
        
        return x4, bev_feature_upscale, fused_features, image_feature_grid
    
    def preprocess_rgb(self, rgb_path: str) -> torch.Tensor:
        """
        
        Args:
            
        Returns:
        """
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = t_u.crop_array(self.config, image)
        
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        return image
    
    def preprocess_lidar(self, lidar_path: str) -> torch.Tensor:
        """
        
        Args:
            
        Returns:
        """
        import laspy

        las_object = laspy.read(lidar_path)
        lidar = las_object.xyz
        
        lidar_bev = self.lidar_to_histogram_features(lidar, use_ground_plane=self.config.use_ground_plane)
        
        lidar_bev = torch.from_numpy(lidar_bev).float().unsqueeze(0)
        
        return lidar_bev
    
    def lidar_to_histogram_features(self, lidar: np.ndarray, use_ground_plane: bool) -> np.ndarray:
        """
        
        Args:
            
        Returns:
        """
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(self.config.min_x, self.config.max_x,
                                (self.config.max_x - self.config.min_x) * int(self.config.pixels_per_meter) + 1)
            ybins = np.linspace(self.config.min_y, self.config.max_y,
                                (self.config.max_y - self.config.min_y) * int(self.config.pixels_per_meter) + 1)
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self.config.hist_max_per_pixel] = self.config.hist_max_per_pixel
            overhead_splat = hist / self.config.hist_max_per_pixel
            return overhead_splat.T
        
        lidar = lidar[lidar[..., 2] < self.config.max_height_lidar]
        below = lidar[lidar[..., 2] <= self.config.lidar_split_height]
        above = lidar[lidar[..., 2] > self.config.lidar_split_height]
        below_features = splat_points(below)
        above_features = splat_points(above)
        
        if use_ground_plane:
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features
    
    def extract_features(self, rgb_path: str, lidar_path: str):
        """
        
        Args:
            
        Returns:
        """
        rgb = self.preprocess_rgb(rgb_path)
        lidar_bev = self.preprocess_lidar(lidar_path)
        
        return self.forward(rgb, lidar_bev)
