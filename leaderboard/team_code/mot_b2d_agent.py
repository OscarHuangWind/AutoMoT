import os
import sys
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import imageio
import random
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

projects_root = str(pathlib.Path(__file__).parent.parent.parent)
leaderboard_root = str(os.path.join(projects_root, 'leaderboard'))
scenario_runner_root = str(os.path.join(projects_root, 'scenario_runner'))
mot_dp_root = str(os.path.join(projects_root, 'Automot'))
carla_api_root = str(os.path.join(projects_root.replace('Bench2Drive', 'carla'), 'PythonAPI', 'carla'))

for path in [projects_root, leaderboard_root, scenario_runner_root, mot_dp_root, carla_api_root]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

sys.path = [str(p) for p in sys.path]

from leaderboard.autoagents import autonomous_agent
from team_code.nav_planner import RoutePlanner, LateralPIDController  
from agents.navigation.local_planner import RoadOption
import team_code.automot_utils as t_u  
from team_code.render import render, render_self_car, render_waypoints
from preprocess.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import xml.etree.ElementTree as ET  
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  

# BEV encoder backbone
from mot.modeling.bev_encoder.backbone_extractor import BEVEncoderBackboneExtractor
from mot.modeling.bev_encoder.config import GlobalConfig as BEVEncoderConfig
import mot.modeling.bev_encoder.bev_encoder_utils as bev_encoder_t_u

# AutoMoT dependencies
projects_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projects_root)
mot_dp_path = str(os.path.join(os.path.dirname(projects_root), 'Automot'))
mot_path = str(os.path.join(mot_dp_path, 'mot'))
sys.path.append(mot_dp_path)
sys.path.append(mot_path)
sys.path = [str(p) for p in sys.path]

from transformers import HfArgumentParser
import json
from dataclasses import dataclass, field
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from PIL import Image
from safetensors import safe_open
import glob
from data.automot.data_utils import add_special_tokens
from mot.modeling.automot import (
    AutoMoTConfig, AutoMoT,
    Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLForConditionalGenerationMoT
)
from evaluation.inference import InterleaveInferencer
from transformers import AutoTokenizer

from team_code.bev_data_utils import lidar_to_histogram_features as lidar_to_bev_histogram

from team_code.automot_utils import (
    ModelArguments, InferenceArguments,
    load_model_mot, build_cleaned_prompt_and_modes,
    parse_decision_sequence, split_prompt
)
from team_code.lidar_utils import lidar_to_ego_coordinate, algin_lidar
from team_code.ukf_utils import (
    bicycle_model_forward, measurement_function_hx,
    state_mean, measurement_mean,
    residual_state_x, residual_measurement_h
)
from team_code.display_interface import DisplayInterface

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
EARTH_RADIUS_EQUA = 6378137.0
USE_UKF = True  # Enable Unscented Kalman Filter for GPS/compass smoothing

# Entry point
def get_entry_point():
	return 'MOTAgent'


class MOTAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			now = datetime.datetime.now()
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		import gc
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Release cached memory before loading the model stack.
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()

		# Load MoT model
		print("Loading MoT model...")
		parser = HfArgumentParser((ModelArguments, InferenceArguments))
		model_args, inference_args = parser.parse_args_into_dataclasses(args=[])
		self.inference_args = inference_args  
		self.AutoMoT = load_model_mot(device)
		tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
		tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
		self.AutoMoT.language_model.tokenizer = tokenizer
		self.inferencer = InterleaveInferencer(
			model=self.AutoMoT,
			tokenizer=tokenizer,
			vit_transform=None,
			new_token_ids=new_token_ids,
			max_num_tokens=inference_args.max_num_tokens,
		)
		print("MoT model loaded.")

		print("Loading BEV encoder backbone...")
		bev_encoder_config_path = os.path.abspath(os.path.expanduser(model_args.model_path))
		combined_ckpt_path = os.path.join(bev_encoder_config_path, 'model.safetensors')
		bev_state_dict = {}
		with safe_open(combined_ckpt_path, framework="pt", device="cpu") as f:
			for key in f.keys():
				if key.startswith('bev_encoder.'):
					bev_state_dict[key[len('bev_encoder.'):]] = f.get_tensor(key)
		self.bev_encoder = BEVEncoderBackboneExtractor(
			config_path=bev_encoder_config_path,
			device='cuda:0',
			state_dict=bev_state_dict
		)
		del bev_state_dict
		self.bev_encoder.eval()
		self.bev_encoder = self.bev_encoder.to(torch.bfloat16)
		self.bev_encoder_config = self.bev_encoder.config
		print("BEV encoder backbone loaded, frozen, and converted to bfloat16.")
		
		# Initialize bev_encoder lidar buffer for temporal alignment
		self.bev_encoder_lidar_buffer = deque(maxlen=self.bev_encoder_config.lidar_seq_len * self.bev_encoder_config.data_save_freq)
		self.bev_encoder_lidar_last = None
		self.bev_encoder_state_log = deque(maxlen=max((self.bev_encoder_config.lidar_seq_len * self.bev_encoder_config.data_save_freq), 2))
		
		# Report GPU memory after both model components are resident.
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			allocated = torch.cuda.memory_allocated() / 1024**3
			reserved = torch.cuda.memory_reserved() / 1024**3
			print(f"[GPU Memory] After BEV encoder: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

		self.turn_controller = LateralPIDController(
			inference_mode=False, 
			k_p=3.118,
			speed_offset=1.195,
			default_lookahead=24
		)
		self.speed_controller = t_u.PIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20) 
		
		# Control config 
		self.carla_fps = 20
		self.wp_dilation = 1
		self.data_save_freq = 5
		self.brake_speed = 0.4
		self.brake_ratio = 1.1
		self.clip_delta = 1.0
		self.clip_throttle = 1.0
		self.stuck_threshold = 300
		self.stuck_helper_threshold = 100
		self.creep_duration = 14
		self.creep_throttle = 0.4
		
		# Stuck detection
		self.stuck_detector = 0
		self.stuck_helper = 0
		self.force_move = 0

		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = 0
		
		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0
		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
		self.lat_ref, self.lon_ref = 42.0, 2.0
		control = carla.VehicleControl()
		control.steer = 0.0
		control.throttle = 0.0
		control.brake = 0.0	
		self.prev_control = control
		self.control = control  # Store control for UKF prediction
		
		# Initialize Unscented Kalman Filter 
		self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate
		if USE_UKF:
			self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
			self.ukf = UKF(dim_x=4,
						   dim_z=4,
						   fx=bicycle_model_forward,
						   hx=measurement_function_hx,
						   dt=self.carla_frame_rate,
						   points=self.points,
						   x_mean_fn=state_mean,
						   z_mean_fn=measurement_mean,
						   residual_x=residual_state_x,
						   residual_z=residual_measurement_h)
			# State noise, same as measurement because we initialize with the first measurement later
			self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
			# Measurement noise
			self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
			self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
			# Used to set the filter state equal the first measurement
			self.filter_initialized = False
			# Stores the last filtered positions of the ego vehicle
			self.state_log = deque(maxlen=20)

		self.save_path = None
		if SAVE_PATH:
			string = self.save_name
			print(string)
			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			(self.save_path / 'rgb_front').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()
			(self.save_path / 'lidar_bev').mkdir()
		
		# Initialize lidar buffer for combining two frames
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None
		
		# Observation history buffers for MoT multi-frame input
		# MoT needs 4 RGB frames (sampled every 5 steps) and 1 lidar frame
		# Buffer needs 31 frames minimum (obs_horizon=4, sampled every 10 steps)
		obs_horizon = 4
		self.obs_horizon = obs_horizon
		self.lidar_bev_history = deque(maxlen=obs_horizon*10)
		self.rgb_history = deque(maxlen=obs_horizon*10)
		self.speed_history = deque(maxlen=obs_horizon*10)
		self.theta_history = deque(maxlen=obs_horizon*10)
		self.next_command_history = deque(maxlen=obs_horizon*10)
		self.target_point_history = deque(maxlen=obs_horizon*10)
		self.next_target_point_history = deque(maxlen=obs_horizon*10)
		self.waypoint_history = deque(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10)
		self.brake_history = deque(maxlen=obs_horizon*10)

		# Store predicted trajectory for BEV visualization
		self.last_pred_traj = None  # Store the last predicted trajectory (in ego frame)
		self.last_target_point = None  # Store the last target point (in ego frame)
		self.last_next_target_point = None  # Store the last next target point (in ego frame)
		self.last_route_pred = None  # Store the last route prediction (20 waypoints for lateral control)

		# Parking escape uses long-term displacement detection.
		self.parking_escape_active = False
		self.parking_escape_phase = 0            # 1=lateral, 2=forward
		self.parking_escape_timer = 0
		self.parking_escape_anchor = None
		self.parking_escape_start_compass = None
		self.parking_escape_attempt = 0
		self.parking_escape_cooldown = 0
		self.parking_escape_direction = 1.0      # +1 = escape left, -1 = escape right
		self.pos_snapshot_interval = 200
		self.pos_snapshots = []                  # [(step, pos), ...]
		self.parking_deadlock_window = 1500
		self.parking_deadlock_max_disp = 5.0
		self.parking_start_check_frame = 200
		self.parking_start_disp_thresh = 6.0
		self.parking_start_detected = False
		self.parking_start_checked = False
		self.parking_start_anchor = None

	def _init(self):
		# Prefer the CARLA-frame global plan and read the map georeference when available.
		try:
			world_map = CarlaDataProvider.get_map()
			xodr = world_map.to_opendrive()
			tree = ET.ElementTree(ET.fromstring(xodr))
			
			# Default values if not found in OpenDRIVE
			self.lat_ref = 42.0
			self.lon_ref = 2.0
			
			for opendrive in tree.iter('OpenDRIVE'):
				for header in opendrive.iter('header'):
					for georef in header.iter('geoReference'):
						if georef.text:
							str_list = georef.text.split(' ')
							for item in str_list:
								if '+lat_0' in item:
									self.lat_ref = float(item.split('=')[1])
								if '+lon_0' in item:
									self.lon_ref = float(item.split('=')[1])
		except Exception as e:
			# Fallback to estimating the map georeference from the first waypoint pair.
			try:
				locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
				lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
				earth_radius_equa = 6378137.0
				def equations(variables):
					x, y = variables
					eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
								 - math.cos(x * math.pi / 180.0) * y)
					eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
								 * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
								 * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
					return [eq1, eq2]
				initial_guess = [0.0, 0.0]
				solution = fsolve(equations, initial_guess)
				self.lat_ref, self.lon_ref = solution[0], solution[1]
			except Exception as e2:
				self.lat_ref, self.lon_ref = 0.0, 0.0
		

		self.route_planner_min_distance = 7.5 
		self.route_planner_max_distance = 50.0
		self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
										   self.lat_ref, self.lon_ref)
		
		if len(self._global_plan_world_coord) > 0:
			first_wp = self._global_plan_world_coord[0]
		
		# Route planner receives CARLA-frame waypoints directly.
		self._route_planner.set_route(self._global_plan_world_coord, gps=False)
		
				
		# Initialize command tracking 
		self.commands = deque(maxlen=2)
		self.commands.append(4)
		self.commands.append(4)
		self.target_point_prev = [1e5, 1e5, 1e5]
		self.last_command = -1
		self.last_command_tmp = -1
		
		self.initialized = True
		self.metric_info = {}
		self._hic = DisplayInterface()

	def sensors(self):
		sensors =  [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.50, 'y': 0.0, 'z': 2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1024, 'height': 512, 'fov': 110,
					'id': 'CAM_FRONT'
					},
				# lidar
				{
          			'type': 'sensor.lidar.ray_cast',
          			'x': 0.0, 'y': 0.0, 'z': 2.5,
          			'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
          			'id': 'LIDAR'
      				},
				# imu
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
					},
				# gps
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
					},
				# speed
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
					},
				]
		
		if IS_BENCH2DRIVE:
			sensors += [
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, input_data):
		self.step += 1
		rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])
		
		gps_full = input_data['GPS'][1]  # [lat, lon, altitude]
		gps_pos = self._route_planner.convert_gps_to_carla(gps_full)
		
		# Handle compass NaN 
		compass_raw = input_data['IMU'][1][-1]
		if math.isnan(compass_raw):
			print("compass sends nan!!!")
			compass_raw = 0.0
		
		# Preprocess compass to CARLA coordinate system
		compass = t_u.preprocess_compass(compass_raw)
		
		# Get speed for UKF
		speed = input_data['SPEED'][1]['speed']
		
		# Apply Unscented Kalman Filter 
		if USE_UKF:
			if not self.filter_initialized:
				self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
				self.filter_initialized = True

			self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
			self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
			filtered_state = self.ukf.x

			self.state_log.append(filtered_state)
			gps_filtered = filtered_state[0:2]
			compass_filtered = filtered_state[2]
		else:
			gps_filtered = np.array([gps_pos[0], gps_pos[1]])
			compass_filtered = compass
		
		# Combine two frames of lidar data using algin_lidar
		# Use filtered GPS for lidar alignment
		if self.last_lidar is not None and self.last_ego_transform is not None:
			# Calculate relative transformation between current and last frame
			current_pos = np.array([gps_filtered[0], gps_filtered[1], 0.0])
			last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
			relative_translation = current_pos - last_pos
			
			# Calculate relative rotation using filtered compass
			current_yaw = compass_filtered
			last_yaw = self.last_ego_transform['compass']
			relative_rotation = current_yaw - last_yaw
			
			# Rotate difference vector from global to local coordinate system
			rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0.0],
										[np.sin(current_yaw), np.cos(current_yaw), 0.0], 
										[0.0, 0.0, 1.0]])
			relative_translation_local = rotation_matrix.T @ relative_translation
			
			# Align the last lidar to current coordinate system
			lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
			# Combine lidar frames
			lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
		else:
			lidar_combined = lidar_ego
		
		# Store current frame for next iteration (use filtered values)
		self.last_lidar = lidar_ego
		self.last_ego_transform = {'gps': gps_filtered, 'compass': compass_filtered}
		
		# Generate lidar BEV image from combined lidar data
		lidar_bev_img = generate_lidar_bev_images(
			np.copy(lidar_combined), 
			saving_name=None, 
			img_height=448, 
			img_width=448
		)
		# Convert BEV image to tensor format for the BEV encoder backbone
		lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0
		
		# ========== BEV encoder style processing for DP features ==========
		# Process RGB for the BEV encoder backbone
		bev_encoder_rgb = input_data['CAM_FRONT'][1][:, :, :3]
		# Add jpg artifacts at test time, because the training data was saved as jpg
		_, compressed_image = cv2.imencode('.jpg', bev_encoder_rgb)
		bev_encoder_rgb = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
		bev_encoder_rgb = cv2.cvtColor(bev_encoder_rgb, cv2.COLOR_BGR2RGB)
		# Crop RGB image (same as bev_encoder training)
		bev_encoder_rgb = bev_encoder_t_u.crop_array(self.bev_encoder_config, bev_encoder_rgb)
		# Convert to PyTorch format (C, H, W) and batch
		bev_encoder_rgb = np.transpose(bev_encoder_rgb, (2, 0, 1))
		bev_encoder_rgb_tensor = torch.from_numpy(bev_encoder_rgb).float().unsqueeze(0).to('cuda')
		
		# Process LiDAR for the BEV encoder backbone
		bev_encoder_lidar = bev_encoder_t_u.lidar_to_ego_coordinate(self.bev_encoder_config, input_data['LIDAR'])
		
		# Store state for lidar alignment
		self.bev_encoder_state_log.append([gps_filtered[0], gps_filtered[1], compass_filtered, speed])
		
		# We only get half a LiDAR at every time step. Align the last half into the current frame.
		if self.bev_encoder_lidar_last is not None and len(self.bev_encoder_state_log) >= 2:
			ego_x = self.bev_encoder_state_log[-1][0]
			ego_y = self.bev_encoder_state_log[-1][1]
			ego_theta = self.bev_encoder_state_log[-1][2]
			
			ego_x_last = self.bev_encoder_state_log[-2][0]
			ego_y_last = self.bev_encoder_state_log[-2][1]
			ego_theta_last = self.bev_encoder_state_log[-2][2]
			
			bev_encoder_lidar_last_aligned = self._align_lidar_bev_encoder(
				self.bev_encoder_lidar_last, 
				ego_x_last, ego_y_last, ego_theta_last,
				ego_x, ego_y, ego_theta
			)
			bev_encoder_lidar_full = np.concatenate((bev_encoder_lidar, bev_encoder_lidar_last_aligned), axis=0)
		else:
			bev_encoder_lidar_full = bev_encoder_lidar
		
		self.bev_encoder_lidar_last = bev_encoder_lidar.copy()
		self.bev_encoder_lidar_buffer.append(bev_encoder_lidar_full)
		
		# Convert to histogram BEV (same as sensor_agent.py)
		bev_encoder_lidar_bev = lidar_to_bev_histogram(bev_encoder_lidar_full, self.bev_encoder_config)
		bev_encoder_lidar_bev_tensor = torch.from_numpy(bev_encoder_lidar_bev).float().unsqueeze(0).to('cuda')
		
		# Process other sensors
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		
		result = {
				'rgb_front': rgb_front,
				'lidar_bev': lidar_bev_tensor,
				'gps': gps_filtered,  # Use UKF filtered CARLA coordinates
				'speed': speed,
				'compass': compass_filtered,  # Use UKF filtered compass
				'bev': bev,
				# BEV encoder processed data for DP
				'bev_encoder_rgb': bev_encoder_rgb_tensor,  # (1, 3, H, W) on GPU
				'bev_encoder_lidar_bev': bev_encoder_lidar_bev_tensor,  # (1, C, H, W) on GPU
				}
		
		waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))
		

		
		if len(waypoint_route) > 2:
			target_point, far_command = waypoint_route[1]
			next_target_point, next_far_command = waypoint_route[2]
		elif len(waypoint_route) > 1:
			target_point, far_command = waypoint_route[1]
			# Only target_point available, generate virtual next_target_point
			# Extend 50m along the direction from ego to target_point (in world frame)
			ego_pos = result['gps'][:2]
			direction = target_point[:2] - ego_pos
			dist = np.linalg.norm(direction)
			if dist > 1e-3:
				direction_normalized = direction / dist
			else:
				# If target_point is too close, use forward direction based on compass
				direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
			next_target_point = target_point[:2] + direction_normalized * 5.0
			next_far_command = far_command
		elif len(waypoint_route) > 0:
			target_point, far_command = waypoint_route[0]
			# Only one waypoint available, generate virtual next_target_point
			# Extend 50m along the direction from ego to target_point (in world frame)
			ego_pos = result['gps'][:2]
			direction = target_point[:2] - ego_pos
			dist = np.linalg.norm(direction)
			if dist > 1e-3:
				direction_normalized = direction / dist
			else:
				# If target_point is too close, use forward direction based on compass
				direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
			next_target_point = target_point[:2] + direction_normalized * 5.0
			next_far_command = far_command
		else:
			target_point, far_command = (result['gps'][:2], RoadOption.LANEFOLLOW)
			# Generate virtual next_target_point 50m ahead in ego's forward direction
			direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
			next_target_point = result['gps'][:2] + direction_normalized * 50.0
			next_far_command = RoadOption.LANEFOLLOW

		if self.last_command_tmp != far_command:
			self.last_command = self.last_command_tmp
		self.last_command_tmp = far_command
		
		if hasattr(target_point, '__iter__') and len(target_point) >= 2:
			if (target_point[:2] != self.target_point_prev[:2]).any() if isinstance(target_point, np.ndarray) else (list(target_point[:2]) != list(self.target_point_prev[:2])):
				self.target_point_prev = target_point
				self.commands.append(far_command.value)
		
		result['next_command'] = self.commands[-2]
		ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
		ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])
		
		result['target_point'] = ego_target_point
		result['next_target_point'] = ego_next_target_point
		result['theta'] = compass_filtered

		return result

	def _align_lidar_bev_encoder(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
		"""
		Align lidar from past frame to current frame (same as sensor_agent.py).
		
		Args:
			lidar: numpy LiDAR point cloud (N, 3)
			x, y, orientation: past frame ego pose
			x_target, y_target, orientation_target: current frame ego pose
			
		Returns:
			aligned_lidar: numpy LiDAR point cloud in current frame coordinates
		"""
		pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
		rot_diff = bev_encoder_t_u.normalize_angle(orientation_target - orientation)
		
		# Rotate difference vector from global to local coordinate system.
		rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
		                            [np.sin(orientation_target), np.cos(orientation_target), 0.0], 
		                            [0.0, 0.0, 1.0]])
		pos_diff = rotation_matrix.T @ pos_diff
		
		return bev_encoder_t_u.algin_lidar(lidar, pos_diff, rot_diff)
	
	def _truncate_route_by_target_point(self, route_waypoints_np, target_point_np):
		"""
		Truncate route_pred based on the target point projection.
		
		Logic:
		- Project target_point onto the polyline formed by route_pred.
		- If projection falls inside route_pred, drop the portion after it.
		- If projection falls beyond route_pred's end, keep the whole route.
		
		Protection mechanism:
		- If truncated route has too few points (< MIN_POINTS_THRESHOLD) or
		  is too short (< MIN_LENGTH_THRESHOLD), skip truncation and use original route
		- This handles edge cases near the destination where target_point is very close
		
		Args:
			route_waypoints_np: (N, 2) numpy array in ego frame [x_forward, y_left]
			target_point_np: (2,) numpy array in ego frame [x_forward, y_left]
		
		Returns:
			truncated_route: (M, 2) numpy array, M <= N, the valid portion of route_pred
			truncation_idx: int, the index up to which the route is valid (-1 if no truncation)
		"""
		MIN_POINTS_THRESHOLD = 5  # Minimum number of points needed for reliable control
		MIN_LENGTH_THRESHOLD = 3.0  # Minimum route length in meters for reliable lookahead
		
		if len(route_waypoints_np) < 2:
			return route_waypoints_np, -1
		
		# Find the closest segment to target_point
		min_dist = float('inf')
		best_segment_idx = -1
		best_t = 0.0  # Parameter along segment [0, 1]
		best_proj_point = None
		
		for i in range(len(route_waypoints_np) - 1):
			p1 = route_waypoints_np[i]
			p2 = route_waypoints_np[i + 1]
			
			# Vector from p1 to p2
			v = p2 - p1
			# Vector from p1 to target_point
			w = target_point_np - p1
			
			# Length squared of segment
			l2 = np.dot(v, v)
			if l2 < 1e-10:  # Degenerate segment
				t = 0.0
				proj = p1
			else:
				# Project target_point onto the line containing the segment.
				t = np.dot(w, v) / l2
				proj = p1 + t * v
			
			# Distance from target_point to projection.
			dist = np.linalg.norm(target_point_np - proj)
			
			if dist < min_dist:
				min_dist = dist
				best_segment_idx = i
				best_t = t
				best_proj_point = proj
		
		if best_segment_idx == -1:
			return route_waypoints_np, -1
		
		is_on_last_segment = (best_segment_idx == len(route_waypoints_np) - 2)
		
		if best_t > 1.0 and is_on_last_segment:
			return route_waypoints_np, -1
		
		if best_t <= 0.0:
			truncation_idx = best_segment_idx
		elif best_t >= 1.0:
			truncation_idx = best_segment_idx + 1
		else:
			truncation_idx = best_segment_idx
		
		if truncation_idx >= len(route_waypoints_np) - 1:
			return route_waypoints_np, -1
		
		truncated = route_waypoints_np[:truncation_idx + 1].copy()
		
		# Add the projection point if it is distinct from the last included point.
		if best_proj_point is not None and len(truncated) > 0:
			dist_to_last = np.linalg.norm(best_proj_point - truncated[-1])
			if dist_to_last > 0.1:
				truncated = np.vstack([truncated, best_proj_point])
		
		truncated_length = 0.0
		for i in range(len(truncated) - 1):
			truncated_length += np.linalg.norm(truncated[i + 1] - truncated[i])
		
		if len(truncated) < MIN_POINTS_THRESHOLD or truncated_length < MIN_LENGTH_THRESHOLD:
			print(f"[Lateral] Skip truncation: points={len(truncated)}, length={truncated_length:.2f}m "
				  f"(thresholds: {MIN_POINTS_THRESHOLD} points, {MIN_LENGTH_THRESHOLD}m)")
			return route_waypoints_np, -1
		
		return truncated, truncation_idx
	
	def control_pid(self, route_waypoints, velocity, speed_waypoints, target_point=None):
		"""
		Predicts vehicle control with a PID controller.
		
		Args:
			route_waypoints: (1, N, 2) tensor in ego frame [x_forward, y_left]
			velocity: float, current speed in m/s
			speed_waypoints: (1, N, 2) tensor for speed calculation
			target_point: (1, 2) tensor in ego frame [x_forward, y_left], unused (kept for API compat)
		"""
		assert route_waypoints.size(0) == 1
		route_waypoints_np = route_waypoints[0].data.cpu().numpy()
		speed = velocity
		speed_waypoints_np = speed_waypoints[0].data.cpu().numpy()
		
		# MoT trajectory: 6 points, 0.5s interval each, total 3s
		# Point indices: 0(0.5s), 1(1.0s), 2(1.5s), 3(2.0s), 4(2.5s), 5(3.0s)
		one_second_idx = 1
		half_second_idx = 0
		
		if speed_waypoints_np.shape[0] >= 2:
			desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx]) * 2.0
		else:
			desired_speed = np.linalg.norm(speed_waypoints_np[0]) * 2.0

		brake = ((desired_speed < self.brake_speed) or ((speed / max(desired_speed, 1e-5)) > self.brake_ratio))
		
		delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.clip_throttle)
		throttle = throttle if not brake else 0.0
		

		route_interp = self.interpolate_waypoints(route_waypoints_np)
		
		
		steer = self.turn_controller.step(route_interp, speed)
		steer = np.clip(steer, -1.0, 1.0)
		steer = round(steer, 3)
		
		
		return steer, throttle, brake
	
	def interpolate_waypoints(self, waypoints):
		"""
		Interpolate waypoints to be 0.1m apart
		
		Args:
			waypoints: (N, 2) numpy array in ego frame [x_forward, y_left]
			
		Returns:
			interp_points: (M, 2) numpy array with points 0.1m apart
		"""
		waypoints = waypoints.copy()
		# Add origin point at the beginning
		waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
		shift = np.roll(waypoints, 1, axis=0)
		shift[0] = shift[1]
		
		dists = np.linalg.norm(waypoints - shift, axis=1)
		dists = np.cumsum(dists)
		dists += np.arange(0, len(dists)) * 1e-4  # Prevents dists not being strictly increasing
		
		interp = PchipInterpolator(dists, waypoints, axis=0)
		
		x = np.arange(0.1, dists[-1], 0.1)
		
		interp_points = interp(x)
		
		if interp_points.shape[0] == 0:
			interp_points = waypoints[None, -1]
		
		return interp_points

	# ====== Parking escape methods ======

	def _update_pos_snapshots(self, tick_data):
		"""
		Record position snapshots periodically and detect long-term deadlock.
		
		Every pos_snapshot_interval frames, record GPS position.
		If over the last parking_deadlock_window frames the max displacement
		from the oldest snapshot < threshold -> deadlock.
		
		Safe w.r.t. red lights:
		- A red light lasts 30-60 s max, then car moves -> displacement grows.
		- Our window is 60 s, so even a full red light cycle is borderline;
		  two consecutive reds with driving in between will exceed the threshold.
		"""
		if self.parking_escape_cooldown > 0:
			self.parking_escape_cooldown -= 1
			# Still record snapshots during cooldown so re-detection is faster
			# (but don't trigger escape)
		
		if self.parking_escape_active:
			return False
		
		ego_pos = tick_data['gps'][:2]
		
		# Record snapshot every N frames
		if self.step % self.pos_snapshot_interval == 0:
			self.pos_snapshots.append((self.step, ego_pos.copy()))
			# Only keep snapshots within the window
			cutoff = self.step - self.parking_deadlock_window
			self.pos_snapshots = [(s, p) for s, p in self.pos_snapshots if s >= cutoff]
		
		# Need enough history to judge
		if len(self.pos_snapshots) < 2:
			return False
		
		oldest_step, oldest_pos = self.pos_snapshots[0]
		time_span = self.step - oldest_step
		
		# Haven't accumulated enough time yet
		if time_span < self.parking_deadlock_window:
			return False
		
		# Still in cooldown - do not trigger, just track.
		if self.parking_escape_cooldown > 0:
			return False
		
		# Check max displacement across ALL snapshots
		max_displacement = 0.0
		for _, pos in self.pos_snapshots:
			d = np.linalg.norm(pos - oldest_pos)
			max_displacement = max(max_displacement, d)
		
		if max_displacement < self.parking_deadlock_max_disp:
			print(f"[ParkingDetect] === DEADLOCK === "
			      f"{time_span} frames ({time_span/20:.0f}s), "
			      f"max displacement = {max_displacement:.2f}m < {self.parking_deadlock_max_disp}m")
			return True
		
		return False

	def _activate_parking_escape(self, tick_data):
		"""
		Activate parking escape.
		
		Always escape left; in CARLA right-hand traffic, parking is on the right.
		Phase 1 directly overrides steer (model output is ignored for steering).
		"""
		self.parking_escape_active = True
		self.parking_escape_phase = 1
		self.parking_escape_timer = 40   # Phase 1: ~2.5 seconds at 20 fps (enough to turn out)
		self.parking_escape_anchor = tick_data['gps'][:2].copy()
		self.parking_escape_start_compass = tick_data['compass']  # Record heading for angle check
		self.parking_escape_attempt += 1
		
		# Always escape LEFT
		self.parking_escape_direction = 1.0   # +1 = left
		
		# Advance route planner: pop up to 5 waypoints so next TP is farther
		n_pop = 0
		try:
			n_pop = min(5, max(0, len(self._route_planner.route) - 3))
			for _ in range(n_pop):
				if len(self._route_planner.route) > 3:
					self._route_planner.route.popleft()
					self._route_planner.route_distances.popleft()
			print(f"[ParkingEscape] Popped {n_pop} route WPs")
		except (AttributeError, IndexError) as e:
			print(f"[ParkingEscape] Skip route pop: {e}")
		
		# Clear stuck / force_move to prevent interference
		self.stuck_detector = 0
		self.force_move = 0
		
		dir_str = "LEFT" if self.parking_escape_direction > 0 else "RIGHT"
		print(f"[ParkingEscape] === ACTIVATED (attempt #{self.parking_escape_attempt}, "
		      f"dir={dir_str}, popped {n_pop} route WPs) ===")

	def _get_escape_target_points(self, target_point, next_target_point, gt_velocity):
		"""
		Generate override target points during escape (ego frame).
		
		Phase 1 (lateral): TP offset to the escape side to steer out.
		Phase 2 (merge):   TP mostly forward with mild lateral offset to merge
		                    back into the lane.
		
		Returns (override_tp, override_ntp) as (1,2) tensors, or (None,None).
		"""
		if not self.parking_escape_active:
			return None, None
		
		self.parking_escape_timer -= 1
		d = self.parking_escape_direction  # +1 left, -1 right
		
		if self.parking_escape_phase == 1:
			lat = d * (5.0 + self.parking_escape_attempt * 1.5)
			fwd = 3.0
			
			override_tp = torch.tensor([[fwd, lat]], dtype=torch.float32).to('cuda')
			override_ntp = torch.tensor([[fwd + 3.0, lat]], dtype=torch.float32).to('cuda')
			
			if self.parking_escape_timer <= 0:
				# Phase 1 is complete; return control to the model.
				self._end_parking_escape("phase 1 timeout (3s)")
				return None, None
			elif self.parking_escape_timer % 20 == 0:
				print(f"[ParkingEscape] Phase 1: timer={self.parking_escape_timer}, "
				      f"TP=({fwd:.0f}, {lat:.1f})")
			
			return override_tp, override_ntp
		
		return None, None

	def _check_escape_progress(self, ego_pos, compass=None):
		"""
		During escape, end early if:
		  - Vehicle has moved > 6m from anchor (displacement check), OR
		  - Vehicle heading has changed by more than 25 degrees from the start
		"""
		if not self.parking_escape_active or self.parking_escape_anchor is None:
			return
		
		displacement = np.linalg.norm(ego_pos - self.parking_escape_anchor)
		
		# Check heading change (only in Phase 1 where we force steer)
		if compass is not None and self.parking_escape_start_compass is not None and self.parking_escape_phase == 1:
			heading_diff = abs(compass - self.parking_escape_start_compass)
			# Normalize to [-pi, pi]
			if heading_diff > np.pi:
				heading_diff = 2 * np.pi - heading_diff
			heading_deg = np.degrees(heading_diff)
			if heading_deg > 25.0:
				print(f"[ParkingEscape] Turned {heading_deg:.1f} deg (>25 deg), ending Phase 1 early")
				self._end_parking_escape(f"heading change {heading_deg:.1f} deg, disp={displacement:.1f}m")
				return
		
		if displacement > 6.0:
			self._end_parking_escape(f"success! moved {displacement:.1f}m")

	def _end_parking_escape(self, reason=""):
		"""End parking escape, enter cooldown."""
		print(f"[ParkingEscape] === ENDED: {reason} ===")
		self.parking_escape_active = False
		self.parking_escape_phase = 0
		self.parking_escape_timer = 0
		# Short cooldown (20s); keep snapshots so re-detection is fast
		self.parking_escape_cooldown = 2400  # 2400 frames = 120 seconds (2 min) @ 20fps

	# ====== End parking escape methods ======

	def _build_obs_dict(self, tick_data):
		"""
		Build multi-frame observation from historical buffers for MoT.
		
		MoT needs:
		  - 4 RGB frames sampled every 5 steps from rgb_history (t0, t-5, t-10, t-15)
		  - 1 lidar BEV frame (the latest)
		
		Returns:
			rgb_stacked: (1, N, C, H, W) stacked RGB images (N<=4)
			lidar_last: (C, H, W) the latest lidar BEV tensor
		"""
		rgb_history_list = list(self.rgb_history)
		lidar_history_list = list(self.lidar_bev_history)

		# Sample every 5 for RGB from the end (t0, t-5, t-10, t-15)
		rgb_list = [rgb_history_list[-1 - i*5] for i in range(4) if -1 - i*5 >= -len(rgb_history_list)]
		rgb_list = rgb_list[::-1]  # Reverse to chronological order (oldest to newest)

		rgb_stacked = torch.stack(rgb_list, dim=0).unsqueeze(0)  # (1, N, C, H, W)

		# Use the latest lidar frame
		lidar_last = lidar_history_list[-1]  # (C, H, W)

		return rgb_stacked, lidar_last

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)

		# Prepare current observations
		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		# Encode the delayed high-level command used by the controller.
		one_hot_command = t_u.command_to_one_hot(self.commands[-2])
		cmd_one_hot = torch.from_numpy(one_hot_command[np.newaxis]).to('cuda', dtype=torch.float32)
		# Keep command variable for metadata (convert from 1-6 to 0-5 range)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		theta = torch.FloatTensor([float(tick_data['theta'])]).view(1,1).to('cuda', dtype=torch.float32)
		lidar = tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		
		rgb_front = torch.from_numpy(tick_data['rgb_front']).permute(2, 0, 1).float() / 255.0
		rgb_front = rgb_front.to('cuda', dtype=torch.float32)
		waypoint = torch.from_numpy(tick_data['gps']).float().to('cuda', dtype=torch.float32)
		target_point = torch.from_numpy(tick_data['target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		next_target_point = torch.from_numpy(tick_data['next_target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		
		# Report target-point state at a low frequency.
		if self.step % 20 == 0:
			tp = tick_data['target_point']
			ntp = tick_data['next_target_point']
			print(f"[Target Points] TP=({tp[0]:.1f},{tp[1]:.1f}), NTP=({ntp[0]:.1f},{ntp[1]:.1f})")

		# Accumulate observation history into buffers
		self.lidar_bev_history.append(lidar)
		self.rgb_history.append(rgb_front)
		self.speed_history.append(speed)
		self.target_point_history.append(target_point)
		self.next_target_point_history.append(next_target_point)
		self.next_command_history.append(cmd_one_hot)
		self.theta_history.append(theta)
		self.waypoint_history.append(waypoint)

		# Append throttle and brake from previous step (or 0 for first step)
		if self.step < 1:
			self.throttle_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
		else:
			prev_control = self.prev_control if self.prev_control is not None else carla.VehicleControl()
			self.throttle_history.append(torch.tensor(prev_control.throttle).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(prev_control.brake).view(1, 1).to('cuda'))

		# Buffer size = 31 frames (for obs_horizon=4 with 10x sampling for lidar, 5x for RGB)
		BUFFER_PHASE = 31

		if self.step < BUFFER_PHASE:
			# Brake during warmup so the UKF can stabilize.
			control = carla.VehicleControl(0.0, 0.0, 1.0)
			self.control = control  # Important: UKF uses self.control for prediction
			self.pid_metadata = {}
			self.pid_metadata['agent'] = 'warmup_phase'
			self.pid_metadata['step'] = self.step
		else:
			# ====== Parking-start detection: one-time check ======
			if self.parking_start_anchor is None:
				self.parking_start_anchor = tick_data['gps'][:2].copy()
			if not self.parking_start_checked and self.step >= BUFFER_PHASE + self.parking_start_check_frame:
				disp = np.linalg.norm(tick_data['gps'][:2] - self.parking_start_anchor)
				self.parking_start_detected = (disp < self.parking_start_disp_thresh)
				self.parking_start_checked = True
				if self.parking_start_detected:
					print(f"[ParkingStart] Detected! Displacement in first {self.parking_start_check_frame} frames = {disp:.2f}m < {self.parking_start_disp_thresh}m. force_move DISABLED for this episode.")
				else:
					print(f"[ParkingStart] Normal start. Displacement = {disp:.2f}m. force_move enabled.")

			# ====== Parking escape: snapshot & detect ======
			deadlock_detected = self._update_pos_snapshots(tick_data)
			if deadlock_detected:
				self._activate_parking_escape(tick_data)
			if self.parking_escape_active:
				self._check_escape_progress(tick_data['gps'][:2], compass=tick_data['compass'])

			# Build multi-frame observations from history buffers
			rgb_stacked, lidar_last = self._build_obs_dict(tick_data)

			# Convert rgb_stacked (1, N, C, H, W) to PIL list for MoT inferencer
			rgb_pil_list = []
			for i in range(rgb_stacked.shape[1]):
				rgb_tensor = rgb_stacked[0, i]  # (C, H, W)
				rgb_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				rgb_pil = Image.fromarray(rgb_np, mode='RGB')
				rgb_pil_list.append(rgb_pil)

			# Convert lidar_last to PIL for MoT inferencer
			lidar_tensor = lidar_last.squeeze(0) if lidar_last.dim() == 4 else lidar_last  # (C, H, W)
			lidar_np = (lidar_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
			lidar_pil = Image.fromarray(lidar_np, mode='RGB')
			
			
			lidar_pil_list = [lidar_pil]
			
			if self.stuck_helper > 0:
				target_point_speed=torch.cat([speed, target_point, next_target_point], dim=-1)  # (1, 5)
				print("Get stucked! Trigger the stuck helper!")
			else:
				target_point_speed=torch.cat([speed, target_point, next_target_point], dim=-1)  # (1, 5)

			# ====== Parking escape: override target points for model input ======
			escape_tp, escape_ntp = self._get_escape_target_points(
				target_point, next_target_point, gt_velocity
			)
			if escape_tp is not None:
				# Only override target_point_speed for model input;
				# Preserve target points for BEV rendering and PID control.
				target_point_speed = torch.cat([speed, escape_tp, escape_ntp], dim=-1)  # (1, 5)

			prompt_cleaned, understanding_output, reasoning_output = build_cleaned_prompt_and_modes(target_point_speed)
			t0 = time.time()
			
			# ========== Run BEV encoder backbone to get BEV features ==========
			with torch.no_grad():
				# Convert inputs to bfloat16 to match bev_encoder backbone
				bev_encoder_rgb_bf16 = tick_data['bev_encoder_rgb'].to(torch.bfloat16)
				bev_encoder_lidar_bev_bf16 = tick_data['bev_encoder_lidar_bev'].to(torch.bfloat16)
				bev_encoder_output = self.bev_encoder(
					rgb=bev_encoder_rgb_bf16,  # (1, 3, H, W) on GPU, bfloat16
					lidar_bev=bev_encoder_lidar_bev_bf16  # (1, C, H, W) on GPU, bfloat16
				)
			
			# Extract BEV encoder features consumed by AutoMoT.
			bev_encoder_bev_feature = bev_encoder_output['bev_feature']  # (1, 1512, 8, 8)
			output = self.inferencer(
				image=rgb_pil_list,
				front=[rgb_pil_list[-1]],
				lidar=lidar_pil_list,
				text=prompt_cleaned,
				understanding_output=understanding_output,
				reasoning_output=reasoning_output,
				max_think_token_n=self.inference_args.max_num_tokens,
				v_target_point=target_point_speed,
				bev_encoder_feature=bev_encoder_bev_feature,
				do_sample=False,
				text_temperature=0.0,
				frame_idx=self.step,
			)
			pred_traj = output['traj']
			pred_route = output['route']
			pred_decision = output.get('text', '')
			
			# Store for rendering
			self.last_pred_traj = pred_traj.squeeze(0).float().cpu().numpy()  # (6, 2) in [x, y] format
			self.last_target_point = target_point.squeeze(0).float().cpu().numpy()  # (2,) in [x, y] format
			self.last_next_target_point = next_target_point.squeeze(0).float().cpu().numpy()  # (2,) in [x, y] format
			
			t1 = time.time()
			print(f"[Inference] MoT inference time: {t1 - t0:.2f} seconds")

			# ================== control_pid method ==================
			# - speed_waypoints: use pred_traj from MoT model for speed control (throttle/brake)
			# - route_waypoints: use route_pred from MoT model for lateral angle control (steering)
			speed_waypoints = pred_traj.float()  # (1, 6, 2) for speed control
			route_pred = pred_route
			if isinstance(route_pred, torch.Tensor):
				route_waypoints = route_pred.float().cpu()  # (1, 20, 2) for steering control
			else:
				route_waypoints = torch.from_numpy(route_pred).float()
			self.last_route_pred = route_waypoints.squeeze(0).numpy().copy()  # (20, 2) for visualization

			gt_velocity = tick_data['speed']
			
			# Use target_point to truncate route for lateral control
			steer, throttle, brake = self.control_pid(route_waypoints, gt_velocity, speed_waypoints, target_point=target_point)
			
			# Restart mechanism in case the car got stuck
			if gt_velocity < 0.1:
				self.stuck_detector += 1
			elif gt_velocity > 0.2:
				self.stuck_detector = 0
			
			# ====== Parking escape: suppress normal stuck/force_move, OVERRIDE steer+throttle ======
			if self.parking_escape_active:
				# During escape, prevent force_move from interfering
				self.stuck_detector = 0
				self.force_move = 0
				d = self.parking_escape_direction  # +1=left, -1=right
				# Phase 1: FORCE hard turn + moderate throttle to get out of parking spot
				if self.parking_escape_phase == 1:
					# In CARLA: positive steer = right, negative steer = left
					steer = -d * 0.65  # Strong left turn
					throttle = 0.45
					brake = False
					if self.parking_escape_timer % 20 == 0:
						print(f"[ParkingEscape] Phase1 OVERRIDE: steer={steer:.2f}, throttle={throttle:.2f}")
			else:
				# If stuck for too long, trigger force_move
				# But NOT if this episode started from a parking spot
				if self.stuck_detector > self.stuck_threshold and not self.parking_start_detected:
					self.force_move = self.creep_duration
				
				# Force move: override throttle and brake to get unstuck
				if self.force_move > 0:
					throttle = max(self.creep_throttle, throttle)
					brake = False
					self.force_move -= 1
					print(f"force_move: {self.force_move}")

			print(f"stuck_detector: {self.stuck_detector}")

			
			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)

			# Speed limit enforcement: if current speed > 35 km/h, force brake
			# gt_velocity is in m/s, convert to km/h by multiplying 3.6
			if gt_velocity * 3.6 > 35:
				control.throttle = 0.0
				control.brake = 1.0

			# Store metadata
			self.pid_metadata = {
				'agent': 'mot',
				'steer': control.steer,
				'throttle': control.throttle,
				'brake': control.brake,
				'speed': gt_velocity,
				'command': command,
			}

			self.prev_control = control
			self.control = control  # Update control for UKF prediction in next tick
			metric_info = self.get_metric_info()
			self.metric_info[self.step] = metric_info

			if SAVE_PATH is not None:
				self.save(tick_data)

			##### Rendering ####
			ego_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=10, max_distance=30,
			)

			# Prepare target point for rendering
			tp_for_render = target_point.cpu().float().numpy().copy()
			if tp_for_render.ndim == 2:
				tp_for_render = tp_for_render.squeeze(0)
			tp_for_render[1] = -tp_for_render[1]  # Negate y: left -> right

			# Prepare MoT pred_traj for rendering (green) - speed trajectory
			mot_traj_for_render = pred_traj.squeeze(0).cpu().float().numpy().copy()  # (6, 2)
			mot_traj_for_render[:, 1] = -mot_traj_for_render[:, 1]  # Negate y: left -> right
			mot_traj_trajectory = np.concatenate((mot_traj_for_render, tp_for_render.reshape(1, 2)), axis=0)
			mot_traj_trajectory = mot_traj_trajectory[:, [1, 0]]
			mot_traj_trajectory[:, 0] = -mot_traj_trajectory[:, 0]  # y (now in col 0) 
			mot_traj_trajectory[:, 1] = -mot_traj_trajectory[:, 1]  # x (now in col 1)
			render_mot_traj = render_waypoints(mot_traj_trajectory, pixels_per_meter=30, max_distance=20, color=(0, 255, 0))
			
			# Prepare MoT pred_route for rendering (red) - route trajectory
			mot_route_for_render = pred_route.squeeze(0).cpu().float().numpy().copy()  # (6, 2)
			mot_route_for_render[:, 1] = -mot_route_for_render[:, 1]  # Negate y: left -> right
			mot_route_trajectory = np.concatenate((mot_route_for_render, tp_for_render.reshape(1, 2)), axis=0)
			mot_route_trajectory = mot_route_trajectory[:, [1, 0]]
			mot_route_trajectory[:, 0] = -mot_route_trajectory[:, 0]  # y (now in col 0) 
			mot_route_trajectory[:, 1] = -mot_route_trajectory[:, 1]  # x (now in col 1)
			render_mot_route = render_waypoints(mot_route_trajectory, pixels_per_meter=30, max_distance=20, color=(255, 0, 0))

			ego_car_map = cv2.resize(ego_car_map, (200, 200))
			render_mot_traj = cv2.resize(render_mot_traj, (200, 200))
			render_mot_route = cv2.resize(render_mot_route, (200, 200))

			surround_map = np.clip(
				(
					ego_car_map.astype(np.float32)
					+ render_mot_traj.astype(np.float32)
					+ render_mot_route.astype(np.float32)
				),
				0,
				255,
			).astype(np.uint8)
			tick_data["predicted_trajectory"] = surround_map
			decision_1s, decision_2s, decision_3s = parse_decision_sequence(pred_decision)
			tick_data["decision_1s"] = decision_1s
			tick_data["decision_2s"] = decision_2s
			tick_data["decision_3s"] = decision_3s

			tick_data["rgb_raw"] = tick_data["rgb_front"]

			tick_data["rgb"] = cv2.resize(tick_data["rgb_front"], (800, 600))

			# Generate bev_traj here (before save()) by drawing trajectory on bev image
			bev_img_for_display = tick_data['bev'].copy()
			if self.last_pred_traj is not None:
				bev_img_for_display = self._draw_trajectory_on_bev(
					bev_img_for_display, self.last_pred_traj, self.last_target_point,
					self.last_next_target_point, self.last_route_pred)
			tick_data["bev_traj"] = cv2.resize(bev_img_for_display, (400, 400))

			tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data["speed"] = "speed: %.2f Km/h, target point x: %.2f m, target point y: %.2f m" % (gt_velocity*3.6, target_point.squeeze(0).cpu().float().numpy()[0], target_point.squeeze(0).cpu().float().numpy()[1])
			
			sentence1, sentence2 = split_prompt(prompt_cleaned)
			tick_data["language_1"] = "Instruction: " + sentence1
			tick_data["language_2"] = sentence2

			tick_data["mes"] = "speed: %.2f" % gt_velocity
			tick_data["time"] = "time: %.3f" % timestamp

			surface = self._hic.run_interface(tick_data)
			tick_data["surface"] = surface

		return control

	def save(self, tick_data):
		frame = self.step 
		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
		
		# Draw trajectory on BEV image if available
		bev_img = tick_data['bev'].copy()
		if self.last_pred_traj is not None:
			# Pass last_route_pred for visualization (20 waypoints for lateral control, blue points)
			# Pass both target_point and next_target_point for visualization
			bev_img = self._draw_trajectory_on_bev(bev_img, self.last_pred_traj, self.last_target_point,
			                                        self.last_next_target_point, self.last_route_pred)
		tick_data['bev_traj'] = bev_img
		Image.fromarray(bev_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
		
		if 'lidar_bev' in tick_data:
			lidar_bev_tensor = tick_data['lidar_bev']
			if isinstance(lidar_bev_tensor, torch.Tensor):
				lidar_bev_tensor = lidar_bev_tensor.cpu().numpy()
			lidar_bev_img = (lidar_bev_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
			imageio.imwrite(str(self.save_path / 'lidar_bev' / (f'{frame:04d}.png')), lidar_bev_img)

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# metric info
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def _draw_trajectory_on_bev(self, bev_img, traj, target_point=None, next_target_point=None, route_pred=None):
		"""
		Draw predicted trajectory on BEV image.
		
		BEV camera parameters:
		- Position: x=0, y=0, z=50 (50m height, looking down)
		- FOV: 50 degrees
		- Image size: 512x512
		
		Trajectory is in ego frame: [x, y] where x is forward, y is left (model convention)
		For BEV visualization, we negate y to convert to right-positive convention.
		BEV image: center is ego position, up is forward (negative x in image coords)
		
		Args:
			bev_img: numpy array (512, 512, 3) RGB image
			traj: numpy array (6, 2) trajectory points in ego frame [x_forward, y_left]
			target_point: numpy array (2,) target point in ego frame [x_forward, y_left], optional
			next_target_point: numpy array (2,) next target point in ego frame [x_forward, y_left], optional
			route_pred: numpy array (20, 2) route waypoints for lateral control, optional
		
		Returns:
			bev_img: numpy array with trajectory drawn
		"""
		img_h, img_w = bev_img.shape[:2]  # 512, 512
		
		# BEV camera: z=50m, FOV=50 degrees
		# Calculate meters per pixel
		# FOV = 50 deg means the camera sees 50 degrees width/height
		# At z=50m, the ground coverage is: 2 * z * tan(FOV/2)
		fov_rad = np.deg2rad(50.0)
		ground_size = 2 * 50.0 * np.tan(fov_rad / 2)  # meters covered by the image
		meters_per_pixel = ground_size / img_w  # ~0.093 m/pixel
		
		# Image center is ego position
		cx, cy = img_w // 2, img_h // 2
		
		# Convert trajectory points to pixel coordinates
		# Model ego frame: x is forward, y is LEFT (positive y = left)
		# BEV image: center is ego, up (-row) is forward, right (+col) is right
		# Need to negate y to convert from left-positive to right-positive
		# So: pixel_col = cx + y / meters_per_pixel (negate y: left -> right, then right is +col)
		#     pixel_row = cy - x / meters_per_pixel (x forward -> -row, i.e., up)
		
		pixels = []
		for i in range(len(traj)):
			x, y = traj[i]  # x: forward, y: left (model convention)
			# Negate y for visualization: left-positive -> right-positive
			pixel_col = int(cx + y / meters_per_pixel)  # y_left negated: +y_left -> -col, so use + to flip
			pixel_row = int(cy - x / meters_per_pixel)
			pixels.append((pixel_col, pixel_row))
		
		# Draw trajectory using cv2
		# Draw lines connecting waypoints
		for i in range(len(pixels) - 1):
			pt1 = pixels[i]
			pt2 = pixels[i + 1]
			# Check if points are within image bounds
			if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
				0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
				cv2.line(bev_img, pt1, pt2, (0, 255, 0), 2)  # Green line
		
		# Draw waypoints as circles
		for i, (col, row) in enumerate(pixels):
			if 0 <= col < img_w and 0 <= row < img_h:
				# Color gradient: start (red) -> end (blue)
				color_r = int(255 * (1 - i / (len(pixels) - 1)))
				color_b = int(255 * (i / (len(pixels) - 1)))
				cv2.circle(bev_img, (col, row), 5, (color_r, 0, color_b), -1)
		
		# Draw route_pred waypoints if provided (blue color) - used for lateral/steering control
		if route_pred is not None:
			route_pixels = []
			for i in range(len(route_pred)):
				x, y = route_pred[i]  # x: forward, y: left (model convention)
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				route_pixels.append((pixel_col, pixel_row))
			
			# Draw route_pred trajectory lines (blue)
			for i in range(len(route_pixels) - 1):
				pt1 = route_pixels[i]
				pt2 = route_pixels[i + 1]
				if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
					0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
					cv2.line(bev_img, pt1, pt2, (255, 165, 0), 1)  # Orange line (thinner)
			
			# Draw route_pred waypoints as circles (blue)
			for i, (col, row) in enumerate(route_pixels):
				if 0 <= col < img_w and 0 <= row < img_h:
					# Solid blue points for route_pred
					cv2.circle(bev_img, (col, row), 3, (0, 0, 255), -1)  # Blue circles (smaller)
		
		# Draw target point if provided (cyan/aqua color with larger circle)
		if target_point is not None:
			x, y = target_point[0], target_point[1]  # x: forward, y: left (model convention)
			# Negate y for visualization
			tp_col = int(cx + y / meters_per_pixel)  # Negate y: +y_left -> -col, use + to flip
			tp_row = int(cy - x / meters_per_pixel)
			if 0 <= tp_col < img_w and 0 <= tp_row < img_h:
				cv2.circle(bev_img, (tp_col, tp_row), 10, (0, 255, 255), -1)  # Cyan circle for target point
				cv2.circle(bev_img, (tp_col, tp_row), 12, (255, 255, 255), 2)  # White border
		
		# Draw next target point if provided (magenta/pink color with larger circle)
		if next_target_point is not None:
			x, y = next_target_point[0], next_target_point[1]  # x: forward, y: left (model convention)
			# Negate y for visualization
			ntp_col = int(cx + y / meters_per_pixel)
			ntp_row = int(cy - x / meters_per_pixel)
			if 0 <= ntp_col < img_w and 0 <= ntp_row < img_h:
				cv2.circle(bev_img, (ntp_col, ntp_row), 10, (255, 0, 255), -1)  # Magenta circle for next target point
				cv2.circle(bev_img, (ntp_col, ntp_row), 12, (255, 255, 255), 2)  # White border
		
		# Draw ego position (center)
		cv2.circle(bev_img, (cx, cy), 8, (255, 255, 0), -1)  # Yellow circle for ego
		
		return bev_img

	def destroy(self):
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		# gps content: numpy array: [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])
