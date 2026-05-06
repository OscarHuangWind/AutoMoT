"""
LiDAR utilities for coordinate transformation and alignment.
Contains functions for converting LiDAR points between different coordinate systems.
"""

import numpy as np


def lidar_to_ego_coordinate(lidar):
    """
    Converts the LiDAR points given by the simulator into the ego agent's
    coordinate system.
    
    Args:
        lidar: the LiDAR point cloud as provided in the input of run_step
    
    Returns:
        ego_lidar: lidar where the points are w.r.t. 0/0/0 of the car and the carla
                   coordinate system.
    """
    yaw = np.deg2rad(-90.0)
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ])

    translation = np.array([0.0, 0.0, 2.5])

    # The double transpose is a trick to compute all the points together.
    ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

    return ego_lidar


def algin_lidar(lidar, translation, yaw):
    """
    Translates and rotates a LiDAR into a new coordinate system.
    Rotation is inverse to translation and yaw.
    
    Args:
        lidar: numpy LiDAR point cloud (N, 3)
        translation: translations in meters
        yaw: yaw angle in radians
    
    Returns:
        aligned_lidar: numpy LiDAR point cloud in the new coordinate system.
    """
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ])

    aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T

    return aligned_lidar
