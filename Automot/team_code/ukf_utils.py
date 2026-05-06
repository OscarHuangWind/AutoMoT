"""
Unscented Kalman Filter (UKF) utilities for vehicle state estimation.
Contains functions for bicycle model dynamics and UKF-related computations.
"""

import math
import numpy as np


def bicycle_model_forward(x, dt, steer, throttle, brake):
    """
    Kinematic bicycle model for vehicle dynamics prediction.
    Numbers are the tuned parameters from World on Rails.
    
    Args:
        x: Current state [pos_x, pos_y, yaw, speed]
        dt: Time step in seconds
        steer: Steering angle
        throttle: Throttle input
        brake: Boolean brake flag
    
    Returns:
        next_state_x: Predicted next state [pos_x, pos_y, yaw, speed]
    """
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() if hasattr(locs_0, 'item') else locs_0
    next_locs_0 = next_locs_0 + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() if hasattr(locs_1, 'item') else locs_1
    next_locs_1 = next_locs_1 + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x


def measurement_function_hx(vehicle_state):
    """
    Measurement function for UKF. For now we use the same internal state 
    as the measurement state.
    
    Args:
        vehicle_state: VehicleState vehicle state variable containing
                      an internal state of the vehicle from the filter
    
    Returns:
        np.array: describes the vehicle state as numpy array.
                  0: pos_x, 1: pos_y, 2: rotation, 3: speed
    """
    return vehicle_state


def state_mean(state, wm):
    """
    Calculate mean of vehicle states, using arctan of the average of sin and cos 
    of the angle to calculate the average of orientations.
    
    Args:
        state: array of states to be averaged. First index is the timestep.
        wm: weights
    
    Returns:
        x: averaged state [pos_x, pos_y, yaw, speed]
    """
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def measurement_mean(state, wm):
    """
    Calculate mean of measurements, using arctan of the average of sin and cos 
    of the angle to calculate the average of orientations.
    
    Args:
        state: array of states to be averaged. First index is the timestep.
        wm: weights
    
    Returns:
        x: averaged measurement [pos_x, pos_y, yaw, speed]
    """
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def residual_state_x(a, b):
    """
    Calculate residual between two states, normalizing the angle component.
    
    Args:
        a: First state
        b: Second state
    
    Returns:
        y: Residual with normalized angle
    """
    # Import here to avoid circular dependency
    import sys
    import pathlib
    project_root = str(pathlib.Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import team_code.automot_utils as t_u
    
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    """
    Calculate residual between two measurements, normalizing the angle component.
    
    Args:
        a: First measurement
        b: Second measurement
    
    Returns:
        y: Residual with normalized angle
    """
    # Import here to avoid circular dependency
    import sys
    import pathlib
    project_root = str(pathlib.Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import team_code.automot_utils as t_u
    
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y
