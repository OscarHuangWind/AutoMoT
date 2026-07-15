import numpy as np


def lidar_to_histogram_features(lidar, config):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid.
    Adapted from the BEV encoder LiDAR histogram preprocessing.

    :param lidar: (N, 3) numpy, LiDAR point cloud
    :param config: BEV encoder config (GlobalConfig)
    :return: (2, H, W) numpy, LiDAR as sparse image
    """

    def splat_points(point_cloud):
        xbins = np.linspace(config.min_x, config.max_x,
                            (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1)
        ybins = np.linspace(config.min_y, config.max_y,
                            (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
        overhead_splat = hist / config.hist_max_per_pixel
        return overhead_splat.T

    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if config.use_ground_plane:
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features
