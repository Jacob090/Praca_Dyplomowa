import numpy as np


def is_out_of_bounds(position, bounds):
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    z_min, z_max = bounds["z"]
    x, y, z = position
    return (x < x_min) or (x > x_max) or (y < y_min) or (y > y_max) or (z < z_min) or (z > z_max)


def compute_success(stage, dist_tcp_obj, dist_obj_goal, is_grasped, object_height, lift_height, success_threshold):
    if stage == "reach":
        return dist_tcp_obj < success_threshold
    if stage == "grasp":
        return is_grasped and (object_height > lift_height)
    if stage == "place":
        return dist_obj_goal < success_threshold
    return False


def has_nan(array_like):
    return bool(np.isnan(array_like).any())
