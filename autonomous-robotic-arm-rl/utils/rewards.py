import numpy as np


def reward_stage1(dist_tcp_obj, action, reach_threshold, action_penalty_scale, reach_bonus):
    reward = -dist_tcp_obj
    reward -= action_penalty_scale * float(np.mean(np.square(action[:4])))
    if dist_tcp_obj < reach_threshold:
        reward += reach_bonus
    return float(reward)


def reward_stage2(
    dist_tcp_obj,
    action,
    gripper_cmd,
    is_grasped,
    object_height,
    lift_height,
    reach_threshold,
    action_penalty_scale,
    grasp_bonus,
    lift_bonus,
):
    reward = -dist_tcp_obj
    reward -= action_penalty_scale * float(np.mean(np.square(action[:4])))
    if dist_tcp_obj < reach_threshold and gripper_cmd > 0.7:
        reward += grasp_bonus
    if is_grasped and object_height > lift_height:
        reward += lift_bonus
    return float(reward)


def reward_stage3(
    dist_obj_goal,
    prev_dist_obj_goal,
    action,
    is_grasped,
    object_height,
    drop_height,
    success_threshold,
    action_penalty_scale,
    goal_bonus,
    drop_penalty,
):
    progress = prev_dist_obj_goal - dist_obj_goal
    reward = progress
    reward -= action_penalty_scale * float(np.mean(np.square(action[:4])))
    if dist_obj_goal < success_threshold:
        reward += goal_bonus
    if (not is_grasped) and object_height < drop_height:
        reward -= drop_penalty
    return float(reward)
