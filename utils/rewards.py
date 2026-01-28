import numpy as np


def _arm_action_penalty(action, action_penalty_scale):
    if action is None:
        return 0.0
    action = np.asarray(action, dtype=np.float32)
    arm_action = action[:-1] if action.shape[0] > 1 else action
    return action_penalty_scale * float(np.mean(np.square(arm_action)))


def _apply_action_penalty(reward, action, action_penalty_scale):
    return reward - _arm_action_penalty(action, action_penalty_scale)


def reward_stage1(dist_tcp_obj, action, reach_threshold, action_penalty_scale, reach_bonus):
    reward = -dist_tcp_obj
    reward = _apply_action_penalty(reward, action, action_penalty_scale)
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
    reward = _apply_action_penalty(reward, action, action_penalty_scale)
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
    reward = _apply_action_penalty(reward, action, action_penalty_scale)
    if dist_obj_goal < success_threshold:
        reward += goal_bonus
    if (not is_grasped) and object_height < drop_height:
        reward -= drop_penalty
    return float(reward)