from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from utils.metrics import compute_success, has_nan, is_out_of_bounds
from utils.observations import compute_observation
from utils.rewards import reward_stage1, reward_stage2, reward_stage3


class Arm4DOFPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: Dict[str, Any], stage: str, render_mode: Optional[str] = None):
        if stage not in {"reach", "grasp", "place"}:
            raise ValueError("stage must be one of: reach, grasp, place")

        self.config = config
        self.stage = stage
        self.render_mode = render_mode

        scene_path = config["scene_path"]
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = ["joint1_yaw", "joint2_pitch", "joint3_pitch", "joint4_pitch"]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")

        self.action_scale = float(config["action_scale"])
        self.max_joint_vel = float(config["max_joint_vel"])
        self.sim_steps = int(config["sim_steps_per_action"])
        self.max_steps = int(config["max_steps"])
        self.grasp_distance = float(config["grasp_distance"])
        self.grasp_close_threshold = float(config["grasp_close_threshold"])
        self.grasp_open_threshold = float(config["grasp_open_threshold"])
        self.grasp_offset = np.array(config["grasp_offset"], dtype=np.float32)
        self.workspace_bounds = config["workspace_bounds"]
        self.table_height = float(config["table_height"])
        self.object_size = float(config["object_size"])
        self.object_spawn_range = config["object_spawn_range"]
        self.goal_spawn_range = config["goal_spawn_range"]
        self.fixed_goal = np.array(config["fixed_goal"], dtype=np.float32)
        self.reset_joint_pos = np.array(config["reset_joint_pos"], dtype=np.float32)

        self.reward_cfg = config["reward"]

        obs_dim = 4 + 4 + 3 + 3 + 3 + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.np_random = None
        self.steps = 0
        self.is_grasped = False
        self.prev_obj_goal_dist = None

        self.renderer = None
        self.viewer = None

    def _set_object_position(self, pos):
        qpos_adr = self.model.joint_qposadr[self.object_joint_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1, 0, 0, 0], dtype=np.float32)
        qvel_adr = self.model.joint_dofadr[self.object_joint_id]
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0

    def _set_goal_position(self, pos):
        self.model.site_pos[self.goal_site_id] = pos

    def _sample_xy(self, rng, spawn_range):
        x = rng.uniform(spawn_range["x"][0], spawn_range["x"][1])
        y = rng.uniform(spawn_range["y"][0], spawn_range["y"][1])
        return float(x), float(y)

    def _reset_scene(self):
        mujoco.mj_resetData(self.model, self.data)
        for jid, qpos in zip(self.joint_ids, self.reset_joint_pos):
            qpos_adr = self.model.joint_qposadr[jid]
            self.data.qpos[qpos_adr] = qpos
        obj_x, obj_y = self._sample_xy(self.np_random, self.object_spawn_range)
        obj_z = self.table_height + self.object_size
        self._set_object_position(np.array([obj_x, obj_y, obj_z], dtype=np.float32))
        if self.stage == "place":
            goal_x, goal_y = self._sample_xy(self.np_random, self.goal_spawn_range)
            goal_z = self.table_height + self.object_size
            self._set_goal_position(np.array([goal_x, goal_y, goal_z], dtype=np.float32))
        else:
            self._set_goal_position(self.fixed_goal)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_scene()
        self.steps = 0
        self.is_grasped = False
        obs, dist_tcp_obj, dist_obj_goal = compute_observation(
            self.model, self.data, self.joint_ids, self.tcp_site_id, self.object_body_id, self.goal_site_id
        )
        self.prev_obj_goal_dist = dist_obj_goal
        info = self._info(dist_tcp_obj, dist_obj_goal)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        joint_action = action[:4] * self.action_scale
        gripper_cmd = float((action[4] + 1.0) * 0.5)
        self.data.ctrl[:4] = joint_action
        self.data.ctrl[4] = gripper_cmd

        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
            for jid in self.joint_ids:
                dof_adr = self.model.joint_dofadr[jid]
                self.data.qvel[dof_adr] = np.clip(self.data.qvel[dof_adr], -self.max_joint_vel, self.max_joint_vel)

            if self.is_grasped:
                tcp_pos = np.array(self.data.site_xpos[self.tcp_site_id], dtype=np.float32)
                self._set_object_position(tcp_pos + self.grasp_offset)
                mujoco.mj_forward(self.model, self.data)

        obs, dist_tcp_obj, dist_obj_goal = compute_observation(
            self.model, self.data, self.joint_ids, self.tcp_site_id, self.object_body_id, self.goal_site_id
        )

        object_pos = np.array(self.data.body_xpos[self.object_body_id], dtype=np.float32)
        object_height = float(object_pos[2])

        if self.stage in {"grasp", "place"}:
            if self.is_grasped and gripper_cmd < self.grasp_open_threshold:
                self.is_grasped = False
            if (not self.is_grasped) and gripper_cmd > self.grasp_close_threshold and dist_tcp_obj < self.grasp_distance:
                self.is_grasped = True

        reward = 0.0
        if self.stage == "reach":
            reward = reward_stage1(
                dist_tcp_obj=dist_tcp_obj,
                action=action,
                reach_threshold=self.reward_cfg["reach_threshold"],
                action_penalty_scale=self.reward_cfg["action_penalty_scale"],
                reach_bonus=self.reward_cfg["reach_bonus"],
            )
        elif self.stage == "grasp":
            reward = reward_stage2(
                dist_tcp_obj=dist_tcp_obj,
                action=action,
                gripper_cmd=gripper_cmd,
                is_grasped=self.is_grasped,
                object_height=object_height,
                lift_height=self.reward_cfg["lift_height"],
                reach_threshold=self.reward_cfg["reach_threshold"],
                action_penalty_scale=self.reward_cfg["action_penalty_scale"],
                grasp_bonus=self.reward_cfg["grasp_bonus"],
                lift_bonus=self.reward_cfg["lift_bonus"],
            )
        elif self.stage == "place":
            reward = reward_stage3(
                dist_obj_goal=dist_obj_goal,
                prev_dist_obj_goal=self.prev_obj_goal_dist,
                action=action,
                is_grasped=self.is_grasped,
                object_height=object_height,
                drop_height=self.reward_cfg["drop_height"],
                success_threshold=self.reward_cfg["place_threshold"],
                action_penalty_scale=self.reward_cfg["action_penalty_scale"],
                goal_bonus=self.reward_cfg["goal_bonus"],
                drop_penalty=self.reward_cfg["drop_penalty"],
            )

        self.prev_obj_goal_dist = dist_obj_goal

        success = compute_success(
            stage=self.stage,
            dist_tcp_obj=dist_tcp_obj,
            dist_obj_goal=dist_obj_goal,
            is_grasped=self.is_grasped,
            object_height=object_height,
            lift_height=self.reward_cfg["lift_height"],
            success_threshold=self.reward_cfg["success_threshold"],
        )

        terminated = False
        truncated = False

        if success:
            terminated = True
        if has_nan(obs):
            terminated = True
            reward = float(self.reward_cfg["nan_penalty"])
        if is_out_of_bounds(object_pos, self.workspace_bounds):
            terminated = True
            reward = float(self.reward_cfg["out_of_bounds_penalty"])

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        info = self._info(dist_tcp_obj, dist_obj_goal, success=success)
        return obs, float(reward), terminated, truncated, info

    def _info(self, dist_tcp_obj, dist_obj_goal, success=False):
        return {
            "stage": self.stage,
            "distance_tcp_object": float(dist_tcp_obj),
            "distance_object_goal": float(dist_obj_goal),
            "is_grasped": bool(self.is_grasped),
            "success": bool(success),
        }

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="main")
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
