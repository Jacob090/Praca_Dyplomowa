import numpy as np


def get_joint_state(model, data, joint_ids):
    qpos_list = []
    qvel_list = []
    for jid in joint_ids:
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        qpos_list.append(data.qpos[qpos_adr])
        qvel_list.append(data.qvel[qvel_adr])
    return np.array(qpos_list, dtype=np.float32), np.array(qvel_list, dtype=np.float32)


def get_tcp_position(data, tcp_site_id):
    return np.array(data.site_xpos[tcp_site_id], dtype=np.float32)


def get_object_state(data, body_id):
    pos = np.array(data.xpos[body_id], dtype=np.float32)
    # data.cvel: [ang vel (3), lin vel (3)]
    vel = np.array(data.cvel[body_id][3:6], dtype=np.float32)
    return pos, vel


def get_goal_position(model, goal_site_id, goal_pos=None):
    if goal_site_id is not None:
        return np.array(model.site_pos[goal_site_id], dtype=np.float32)
    if goal_pos is None:
        raise ValueError("Goal position is required when goal_site_id is None")
    return np.array(goal_pos, dtype=np.float32)


def compute_observation(model, data, joint_ids, tcp_site_id, object_body_id, goal_site_id, goal_pos=None):
    joint_pos, joint_vel = get_joint_state(model, data, joint_ids)
    tcp = get_tcp_position(data, tcp_site_id)
    obj_pos, obj_vel = get_object_state(data, object_body_id)
    goal = get_goal_position(model, goal_site_id, goal_pos=goal_pos)
    dist_tcp_obj = np.linalg.norm(tcp - obj_pos).astype(np.float32)
    dist_obj_goal = np.linalg.norm(obj_pos - goal).astype(np.float32)
    obs = np.concatenate(
        [
            joint_pos,
            joint_vel,
            tcp,
            obj_pos,
            obj_vel,
            np.array([dist_tcp_obj, dist_obj_goal], dtype=np.float32),
        ],
        axis=0,
    )
    return obs.astype(np.float32), dist_tcp_obj, dist_obj_goal
