from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Konfigurasi Garis FIRA HuroCup (Adult Size)
# State 0 (Approach): TARGET_LINE_X = 1.0 (Pick-up Line), LINE_STOP_OFFSET = 0.22
# State 2 (Low Carry): TARGET_LINE_X = 2.0 (Lift Line), LINE_STOP_OFFSET = 0.0
# State 4 (Overhead): TARGET_LINE_X = 3.15 (Finish Line), LINE_STOP_OFFSET = 0.0
TARGET_LINE_X = 1.0
LINE_STOP_OFFSET = 0.22
GAIT_PERIOD_S = 0.70  # Siklus langkah kiri-kanan 0.7 detik (0.35s per langkah)


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


_FOOT_INDICES_CACHE: dict[int, tuple[int, int]] = {}


def _get_foot_indices(env: ManagerBasedRlEnv) -> tuple[int, int]:
    env_id = id(env)
    cached = _FOOT_INDICES_CACHE.get(env_id)
    if cached is not None:
        return cached
    robot = env.scene["booster_k1"]
    names = robot.body_names
    left_idx, right_idx = None, None
    for i, name in enumerate(names):
        low = name.lower()
        if "left" in low and ("foot" in low or "ankle_roll" in low):
            left_idx = i
        elif "right" in low and ("foot" in low or "ankle_roll" in low):
            right_idx = i
    if left_idx is None or right_idx is None:
        left_idx = next((i for i, n in enumerate(names) if "left" in n.lower() and "ankle" in n.lower()), 0)
        right_idx = next((i for i, n in enumerate(names) if "right" in n.lower() and "ankle" in n.lower()), 0)
    _FOOT_INDICES_CACHE[env_id] = (left_idx, right_idx)
    return left_idx, right_idx


def _get_line_target_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    target = torch.zeros((env.num_envs, 2), device=env.device)
    target[:, 0] = TARGET_LINE_X - LINE_STOP_OFFSET
    return target


# ==============================================================================
# OBSERVATIONS
# ==============================================================================

def obs_base_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["booster_k1"]
    quat_w = robot.data.root_link_pose_w[:, 3:7]
    lin_vel_w = robot.data.root_link_vel_w[:, :3]
    return _quat_rotate_inverse(quat_w, lin_vel_w) * 2.0


def obs_base_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["booster_k1"]
    quat_w = robot.data.root_link_pose_w[:, 3:7]
    ang_vel_w = robot.data.root_link_vel_w[:, 3:6]
    return _quat_rotate_inverse(quat_w, ang_vel_w) * 0.25


def obs_projected_gravity(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["booster_k1"]
    quat_w = robot.data.root_link_pose_w[:, 3:7]
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    return _quat_rotate_inverse(quat_w, gravity_w)


def obs_target_line_info(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Vektor relatif menuju garis target (Pick-up / Lift / Finish Line) dalam body frame."""
    robot = env.scene["booster_k1"]
    robot_pos_xy = robot.data.root_link_pose_w[:, :2]
    quat_w = robot.data.root_link_pose_w[:, 3:7]

    target_xy = _get_line_target_xy(env)
    diff_w = torch.zeros((env.num_envs, 3), device=env.device)
    diff_w[:, :2] = target_xy - robot_pos_xy
    diff_b = _quat_rotate_inverse(quat_w, diff_w)

    dist = torch.norm(diff_w[:, :2], dim=1, keepdim=True)
    # Arah hadap selalu tegak lurus garis (+X dunia)
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return torch.cat(
        [diff_b[:, :2], dist, torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)],
        dim=1,
    )


def obs_gait_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sinyal jam periodik (sin, cos) untuk ritme ayunan kaki kiri-kanan."""
    robot = env.scene["booster_k1"]
    dist = torch.norm(_get_line_target_xy(env) - robot.data.root_link_pose_w[:, :2], dim=1)
    is_moving = (dist > 0.06).float().unsqueeze(1)

    phase = (env.episode_length_buf.float() * env.step_dt / GAIT_PERIOD_S) % 1.0
    angle = 2.0 * torch.pi * phase.unsqueeze(1)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1) * is_moving


# ==============================================================================
# REWARDS (LINE NAVIGATION + SMOOTH BIPEDAL GAIT)
# ==============================================================================

def reward_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def reward_line_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot_pos_xy = env.scene["booster_k1"].data.root_link_pose_w[:, :2]
    dist = torch.norm(_get_line_target_xy(env) - robot_pos_xy, dim=1)
    return 1.0 - torch.tanh(dist / 0.5)


def reward_line_velocity_profile(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Profil kecepatan mulus: jalan stabil 0.40 m/s lalu melambat halus mendekati garis."""
    robot = env.scene["booster_k1"]
    robot_pos_xy = robot.data.root_link_pose_w[:, :2]
    diff_w = _get_line_target_xy(env) - robot_pos_xy
    dist = torch.norm(diff_w, dim=1, keepdim=True).clamp(min=1e-6)
    dir_w = diff_w / dist

    vel_xy_w = robot.data.root_link_vel_w[:, :2]
    vel_toward = torch.sum(vel_xy_w * dir_w, dim=1)

    # Referensi kecepatan turun mulus ke 0 saat mendekati garis target
    ref_vel = torch.clamp(1.2 * dist.squeeze(1), min=0.0, max=0.40)
    vel_error = torch.square(vel_toward - ref_vel)
    return torch.exp(-vel_error / (0.15**2))


def reward_line_precision(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot_pos_xy = env.scene["booster_k1"].data.root_link_pose_w[:, :2]
    dist = torch.norm(_get_line_target_xy(env) - robot_pos_xy, dim=1)
    return torch.exp(-torch.square(dist) / (0.15**2))


def reward_heading_alignment(env: ManagerBasedRlEnv) -> torch.Tensor:
    quat_w = env.scene["booster_k1"].data.root_link_pose_w[:, 3:7]
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.exp(-torch.square(yaw) / (0.20**2))


def reward_stop_at_line(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["booster_k1"]
    dist = torch.norm(_get_line_target_xy(env) - robot.data.root_link_pose_w[:, :2], dim=1)
    speed = torch.norm(robot.data.root_link_vel_w[:, :2], dim=1)
    at_line = (dist < 0.07).float()
    return at_line * torch.exp(-torch.square(speed) / (0.08**2))


def reward_bipedal_gait_rhythm(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mendorong ayunan kaki kiri-kanan bergantian sesuai fase, dan menapak sejajar saat berhenti."""
    robot = env.scene["booster_k1"]
    l_idx, r_idx = _get_foot_indices(env)

    foot_z_l = robot.data.body_link_pose_w[:, l_idx, 2]
    foot_z_r = robot.data.body_link_pose_w[:, r_idx, 2]

    dist = torch.norm(_get_line_target_xy(env) - robot.data.root_link_pose_w[:, :2], dim=1)
    is_walking = (dist > 0.07).float()

    phase = (env.episode_length_buf.float() * env.step_dt / GAIT_PERIOD_S) % 1.0
    sin_p = torch.sin(2.0 * torch.pi * phase)

    # Saat berjalan: reward jika kaki yang sedang fase swing terangkat lebih tinggi (maks 5 cm)
    swing_mask = (torch.abs(sin_p) > 0.2).float()
    z_diff_signed = (foot_z_l - foot_z_r) * torch.sign(sin_p)
    walk_gait_rew = swing_mask * (torch.clamp(z_diff_signed, min=0.0, max=0.05) / 0.05)

    # Saat berhenti di garis: reward jika kedua kaki menapak sejajar
    both_grounded = torch.exp(-torch.square(foot_z_l - foot_z_r) / (0.015**2))

    return is_walking * walk_gait_rew + (1.0 - is_walking) * both_grounded


def penalize_feet_slide(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mencegah telapak kaki menggeser/terseret di lantai saat menapak."""
    robot = env.scene["booster_k1"]
    l_idx, r_idx = _get_foot_indices(env)

    pos_z_l = robot.data.body_link_pose_w[:, l_idx, 2]
    pos_z_r = robot.data.body_link_pose_w[:, r_idx, 2]
    vel_xy_l = torch.norm(robot.data.body_link_vel_w[:, l_idx, :2], dim=1)
    vel_xy_r = torch.norm(robot.data.body_link_vel_w[:, r_idx, :2], dim=1)

    contact_l = (pos_z_l < 0.055).float()
    contact_r = (pos_z_r < 0.055).float()
    return contact_l * torch.square(vel_xy_l) + contact_r * torch.square(vel_xy_r)


def penalize_stance_width(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Menjaga jarak lateral antar kaki di sekitar lebar pinggul (~0.18 m) agar tidak mengangkang."""
    robot = env.scene["booster_k1"]
    l_idx, r_idx = _get_foot_indices(env)

    foot_y_l = robot.data.body_link_pose_w[:, l_idx, 1]
    foot_y_r = robot.data.body_link_pose_w[:, r_idx, 1]
    lateral_dist = torch.abs(foot_y_l - foot_y_r)
    return torch.square(lateral_dist - 0.18)


def penalize_torso_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
    proj_grav = obs_projected_gravity(env)
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)


def penalize_lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    lin_vel_z = env.scene["booster_k1"].data.root_link_vel_w[:, 2]
    return torch.square(lin_vel_z)


def penalize_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.sum(torch.square(action - prev_action), dim=1)


def penalize_joint_torques(env: ManagerBasedRlEnv) -> torch.Tensor:
    torques = env.scene["booster_k1"].data.actuator_force
    return torch.sum(torch.square(torques), dim=1)


def term_robot_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["booster_k1"]
    height = robot.data.root_link_pose_w[:, 2]
    proj_grav = obs_projected_gravity(env)
    tilt = torch.norm(proj_grav[:, :2], dim=1)
    return (height < 0.32) | (tilt > 0.7)