from pathlib import Path
import mujoco

from mjlab.asset_zoo.robots import get_K1_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import JointPositionActionCfg, reset_scene_to_default
from mjlab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    TerminationTermCfg as DoneTerm,
)
from mjlab.tasks.cartpole.cartpole_env_cfg import _make_env_cfg as make_base_cfg
from mjlab.terrains import TerrainEntityCfg
import mjlab.tasks.booster_k1.mdp.rewards as k1_mdp

from pathlib import Path
import mujoco
import numpy as np

BARBELL_XML_PATH = Path(__file__).parents[2] / "asset_zoo" / "objects" / "barbell.xml"


def _strip_plane_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Menghapus plane bawaan XML agar tidak Z-fighting dengan terrain mjlab."""
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)
    return spec


def _get_clean_k1_cfg() -> EntityCfg:
    cfg = get_K1_cfg()
    orig_spec_fn = cfg.spec_fn
    cfg.spec_fn = lambda: _strip_plane_geoms(orig_spec_fn())
    return cfg


def _get_barbell_and_lines_spec() -> mujoco.MjSpec:
    spec = _strip_plane_geoms(mujoco.MjSpec.from_file(str(BARBELL_XML_PATH)))

    # Tambahkan 4 garis visual standar FIRA HuroCup (Start, Pick-up, Lift, Finish)
    # contype=0 & conaffinity=0 agar murni visual dan tidak mengganggu langkah kaki
    lines = [
        ("start_line", 0.0, [1.0, 1.0, 1.0, 0.9]),     # Putih: X = 0.0m
        ("pickup_line", 1.0, [1.0, 0.85, 0.0, 0.95]),  # Kuning: X = 1.0m (Pick-up)
        ("lift_line", 2.0, [0.0, 0.9, 1.0, 0.95]),     # Cyan: X = 2.0m (Lift)
        ("finish_line", 3.0, [0.2, 1.0, 0.3, 0.95]),   # Hijau: X = 3.0m (Finish)
    ]
    for name, x_pos, rgba in lines:
        geom = spec.worldbody.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = np.array([0.015, 0.85, 0.0005], dtype=np.float64)
        geom.pos = np.array([x_pos, 0.0, 0.0008], dtype=np.float64)
        geom.rgba = np.array(rgba, dtype=np.float32)
        geom.contype = 0
        geom.conaffinity = 0

    return spec


def get_booster_k1_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = make_base_cfg(swing_up=False)

    cfg.events = {}
    cfg.commands = {}
    cfg.actions = {}
    cfg.observations = {}
    cfg.rewards = {}
    cfg.terminations = {}

    # Scene & Entitas (K1 di Start Line X=0.0m, Barbell di Pick-up Line X=1.0m)
    cfg.scene.entities = {
        "booster_k1": _get_clean_k1_cfg(),
        "barbell": EntityCfg(
            init_state=EntityCfg.InitialStateCfg(pos=(1.0, 0.0, 0.06)),
            spec_fn=_get_barbell_and_lines_spec,
        ),
    }
    cfg.scene.num_envs = 1024
    cfg.scene.env_spacing = 4.0
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")

    cfg.episode_length_s = 8.0

    # Event Reset
    cfg.events = {
        "reset_scene": EventTerm(
            func=reset_scene_to_default,
            mode="reset",
        ),
    }

    # Ruang Aksi
    cfg.actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="booster_k1",
            actuator_names=[".*"],
            scale=0.25,
            use_default_offset=True,
        )
    }

    # Ruang Observasi (Line-based + Gait Phase Clock)
    cfg.observations = {
        "policy": ObsGroup(
            terms={
                "base_lin_vel": ObsTerm(func=k1_mdp.obs_base_lin_vel),
                "base_ang_vel": ObsTerm(func=k1_mdp.obs_base_ang_vel),
                "projected_gravity": ObsTerm(func=k1_mdp.obs_projected_gravity),
                "joint_pos": ObsTerm(
                    func=lambda env: env.scene["booster_k1"].data.joint_pos
                    - env.scene["booster_k1"].data.default_joint_pos
                ),
                "joint_vel": ObsTerm(
                    func=lambda env: env.scene["booster_k1"].data.joint_vel * 0.1
                ),
                "target_line": ObsTerm(func=k1_mdp.obs_target_line_info),
                "gait_phase": ObsTerm(func=k1_mdp.obs_gait_phase),
                "last_action": ObsTerm(func=lambda env: env.action_manager.prev_action),
            }
        )
    }

    # Pembobotan Reward (Smooth Line Approach Walking)
    cfg.rewards = {
        "alive": RewardTerm(func=k1_mdp.reward_alive, weight=1.0),
        "line_progress": RewardTerm(func=k1_mdp.reward_line_progress, weight=2.0),
        "line_velocity": RewardTerm(func=k1_mdp.reward_line_velocity_profile, weight=2.0),
        "line_precision": RewardTerm(func=k1_mdp.reward_line_precision, weight=1.5),
        "heading_align": RewardTerm(func=k1_mdp.reward_heading_alignment, weight=0.8),
        "stop_at_line": RewardTerm(func=k1_mdp.reward_stop_at_line, weight=2.0),
        "gait_rhythm": RewardTerm(func=k1_mdp.reward_bipedal_gait_rhythm, weight=1.5),
        "feet_slide": RewardTerm(func=k1_mdp.penalize_feet_slide, weight=-0.5),
        "stance_width": RewardTerm(func=k1_mdp.penalize_stance_width, weight=-1.0),
        "torso_orientation": RewardTerm(func=k1_mdp.penalize_torso_orientation, weight=-1.5),
        "lin_vel_z": RewardTerm(func=k1_mdp.penalize_lin_vel_z, weight=-0.5),
        "action_rate": RewardTerm(func=k1_mdp.penalize_action_rate, weight=-0.06),
        "joint_torques": RewardTerm(func=k1_mdp.penalize_joint_torques, weight=-1e-5),
    }

    # Terminasi
    cfg.terminations = {
        "time_out": DoneTerm(
            func=lambda env: env.episode_length_buf >= env.max_episode_length,
            time_out=True,
        ),
        "robot_fall": DoneTerm(func=k1_mdp.term_robot_fall),
    }

    # Simulasi & Viewer
    cfg.viewer.entity_name = "booster_k1"
    cfg.viewer.body_name = "Trunk"
    cfg.viewer.distance = 2.8
    cfg.viewer.elevation = -18.0

    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.sim.mujoco.disableflags = ()
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4

    return cfg


class BoosterK1EnvCfg(ManagerBasedRlEnvCfg):
    def __new__(cls):
        return get_booster_k1_env_cfg()


class BoosterK1PlayEnvCfg(ManagerBasedRlEnvCfg):
    def __new__(cls):
        cfg = get_booster_k1_env_cfg()
        cfg.scene.num_envs = 1
        return cfg