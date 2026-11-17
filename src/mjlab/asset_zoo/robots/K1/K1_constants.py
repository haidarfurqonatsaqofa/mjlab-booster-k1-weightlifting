import os
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "K1_22dof.xml")


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(XML_PATH))


# 1. Aktuator Kaki (12 DoF): butuh kekakuan tinggi untuk menopang tubuh & beban
K1_LEGS_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"),
    stiffness=80.0,
    damping=4.0,
    effort_limit=60.0,
)

# 2. Aktuator Lengan (8 DoF): kekakuan menengah agar gerakan ayunan halus
K1_ARMS_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_Shoulder_.*", ".*_Elbow_.*"),
    stiffness=30.0,
    damping=1.5,
    effort_limit=25.0,
)

# 3. Aktuator Kepala & Gripper (4 DoF): gain rendah agar tidak bergetar/overflow
K1_HEAD_GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Head_.*", ".*_gripper"),
    stiffness=10.0,
    damping=0.5,
    effort_limit=10.0,
)

K1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        K1_LEGS_ACTUATOR,
        K1_ARMS_ACTUATOR,
        K1_HEAD_GRIPPER_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# 4. Postur Awal Berdiri (Lutut sedikit menekuk & lengan turun di samping badan)
K1_INITIAL_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.53),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        ".*_Hip_Pitch": -0.25,
        ".*_Knee_Pitch": 0.50,
        ".*_Ankle_Pitch": -0.25,
        "Left_Shoulder_Roll": -1.3,
        "Right_Shoulder_Roll": 1.3,
    },
)


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=K1_INITIAL_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=K1_ARTICULATION,
    )