"""
arm_control.py

EL-A3 机械臂控制接口, 封装 robstride_dynamics SDK。
支持关节位置读取、位置控制、夹爪控制。
用于 leader-follower 遥操作数据采集。
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# robstride_dynamics SDK 位于 Python_Sample/ 目录下，添加到 import 路径
_SDK_DIR = str(Path(__file__).resolve().parent.parent / "Python_Sample")
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

try:
    from robstride_dynamics import RobstrideBus, Motor, ParameterType  # type: ignore
except ImportError as e:
    raise ImportError(
        f"无法导入 robstride_dynamics: {e}\n"
        "请确保已安装依赖: pip install python-can numpy tqdm"
    ) from e


# ── EL-A3 默认配置 ──────────────────────────────────────────
# 根据你的实际接线和电机 ID 修改下面的配置

@dataclass
class ArmConfig:
    """单条机械臂的电机配置"""
    channel: str = "can0"
    motors: dict = field(default_factory=lambda: {
        "J1": Motor(id=1, model="rs-00"),
        "J2": Motor(id=2, model="rs-00"),
        "J3": Motor(id=3, model="rs-00"),
        "J4": Motor(id=4, model="rs-00"),
        "J5": Motor(id=5, model="rs-00"),
        "J6": Motor(id=6, model="rs-00"),
        "gripper": Motor(id=7, model="rs-00"),
    })
    # 每个电机的校准参数: direction(+1/-1) 和 homing_offset(rad)
    # 根据你的装配实际情况修改
    calibration: dict = field(default_factory=lambda: {
        "J1": {"direction": -1, "homing_offset": 0.0},
        "J2": {"direction": 1, "homing_offset": 0.0},
        "J3": {"direction": -1, "homing_offset": 0.0},
        "J4": {"direction": 1, "homing_offset": 0.0},
        "J5": {"direction": -1, "homing_offset": 0.0},
        "J6": {"direction": 1, "homing_offset": 0.0},
        "gripper": {"direction": 1, "homing_offset": 0.0},
    })


JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]

# URDF 关节限位 (rad)
JOINT_LIMITS = {
    "J1": (-2.79253, 2.79253),
    "J2": (0.0, 3.66519),
    "J3": (-4.01426, 0.0),
    "J4": (-1.5708, 1.5708),
    "J5": (-1.5708, 1.5708),
    "J6": (-1.5708, 1.5708),
}


class ELA3Arm:
    """EL-A3 6-DOF 机械臂 + 夹爪控制器"""

    def __init__(self, config: ArmConfig, name: str = "arm"):
        self.name = name
        self.config = config
        self.bus: Optional[RobstrideBus] = None
        self._connected = False

    def connect(self):
        self.bus = RobstrideBus(
            channel=self.config.channel,
            motors=self.config.motors,
            calibration=self.config.calibration,
            bitrate=1000000,
        )
        self.bus.connect()
        self._connected = True
        print(f"[{self.name}] 已连接 (channel={self.config.channel})")

    def disconnect(self):
        if self._connected and self.bus is not None:
            self.bus.disconnect(disable_torque=True)
            self._connected = False
            print(f"[{self.name}] 已断开")

    def enable_all(self):
        for joint in JOINT_NAMES:
            self.bus.enable(joint)
        self.bus.enable("gripper")
        print(f"[{self.name}] 所有电机已使能")

    def disable_all(self):
        for motor in JOINT_NAMES + ["gripper"]:
            try:
                self.bus.disable(motor)
            except Exception as e:
                print(f"WARNING: 失能 {motor} 失败: {e}")
        print(f"[{self.name}] 所有电机已失能")

    # ── 校准工具 ───────────────────────────────────────────

    def _apply_calibration(self, joint: str, raw_pos: float) -> float:
        """电机原始角度 → URDF 约定角度：raw * direction + homing_offset"""
        cal = self.config.calibration.get(joint, {})
        direction = cal.get("direction", 1)
        offset = cal.get("homing_offset", 0.0)
        return raw_pos * direction + offset

    def _inverse_calibration(self, joint: str, urdf_pos: float) -> float:
        """URDF 约定角度 → 电机原始角度：(urdf - homing_offset) / direction"""
        cal = self.config.calibration.get(joint, {})
        direction = cal.get("direction", 1)
        offset = cal.get("homing_offset", 0.0)
        return (urdf_pos - offset) / direction

    # ── 读取 ─────────────────────────────────────────────

    def read_joint_positions(self) -> np.ndarray:
        """
        读取 6 个关节的当前角度 (rad)，已应用 calibration 修正。

        Returns:
            np.ndarray: shape (6,), [q1, q2, q3, q4, q5, q6]
        """
        positions = np.zeros(6, dtype=np.float64)
        for i, joint in enumerate(JOINT_NAMES):
            raw = float(self.bus.read(joint, ParameterType.MEASURED_POSITION))
            positions[i] = self._apply_calibration(joint, raw)
        return positions

    def read_joint_positions_mit(self) -> np.ndarray:
        """
        通过发送零力矩的 MIT 帧来同时读取关节位置。
        比逐个 read 更快，适合高频循环。
        bus 层的 read_operation_frame 已处理 calibration，此处不再重复。

        Returns:
            np.ndarray: shape (6,), [q1, q2, q3, q4, q5, q6]
        """
        positions = np.zeros(6, dtype=np.float64)
        for i, joint in enumerate(JOINT_NAMES):
            self.bus.write_operation_frame(joint, position=0, kp=0, kd=0, velocity=0, torque=0)
            pos, vel, torque, temp = self.bus.read_operation_frame(joint)
            positions[i] = pos
        return positions

    def read_gripper_position(self) -> float:
        """读取夹爪位置 (rad)，通过参数读取，已应用 calibration 修正。"""
        raw = float(self.bus.read("gripper", ParameterType.MEASURED_POSITION))
        return self._apply_calibration("gripper", raw)

    def read_gripper_position_mit(self) -> float:
        """
        通过 MIT 帧读取夹爪位置，同时保持夹爪在 MIT 模式下活跃。
        bus 层的 read_operation_frame 已处理 calibration，此处不再重复。
        """
        self.bus.write_operation_frame("gripper", position=0, kp=0, kd=0, velocity=0, torque=0)
        pos, vel, torque, temp = self.bus.read_operation_frame("gripper")
        return pos

    # ── 控制 ─────────────────────────────────────────────

    def set_joint_positions(self, positions: np.ndarray, kp: float = 30.0, kd: float = 1.0):
        """
        MIT 模式位置控制，发送 6 个关节目标位置。
        输入为 URDF 约定角度，bus 层的 write_operation_frame 负责转换为电机原始角度。

        Args:
            positions: shape (6,), 目标关节角度 (rad), URDF 约定
            kp: 位置增益
            kd: 阻尼增益
        """
        for i, joint in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[joint]
            target = float(np.clip(positions[i], lo, hi))
            self.bus.write_operation_frame(joint, position=target, kp=kp, kd=kd)
            self.bus.read_operation_frame(joint)

    def set_gripper(self, position: float, kp: float = 20.0, kd: float = 0.5):
        """
        控制夹爪位置。
        输入为 URDF 约定角度，bus 层的 write_operation_frame 负责转换为电机原始角度。

        Args:
            position: 夹爪目标位置 (rad), 0=关闭, 正值=打开
            kp: 位置增益
            kd: 阻尼增益
        """
        self.bus.write_operation_frame("gripper", position=position, kp=kp, kd=kd)
        self.bus.read_operation_frame("gripper")

    # ── Leader 模式 (低刚度，可手动拖拽) ────────────────

    def set_leader_mode(self, kp: float = 0.0, kd: float = 0.3):
        """
        将机械臂设为 leader 模式：极低刚度 + 轻微阻尼。
        你可以用手自由拖拽机械臂，同时读取关节角度。
        """
        for joint in JOINT_NAMES:
            self.bus.write_operation_frame(joint, position=0, kp=kp, kd=kd)
            self.bus.read_operation_frame(joint)
        self.bus.write_operation_frame("gripper", position=0, kp=kp, kd=kd)
        self.bus.read_operation_frame("gripper")
        print(f"[{self.name}] Leader 模式 (kp={kp}, kd={kd})")

    # ── Follower 模式 (高刚度，跟随目标) ────────────────

    def follow(self, joint_positions: np.ndarray, gripper_pos: float,
               kp: float = 30.0, kd: float = 1.0,
               gripper_kp: float = 20.0, gripper_kd: float = 0.5):
        """
        Follower 模式：跟随给定的关节位置和夹爪位置。
        """
        self.set_joint_positions(joint_positions, kp=kp, kd=kd)
        self.set_gripper(gripper_pos, kp=gripper_kp, kd=gripper_kd)

    # ── 工具方法 ─────────────────────────────────────────

    def get_full_state(self) -> dict:
        """
        读取完整状态: 6 关节角度 + 夹爪位置。

        Returns:
            dict: {"joint_positions": np.ndarray(6,), "gripper_position": float}
        """
        return {
            "joint_positions": self.read_joint_positions_mit(),
            "gripper_position": self.read_gripper_position(),
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
