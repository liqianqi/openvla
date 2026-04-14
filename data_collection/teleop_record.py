"""
teleop_record.py

Leader-Follower 遥操作数据采集脚本。
- Leader 机械臂: 低刚度, 人手拖拽, 读取关节角度
- Follower 机械臂: 高刚度, 跟随 Leader 的关节角度
- RealSense 相机: 拍摄 Follower 的工作场景
- 记录: 每帧的图像 + 末端绝对位姿 + 夹爪状态

采集完成后用 postprocess.py 转成增量动作 (delta action)。
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np

from data_collection.arm_control import ELA3Arm, ArmConfig, JOINT_NAMES
from data_collection.forward_kinematics import forward_kinematics
from data_collection.realsense_camera import RealSenseCamera, DummyCamera


@dataclass
class RecordConfig:
    hz: float = 5.0                         # 采集频率
    raw_data_dir: str = "raw_data"          # 原始数据保存目录
    use_dummy_camera: bool = False          # 是否用假相机 (调试用)
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_serial: str = ""                 # RealSense 序列号

    # Leader 机械臂配置
    leader_channel: str = "can0"
    leader_motor_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    leader_motor_model: str = "rs-00"

    # Follower 机械臂配置
    follower_channel: str = "can1"
    follower_motor_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    follower_motor_model: str = "rs-00"

    # Leader 模式参数
    leader_kp: float = 0.0
    leader_kd: float = 0.3

    # Follower 模式参数
    follower_kp: float = 30.0
    follower_kd: float = 1.0
    follower_gripper_kp: float = 20.0
    follower_gripper_kd: float = 0.5

    # 夹爪阈值: leader 夹爪 > 此值 → 视为打开 (1.0), 否则关闭 (0.0)
    gripper_open_threshold: float = 0.3


def _make_arm_config(channel: str, motor_ids: list, model: str) -> ArmConfig:
    from data_collection.arm_control import Motor
    motors = {}
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]
    for name, mid in zip(joint_names, motor_ids):
        motors[name] = Motor(id=mid, model=model)
    return ArmConfig(channel=channel, motors=motors)


class TeleopRecorder:
    """Leader-Follower 遥操作录制器"""

    def __init__(self, config: RecordConfig):
        self.config = config
        self.leader: Optional[ELA3Arm] = None
        self.follower: Optional[ELA3Arm] = None
        self.camera = None

    def setup(self):
        """初始化所有硬件"""
        # Leader 机械臂
        leader_cfg = _make_arm_config(
            self.config.leader_channel,
            self.config.leader_motor_ids,
            self.config.leader_motor_model,
        )
        self.leader = ELA3Arm(leader_cfg, name="leader")
        self.leader.connect()
        self.leader.enable_all()

        # Follower 机械臂
        follower_cfg = _make_arm_config(
            self.config.follower_channel,
            self.config.follower_motor_ids,
            self.config.follower_motor_model,
        )
        self.follower = ELA3Arm(follower_cfg, name="follower")
        self.follower.connect()
        self.follower.enable_all()

        # 相机
        if self.config.use_dummy_camera:
            self.camera = DummyCamera(self.config.camera_width, self.config.camera_height)
        else:
            self.camera = RealSenseCamera(
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                serial=self.config.camera_serial,
            )
        self.camera.connect()

        # 设置 leader 为低刚度拖拽模式
        self.leader.set_leader_mode(kp=self.config.leader_kp, kd=self.config.leader_kd)

        print("\n硬件初始化完成!")
        print(f"  Leader:   {self.config.leader_channel}")
        print(f"  Follower: {self.config.follower_channel}")
        print(f"  Camera:   {'Dummy' if self.config.use_dummy_camera else 'RealSense'}")
        print(f"  频率:     {self.config.hz} Hz")

    def teardown(self):
        """关闭所有硬件"""
        if self.leader:
            self.leader.disable_all()
            self.leader.disconnect()
        if self.follower:
            self.follower.disable_all()
            self.follower.disconnect()
        if self.camera:
            self.camera.disconnect()

    def record_one_trajectory(self, instruction: str, trajectory_idx: int) -> str:
        """
        录制一条完整轨迹。

        流程:
        1. 读取 leader 关节角度
        2. 发送给 follower 跟随
        3. 相机拍照
        4. 用 FK 计算末端绝对位姿
        5. 记录夹爪状态
        6. 按频率循环，直到用户按 Enter 结束

        保存:
        - images:             (T, H, W, 3) uint8
        - eef_poses:          (T, 6) float64  [x, y, z, roll, pitch, yaw]
        - gripper_states:     (T,)   float64  夹爪位置 (raw)
        - joint_positions:    (T, 6) float64  关节角度 (用于调试)
        - language_instruction: str

        Args:
            instruction: 语言指令
            trajectory_idx: 轨迹编号

        Returns:
            保存的文件路径
        """
        dt = 1.0 / self.config.hz
        images, eef_poses, gripper_states, joint_positions_list = [], [], [], []

        print(f"\n{'='*50}")
        print(f"轨迹 #{trajectory_idx:04d}")
        print(f"指令: \"{instruction}\"")
        print(f"按 Enter 开始录制...")
        input()
        print("录制中... (按 Enter 停止)")

        # 非阻塞检测 Enter
        import threading
        stop_flag = threading.Event()

        def _wait_for_enter():
            input()
            stop_flag.set()

        t = threading.Thread(target=_wait_for_enter, daemon=True)
        t.start()

        step = 0
        try:
            while not stop_flag.is_set():
                t_start = time.time()

                # 1) 读取 leader 状态
                leader_joints = self.leader.read_joint_positions_mit()
                leader_gripper = self.leader.read_gripper_position()

                # 2) Follower 跟随
                self.follower.follow(
                    leader_joints,
                    leader_gripper,
                    kp=self.config.follower_kp,
                    kd=self.config.follower_kd,
                    gripper_kp=self.config.follower_gripper_kp,
                    gripper_kd=self.config.follower_gripper_kd,
                )

                # 3) 拍照 (拍 follower 的工作场景)
                image = self.camera.capture_rgb()

                # 4) FK 计算末端位姿
                eef_pose = forward_kinematics(leader_joints)

                # 5) 记录
                images.append(image)
                eef_poses.append(eef_pose)
                gripper_states.append(leader_gripper)
                joint_positions_list.append(leader_joints.copy())

                step += 1
                if step % 10 == 0:
                    print(f"  step {step:4d} | EEF: [{eef_pose[0]:.3f}, {eef_pose[1]:.3f}, {eef_pose[2]:.3f}]"
                          f" | gripper: {leader_gripper:.3f}")

                # 控制频率
                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n录制被中断")

        print(f"录制完成: {step} 步 ({step / self.config.hz:.1f} 秒)")

        # 保存到 HDF5
        filepath = os.path.join(
            self.config.raw_data_dir,
            f"trajectory_{trajectory_idx:04d}.hdf5",
        )
        os.makedirs(self.config.raw_data_dir, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "observation/image",
                data=np.array(images, dtype=np.uint8),
                chunks=(1, images[0].shape[0], images[0].shape[1], 3),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset("eef_poses", data=np.array(eef_poses, dtype=np.float64))
            f.create_dataset("gripper_states", data=np.array(gripper_states, dtype=np.float64))
            f.create_dataset("joint_positions", data=np.array(joint_positions_list, dtype=np.float64))
            f.attrs["language_instruction"] = instruction
            f.attrs["hz"] = self.config.hz
            f.attrs["num_steps"] = step

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"已保存: {filepath} ({file_size_mb:.1f} MB)")
        return filepath

    def run(self, num_trajectories: int = 10, default_instruction: str = ""):
        """
        交互式录制多条轨迹。

        Args:
            num_trajectories: 要录制的轨迹总数
            default_instruction: 默认指令 (为空则每次手动输入)
        """
        self.setup()

        try:
            for i in range(num_trajectories):
                print(f"\n{'='*60}")
                print(f"准备录制第 {i+1}/{num_trajectories} 条轨迹")
                print(f"{'='*60}")

                if default_instruction:
                    instruction = default_instruction
                    print(f"使用默认指令: \"{instruction}\"")
                else:
                    instruction = input("请输入任务指令 (英文): ").strip()
                    if not instruction:
                        instruction = "do the task"

                self.record_one_trajectory(instruction, trajectory_idx=i)

                if i < num_trajectories - 1:
                    print("\n请将物体复位, 准备下一条轨迹...")
                    input("按 Enter 继续...")

        except KeyboardInterrupt:
            print("\n\n采集被用户中断")
        finally:
            self.teardown()

        print(f"\n采集完成! 数据保存在: {self.config.raw_data_dir}/")
        print(f"接下来运行 postprocess.py 将绝对位姿转换为增量动作。")


# ── 命令行入口 ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EL-A3 Leader-Follower 遥操作数据采集")
    parser.add_argument("--num", type=int, default=10, help="采集轨迹数量")
    parser.add_argument("--hz", type=float, default=5.0, help="采集频率")
    parser.add_argument("--instruction", type=str, default="", help="默认指令 (为空则每次手动输入)")
    parser.add_argument("--output_dir", type=str, default="raw_data", help="输出目录")
    parser.add_argument("--leader_channel", type=str, default="can0", help="Leader CAN 通道")
    parser.add_argument("--follower_channel", type=str, default="can1", help="Follower CAN 通道")
    parser.add_argument("--dummy_camera", action="store_true", help="使用假相机 (调试)")
    args = parser.parse_args()

    config = RecordConfig(
        hz=args.hz,
        raw_data_dir=args.output_dir,
        use_dummy_camera=args.dummy_camera,
        leader_channel=args.leader_channel,
        follower_channel=args.follower_channel,
    )

    recorder = TeleopRecorder(config)
    recorder.run(num_trajectories=args.num, default_instruction=args.instruction)
