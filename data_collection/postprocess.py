"""
postprocess.py

后处理：将 teleop_record.py 采集的原始数据 (绝对位姿)
转换为 OpenVLA 需要的增量动作 (delta action)。

方式 B: 逐帧计算相邻帧的位姿差值
    action[t] = eef_pose[t+1] - eef_pose[t]   (位置和旋转的增量)
    action[6] = gripper_state[t]                (夹爪绝对值: 1=打开, 0=关闭)

输入:  raw_data/trajectory_XXXX.hdf5  (含 eef_poses, gripper_states)
输出:  raw_data/trajectory_XXXX.hdf5  (追加 action 字段, 可直接用于训练)
"""

import os
import glob

import h5py
import numpy as np


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_delta_actions(
    eef_poses: np.ndarray,
    gripper_states: np.ndarray,
    gripper_open_threshold: float = 0.3,
) -> np.ndarray:
    """
    从绝对位姿序列计算增量动作。

    Args:
        eef_poses:     (T, 6) [x, y, z, roll, pitch, yaw], 每帧的末端绝对位姿
        gripper_states: (T,)   夹爪原始位置 (rad)
        gripper_open_threshold: 夹爪二值化阈值

    Returns:
        actions: (T-1, 7) [dx, dy, dz, droll, dpitch, dyaw, gripper]
                 前 6 维是增量, 第 7 维是二值化的夹爪状态 (1=打开, 0=关闭)
    """
    T = len(eef_poses)
    assert T >= 2, f"至少需要 2 帧才能计算增量, 当前 {T} 帧"
    assert len(gripper_states) == T

    actions = np.zeros((T - 1, 7), dtype=np.float32)

    for t in range(T - 1):
        # 位置增量
        delta_pos = eef_poses[t + 1, :3] - eef_poses[t, :3]

        # 旋转增量 (带角度归一化)
        delta_rot = np.array([
            normalize_angle(eef_poses[t + 1, 3] - eef_poses[t, 3]),
            normalize_angle(eef_poses[t + 1, 4] - eef_poses[t, 4]),
            normalize_angle(eef_poses[t + 1, 5] - eef_poses[t, 5]),
        ])

        # 夹爪: 二值化, 1.0 = 打开, 0.0 = 关闭
        gripper = 1.0 if abs(gripper_states[t]) > gripper_open_threshold else 0.0

        actions[t, :3] = delta_pos
        actions[t, 3:6] = delta_rot
        actions[t, 6] = gripper

    return actions


def postprocess_trajectory(filepath: str, gripper_open_threshold: float = 0.3) -> None:
    """
    对单个 HDF5 轨迹文件进行后处理：计算增量动作并写入文件。

    处理后文件新增:
      - action:     (T-1, 7) float32
      - observation/image 截断为 (T-1, H, W, 3) 与 action 对齐

    Args:
        filepath: HDF5 文件路径
        gripper_open_threshold: 夹爪二值化阈值
    """
    with h5py.File(filepath, "r") as f:
        eef_poses = f["eef_poses"][:]
        gripper_states = f["gripper_states"][:]
        images = f["observation/image"][:]
        joint_positions = f["joint_positions"][:]
        instruction = f.attrs["language_instruction"]
        hz = f.attrs["hz"]

    T = len(eef_poses)
    if T < 2:
        print(f"  跳过 {filepath}: 不足 2 帧")
        return

    # 计算增量动作
    actions = compute_delta_actions(eef_poses, gripper_states, gripper_open_threshold)

    # 截断到 T-1 帧 (最后一帧没有对应的 action)
    images_trimmed = images[:T - 1]
    eef_trimmed = eef_poses[:T - 1]
    gripper_trimmed = gripper_states[:T - 1]
    joints_trimmed = joint_positions[:T - 1]

    # 重写文件 (包含 action)
    with h5py.File(filepath, "w") as f:
        f.create_dataset(
            "observation/image",
            data=images_trimmed,
            dtype=np.uint8,
            chunks=(1, images_trimmed.shape[1], images_trimmed.shape[2], 3),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("action", data=actions, dtype=np.float32)
        f.create_dataset("eef_poses", data=eef_trimmed, dtype=np.float64)
        f.create_dataset("gripper_states", data=gripper_trimmed, dtype=np.float64)
        f.create_dataset("joint_positions", data=joints_trimmed, dtype=np.float64)
        f.attrs["language_instruction"] = instruction
        f.attrs["hz"] = hz
        f.attrs["num_steps"] = T - 1

    print(f"  已处理: {filepath} ({T} → {T-1} 帧)")


def postprocess_all(data_dir: str, gripper_open_threshold: float = 0.3) -> None:
    """
    批量后处理目录下所有轨迹文件。
    """
    pattern = os.path.join(data_dir, "trajectory_*.hdf5")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"在 {data_dir}/ 中未找到轨迹文件")
        return

    print(f"找到 {len(files)} 个轨迹文件")
    print(f"夹爪阈值: {gripper_open_threshold}")
    print()

    for filepath in files:
        postprocess_trajectory(filepath, gripper_open_threshold)

    # 统计动作范围
    print("\n" + "=" * 50)
    print("动作统计")
    print("=" * 50)

    all_actions = []
    for filepath in files:
        with h5py.File(filepath, "r") as f:
            if "action" in f:
                all_actions.append(f["action"][:])

    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)
        labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        print(f"\n总步数: {len(all_actions)}")
        print(f"\n{'维度':<10} {'最小值':>10} {'最大值':>10} {'均值':>10} {'标准差':>10}")
        print("-" * 55)
        for i, label in enumerate(labels):
            print(f"{label:<10} {all_actions[:, i].min():>10.5f} {all_actions[:, i].max():>10.5f} "
                  f"{all_actions[:, i].mean():>10.5f} {all_actions[:, i].std():>10.5f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="后处理: 绝对位姿 → 增量动作")
    parser.add_argument("--data_dir", type=str, default="raw_data", help="数据目录")
    parser.add_argument("--gripper_threshold", type=float, default=0.3, help="夹爪二值化阈值")
    args = parser.parse_args()

    postprocess_all(args.data_dir, args.gripper_threshold)
