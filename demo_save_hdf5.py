"""
demo_save_hdf5.py

EL-A3 数据采集主入口。

用法:
  # 步骤 1: Leader-Follower 遥操作录制原始数据
  python demo_save_hdf5.py record --num 10 --instruction "pick up the red block"

  # 步骤 2: 后处理 → 绝对位姿转增量动作
  python demo_save_hdf5.py postprocess

  # 步骤 3: 验证数据
  python demo_save_hdf5.py verify

  # 调试模式 (不需要真实硬件)
  python demo_save_hdf5.py record --num 3 --dummy --instruction "pick up the red block"
"""

import os
import sys
import glob
import argparse

import h5py
import numpy as np


def cmd_record(args):
    """录制轨迹数据"""
    from data_collection.teleop_record import TeleopRecorder, RecordConfig

    config = RecordConfig(
        hz=args.hz,
        raw_data_dir=args.output_dir,
        use_dummy_camera=args.dummy,
        leader_channel=args.leader_channel,
        follower_channel=args.follower_channel,
    )

    recorder = TeleopRecorder(config)
    recorder.run(num_trajectories=args.num, default_instruction=args.instruction)

def cmd_postprocess(args):
    """后处理: 绝对位姿 → 增量动作"""
    from data_collection.postprocess import postprocess_all

    postprocess_all(args.data_dir, args.gripper_threshold)


def cmd_verify(args):
    """验证 HDF5 数据集"""
    pattern = os.path.join(args.data_dir, "trajectory_*.hdf5")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"在 {args.data_dir}/ 中未找到轨迹文件")
        return

    print(f"找到 {len(files)} 个轨迹文件\n")

    total_steps = 0
    has_action = True

    for filepath in files:
        with h5py.File(filepath, "r") as f:
            num_steps = f.attrs.get("num_steps", "?")
            instruction = f.attrs.get("language_instruction", "?")
            hz = f.attrs.get("hz", "?")

            has_img = "observation/image" in f
            has_act = "action" in f
            has_eef = "eef_poses" in f

            if has_img:
                img_shape = f["observation/image"].shape
            else:
                img_shape = "N/A"

            if has_act:
                act_shape = f["action"].shape
            else:
                act_shape = "N/A"
                has_action = False

            if has_eef:
                eef_shape = f["eef_poses"].shape
            else:
                eef_shape = "N/A"

            total_steps += num_steps if isinstance(num_steps, int) else 0

        filename = os.path.basename(filepath)
        print(f"{filename}:")
        print(f"  指令:     \"{instruction}\"")
        print(f"  步数:     {num_steps} @ {hz} Hz")
        print(f"  图像:     {img_shape}")
        print(f"  动作:     {act_shape}")
        print(f"  位姿:     {eef_shape}")
        print()

    print("=" * 50)
    print(f"总轨迹数: {len(files)}")
    print(f"总步数:   {total_steps}")
    if not has_action:
        print("\n注意: 部分文件还没有 action 字段，请先运行 postprocess 命令!")

    # 详细查看第一个文件
    if files and args.detail:
        print(f"\n{'='*50}")
        print(f"详细查看: {os.path.basename(files[0])}")
        print(f"{'='*50}")
        with h5py.File(files[0], "r") as f:
            def print_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name:30s}  shape={obj.shape}  dtype={obj.dtype}")
            f.visititems(print_item)

            for key in f.attrs:
                print(f"  attr: {key} = {f.attrs[key]}")

            if "action" in f:
                actions = f["action"][:]
                print(f"\n  前 5 个 action:")
                labels = ["dx", "dy", "dz", "dr", "dp", "dy", "grip"]
                print(f"  {'step':>4s}  " + "  ".join(f"{l:>8s}" for l in labels))
                for t in range(min(5, len(actions))):
                    vals = "  ".join(f"{v:>8.4f}" for v in actions[t])
                    print(f"  {t:>4d}  {vals}")


def main():
    parser = argparse.ArgumentParser(
        description="EL-A3 OpenVLA 数据采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 录制 10 条轨迹
  python demo_save_hdf5.py record --num 10 --instruction "pick up the red block"

  # 后处理
  python demo_save_hdf5.py postprocess

  # 验证
  python demo_save_hdf5.py verify --detail

  # 调试模式 (无需硬件)
  python demo_save_hdf5.py record --num 3 --dummy --instruction "test"
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # record
    p_record = subparsers.add_parser("record", help="Leader-Follower 遥操作录制")
    p_record.add_argument("--num", type=int, default=10, help="录制轨迹数量")
    p_record.add_argument("--hz", type=float, default=5.0, help="采集频率 (Hz)")
    p_record.add_argument("--instruction", type=str, default="", help="默认语言指令")
    p_record.add_argument("--output_dir", type=str, default="raw_data", help="输出目录")
    p_record.add_argument("--leader_channel", type=str, default="can0", help="Leader CAN 通道")
    p_record.add_argument("--follower_channel", type=str, default="can1", help="Follower CAN 通道")
    p_record.add_argument("--dummy", action="store_true", help="调试模式 (假相机+假电机)")

    # postprocess
    p_post = subparsers.add_parser("postprocess", help="后处理: 绝对位姿 → 增量动作")
    p_post.add_argument("--data_dir", type=str, default="raw_data", help="数据目录")
    p_post.add_argument("--gripper_threshold", type=float, default=0.3, help="夹爪二值化阈值")

    # verify
    p_verify = subparsers.add_parser("verify", help="验证数据集")
    p_verify.add_argument("--data_dir", type=str, default="raw_data", help="数据目录")
    p_verify.add_argument("--detail", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args)
    elif args.command == "postprocess":
        cmd_postprocess(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
