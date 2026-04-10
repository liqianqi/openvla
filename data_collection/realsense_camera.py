"""
realsense_camera.py

Intel RealSense 相机接口。
采集 RGB 图像用于 OpenVLA 数据集。
"""

import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


class RealSenseCamera:
    """Intel RealSense RGB 相机封装"""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, serial: str = ""):
        """
        Args:
            width:  图像宽度
            height: 图像高度
            fps:    帧率
            serial: 相机序列号 (空字符串=自动选择第一个)
        """
        if not HAS_REALSENSE:
            raise ImportError(
                "pyrealsense2 未安装。请运行: pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial

        self.pipeline = None
        self.config = None
        self._started = False

    def connect(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial:
            self.config.enable_device(self.serial)

        self.config.enable_stream(
            rs.stream.color,
            self.width, self.height,
            rs.format.rgb8,
            self.fps,
        )

        self.pipeline.start(self.config)
        self._started = True

        # 丢弃前几帧 (自动曝光稳定)
        for _ in range(30):
            self.pipeline.wait_for_frames()

        print(f"[RealSense] 已连接 ({self.width}x{self.height} @ {self.fps}fps)")

    def disconnect(self):
        if self._started and self.pipeline is not None:
            self.pipeline.stop()
            self._started = False
            print("[RealSense] 已断开")

    def capture_rgb(self) -> np.ndarray:
        """
        拍摄一帧 RGB 图像。

        Returns:
            np.ndarray: shape (H, W, 3), dtype uint8, RGB 格式
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("未能获取彩色帧")
        return np.asanyarray(color_frame.get_data())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


class DummyCamera:
    """
    模拟相机，用于没有 RealSense 时的调试。
    返回随机噪声图像。
    """

    def __init__(self, width: int = 640, height: int = 480, **kwargs):
        self.width = width
        self.height = height

    def connect(self):
        print(f"[DummyCamera] 已连接 ({self.width}x{self.height})")

    def disconnect(self):
        print("[DummyCamera] 已断开")

    def capture_rgb(self) -> np.ndarray:
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
