"""
图像预处理工具模块。

本模块面向单张 2D 图像的轻量预处理，并预留若干扩展点，用于：
- 背景移除
- 边缘图提取
- 深度估计
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps


def resize_and_normalize(image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    对单张图像进行基础缩放与规范化。

    参数:
        image: 输入 PIL 图像。
        size: 目标尺寸 (width, height)。

    返回:
        已缩放、RGB 模式的图像，可直接送入主干模型。
    """
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, method=Image.BICUBIC)
    return image


def compute_edge_map(image: Image.Image) -> np.ndarray:
    """
    基于边缘的预处理占位实现。

    返回:
        一个非常简单的梯度幅值近似（二维数组）。

    TODO:
        替换为更可靠的边缘检测算法（例如 Canny），或
        可学习的结构/边缘提取器，用于为 3D 生成器提供先验。
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # Simple finite differences as a placeholder.
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, :-1] = arr[:, 1:] - arr[:, :-1]
    gy[:-1, :] = arr[1:, :] - arr[:-1, :]
    mag = np.sqrt(gx**2 + gy**2)
    return mag


def preprocess_image_for_model(image: Image.Image, config: Dict | None = None) -> Dict:
    """
    供流水线调用的高层图像预处理入口。

    参数:
        image: 输入 PIL 图像。
        config: 可选的预处理配置字典。

    返回:
        字典，包含：
            - 'image': 预处理后的图像
            - 'edge_map': 简单的边缘图（numpy 数组）
    """
    _ = config  # placeholder
    resized = resize_and_normalize(image)
    edges = compute_edge_map(resized)
    return {"image": resized, "edge_map": edges}


__all__ = ["resize_and_normalize", "compute_edge_map", "preprocess_image_for_model"]


