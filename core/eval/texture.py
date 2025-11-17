"""
纹理评估工具模块。

包含简单的图像级指标，如 PSNR，用于比较渲染纹理或视图
与参考图像之间的差异。
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray]


def psnr(img_a: ArrayLike, img_b: ArrayLike, max_val: float = 1.0) -> float:
    """
    计算两张图像之间的峰值信噪比（PSNR）。

    参数:
        img_a: 第一张图像，对应数值范围 [0, max_val]。
        img_b: 第二张图像，对应数值范围 [0, max_val]。
        max_val: 像素值的最大可能值。
    """
    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(f"Image shapes must match, got {a.shape} and {b.shape}")

    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")

    return float(20 * math.log10(max_val) - 10 * math.log10(mse))


__all__ = ["psnr"]


