"""
几何评估工具模块。

用于计算几何质量指标，例如生成形状与参考形状之间的
Chamfer 距离等。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    计算两个点云之间的对称 Chamfer 距离（简化实现）。

    参数:
        points_a: 形状为 (Na, 3) 的 float32 数组。
        points_b: 形状为 (Nb, 3) 的 float32 数组。

    返回:
        标量 Chamfer 距离。

    说明:
        为了简洁，这里采用了最直接的实现方式，若需要可
        替换为更高效的批量版本。
    """
    points_a = np.asarray(points_a, dtype=np.float32)
    points_b = np.asarray(points_b, dtype=np.float32)

    if points_a.size == 0 or points_b.size == 0:
        return float("inf")

    # (Na, Nb, 3)
    diff = points_a[:, None, :] - points_b[None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    a_to_b = dist_sq.min(axis=1).mean()
    b_to_a = dist_sq.min(axis=0).mean()
    return float(a_to_b + b_to_a)


__all__ = ["chamfer_distance"]


