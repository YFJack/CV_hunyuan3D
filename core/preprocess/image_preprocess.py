"""
图像预处理工具模块。

本模块面向单张 2D 图像的预处理，实现了官方推荐的流程：
1. 背景移除 (使用 rembg)
2. 主体裁剪与居中 (Recenter)
3. 规范化到 768x768 并填充白色背景
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2

# 尝试导入 rembg，如果未安装则在运行时报错或降级
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# 全局 session 缓存
_REMBG_SESSION = None

def get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and REMBG_AVAILABLE:
        _REMBG_SESSION = new_session()
    return _REMBG_SESSION

def remove_background(image: Image.Image) -> Image.Image:
    """
    使用 rembg 移除背景。
    """
    if not REMBG_AVAILABLE:
        print("Warning: rembg not installed. Skipping background removal.")
        return image.convert("RGBA")
    
    session = get_rembg_session()
    return remove(image, session=session)

def recenter_image(image: Image.Image, size: int = 768, border_ratio: float = 0.2) -> Image.Image:
    """
    将图像主体居中并缩放，同时保持长宽比，填充为正方形。
    参考官方 hy3dgen/shapegen/preprocessors.py 中的实现。
    """
    # 确保是 RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    np_image = np.array(image)
    
    # 提取 Alpha 通道作为 Mask
    mask = np_image[:, :, 3]
    
    # 如果全透明，直接返回白图
    if np.max(mask) == 0:
        return Image.new("RGB", (size, size), (255, 255, 255))

    # 找到包围盒
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    
    if h == 0 or w == 0:
        return Image.new("RGB", (size, size), (255, 255, 255))

    # 计算目标尺寸
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    
    h2 = int(h * scale)
    w2 = int(w * scale)
    
    # 裁剪并缩放主体
    cropped = np_image[x_min:x_max, y_min:y_max]
    resized = cv2.resize(cropped, (w2, h2), interpolation=cv2.INTER_AREA)
    
    # 创建白色背景画布
    result = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # 计算居中位置
    x_start = (size - h2) // 2
    y_start = (size - w2) // 2
    
    # 混合前景与背景
    alpha = resized[:, :, 3].astype(np.float32) / 255.0
    foreground = resized[:, :, :3].astype(np.float32)
    
    # 目标区域背景
    bg_slice = result[x_start:x_start+h2, y_start:y_start+w2].astype(np.float32)
    
    # Alpha Blending
    blended = foreground * alpha[:, :, np.newaxis] + bg_slice * (1 - alpha[:, :, np.newaxis])
    
    result[x_start:x_start+h2, y_start:y_start+w2] = blended.astype(np.uint8)
    
    return Image.fromarray(result)

def resize_and_normalize(image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    保留旧接口以兼容，但建议使用新的处理流程。
    """
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, method=Image.BICUBIC)
    return image

def compute_edge_map(image: Image.Image) -> np.ndarray:
    """
    计算边缘图。
    """
    # 简单实现，后续可替换为 Canny
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, :-1] = arr[:, 1:] - arr[:, :-1]
    gy[:-1, :] = arr[1:, :] - arr[:-1, :]
    mag = np.sqrt(gx**2 + gy**2)
    return mag

def preprocess_image_for_model(image: Image.Image, config: Dict | None = None) -> Dict:
    """
    供流水线调用的高层图像预处理入口。
    
    流程:
    1. 移除背景
    2. 居中并缩放到 768x768 (默认)
    3. 计算边缘图
    """
    _ = config  # placeholder
    
    # 1. 移除背景
    image_no_bg = remove_background(image)
    
    # 2. 居中与规范化 (默认 768)
    # 注意：官方模型通常在 768 分辨率下训练或微调效果更好
    processed_image = recenter_image(image_no_bg, size=768, border_ratio=0.2)
    
    # 3. 计算边缘图 (基于处理后的图像)
    edges = compute_edge_map(processed_image)
    
    return {"image": processed_image, "edge_map": edges}

__all__ = ["resize_and_normalize", "compute_edge_map", "preprocess_image_for_model", "remove_background", "recenter_image"]
