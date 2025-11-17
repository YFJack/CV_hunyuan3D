"""
语义评估工具模块（CLIP 风格相似度）。

为保持模板轻量且开箱即用，这里使用了非常简单的嵌入占位实现。
在真实系统中，本模块应包装 CLIP 或类似的视觉-语言模型来
计算 CLIPScore。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _dummy_text_embedding(text: str) -> np.ndarray:
    """
    基于哈希的简易文本嵌入占位实现。

    TODO:
        替换为真实的 CLIP 文本编码器（例如 OpenCLIP、
        HuggingFace transformers 等开源实现）。
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(256, dtype=np.float32)


def _dummy_image_embedding(image_array: np.ndarray) -> np.ndarray:
    """
    基于低维投影的简易图像嵌入占位实现。

    TODO:
        替换为真实的 CLIP 图像编码器。
    """
    flat = image_array.astype(np.float32).ravel()
    if flat.size == 0:
        return np.zeros(256, dtype=np.float32)
    # Deterministic projection via fixed random seed.
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((flat.size, 256), dtype=np.float32)
    emb = flat @ proj
    return emb.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个一维向量之间的余弦相似度。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D vectors.")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(a.dot(b) / denom)


def clipscore_stub(prompt: str, rendered_views: Iterable[np.ndarray]) -> float:
    """
    使用占位嵌入计算 CLIP 风格的语义相似度。

    参数:
        prompt: 文本提示词输入。
        rendered_views: 渲染得到的 RGB 图像数组迭代器。

    返回:
        一个标量相似度分数（[-1, 1]，余弦相似度）。

    说明:
        当前实现刻意保持轻量且确定性，但并不是实际的 CLIPScore。
        该接口可在未来无缝替换为真正的 CLIP 实现。
    """
    text_emb = _dummy_text_embedding(prompt)
    scores = []
    for img in rendered_views:
        img_emb = _dummy_image_embedding(img)
        scores.append(cosine_similarity(text_emb, img_emb))
    if not scores:
        return 0.0
    return float(np.mean(scores))


__all__ = ["cosine_similarity", "clipscore_stub"]


