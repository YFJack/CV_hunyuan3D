"""
3D 资产生成系统的输入工具模块。

本模块保持极简，只负责：
- 文本提示词读取
- 单张图像加载

设计上尽量轻量，以便后续轻松扩展更复杂的输入形式
（例如多视角图像、相机位姿等），而无需修改对外接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


def load_text_prompt(prompt: Optional[str] = None, *, prompt_file: Optional[str] = None) -> str:
    """
    从字符串或文件中读取文本提示词。

    参数:
        prompt: 用户直接传入的提示词字符串。
        prompt_file: 包含提示词的文本文件路径。

    返回:
        处理后的提示词字符串。
    """
    if prompt is not None:
        return prompt.strip()

    if prompt_file is None:
        raise ValueError("Either `prompt` or `prompt_file` must be provided.")

    text = Path(prompt_file).read_text(encoding="utf-8")
    return text.strip()


def load_single_image(image_path: str) -> Image.Image:
    """
    从给定路径加载单张 RGB 图像。

    参数:
        image_path: 图像文件路径。

    返回:
        RGB 模式的 PIL Image 对象。
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert("RGB")
    return img


__all__ = ["load_text_prompt", "load_single_image"]


