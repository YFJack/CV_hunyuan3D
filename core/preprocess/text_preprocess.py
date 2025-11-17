"""
文本预处理工具模块。

本模块保持极简，但预留了清晰的扩展接口，用于：
- 提示词工程
- 词元级变换
- 基于模板的提示词构造
"""

from __future__ import annotations

from typing import Dict


def basic_clean(prompt: str) -> str:
    """
    对文本提示词做基础清洗。

    当前仅进行非常轻量的规范化；未来可以扩展为：
    - 自动添加风格修饰词
    - 安全/敏感内容过滤
    - 任务相关的模板化提示词
    """
    return " ".join(prompt.strip().split())


def apply_prompt_engineering(prompt: str, config: Dict | None = None) -> str:
    """
    提示词工程扩展入口。

    参数:
        prompt: 用户原始提示词。
        config: 可选的提示词策略配置字典。

    返回:
        可能被改写后的提示词字符串。

    TODO:
        实现更高级的策略，例如：
        - 自动添加相机/光照等描述
        - 材质与风格增强
        - 基于 CLIP 的多候选提示词重排序
    """
    _ = config  # placeholder
    return basic_clean(prompt)


__all__ = ["basic_clean", "apply_prompt_engineering"]


