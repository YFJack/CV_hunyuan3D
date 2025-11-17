"""
高层生成流水线模块。

整体流程:
    输入 -> 预处理 -> Hunyuan3D 推理
         -> （可选精炼）-> 导出

本模块负责在 IO、预处理、主干模型、精炼与导出之间进行调度，
同时保持各组件解耦，便于未来替换或扩展。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from core.io.outputs import ensure_dir, render_preview_images, save_asset_as_mesh, save_asset_metadata
from core.models.hunyuan3d_wrapper import RawAsset, load_hunyuan3d_from_config
from core.preprocess.image_preprocess import preprocess_image_for_model
from core.preprocess.text_preprocess import apply_prompt_engineering
from core.refine.mv_refine import refine as mv_refine


class GenerationPipeline:
    """
    负责将文本或图像输入转换为 3D 资产的总控流水线。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = load_hunyuan3d_from_config(config)
        self.output_dir = ensure_dir(config.get("default_output_dir", "outputs"))

    @classmethod
    def from_config_file(cls, path: str | Path) -> "GenerationPipeline":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _run_refinement(self, asset: RawAsset) -> RawAsset:
        """
        调用可选的精炼步骤（当前仅为多视图占位实现）。
        """
        refined = mv_refine(asset, self.config)
        return refined

    def generate_from_text(
        self,
        prompt: str,
        *,
        output_name: str = "asset_from_text",
        file_format: Literal["obj", "glb"] = "obj",
    ) -> Dict[str, Any]:
        """
        完整的文本到资产生成流水线。

        返回:
            字典，包含:
                - 'asset': RawAsset 对象
                - 'mesh_path': 导出的网格文件路径
                - 'metadata_path': 元数据 JSON 路径
                - 'previews': 预览图路径列表
        """
        processed_prompt = apply_prompt_engineering(prompt, config=self.config)
        raw_asset = self.model.generate_from_text(processed_prompt)
        refined_asset = self._run_refinement(raw_asset)

        mesh_path = save_asset_as_mesh(
            refined_asset,
            self.output_dir / output_name,
            file_format=file_format,
        )
        meta_path = save_asset_metadata(refined_asset, self.output_dir, name=f"{output_name}_metadata")
        previews = render_preview_images(refined_asset, self.output_dir)

        return {
            "asset": refined_asset,
            "mesh_path": mesh_path,
            "metadata_path": meta_path,
            "previews": previews,
        }

    def generate_from_image(
        self,
        image,
        *,
        output_name: str = "asset_from_image",
        file_format: Literal["obj", "glb"] = "obj",
    ) -> Dict[str, Any]:
        """
        完整的图像到资产生成流水线。
        """
        preprocessed = preprocess_image_for_model(image, config=self.config)
        raw_asset = self.model.generate_from_image(preprocessed)
        refined_asset = self._run_refinement(raw_asset)

        mesh_path = save_asset_as_mesh(
            refined_asset,
            self.output_dir / output_name,
            file_format=file_format,
        )
        meta_path = save_asset_metadata(refined_asset, self.output_dir, name=f"{output_name}_metadata")
        previews = render_preview_images(refined_asset, self.output_dir)

        return {
            "asset": refined_asset,
            "mesh_path": mesh_path,
            "metadata_path": meta_path,
            "previews": previews,
        }


__all__ = ["GenerationPipeline"]


