"""
用于评估生成资产的命令行入口脚本。

该脚本示例化了几何、纹理与语义评估模块的用法。为简化流程，
它基于预先保存的数据路径，并使用占位的 CLIP 风格打分。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np

from core.eval.geometry import chamfer_distance
from core.eval.semantic import clipscore_stub
from core.eval.texture import psnr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估生成的 3D 资产。")
    parser.add_argument("--metadata", type=str, required=True, help="资产元数据 JSON 路径。")
    parser.add_argument("--ref_points", type=str, help="参考点云文件路径（.npy）。")
    parser.add_argument("--ref_images", type=str, nargs="*", help="参考图像路径列表。")
    parser.add_argument("--prompt", type=str, help="用于语义评估的原始文本提示词。")
    return parser.parse_args()


def load_points_from_metadata(meta_path: Path) -> np.ndarray:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    vertices = np.asarray(data["mesh"]["vertices"], dtype=np.float32)
    return vertices


def main() -> None:
    args = parse_args()
    meta_path = Path(args.metadata)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    generated_points = load_points_from_metadata(meta_path)

    # 几何指标: Chamfer 距离
    if args.ref_points:
        ref_points = np.load(args.ref_points).astype(np.float32)
        cd = chamfer_distance(generated_points, ref_points)
        print(f"Chamfer Distance: {cd:.6f}")

    # 纹理指标: 使用第一张参考图与第一张预览图计算 PSNR（需用户提供）
    if args.ref_images and len(args.ref_images) >= 2:
        a = imageio.imread(args.ref_images[0]).astype(np.float32) / 255.0
        b = imageio.imread(args.ref_images[1]).astype(np.float32) / 255.0
        val = psnr(a, b, max_val=1.0)
        print(f"Texture PSNR: {val:.2f} dB")

    # 语义指标: CLIPScore 风格的占位打分
    if args.prompt and args.ref_images:
        rendered_arrays: List[np.ndarray] = []
        for p in args.ref_images:
            arr = imageio.imread(p).astype(np.float32) / 255.0
            rendered_arrays.append(arr)
        score = clipscore_stub(args.prompt, rendered_arrays)
        print(f"Semantic similarity (stub CLIPScore): {score:.4f}")


if __name__ == "__main__":
    main()


