## 基于 Hunyuan3D-2 的 3D 资产生成模板

本项目是一个**最小化、模块化且易扩展**的代码模板，用于封装
开源的 **Hunyuan3D-2** 模型，实现 3D 资产生成。

整体流水线为：

> 输入 → 预处理 → Hunyuan3D构建3d → 渲染纹理 → （可选精炼）→ 导出

限于原有hunyuan项目限制，目前仅支持 **单张图像** 两种输入形式，可输出可用的
3D 网格（`.obj` 或 `.glb`），并预留预览图渲染接口。
目前调用的情况来看，gpu构建3d很快，但是渲染纹理阶段由于项目采用的算子大都是用cpu的，因此会较慢，默认设置下约20分钟（权重会存在显存，但gpu利用率为0）
---

### 项目结构

- `configs/hunyuan3d_default.yaml` – 基础配置（模型路径、设备、采样步数、默认输出目录）。
- `core/io/inputs.py` – 文本提示词与图像的读取工具。
- `core/io/outputs.py` – 网格/元数据的保存与预览渲染占位实现。
- `core/preprocess/text_preprocess.py` – 文本清洗与提示词工程扩展点。
- `core/preprocess/image_preprocess.py` – 图像缩放与边缘图占位实现。
- `core/models/hunyuan3d_wrapper.py` – `RawAsset` 定义与 Hunyuan3D 模型封装。
- `core/pipeline/generation_pipeline.py` – 负责完整生成流水线的调度。
- `core/refine/mv_refine.py` – 多视图一致性精炼模块占位实现。
- `core/eval/geometry.py` – Chamfer 距离几何评估工具。
- `core/eval/texture.py` – PSNR 纹理评估工具。
- `core/eval/semantic.py` – CLIP 风格语义相似度占位实现。
- `scripts/run_text2asset.py` – 文本 → 3D 资产的命令行脚本。
- `scripts/run_image2asset.py` – 图像 → 3D 资产的命令行脚本。
- `scripts/evaluate_assets.py` – 资产评估命令行脚本。

---

### 安装与依赖

建议创建虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

最小依赖（详见 `requirements.txt`）：

- `numpy`
- `Pillow`
- `PyYAML`
- `trimesh`
- `imageio`

> **注意**：`requirements.txt` 仅包含与模板逻辑相关的轻量依赖。
> Hunyuan3D-2 主干、PyTorch、`hy3dgen` 以及纹理渲染所需的原生模块
> 需要在官方仓库中单独安装（见下一小节）。

### 在 autodl（5090）上准备官方 Hunyuan3D-2 依赖

1. **准备虚拟环境与 PyTorch**
   ```bash
   conda create -n hy3d python=3.10 -y
   conda activate hy3d
   # 根据 5090 驱动选择合适的 CUDA 版本，下例假设 CUDA 12.4
   pip install torch==2.3.1+cu124 torchvision==0.18.1+cu124 \
       --extra-index-url https://download.pytorch.org/whl/cu124
   ```
   若在autodl上复现项目可略过此步（强烈建议autodl或者其他服务器平台，该项目本身在特定cuda版本时会有报错）
2. **克隆并安装官方 Hunyuan3D-2 仓库**（位于 `~/work/Hunyuan3D-2` 等任意路径）
   ```bash
   git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
   cd Hunyuan3D-2
   pip install -r requirements.txt
   pip install -e .
   ```
3. **纹理管线（Hunyuan3D-Paint），编译自定义模块**
注意，需要先确认硬件的sm版本，可运行
python - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)
PY
确认后，根据终端输出的版本，例如12 0，运行下面的编译
   ```bash
   export TORCH_CUDA_ARCH_LIST="12.0"  
   cd hy3dgen/texgen/custom_rasterizer
   python setup.py install
   cd ../../..
   cd hy3dgen/texgen/differentiable_renderer
   python setup.py install
   cd ../../..
   ```

4. （可选）设置 HuggingFace 访问令牌，以便自动下载官方权重：
   ```bash
   export HUGGING_FACE_HUB_TOKEN=xxxxxxxx
   ```
   推荐魔塔社区下载，huggingface下载连接不稳定
5. 返回本模板根目录，执行 `pip install -r requirements.txt`（如上）。

> 更多官方安装方法与模型列表参见
> [Hunyuan3D-2 官方仓库](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)。

---

### 配置

默认配置位于 `configs/hunyuan3d_default.yaml`：

- `model_path` / `model_subfolder`: HuggingFace repo id 或本地路径以及子目录（如 `hunyuan3d-dit-v2-0`）。
- `device`: 计算设备，例如 `"cuda"`。
- `torch_dtype`: 将 pipeline 转到 `float16` / `bfloat16` / `float32` 的快捷方式。
- `sampling_steps`: 主干推理采样步数。
- `default_output_dir`: 网格与元数据的输出目录。
- `enable_texture`: 是否在生成完 mesh 后调用 Hunyuan3D-Paint。
- `texture_model_path` / `texture_subfolder`: 纹理模型来源（默认仍指向 `tencent/Hunyuan3D-2` 内的 `hunyuan3d-paint-v2-0`）。
- `texture_reference_field`: 纹理阶段默认使用的参考图像键，图像输入场景下默认为预处理后的 `"image"`。
- `low_vram_mode` / `enable_flashvdm`: 映射到官方 pipeline 的同名开关（适用于turbo模型，其他开关影响不大）。
- `hf_auth_token`: 若你需要访问私有模型，可在此填入或依赖环境变量。

你可以根据实验需要新增 YAML 配置文件，进行不同实验设置。

---

### 运行文本 → 3D 资产（尚未实现，需要文生图模型）

在项目根目录下执行：

```bash
python scripts/run_text2asset.py --prompt "a sci-fi chair" \
    --config configs/hunyuan3d_default.yaml \
    --format obj \
    --name scifi_chair
```

或使用外部提示词文件：

```bash
python scripts/run_text2asset.py --prompt_file prompt.txt
```

输出文件：

- `outputs/scifi_chair.obj` – 导出的网格文件。
- `outputs/scifi_chair_metadata.json` – 含 `RawAsset` 字段的元数据。

---

### 运行图像 → 3D 资产

```bash
python scripts/run_image2asset.py --image path/to/input.png \
    --config configs/hunyuan3d_default.yaml \
    --format glb \
    --name chair_from_image
```

输出文件：

- `outputs/chair_from_image.glb` – 导出的网格文件，若开启纹理会自动绑定贴图。
- `outputs/chair_from_image_albedo.png` – 纹理阶段导出的主贴图（仅在 `enable_texture: true` 时生成）。
- `outputs/chair_from_image_metadata.json` – 元数据，仅包含纹理文件路径与形状信息，不再内嵌整幅贴图数据。

---

### 评估生成的资产

评估脚本展示了如何挂接以下模块：

- **几何**：Chamfer 距离（生成点云 vs 参考点云）。
- **纹理**：PSNR（两张图像之间的质量比较）。
- **语义**：CLIP 风格相似度（占位实现，比较提示词与一个或多个图像）。

示例：

```bash
python scripts/evaluate_assets.py \
    --metadata outputs/scifi_chair_metadata.json \
    --ref_points data/reference_chair_points.npy \
    --ref_images data/render1.png data/render2.png \
    --prompt "a sci-fi chair"
```

---

### 创新扩展点

本模板在设计时刻意模块化，方便在不重构核心逻辑的前提下插入
新的研究思路。

- **提示词工程** – 扩展 `core/preprocess/text_preprocess.py`：
  - 在 `apply_prompt_engineering` 中实现更丰富的策略（模板、风格修饰、CLIP 重排序等）。

- **图像预处理** – 扩展 `core/preprocess/image_preprocess.py`：
  - 挂接背景移除、深度图、更加鲁棒的边缘检测等。
  - 通过统一入口 `preprocess_image_for_model` 集中管理预处理逻辑。

- **多视图纹理一致性** – 扩展 `core/refine/mv_refine.py`：
  - 实现多视图渲染 + 纹理优化。
  - 引入可微渲染与光度/感知损失。

- **高斯 ↔ 网格表示** – 扩展 `core/models/hunyuan3d_wrapper.py`：
  - 实现 `convert_mesh_to_gaussians`，输出 3D 高斯参数。
  - 可选添加反向转换，支持在高斯空间中编辑与优化。

- **评估模块** – 扩展 `core/eval/*`：
  - 将 `clipscore_stub` 替换为真实的 CLIP 模型。
  - 增加更多几何指标（如法线一致性、IoU 等）。
  - 增加更多纹理指标（如 SSIM 等）。

这些扩展都可以在**不修改命令行脚本和主流水线 API**的前提下完成，
从而保持实验过程简洁高效。

---

### 将占位实现替换为真实 Hunyuan3D-2

在 `core/models/hunyuan3d_wrapper.py` 中：

- 使用真实的 Hunyuan3D-2 推理代码替换 `_dummy_mesh`。
- 在 `Hunyuan3DModel.__init__` 中实现真实的模型加载逻辑。
- 在以下函数中填充真实推理：
  - `Hunyuan3DModel.generate_from_text`
  - `Hunyuan3DModel.generate_from_image`

只要这些函数能够返回包含有效 `mesh` 的 `RawAsset`，其余系统
（流水线、精炼、导出、评估）即可无缝工作。


