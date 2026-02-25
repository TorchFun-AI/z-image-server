<div align="center">

# ⚡ Z-Image-Turbo

[English](README.en.md) | 简体中文

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Model%20License-red)]()

**高效图像生成基础模型 | 8步推理 | 支持中英双语**

</div>

## 📖 简介

Z-Image-Turbo 是一个高效的图像生成基础模型，基于先进的扩散模型架构优化。该模型通过解耦蒸馏技术（Decoupled Model Distillation with Replay）实现了**仅需8步推理**即可生成高质量图像，推理速度比传统扩散模型提升数倍。

### 核心特点

- ⚡ **极速推理**: 仅需8步DiT前向传播即可生成高质量图像
- 🌐 **双语支持**: 原生支持中文和英文提示词
- 🎨 **高质量输出**: 在多种图像风格上表现优异
- 🚀 **易于部署**: 基于 Diffusers 库，API 服务一键启动
- 💾 **内存优化**: 支持 CPU 卸载、BF16精度等内存节省模式

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 模型参数 | 6B 总参数（3.7B DiT + 1.7B Text Encoder） |
| 推理步数 | 默认 9 步（8次 DiT 前向 + 1次初始处理） |
| 支持分辨率 | 256x256 至 2048x2048 |
| 数据类型 | FP16 / BF16 / FP32 |
| 特殊要求 | guidance_scale 必须设为 0（Turbo 模型特性） |

## 🖥️ 环境要求

### 最低配置

| 组件 | 要求 |
|------|------|
| Python | 3.10 或更高 |
| CUDA | 12.1 或更高 |
| PyTorch | 2.0.0 或更高 |
| 显存 | 16GB（BF16 模式）|

### 推荐配置

| 组件 | 推荐 |
|------|------|
| GPU | NVIDIA RTX 4090 / A100 / H100 |
| 显存 | 24GB+ |
| 内存 | 32GB+ |
| 存储 | 100GB+ 可用空间（模型文件约 32GB）|

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone <repository-url>
cd z-image-server
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r server/requirements.txt
```

### 3. 下载模型权重

模型权重文件通过 Git LFS 管理。如果本地仓库中的权重文件是 LFS 指针（约135字节），请从源目录复制实际文件：

```bash
# 示例：从源目录复制（根据您的实际环境调整路径）
cp /path/to/source/Z-Image-Turbo/transformer/*.safetensors Z-Image-Turbo/transformer/
cp /path/to/source/Z-Image-Turbo/text_encoder/*.safetensors Z-Image-Turbo/text_encoder/
cp /path/to/source/Z-Image-Turbo/vae/*.safetensors Z-Image-Turbo/vae/
cp /path/to/source/Z-Image-Turbo/tokenizer/* Z-Image-Turbo/tokenizer/
cp /path/to/source/Z-Image-Turbo/assets/* Z-Image-Turbo/assets/
```

### 4. 启动服务

```bash
# 使用启动脚本
./start.sh

# 或手动启动
cd server
python api_server.py
```

服务启动后，可以通过以下地址访问：
- API 服务: http://localhost:8002
- WebUI 界面: http://localhost:8002/ui
- API 文档: http://localhost:8002/docs

## 📡 部署指南

### API 服务部署 (api_server.py)

API 服务器提供 HTTP RESTful API 接口，支持文本生成图像。

#### 环境变量配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DEVICE` | `cuda:0` | 运行设备，可选 `cuda:0`, `cuda:1`, `cpu` |
| `DTYPE` | `bfloat16` | 数据类型，可选 `float16`, `bfloat16`, `float32` |
| `PORT` | `8002` | 服务端口 |
| `COMPILE_MODEL` | `false` | 是否启用模型编译（首次推理较慢，后续更快） |

#### 启动示例

```bash
# 使用 FP16 精度（节省显存）
DTYPE=float16 python api_server.py

# 指定设备为第二张显卡
DEVICE=cuda:1 python api_server.py

# 启用模型编译
COMPILE_MODEL=true python api_server.py

# 修改端口
PORT=8080 python api_server.py

# 组合使用
DTYPE=float16 PORT=8080 python api_server.py
```

### 命令行使用 (generate.py)

命令行工具用于直接生成图像，无需启动服务。

```bash
cd server

# 基础用法
python generate.py --prompt "一只可爱的猫咪，戴着红色蝴蝶结"

# 指定参数
python generate.py \
  --prompt "年轻的中国女性，穿着红色汉服，精致刺绣，霓虹灯背景" \
  --width 1024 \
  --height 1024 \
  --steps 9 \
  --seed 42 \
  --output output.png \
  --dtype bfloat16

# 使用 FP16 节省显存
python generate.py --prompt "山水风景画" --dtype float16

# 启用模型编译（提速）
python generate.py --prompt "未来城市夜景" --compile

# 低显存模式（CPU 卸载）
python generate.py --prompt "抽象艺术作品" --cpu_offload
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | 必填 | 图像生成提示词 |
| `--width` | 1024 | 图像宽度（256-2048）|
| `--height` | 1024 | 图像高度（256-2048）|
| `--steps` | 9 | 推理步数（推荐 9，即 8 次 DiT 前向）|
| `--seed` | 42 | 随机种子 |
| `--output` | output.png | 输出文件路径 |
| `--model_path` | ../Z-Image-Turbo | 模型权重路径 |
| `--dtype` | bfloat16 | 数据类型 |
| `--compile` | false | 启用模型编译 |
| `--cpu_offload` | false | 启用 CPU 卸载（低显存模式）|

## 📚 API 文档

### 端点说明

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET /` | GET | 服务状态检查 |
| `GET /health` | GET | 健康检查 |
| `POST /v1/text2image` | POST | 文本生成图像 |
| `GET /ui` | GET | WebUI 界面 |
| `GET /docs` | GET | Swagger API 文档 |

### 请求/响应示例

#### 生成图像 (POST /v1/text2image)

**请求体:**

```json
{
  "prompt": "一只可爱的猫咪，戴着红色蝴蝶结",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 9,
  "seed": 42
}
```

**响应:**

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAABQ...",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "prompt": "一只可爱的猫咪，戴着红色蝴蝶结",
  "generation_time": 2.34
}
```

#### cURL 示例

```bash
curl -X POST "http://localhost:8002/v1/text2image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一只可爱的猫咪，戴着红色蝴蝶结",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 9,
    "seed": 42
  }'
```

#### Python 示例

```python
import requests
import base64
from PIL import Image
import io

# 发送请求
response = requests.post(
    "http://localhost:8002/v1/text2image",
    json={
        "prompt": "一只可爱的猫咪，戴着红色蝴蝶结",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 9,
        "seed": 42
    }
)

# 解析响应
data = response.json()
image_base64 = data["image_base64"]

# 保存图像
image_bytes = base64.b64decode(image_base64)
image = Image.open(io.BytesIO(image_bytes))
image.save("output.png")

print(f"生成时间: {data['generation_time']:.2f}秒")
print(f"种子: {data['seed']}")
```

## ⚡ 性能优化

### 1. 数据类型选择

| 数据类型 | 显存占用 | 速度 | 质量 | 推荐场景 |
|----------|----------|------|------|----------|
| `float32` | 最高 | 慢 | 最佳 | 追求质量 |
| `bfloat16` | 中等 | 快 | 优秀 | **推荐默认** |
| `float16` | 最低 | 最快 | 良好 | 显存受限 |

### 2. 模型编译

启用 `COMPILE_MODEL=true` 可以在首次推理后显著提升速度（可能提升 20-50%），但首次推理会更慢。

```bash
# 首次启动较慢，后续推理更快
COMPILE_MODEL=true python api_server.py
```

### 3. CPU 卸载模式

对于显存 < 16GB 的设备，可以启用 CPU 卸载：

```bash
# 命令行工具
python generate.py --prompt "..." --cpu_offload

# 修改 api_server.py 添加启用逻辑（需要自行修改代码）
# pipe.enable_model_cpu_offload()
```

### 4. 显存占用参考

| 配置 | 显存占用 | 适合 GPU |
|------|----------|----------|
| BF16 标准模式 | ~16GB | RTX 4090 (24GB) |
| FP16 标准模式 | ~14GB | RTX 3090 (24GB) |
| BF16 + CPU 卸载 | ~8GB | RTX 3070 (8GB) |

## 📂 目录结构

```
z-image-server/
├── README.md                    # 中文文档（本文件）
├── README.en.md                 # 英文文档
├── .gitignore                   # Git 忽略规则
├── start.sh                     # 启动脚本
├── server/                      # 服务端代码
│   ├── api_server.py           # API 服务主文件
│   ├── generate.py             # 命令行工具
│   └── requirements.txt        # Python 依赖
└── Z-Image-Turbo/              # 模型目录
    ├── transformer/            # DiT 模型权重
    ├── text_encoder/           # 文本编码器权重
    ├── vae/                    # VAE 权重
    ├── tokenizer/              # 分词器文件
    ├── assets/                 # 资源文件
    ├── model_index.json        # 模型索引
    └── configuration.json      # 配置文件
```

## ❓ 常见问题

### Q: 启动时出现 OOM（显存不足）错误

**A:** 尝试以下方法：
1. 使用 FP16 精度：`DTYPE=float16 python api_server.py`
2. 启用 CPU 卸载（需要修改代码添加 `pipe.enable_model_cpu_offload()`）
3. 降低生成分辨率：使用 512x512 而非 1024x1024
4. 关闭其他占用显存的程序

### Q: 首次生成图像很慢

**A:** 这是正常现象。首次推理需要加载模型到显存并进行初始化。如果启用了模型编译，首次会额外慢一些。

### Q: 生成的图像质量不佳

**A:** 检查以下几点：
1. 确保使用推荐的 9 步推理
2. guidance_scale 必须设为 0（Turbo 模型特性，不可更改）
3. 尝试改进提示词，使用更详细的描述
4. 确保模型权重文件完整（检查文件大小）

### Q: 提示词应该用什么语言？

**A:** Z-Image-Turbo 原生支持中英文，可以使用：
- 中文："一只可爱的猫咪，戴着红色蝴蝶结"
- 英文："A cute cat wearing a red bow tie"
- 中英文混合也可以

### Q: 如何检查模型文件是否完整？

**A:** 检查关键文件大小：

```bash
# Transformer 权重（约 24.6GB）
ls -lh Z-Image-Turbo/transformer/*.safetensors

# Text Encoder 权重（约 8GB）
ls -lh Z-Image-Turbo/text_encoder/*.safetensors

# VAE 权重（约 170MB）
ls -lh Z-Image-Turbo/vae/*.safetensors

# Tokenizer（约 17MB）
ls -lh Z-Image-Turbo/tokenizer/vocab.json
```

如果文件只有约 135 字节，说明是 Git LFS 指针文件，需要下载实际文件。

## 📄 许可证

本项目遵循 Z-Image-Turbo 模型的原始许可证。请在使用前查看模型许可证文件。

## 🙏 致谢

- 基于 [Diffusers](https://github.com/huggingface/diffusers) 库构建
- 使用 [FastAPI](https://fastapi.tiangolo.com/) 提供 API 服务
- Z-Image-Turbo 模型由原始作者开发

---

<div align="center">

**Made with ❤️ for efficient image generation**

</div>
