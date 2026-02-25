<div align="center">

# ⚡ Z-Image-Turbo

English | [简体中文](README.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Model%20License-red)]()

**Efficient Image Generation Foundation Model | 8-Step Inference | Bilingual Support (Chinese & English)**

</div>

## 📖 Introduction

Z-Image-Turbo is an efficient image generation foundation model optimized based on advanced diffusion model architecture. Through Decoupled Model Distillation with Replay (DMDR) technology, this model achieves **high-quality image generation with only 8 inference steps**, offering several times faster inference speed compared to traditional diffusion models.

### Key Features

- ⚡ **Lightning-fast Inference**: Generate high-quality images with only 8 DiT forward passes
- 🌐 **Bilingual Support**: Native support for both Chinese and English prompts
- 🎨 **High-quality Output**: Excellent performance across various image styles
- 🚀 **Easy Deployment**: Based on Diffusers library, one-click API service startup
- 💾 **Memory Optimization**: Supports CPU offloading, BF16 precision, and other memory-saving modes

## ✨ Features

| Feature | Description |
|---------|-------------|
| Model Parameters | 6B total (3.7B DiT + 1.7B Text Encoder) |
| Inference Steps | Default 9 steps (8 DiT forwards + 1 initial processing) |
| Supported Resolutions | 256x256 to 2048x2048 |
| Data Types | FP16 / BF16 / FP32 |
| Special Requirement | guidance_scale must be set to 0 (Turbo model characteristic) |

## 🖥️ Requirements

### Minimum Configuration

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 or higher |
| CUDA | 12.1 or higher |
| PyTorch | 2.0.0 or higher |
| GPU Memory | 16GB (BF16 mode) |

### Recommended Configuration

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA RTX 4090 / A100 / H100 |
| GPU Memory | 24GB+ |
| RAM | 32GB+ |
| Storage | 100GB+ free space (model files ~32GB) |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd z-image-server
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r server/requirements.txt
```

### 3. Download Model Weights

This project uses [Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) as the base model. Due to the large model size (~32GB), please obtain the weights using one of the following methods:

#### Method 1: Download from ModelScope (Recommended)

Visit the official model page: https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo

```bash
# Install ModelScope CLI
pip install modelscope

# Download the full model
modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir ./Z-Image-Turbo
```

Or clone using Git LFS:

```bash
# Ensure Git LFS is installed
git lfs install

# Clone the model repository
git clone https://www.modelscope.cn/Tongyi-MAI/Z-Image-Turbo.git

# Move the downloaded model to your project directory
mv Z-Image-Turbo /path/to/your/project/
```

#### Method 2: Manual Download

Manually download the following files from the ModelScope page and place them in the corresponding directories:

| File Path | Description | Size |
|-----------|-------------|------|
| `transformer/diffusion_pytorch_model*.safetensors` | DiT model weights | ~24.6 GB |
| `text_encoder/model*.safetensors` | Text encoder weights | ~8 GB |
| `vae/diffusion_pytorch_model*.safetensors` | VAE weights | ~170 MB |
| `tokenizer/*` | Tokenizer files | ~17 MB |
| `assets/*` | Asset files | ~1 MB |
| `model_index.json` | Model index | ~1 KB |
| `configuration.json` | Configuration file | ~1 KB |

#### Method 3: Use Existing Copy

If you already have a local copy of the Z-Image-Turbo model, simply copy it to the project root:

```bash
cp -r /path/to/your/Z-Image-Turbo ./Z-Image-Turbo
```

#### Verify Download

After downloading, verify the key files are complete:

```bash
# Check file sizes (reference)
ls -lh Z-Image-Turbo/transformer/*.safetensors  # Should be ~24.6GB
ls -lh Z-Image-Turbo/text_encoder/*.safetensors # Should be ~8GB
ls -lh Z-Image-Turbo/vae/*.safetensors          # Should be ~170MB

# If files are only ~135 bytes, they are Git LFS pointers and need to be re-downloaded
```

### 4. Start the Service

```bash
# Using startup script
./start.sh

# Or start manually
cd server
python api_server.py
```

Once the service is running, access it at:
- API Service: http://localhost:8002
- WebUI Interface: http://localhost:8002/ui
- API Documentation: http://localhost:8002/docs

## 📡 Deployment Guide

### API Service Deployment (api_server.py)

The API server provides HTTP RESTful API interfaces for text-to-image generation.

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda:0` | Device to run on, options: `cuda:0`, `cuda:1`, `cpu` |
| `DTYPE` | `bfloat16` | Data type, options: `float16`, `bfloat16`, `float32` |
| `PORT` | `8002` | Service port |
| `COMPILE_MODEL` | `false` | Enable model compilation (slower first inference, faster subsequent) |

#### Startup Examples

```bash
# Use FP16 precision (save VRAM)
DTYPE=float16 python api_server.py

# Use second GPU
DEVICE=cuda:1 python api_server.py

# Enable model compilation
COMPILE_MODEL=true python api_server.py

# Change port
PORT=8080 python api_server.py

# Combined usage
DTYPE=float16 PORT=8080 python api_server.py
```

### Command Line Usage (generate.py)

The command-line tool is used for direct image generation without starting a service.

```bash
cd server

# Basic usage
python generate.py --prompt "A cute cat wearing a red bow tie"

# With specific parameters
python generate.py \
  --prompt "Young Chinese woman wearing red Hanfu, intricate embroidery, neon background" \
  --width 1024 \
  --height 1024 \
  --steps 9 \
  --seed 42 \
  --output output.png \
  --dtype bfloat16

# Use FP16 to save VRAM
python generate.py --prompt "Landscape painting" --dtype float16

# Enable model compilation (speed boost)
python generate.py --prompt "Futuristic city at night" --compile

# Low VRAM mode (CPU offloading)
python generate.py --prompt "Abstract art piece" --cpu_offload
```

#### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | Required | Image generation prompt |
| `--width` | 1024 | Image width (256-2048) |
| `--height` | 1024 | Image height (256-2048) |
| `--steps` | 9 | Inference steps (recommended 9, i.e., 8 DiT forwards) |
| `--seed` | 42 | Random seed |
| `--output` | output.png | Output file path |
| `--model_path` | ../Z-Image-Turbo | Path to model weights |
| `--dtype` | bfloat16 | Data type |
| `--compile` | false | Enable model compilation |
| `--cpu_offload` | false | Enable CPU offloading (low VRAM mode) |

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Service status check |
| `GET /health` | GET | Health check |
| `POST /v1/text2image` | POST | Text to image generation |
| `GET /ui` | GET | WebUI interface |
| `GET /docs` | GET | Swagger API documentation |

### Request/Response Examples

#### Generate Image (POST /v1/text2image)

**Request Body:**

```json
{
  "prompt": "A cute cat wearing a red bow tie",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 9,
  "seed": 42
}
```

**Response:**

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAABQ...",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "prompt": "A cute cat wearing a red bow tie",
  "generation_time": 2.34
}
```

#### cURL Example

```bash
curl -X POST "http://localhost:8002/v1/text2image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute cat wearing a red bow tie",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 9,
    "seed": 42
  }'
```

#### Python Example

```python
import requests
import base64
from PIL import Image
import io

# Send request
response = requests.post(
    "http://localhost:8002/v1/text2image",
    json={
        "prompt": "A cute cat wearing a red bow tie",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 9,
        "seed": 42
    }
)

# Parse response
data = response.json()
image_base64 = data["image_base64"]

# Save image
image_bytes = base64.b64decode(image_base64)
image = Image.open(io.BytesIO(image_bytes))
image.save("output.png")

print(f"Generation time: {data['generation_time']:.2f}s")
print(f"Seed: {data['seed']}")
```

## ⚡ Performance Optimization

### 1. Data Type Selection

| Data Type | VRAM Usage | Speed | Quality | Recommended Scenario |
|-----------|------------|-------|---------|---------------------|
| `float32` | Highest | Slow | Best | Quality-focused |
| `bfloat16` | Medium | Fast | Excellent | **Recommended default** |
| `float16` | Lowest | Fastest | Good | VRAM-constrained |

### 2. Model Compilation

Enabling `COMPILE_MODEL=true` can significantly improve speed after the first inference (20-50% improvement possible), but the first inference will be slower.

```bash
# Slower first startup, faster subsequent inferences
COMPILE_MODEL=true python api_server.py
```

### 3. CPU Offload Mode

For devices with < 16GB VRAM, enable CPU offloading:

```bash
# Command-line tool
python generate.py --prompt "..." --cpu_offload

# Modify api_server.py to add enable logic (requires code modification)
# pipe.enable_model_cpu_offload()
```

### 4. VRAM Usage Reference

| Configuration | VRAM Usage | Suitable GPU |
|---------------|------------|--------------|
| BF16 Standard | ~16GB | RTX 4090 (24GB) |
| FP16 Standard | ~14GB | RTX 3090 (24GB) |
| BF16 + CPU Offload | ~8GB | RTX 3070 (8GB) |

## 📂 Directory Structure

```
z-image-server/
├── README.md                    # Chinese documentation
├── README.en.md                 # English documentation (this file)
├── .gitignore                   # Git ignore rules
├── start.sh                     # Startup script
├── server/                      # Server code
│   ├── api_server.py           # API service main file
│   ├── generate.py             # Command-line tool
│   └── requirements.txt        # Python dependencies
└── Z-Image-Turbo/              # Model directory
    ├── transformer/            # DiT model weights
    ├── text_encoder/           # Text encoder weights
    ├── vae/                    # VAE weights
    ├── tokenizer/              # Tokenizer files
    ├── assets/                 # Asset files
    ├── model_index.json        # Model index
    └── configuration.json      # Configuration file
```

## ❓ FAQ

### Q: OOM (Out of Memory) error at startup

**A:** Try the following methods:
1. Use FP16 precision: `DTYPE=float16 python api_server.py`
2. Enable CPU offloading (requires code modification to add `pipe.enable_model_cpu_offload()`)
3. Lower generation resolution: use 512x512 instead of 1024x1024
4. Close other VRAM-consuming programs

### Q: First image generation is very slow

**A:** This is normal. The first inference needs to load the model into VRAM and initialize. If model compilation is enabled, the first inference will be even slower.

### Q: Generated image quality is poor

**A:** Check the following:
1. Ensure using the recommended 9 inference steps
2. guidance_scale must be set to 0 (Turbo model characteristic, cannot be changed)
3. Try improving the prompt with more detailed descriptions
4. Ensure model weight files are complete (check file sizes)

### Q: What language should prompts be in?

**A:** Z-Image-Turbo natively supports both Chinese and English:
- Chinese: "一只可爱的猫咪，戴着红色蝴蝶结"
- English: "A cute cat wearing a red bow tie"
- Mixed Chinese-English is also acceptable

### Q: How to check if model files are complete?

**A:** Check the key file sizes:

```bash
# Transformer weights (~24.6GB)
ls -lh Z-Image-Turbo/transformer/*.safetensors

# Text Encoder weights (~8GB)
ls -lh Z-Image-Turbo/text_encoder/*.safetensors

# VAE weights (~170MB)
ls -lh Z-Image-Turbo/vae/*.safetensors

# Tokenizer (~17MB)
ls -lh Z-Image-Turbo/tokenizer/vocab.json
```

If files are only ~135 bytes, they are Git LFS pointer files and need actual files downloaded.

## 📄 License

This project follows the original Z-Image-Turbo model license. Please review the model license file before use.

## 🙏 Acknowledgments

- Built on [Diffusers](https://github.com/huggingface/diffusers) library
- API service provided by [FastAPI](https://fastapi.tiangolo.com/)
- Z-Image-Turbo model developed by original authors

---

<div align="center">

**Made with ❤️ for efficient image generation**

</div>
