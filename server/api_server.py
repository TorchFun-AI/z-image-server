"""
Z-Image-Turbo API 服务器
支持 Text2Image 图像生成
使用 diffusers ZImagePipeline
"""

import base64
import io
import os
import time
import uuid
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from diffusers import ZImagePipeline

# ================= 配置 =================
MODEL_PATH = "../Z-Image-Turbo"  # 默认当前目录，可以通过环境变量覆盖
DEVICE = os.getenv("DEVICE", "cuda:0")
DTYPE_STR = os.getenv("DTYPE", "bfloat16")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
COMPILE_MODEL = os.getenv("COMPILE_MODEL", "false").lower() == "true"

# 解析 dtype
dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
DTYPE = dtype_map.get(DTYPE_STR, torch.bfloat16)

# ================= 加载模型 =================
print("=" * 60)
print("Z-Image-Turbo API Server (diffusers)")
print("=" * 60)
print(f"模型路径: {MODEL_PATH}")
print(f"设备: {DEVICE}")
print(f"数据类型: {DTYPE_STR}")
print(f"模型编译: {COMPILE_MODEL}")
print()

print("正在加载模型...")
pipe = ZImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=False,
)
pipe.to(DEVICE)

if COMPILE_MODEL:
    print("启用模型编译（首次推理较慢）...")
    pipe.transformer.compile()

print("模型加载完成!")
print("=" * 60)

# ================= FastAPI =================
app = FastAPI(title="Z-Image-Turbo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= 数据模型 =================
class Text2ImageRequest(BaseModel):
    prompt: str
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(default=9, ge=1, le=50, description="默认9步（实际8次DiT前向）")
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    width: int
    height: int
    prompt: str
    generation_time: float


# ================= 辅助函数 =================
def image_to_base64(img: Image.Image) -> str:
    """PIL Image 转 base64"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode()


# ================= API 端点 =================
@app.get("/")
async def root():
    return {"message": "Z-Image-Turbo API Server", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "z-image-turbo", "device": DEVICE}


@app.post("/v1/text2image", response_model=GenerateResponse)
async def text2image(request: Text2ImageRequest):
    """Text to Image 生成"""
    try:
        start_time = time.time()

        seed = request.seed if request.seed is not None else torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        image = pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=0.0,  # Turbo 模型必须设为 0
            generator=generator,
        ).images[0]

        generation_time = time.time() - start_time

        # 控制台打印耗时信息
        print(f"✅ [Text2Image] 生成完成 | 耗时: {generation_time:.2f}秒 | 尺寸: {request.width}x{request.height} | 步数: {request.num_inference_steps} | Seed: {seed}")

        return GenerateResponse(
            image_base64=image_to_base64(image),
            seed=seed,
            width=request.width,
            height=request.height,
            prompt=request.prompt,
            generation_time=generation_time,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================= WebUI =================
WEBUI_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Z-Image-Turbo</title>
    <style>
        :root {
            --bg: #0f0f13;
            --bg2: #1a1a22;
            --accent: #10b981;
            --accent2: #34d399;
            --text: #f1f5f9;
            --text2: #94a3b8;
            --border: #2d2d3a;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        h1 {
            text-align: center;
            font-size: 2rem;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: var(--text2);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }
        .form-group { margin-bottom: 1rem; }
        label { display: block; font-size: 0.9rem; color: var(--text2); margin-bottom: 0.5rem; }
        input, textarea, select {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            color: var(--text);
            font-size: 0.95rem;
        }
        textarea { min-height: 120px; resize: vertical; font-family: inherit; }
        input:focus, textarea:focus { outline: none; border-color: var(--accent); }
        .row { display: flex; gap: 1rem; }
        .row > * { flex: 1; }
        .btn {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--accent), #059669);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .result-img {
            max-width: 100%;
            border-radius: 10px;
            display: none;
        }
        .result-placeholder {
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text2);
            border: 1px dashed var(--border);
            border-radius: 10px;
        }
        .status { margin-top: 1rem; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem; display: none; }
        .status.success { display: block; background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent); }
        .status.error { display: block; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; }
        .loading { display: none; text-align: center; padding: 2rem; }
        .loading.active { display: block; }
        .spinner {
            width: 40px; height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .info-box {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            color: var(--text2);
        }
        .info-box strong { color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡️ Z-Image-Turbo</h1>
        <p class="subtitle">高效图像生成基础模型 | 8步推理 | 支持中英双语</p>

        <div class="grid">
            <div class="card">
                <div class="info-box">
                    <strong>提示：</strong>Turbo 模型默认使用 9 步推理（实际为 8 次 DiT 前向传播），guidance_scale 固定为 0。支持中英文提示词。
                </div>

                <div class="form-group">
                    <label>Prompt (提示词)</label>
                    <textarea id="prompt" placeholder="描述你想生成的图片...">年轻的中国女性，穿着红色汉服，精致刺绣，霓虹灯背景</textarea>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>宽度</label>
                        <input type="number" id="width" value="1024" min="256" max="2048" step="64">
                    </div>
                    <div class="form-group">
                        <label>高度</label>
                        <input type="number" id="height" value="1024" min="256" max="2048" step="64">
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>步数 (推荐 9)</label>
                        <input type="number" id="steps" value="9" min="1" max="50">
                    </div>
                    <div class="form-group">
                        <label>Seed (空=随机)</label>
                        <input type="number" id="seed" placeholder="随机">
                    </div>
                </div>
                <button class="btn" onclick="generate()">✨ 生成图像</button>
            </div>
            <div class="card">
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>生成中...</p>
                </div>
                <img id="result" class="result-img" alt="Generated">
                <div id="placeholder" class="result-placeholder">生成的图片将显示在这里</div>
                <div id="status" class="status"></div>
            </div>
        </div>
    </div>

    <script>
        async function generate() {
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const placeholder = document.getElementById('placeholder');
            const status = document.getElementById('status');
            const btn = document.querySelector('.btn');

            loading.classList.add('active');
            placeholder.style.display = 'none';
            result.style.display = 'none';
            status.className = 'status';
            btn.disabled = true;

            const params = {
                prompt: document.getElementById('prompt').value,
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                num_inference_steps: parseInt(document.getElementById('steps').value),
            };
            const seed = document.getElementById('seed').value;
            if (seed) params.seed = parseInt(seed);

            try {
                const res = await fetch('/v1/text2image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail);

                result.src = 'data:image/png;base64,' + data.image_base64;
                result.style.display = 'block';
                status.className = 'status success';
                status.textContent = `✓ Seed: ${data.seed} | 耗时: ${data.generation_time.toFixed(2)}s | 尺寸: ${data.width}x${data.height}`;
            } catch (e) {
                status.className = 'status error';
                status.textContent = '✗ ' + e.message;
                placeholder.style.display = 'flex';
            }
            loading.classList.remove('active');
            btn.disabled = false;
        }

        // Enter 键生成
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generate();
            }
        });
    </script>
</body>
</html>
'''


@app.get("/ui", response_class=HTMLResponse)
async def webui():
    """WebUI 界面"""
    return WEBUI_HTML


# ================= 启动 =================
if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"Z-Image-Turbo API Server")
    print(f"{'=' * 60}")
    print(f"API 地址: http://{HOST}:{PORT}")
    print(f"WebUI: http://{HOST}:{PORT}/ui")
    print(f"API 文档: http://{HOST}:{PORT}/docs")
    print(f"{'=' * 60}\n")

    uvicorn.run(app, host=HOST, port=PORT)
