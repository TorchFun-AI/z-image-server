#!/usr/bin/env python3
"""
Z-Image-Turbo 图像生成脚本
"""

import torch
from diffusers import ZImagePipeline
from PIL import Image
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Generate images using Z-Image-Turbo")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=9, help="Number of inference steps (9 steps = 8 DiT forwards)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--model_path", type=str, default="../Z-Image-Turbo", help="Path to model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--compile", action="store_true", help="Enable model compilation for faster inference")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for low memory")

    args = parser.parse_args()

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_path}...")

    # Load pipeline
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # Optional: Model compilation for faster inference
    if args.compile:
        print("Compiling model (first run will be slower)...")
        pipe.transformer.compile()

    print(f"Generating image with prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}, Steps: {args.steps}, Seed: {args.seed}")

    # Generate image
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,  # Must be 0 for Turbo models
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]

    # Save image
    image.save(args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
