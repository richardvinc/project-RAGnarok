from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..config import settings

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"


def execute_generate_story_image(prompt: str) -> str:
    settings.image_output_dir.mkdir(parents=True, exist_ok=True)

    raw_pipeline = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline: Any = raw_pipeline
    pipeline.enable_model_cpu_offload()

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    styled_prompt = (
        f"{prompt}, in the style of a beautiful children's storybook illustration, "
        "rich colors, highly detailed"
    )
    negative_prompt = (
        "low quality, blurry, photo, realistic, modern clothing, modern items, text, "
        "watermark, mutated"
    )

    result = pipeline(
        prompt=styled_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.0,
    )

    image = result.images[0]
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    output_path = settings.image_output_dir / f"{random_id}.png"
    image.save(output_path)
    return str(output_path)


def to_public_image_url(saved_image_path: str) -> str:
    relative_path = Path(saved_image_path).resolve().relative_to(settings.root_dir)
    return f"{settings.resolved_backend_public_url}/{relative_path.as_posix()}"
