import os
import random
import string
import torch
from typing import Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

def generate_story_image(prompt: str) -> str:
    """
    Generates an image based on a descriptive story prompt and saves it locally.
    
    Args:
        prompt (str): Detailed visual description generated from the story context.
        
    Returns:
        str: File path to the saved PNG image.
    """
    print(f"Processing for prompt: '{prompt[:60]}...'")
    
    # Ensure the output directory exists
    os.makedirs("images_generated", exist_ok=True)

    # Load pipeline with FP16 variants
    raw_pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    )
    pipe: Any = raw_pipe
    
    # This offloads sub-modules to CPU when not active, saving massive VRAM
    pipe.enable_model_cpu_offload() 
    
    # Reduces memory overhead during the attention mechanism layer calculation
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    # Append stylistic guardrails to match children's book aesthetic
    styled_prompt = f"{prompt}, in the style of a beautiful children's storybook illustration, rich colors, highly detailed"
    negative_prompt = "low quality, blurry, photo, realistic, modern clothing, modern items, text, watermark, mutated"
    
    # Run Inference
    output = pipe(
        prompt=styled_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.0
    )
    
    # Save and export the asset
    image = output.images[0]
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    output_filename = f"images_generated/{random_id}.png"
    image.save(output_filename)
    
    print(f"🎉 Asset generated successfully at: {output_filename}")
    return output_filename