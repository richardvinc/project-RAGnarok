import torch
from typing import Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

def run_official_sdxl():
    print("🔄 Step 1: Loading official StabilityAI SDXL Base 1.0 (FP16)...")
    
    # 1. We load the pipeline standard configuration structure
    raw_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    )
    
    # 2. FIX: Cast the object to 'Any' type to shut down strict linter errors.
    # This prevents Pyright/Pylance from claiming it isn't callable.
    pipe: Any = raw_pipe

    print("🚀 Step 2: Offloading execution onto your RTX 3060...")
    pipe.to("cuda")

    prompt = (
        "A gorgeous classic book illustration of Mowgli and Bagheera the panther "
        "sitting together on a massive mossy tree branch, sunbeams breaking through "
        "the dense jungle foliage, oil painting style"
    )
    negative_prompt = "low quality, blurry, modern clothing, sunglasses, text, watermark"
    
    print("🎨 Step 3: Running text-to-image diffusion passes across CUDA...")
    # The linter can now cleanly track the dynamic attributes without throwing errors
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    # Extract and save your final asset image file
    image = output.images[0]
    output_filename = "sdxl_base_jungle.png"
    image.save(output_filename)
    
    print(f"\n🎉 Success! Image generated locally and saved to: '{output_filename}'")

if __name__ == "__main__":
    try:
        run_official_sdxl()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")