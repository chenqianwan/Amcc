"""
Bonus: Side-by-side comparison of Oasis (world model) vs CogVideoX (video generator).
Run this after generate_demo.py.

This generates a Minecraft-style video using CogVideoX for comparison.

Usage:
    python generate_comparison.py
"""

import os
import time
import torch
import sys

def generate_cogvideox():
    """Generate a Minecraft-style video using CogVideoX-2b."""

    output_dir = os.path.expanduser("~/world_model/outputs/comparison")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("  CogVideoX-2b: Minecraft Scene Generation")
    print("=" * 50)
    print()

    print("Loading CogVideoX-2b pipeline...")
    start_load = time.time()

    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")
    print()

    prompts = [
        "First person view walking forward in a Minecraft world with green grass blocks, oak trees, blue sky, pixelated graphics style, game footage",
        "First person view of mining stone blocks underground in Minecraft, torch light, cave environment, pixelated voxel graphics",
        "First person view looking at a Minecraft village with houses and villagers, daytime, clear sky, blocky 3D graphics",
    ]

    for i, prompt in enumerate(prompts):
        print(f"--- Generating video {i+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt[:80]}...")

        start_gen = time.time()

        video_frames = pipe(
            prompt=prompt,
            num_frames=49,
            guidance_scale=6.0,
            num_inference_steps=50,
        ).frames[0]

        gen_time = time.time() - start_gen
        output_path = os.path.join(output_dir, f"cogvideox_minecraft_{i+1}.mp4")
        export_to_video(video_frames, output_path, fps=8)

        print(f"Generated in {gen_time:.1f}s -> {output_path}")
        print()

    print("=" * 50)
    print("CogVideoX generation complete!")
    print(f"Videos saved to: {output_dir}")
    print()
    print("Now you can compare:")
    print("  - ~/world_model/outputs/oasis_demo_*.mp4  (world model, action-conditioned)")
    print("  - ~/world_model/outputs/comparison/cogvideox_*.mp4  (video generator, text-only)")
    print()
    print("Key differences to note in your report:")
    print("  1. Oasis responds to actions; CogVideoX just generates a fixed trajectory")
    print("  2. Oasis maintains consistent world state; CogVideoX may drift/hallucinate")
    print("  3. Oasis frame-by-frame is autoregressive; CogVideoX generates all at once")


if __name__ == "__main__":
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on a GPU server.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    generate_cogvideox()
