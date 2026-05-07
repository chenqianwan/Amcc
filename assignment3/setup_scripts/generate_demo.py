"""
Generate demo videos using Oasis 500M.
Run this on the rented GPU server after setup_all.sh completes.

Usage:
    python generate_demo.py
    python generate_demo.py --num-videos 5
    python generate_demo.py --prompt-path my_screenshot.png
"""

import argparse
import os
import time
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate Oasis demo videos")
    parser.add_argument("--num-videos", type=int, default=3, help="Number of videos to generate")
    parser.add_argument("--prompt-path", type=str, default=None, help="Custom prompt image path")
    parser.add_argument("--oasis-dir", type=str, default=os.path.expanduser("~/world_model/open-oasis"))
    args = parser.parse_args()

    oasis_dir = args.oasis_dir
    weights_dir = os.path.join(oasis_dir, "weights")
    output_dir = os.path.expanduser("~/world_model/outputs")
    os.makedirs(output_dir, exist_ok=True)

    oasis_ckpt = os.path.join(weights_dir, "oasis500m.safetensors")
    vae_ckpt = os.path.join(weights_dir, "vit-l-20.safetensors")

    if not os.path.exists(oasis_ckpt):
        print(f"ERROR: Model weights not found at {oasis_ckpt}")
        print("Run setup_all.sh first.")
        sys.exit(1)

    print(f"Oasis dir: {oasis_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Generating {args.num_videos} video(s)...")
    print()

    for i in range(args.num_videos):
        output_path = os.path.join(output_dir, f"oasis_demo_{i+1}.mp4")
        print(f"--- Video {i+1}/{args.num_videos} ---")

        cmd = [
            sys.executable, os.path.join(oasis_dir, "generate.py"),
            "--oasis-ckpt", oasis_ckpt,
            "--vae-ckpt", vae_ckpt,
        ]

        if args.prompt_path and os.path.exists(args.prompt_path):
            cmd.extend(["--prompt-path", args.prompt_path])

        start_time = time.time()
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=oasis_dir, capture_output=False)

        elapsed = time.time() - start_time
        print(f"Elapsed: {elapsed:.1f}s")

        # Move output to our directory
        default_output = os.path.join(oasis_dir, "video.mp4")
        if os.path.exists(default_output):
            os.rename(default_output, output_path)
            print(f"Saved: {output_path}")
        else:
            print(f"WARNING: Expected output not found at {default_output}")

        print()

    print("=" * 50)
    print("All videos generated!")
    print(f"Check: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Download the .mp4 files to your local machine")
    print("  2. Take screenshots of interesting frames for the report")
    print("  3. Run generate_comparison.py for the bonus")


if __name__ == "__main__":
    main()
