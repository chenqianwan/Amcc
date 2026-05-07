"""
Patch open-oasis generate.py for torchvision 0.26+ compatibility, then run generation.
Usage: python patch_and_run.py [--num-videos N]
"""
import os
import sys
import time
import subprocess

OASIS_DIR = os.path.expanduser("~/world_model/open-oasis")
OUTPUT_DIR = os.path.expanduser("~/world_model/outputs")

def patch_files():
    """Replace torchvision.io imports with av-based alternatives."""

    # Create a compatibility module
    compat_path = os.path.join(OASIS_DIR, "video_compat.py")
    with open(compat_path, "w") as f:
        f.write('''
import torch
import numpy as np

def write_video(filename, video_array, fps=20):
    """Drop-in replacement for torchvision.io.write_video using av."""
    import av
    if isinstance(video_array, torch.Tensor):
        video_array = video_array.cpu().numpy()
    T, H, W, C = video_array.shape
    container = av.open(filename, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    for t in range(T):
        frame = av.VideoFrame.from_ndarray(video_array[t], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def read_video(filename, pts_unit="sec"):
    """Drop-in replacement for torchvision.io.read_video using av."""
    import av
    container = av.open(filename)
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(arr)
    container.close()
    video = torch.from_numpy(np.stack(frames))
    return video, None, {"video_fps": 30}
''')
    print(f"Created {compat_path}")

    # Patch generate.py
    gen_path = os.path.join(OASIS_DIR, "generate.py")
    with open(gen_path, "r") as f:
        content = f.read()

    if "video_compat" not in content:
        content = content.replace(
            "from torchvision.io import read_video, write_video",
            "from video_compat import read_video, write_video"
        )
        with open(gen_path, "w") as f:
            f.write(content)
        print("Patched generate.py")
    else:
        print("generate.py already patched")

    # Patch utils.py if needed
    utils_path = os.path.join(OASIS_DIR, "utils.py")
    with open(utils_path, "r") as f:
        ucontent = f.read()
    if "torchvision.io" in ucontent:
        ucontent = ucontent.replace(
            "from torchvision.io import read_video",
            "from video_compat import read_video"
        ).replace(
            "from torchvision.io import read_video, write_video",
            "from video_compat import read_video, write_video"
        )
        with open(utils_path, "w") as f:
            f.write(ucontent)
        print("Patched utils.py")


def run_generation(num_videos=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weights_dir = os.path.join(OASIS_DIR, "weights")
    oasis_ckpt = os.path.join(weights_dir, "oasis500m.safetensors")
    vae_ckpt = os.path.join(weights_dir, "vit-l-20.safetensors")

    if not os.path.exists(oasis_ckpt):
        print(f"ERROR: {oasis_ckpt} not found")
        sys.exit(1)

    for i in range(num_videos):
        output_path = os.path.join(OUTPUT_DIR, f"oasis_demo_{i+1}.mp4")
        print(f"\n{'='*50}")
        print(f"  Generating video {i+1}/{num_videos}")
        print(f"{'='*50}")

        cmd = [
            sys.executable, os.path.join(OASIS_DIR, "generate.py"),
            "--oasis-ckpt", oasis_ckpt,
            "--vae-ckpt", vae_ckpt,
            "--output-path", output_path,
            "--num-frames", "32",
        ]

        start = time.time()
        result = subprocess.run(cmd, cwd=OASIS_DIR)
        elapsed = time.time() - start

        if result.returncode == 0 and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"OK: {output_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        else:
            print(f"FAILED (exit code {result.returncode}, {elapsed:.1f}s)")

    print(f"\nAll done! Videos in {OUTPUT_DIR}")
    subprocess.run(["ls", "-lh", OUTPUT_DIR])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=3)
    args = parser.parse_args()

    print("Step 1: Patching torchvision compatibility...")
    patch_files()

    print("\nStep 2: Generating Oasis demo videos...")
    run_generation(args.num_videos)
