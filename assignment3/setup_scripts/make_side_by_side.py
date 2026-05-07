"""
Create a side-by-side comparison video: Oasis vs CogVideoX.
Run this after both generate_demo.py and generate_comparison.py.

Requires: pip install moviepy

Usage:
    python make_side_by_side.py
"""

import os
import sys

try:
    from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
except ImportError:
    print("Installing moviepy...")
    os.system(f"{sys.executable} -m pip install moviepy")
    from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip


def make_comparison():
    output_dir = os.path.expanduser("~/world_model/outputs")
    comparison_dir = os.path.join(output_dir, "comparison")

    oasis_path = os.path.join(output_dir, "oasis_demo_1.mp4")
    cogvideo_path = os.path.join(comparison_dir, "cogvideox_minecraft_1.mp4")

    if not os.path.exists(oasis_path):
        print(f"ERROR: Oasis video not found: {oasis_path}")
        sys.exit(1)
    if not os.path.exists(cogvideo_path):
        print(f"ERROR: CogVideoX video not found: {cogvideo_path}")
        sys.exit(1)

    print("Loading videos...")
    oasis_clip = VideoFileClip(oasis_path)
    cogvideo_clip = VideoFileClip(cogvideo_path)

    # Match durations (use the shorter one)
    min_duration = min(oasis_clip.duration, cogvideo_clip.duration)
    oasis_clip = oasis_clip.subclip(0, min_duration)
    cogvideo_clip = cogvideo_clip.subclip(0, min_duration)

    # Resize to same height
    target_height = 360
    oasis_clip = oasis_clip.resize(height=target_height)
    cogvideo_clip = cogvideo_clip.resize(height=target_height)

    # Side by side
    final = clips_array([[oasis_clip, cogvideo_clip]])

    output_path = os.path.join(output_dir, "side_by_side_comparison.mp4")
    print(f"Rendering side-by-side video to: {output_path}")
    final.write_videofile(output_path, fps=8, codec="libx264")

    print(f"\nDone! Output: {output_path}")
    print("Left = Oasis (world model, action-conditioned)")
    print("Right = CogVideoX (video generator, text-only)")


if __name__ == "__main__":
    make_comparison()
