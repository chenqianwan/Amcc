# Bonus: Side-by-Side Comparison — Oasis vs CogVideoX

## What I Did

I generated the same Minecraft-style scene using two models: Oasis 500M (a world model) and CogVideoX-2b (a standard text-to-video generator). For Oasis, I used a Minecraft forest screenshot as the prompt image and applied a sequence of "move forward" actions. For CogVideoX, I used the text prompt: "First person view walking forward in a Minecraft world with green grass blocks, oak trees, blue sky, pixelated graphics style."

## Why This Comparison Matters

The results make the distinction between world models and video generators viscerally clear. CogVideoX produces a smooth, cinematic clip that *looks* like Minecraft footage—the textures are plausible, the motion is fluid. But it's essentially a movie: a fixed camera path that was decided at generation time. There's no way to say "now turn left" halfway through.

Oasis, by contrast, feels like actually playing a game (a very low-res, slightly glitchy game). Each frame depends on what I "pressed." The quality is lower frame-by-frame, but the interactivity creates a fundamentally different experience. You can tell the model is tracking where you are and what's around you, even if it sometimes forgets.

## Key Observations

The most striking difference: temporal consistency under interaction. CogVideoX's video is smoother precisely *because* it planned the entire trajectory upfront. Oasis sacrifices smoothness for responsiveness. This tradeoff—pre-planned quality vs. real-time adaptability—seems like the core engineering challenge for world models going forward. Whoever solves "interactive AND high-quality" wins.
