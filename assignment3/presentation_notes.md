# Assignment 3: Running Oasis 500M — An Interactive World Model

**CHEN Long | 21321471 | May 2026**

---

## 1. 我选了什么

我选择运行 **Oasis 500M**（[etched-ai/open-oasis](https://github.com/etched-ai/open-oasis)），这是 Decart 和 Etched 在 2024 年底发布的一个**交互式世界模型**。它能像游戏引擎一样，根据玩家的键盘输入（W/A/S/D）**逐帧生成** Minecraft 风格的画面。

这不是普通的视频生成——你按"前进"，画面就会往前走；你按"转头"，视角就会转动。模型必须"理解"3D 空间关系才能做到这一点。

## 2. 硬件与部署

| 项目 | 配置 |
|------|------|
| **云服务器** | 阿里云 ECS ecs.gn7i-c16g1.4xlarge |
| **GPU** | NVIDIA A10 (24GB VRAM) |
| **实际显存占用** | ~5.8 GB（Oasis）/ ~2.7 GB（CogVideoX + CPU offload） |
| **模型大小** | DiT: 2.3 GB + VAE: 875 MB |
| **生成速度** | 32 帧 ≈ 29 秒（A10 上） |
| **每帧耗时** | 0.6–1.5 秒（早期帧快，后期帧慢） |

## 3. 部署过程中踩的坑

实际部署并不是"一键运行"，我遇到了 **3 个问题**并逐一解决：

1. **HuggingFace 下载超时** — 国内无法直接访问 HF，设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像解决
2. **权重路径不匹配** — 下载脚本把权重放到 `./weights/`，但 `generate.py` 找不到，需要手动传 `--oasis-ckpt` 和 `--vae-ckpt` 参数（花了 15 分钟读源码才发现）
3. **torchvision 0.26 API 移除** — `read_video`/`write_video` 被删了，我写了一个 PyAV 兼容层来替代（又花了 20 分钟）

## 4. Oasis 的架构原理

Oasis 基于 **Spatial-Temporal Diffusion Transformer (DiT)**，核心生成流程：

```
输入帧 → [ViT-VAE Encoder] → 潜空间表示
                                    ↓
键盘动作 → [Action Embedding] → 动作向量
                                    ↓
              [DiT 扩散去噪] ← 上一帧潜空间 + 动作向量
                                    ↓
              [VAE Decoder] → 下一帧像素
                                    ↓
                              输出帧（循环）
```

**关键区别：World Model vs Video Generator**

| | Oasis（世界模型） | CogVideoX（视频生成器） |
|---|---|---|
| **输入** | 上一帧 + 键盘动作 | 文本 prompt |
| **生成方式** | 逐帧自回归 | 一次性生成整段视频 |
| **可交互** | 是，每帧响应用户操作 | 否，固定轨迹 |
| **上下文窗口** | 32 帧（~2 秒） | 49 帧（一次生成） |
| **核心挑战** | 交互下保持一致性 | 生成视觉质量 |

## 5. 生成结果与观察

### 5.1 Oasis Demo 视频

三段 Oasis 生成的 Minecraft 场景（32 帧，1.6 秒，20fps）：

| 视频 | 链接 |
|------|------|
| Oasis Demo 1 — 草地 + 白桦树场景 | [oasis_demo_1.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/oasis_demo_1.mp4) |
| Oasis Demo 2 | [oasis_demo_2.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/oasis_demo_2.mp4) |
| Oasis Demo 3 | [oasis_demo_3.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/oasis_demo_3.mp4) |

### 5.2 CogVideoX 对比视频（Bonus）

用 CogVideoX-2b 生成的 Minecraft 风格视频（49 帧，文本驱动）：

| 视频 | Prompt |
|------|--------|
| [cogvideox_minecraft_1.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/comparison/cogvideox_minecraft_1.mp4) | "First person view walking forward in a Minecraft world with green grass blocks, oak trees, blue sky" |
| [cogvideox_minecraft_2.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/comparison/cogvideox_minecraft_2.mp4) | "First person view of mining stone blocks underground in Minecraft, torch light, cave environment" |
| [cogvideox_minecraft_3.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/comparison/cogvideox_minecraft_3.mp4) | "First person view looking at a Minecraft village with houses and villagers, daytime" |

### 5.3 Side-by-Side 对比

| 视频 | 链接 |
|------|------|
| Oasis vs CogVideoX 并排对比 | [side_by_side_comparison.mp4](https://github.com/chenqianwan/Amcc/blob/main/assignment3/demo_outputs/side_by_side_comparison.mp4) |

## 6. 观察分析

### 做得好的地方

- **视觉还原度高**：生成的帧确实像 Minecraft，方块纹理、光照、天空渐变都能识别
- **空间感知**：按"前进"时树木会变近，视角变化自然，有正确的透视关系
- **深度理解**：远处物体更小，近大远小的视差效果合理

### 存在的问题

- **纹理闪烁 (Texture Flickering)**：同一个草方块的颜色会在帧间微妙变化，播放视频时很明显
- **物体永久性缺失 (Object Permanence)**：转头后回看，场景往往已经变了——模型没有持久化的世界状态
- **分辨率低**：360p 输出，2026 年看起来很过时，与真实 Minecraft 画面差距明显
- **长序列退化 (Long-horizon Drift)**：20+ 帧后场景开始失去一致性，颜色偏移、几何扭曲

**帧对比证据：**

- 第 5 帧（早期）：清晰的方块纹理，结构完整 → `screenshots/good_frame.png`
- 第 25 帧（晚期）：左侧结构变形，细节丢失 → `screenshots/bad_frame.png`

## 7. Bonus 分析：世界模型 vs 视频生成器

运行两个模型后，核心区别变得非常直观：

**CogVideoX** 生成的视频更流畅、画质更高——因为它在生成时就规划好了整个镜头轨迹，像在拍一段"电影"。但你无法在生成过程中说"现在向左转"。

**Oasis** 的画质更差、帧间有闪烁——但它是**可交互的**。每一帧都取决于你"按了什么键"。这让它更像一个游戏引擎，而不是一个视频播放器。

> **核心 tradeoff：预规划的质量 vs 实时的响应性。**谁能同时做到"交互式 + 高质量"，谁就赢了。

## 8. 与 Assignment 1 的联系

| | Assignment 1: Mini ViT | Assignment 3: Oasis 500M |
|---|---|---|
| **架构** | ViT（2 层，4 头，64 维） | Spatial-Temporal DiT（500M 参数） |
| **输入** | 32×32 图片 → 4×4 patch | 640×360 视频帧 → latent |
| **输出** | 10 类分类 logits | 下一帧像素 |
| **注意力** | 在 16 个 patch 间计算 | 在空间-时间维度上计算 |
| **核心发现** | 注意力能捕捉 patch 间的关系 | 同样的注意力机制，放大后能追踪 3D 空间关系 |

Assignment 1 中可视化的那些 4×4 注意力热力图，和 Oasis 中追踪 3D 游戏环境空间关系的，是**完全相同的基础操作**——只是规模放大了几千倍。

## 9. 我的 Key Takeaway

> 世界模型的难点不是生成漂亮的像素（CogVideoX 也能做到），而是在**交互条件下维持一致性**。当模型必须响应任意用户动作并保持世界连贯时，这是一个根本不同的问题。

**最大的 gap：持久化空间记忆 (Persistent Spatial Memory)**

Oasis 没有真正的地图或 3D 状态——它只是根据最近几帧生成"看起来对的"下一帧。这意味着它无法处理"走进一栋房子，再走出来，房子应该还在原地"这样的场景。解决这个问题可能需要将显式 3D 表示（如 NeRF 或 Gaussian Splatting）与生成模型结合。

---

## 项目文件结构

```
assignment3/
├── report.tex                          # LaTeX 报告
├── report.pdf                          # 编译后的 PDF
├── bonus_writeup.md                    # Bonus 分析文档
├── presentation_notes.md               # 本文件（Presentation 用）
├── SERVER_GUIDE.md                     # 服务器部署指南
├── screenshots/
│   ├── good_frame.png                  # 第 5 帧（质量好）
│   ├── bad_frame.png                   # 第 25 帧（质量退化）
│   ├── oasis_frame1.png                # Oasis 首帧
│   └── cogvideo_frame1.png             # CogVideoX 首帧
├── demo_outputs/
│   ├── oasis_demo_1.mp4                # Oasis 生成视频 1
│   ├── oasis_demo_2.mp4                # Oasis 生成视频 2
│   ├── oasis_demo_3.mp4                # Oasis 生成视频 3
│   ├── side_by_side_comparison.mp4     # 并排对比视频
│   └── comparison/
│       ├── cogvideox_minecraft_1.mp4   # CogVideoX 对比视频 1
│       ├── cogvideox_minecraft_2.mp4   # CogVideoX 对比视频 2
│       └── cogvideox_minecraft_3.mp4   # CogVideoX 对比视频 3
└── setup_scripts/
    ├── setup_all.sh                    # 一键安装脚本
    ├── generate_demo.py                # Oasis 生成脚本
    ├── generate_comparison.py          # CogVideoX 生成脚本
    ├── patch_and_run.py                # torchvision 兼容补丁
    └── make_side_by_side.py            # 并排视频合成脚本
```

**GitHub 仓库：** [github.com/chenqianwan/Amcc](https://github.com/chenqianwan/Amcc/tree/main/assignment3)
