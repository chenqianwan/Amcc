# 阿里云 GPU 服务器操作指南

## 第一步：租服务器

1. 登录阿里云 → 搜索 "GPU 云服务器" 或 "PAI-DSW"（交互式建模）
2. 推荐配置：
   - **GPU：** NVIDIA V100 16GB 或 A10 24GB（够用且便宜）
   - **镜像：** Ubuntu 22.04 + CUDA 12.1 + PyTorch（选预装镜像省时间）
   - **按量付费**即可，预计用 2-3 小时
3. 如果用 PAI-DSW（阿里云的 Notebook 环境），选 V100 或 A10 实例，自带 PyTorch 环境更省事

## 第二步：连接服务器

```bash
# SSH 连接（用阿里云给的公网 IP）
ssh root@<your-server-ip>

# 或者用 PAI-DSW 的 Terminal 直接操作
```

## 第三步：上传脚本

方式 A：从 GitHub clone（如果你的 amcc 仓库已 push）
```bash
git clone https://github.com/chenqianwan/Amcc.git
cd Amcc/assignment3/setup_scripts
```

方式 B：用 scp 从本地上传
```bash
# 在你的 Mac 上执行：
scp -r /Users/chenlong/WorkSpace/amcc/assignment3/setup_scripts root@<server-ip>:~/
```

## 第四步：一键安装

```bash
cd ~/setup_scripts   # 或者 cd ~/Amcc/assignment3/setup_scripts
chmod +x setup_all.sh
./setup_all.sh
```

这个脚本会：
- 检测 GPU
- clone open-oasis 仓库
- 安装所有依赖
- 下载模型权重（约 2GB）
- 安装 CogVideoX 依赖

预计耗时：10-15 分钟（主要是下载权重）

## 第五步：生成 Oasis Demo

```bash
cd ~/setup_scripts
python generate_demo.py --num-videos 3
```

生成 3 个视频，每个约 3 分钟（在 V100 上）。输出在 `~/world_model/outputs/`

**重要：截图/录屏！**
- 截图终端运行过程（显示 GPU 信息、耗时等）
- 记录每个视频的生成时间

## 第六步：生成 CogVideoX 对比（Bonus）

```bash
python generate_comparison.py
```

生成 3 个 CogVideoX 视频用于对比。输出在 `~/world_model/outputs/comparison/`

## 第七步：下载结果到本地

```bash
# 在你的 Mac 上执行：
mkdir -p /Users/chenlong/WorkSpace/amcc/assignment3/demo_outputs
scp -r root@<server-ip>:~/world_model/outputs/* /Users/chenlong/WorkSpace/amcc/assignment3/demo_outputs/
```

## 第八步：释放服务器

**跑完就释放！** 按量付费，别忘了停机或释放实例。

---

## 常见问题

**Q: HuggingFace 下载慢/超时？**
```bash
# 用镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
./setup_all.sh
```

**Q: CUDA 版本不匹配？**
```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi
# 如果是 CUDA 11.x，改用：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: 显存不够？**
Oasis 500M 只需 ~6GB，V100 16GB 绰绰有余。
CogVideoX-2b 加了 cpu_offload，12GB 显存也能跑。

**Q: generate.py 报错找不到文件？**
确保权重下载完整：
```bash
ls -lh ~/world_model/open-oasis/weights/
# 应该看到 oasis500m.safetensors (~1.0GB) 和 vit-l-20.safetensors (~xxx MB)
```
