# Mini Vision Transformer — Forward Pass Only

Short class project: a **minimal Vision Transformer** in **PyTorch** that runs **inference only** (no training). You get patch embedding, a **CLS** token, learnable **positional** vectors on patches, **two** transformer blocks with **`nn.MultiheadAttention` (`batch_first=True`)**, and a **10-way** linear head—plus scripts that print tensor shapes and save figures for demos and reports.

**Repository:** `https://github.com/chenqianwan/Amcc` (clone with SSH: `git@github.com:chenqianwan/Amcc.git`)

## Setup

```bash
git clone git@github.com:chenqianwan/Amcc.git
cd Amcc
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generated PNGs are ignored by git (except `outputs/.gitkeep`); run **`python demo_forward.py`** locally to recreate them.

## Run (main demo)

```bash
python demo_forward.py
```

This writes PNGs under **`outputs/`** and prints a **shape table** in the terminal. Help:

```bash
python demo_forward.py -h
```

**Optional scripts**

| Command | Purpose |
|---------|---------|
| `python validate_project.py` | Quick checks: tensor shapes + subprocess demo into a temp dir |
| `python compare_inputs.py` | One figure comparing 3 synthetic inputs (see `outputs/compare/`) |

## Repository layout

| Path | Role |
|------|------|
| `requirements.txt` | torch, torchvision, matplotlib, pillow, numpy |
| `demo_forward.py` | CLI demo: forward pass, figures, shape printout |
| `validate_project.py` | Lightweight validation (no pytest) |
| `compare_inputs.py` | Multi-input comparison figure |
| `src/mini_vit/model.py` | `MiniViT`, `PatchEmbed`, transformer blocks |
| `src/mini_vit/attention_viz.py` | `AttentionWeights` (per-head + mean tensors) |
| `src/mini_vit/utils.py` | Synthetic images, plotting, shape table |
| `src/mini_vit/__init__.py` | Package exports |
| `.gitignore` | Ignores `.venv`, `__pycache__`, generated files under `outputs/` |
| `submission_notes.md`, `presentation_outline.md`, `vibe_coding_log_draft.md` | Submission helpers |

## Generated outputs

After **`python demo_forward.py`** (default `outputs/`):

| File | What it shows |
|------|----------------|
| `input_image.png` | 32×32 RGB demo input |
| `patch_grid.png` | Same image with 4×4 patch grid |
| `attention_block0_head0.png` | Block 0, head 0: CLS→patch attention (8×8) |
| `attention_block0_mean_heads.png` | Block 0, mean over heads |
| `attention_cls_to_patches_only.png` | Teaching view: CLS→patches only |
| `attention_block0_patch_grid_heatmap.png` | Mean map with patch-cell borders |
| `attention_block0_overlay_on_input.png` | Mean attention upsampled over the RGB image |
| `logits_bar_chart.png` | 10 classifier logits (untrained) |

After **`python compare_inputs.py`**:

| File | What it shows |
|------|----------------|
| `outputs/compare/comparison.png` | 3 columns: input / attention / logits for checkerboard, stripes, gradient |

CLI flags for `demo_forward.py`: `--seed`, `--image-type`, `--save-prefix`, `--output-dir`, `--print-shapes` / `--no-print-shapes` (see `-h`).

## Known limitations

- **No training** — weights are random/init only; logits are **not** meaningful class predictions.
- **Fixed input size** — **32×32** RGB; other sizes raise a clear error.
- **torchvision** — `auto` image mode tries `FakeData` if import works; broken `_lzma` builds fall back to procedural noise (still no manual downloads).
- **Toy model** — 64-dim embeddings, 2 layers, 4 heads; for teaching, not benchmark accuracy.

## Documentation for submission

- **`submission_notes.md`** — screenshots, commands, files to show in a demo video.
- **`presentation_outline.md`** — 1–2 minute talk structure.
- **`vibe_coding_log_draft.md`** — short reflection on iterative AI-assisted development.
