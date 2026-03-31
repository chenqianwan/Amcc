# Submission notes — what to capture

Use this as a checklist before you record or zip your submission.

## Screenshots to take

1. **Terminal — main demo**  
   Full scrollback (or two stacked screenshots) of:
   - `python demo_forward.py`  
   - The bordered **shape table** (tensor names and shapes).  
   - The final line: `Wrote demo artifacts to: .../outputs`

2. **Terminal — help (optional but nice)**  
   `python demo_forward.py -h` showing available flags.

3. **Terminal — validation (optional)**  
   `python validate_project.py` ending with **All checks passed**.

4. **Images from `outputs/`** (after a fresh `python demo_forward.py`):
   - `input_image.png`
   - `patch_grid.png`
   - `attention_block0_head0.png`
   - `attention_block0_mean_heads.png`
   - `attention_cls_to_patches_only.png`
   - `attention_block0_patch_grid_heatmap.png`
   - `attention_block0_overlay_on_input.png`
   - `logits_bar_chart.png`

5. **Comparison figure (optional)**  
   Run `python compare_inputs.py`, then screenshot or embed **`outputs/compare/comparison.png`**.

6. **IDE / file tree (if required)**  
   Show `src/mini_vit/`, `demo_forward.py`, and `README.md` in the explorer.

---

## Terminal commands to run during a live or recorded demo

Run from the project root with the venv activated:

```bash
source .venv/bin/activate          # adjust for Windows
pip install -r requirements.txt    # only if first time
python demo_forward.py
```

Optional second line to show checks:

```bash
python validate_project.py
```

Optional third line for the comparison slide:

```bash
python compare_inputs.py
```

Optional one-liner to show CLI variety:

```bash
python demo_forward.py --image-type checkerboard --seed 0 --save-prefix demo_
```

---

## Output files to show (in order)

When flipping through the folder or slides, this order matches the pipeline story:

1. **`input_image.png`** — what goes into the model.  
2. **`patch_grid.png`** — how the image is cut into 4×4-pixel patches (8×8 grid).  
3. **`attention_block0_head0.png`** — one attention head: where **CLS** looks at patches.  
4. **`attention_block0_mean_heads.png`** — average over heads (smoother summary).  
5. **`attention_cls_to_patches_only.png`** / **`attention_block0_patch_grid_heatmap.png`** — same idea with clearer teaching labels or grid lines.  
6. **`attention_block0_overlay_on_input.png`** — attention mapped back onto the picture.  
7. **`logits_bar_chart.png`** — raw scores before softmax (untrained).  

If you ran **`compare_inputs.py`**, add **`outputs/compare/comparison.png`** at the end as a “different inputs → different internals” capstone.

---

## Short video storyboard (~2 minutes)

1. README or project tree (5–10 s).  
2. Run `python demo_forward.py`; pause on shape table (20–30 s).  
3. Open `outputs/` and walk through the PNGs in the order above (45–60 s).  
4. *(Optional)* `validate_project.py` or `compare_inputs.py` (15–20 s).  
5. *(Optional)* Open `model.py` and point at `MiniViT.forward` / `MultiheadAttention` without reading every line (10–15 s).
