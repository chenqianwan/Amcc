# Presentation outline (about 1–2 minutes)

Use this as speaker notes or slide bullets. Adjust timing to your rubric.

---

## 0:00–0:15 — Hook & goal

- **Title:** Mini Vision Transformer, forward pass only.  
- **One sentence:** I built a tiny ViT in PyTorch that turns a **32×32 image** into **patch tokens**, runs **self-attention**, and outputs **10 logits**—without training, so we can **see** tensors and attention maps.

---

## 0:15–0:45 — Four ideas (simple language)

**1. Patch embedding**  
- The image is split into an **8×8 grid** of small squares (each square is **4×4 pixels**).  
- Each square becomes one **token** (a vector of length 64). So we go from “pixels” to “**64 patch tokens**.”

**2. CLS token**  
- We add one **special token** at the start: **[CLS]**.  
- The full sequence is **1 + 64 = 65 tokens**. Later we read the **CLS** row as a summary of the whole image (here it feeds the classifier).

**3. Self-attention**  
- Every token can **look at** every other token and **mix** information.  
- **Multi-head attention** means several parallel “ways of looking.”  
- My figures show **CLS → patches**: how much the summary token focuses on each part of the image (block 0, mean over heads).

**4. Logits**  
- After the blocks, we take the **CLS** vector and apply a **linear layer** to get **10 numbers** (logits).  
- With **no training**, those numbers are **random-ish**—the point is to show the **pipeline**, not accuracy.

---

## 0:45–1:30 — Demo (what to show)

- Run **`python demo_forward.py`** (or show a screenshot of the **shape table**).  
- Flash **input** → **patch grid** → **one attention heatmap** → **overlay on image** → **logits bar chart**.  
- Say once: *“Same code path every time; only forward pass, no optimizer.”*

---

## 1:30–2:00 — Limitations & wrap-up

- **Limitations:** not trained; tiny model; 32×32 only.  
- **Takeaway:** ViT = **patches + position + attention + head**; I visualized the middle steps for class.

---

**Total:** ~90–120 seconds. Practice once with your actual PNGs so transitions match your slides.
