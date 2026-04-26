# Visuals to Make

Build these in Keynote / Google Slides / Figma — whatever you're fastest in. Keep a consistent look across both videos: one font, two colors, lots of whitespace. Suggested palette: deep green `#15803d` (matches the app's button) + slate gray `#1e293b` on white.

---

## Demo video slides

### SLIDE 1 — Title
- **Title:** ceol-gpt
- **Subtitle:** A transformer that writes Irish folk tunes
- **Footer:** your name · CS 372 · Spring 2026
- Keep it austere — black/dark text on white, lots of space. Maybe a small treble-clef glyph.

### SLIDE 2 — "What is ABC notation"
Two-column layout.
- **Left column header:** "ABC notation (text)"
  - Show 4–6 lines of ABC for a real, well-known tune (e.g., the opening of "The Kesh"). Monospace font, ~18pt.
  - Example block:
    ```
    X:1
    T:The Kesh
    M:6/8
    K:Gmaj
    |:G3 GAB|A3 ABd|edd gdd|edB dBA|
    ```
- **Right column header:** "Rendered sheet music"
  - Render the same ABC at https://abcjs.net/abcjs-editor.html, screenshot the staff, paste it in.
- Caption underneath spanning both columns, small italic: *"Same tune, two representations. The model reads and writes the left."*

### SLIDE 3 — "Why this is interesting"
Three bullets, fade in one at a time if your tool supports it:
- Generative models for **structured creative domains** — not just chat or images
- A **living oral tradition** — composition has always been part of Irish music
- A **tool for musicians** — give it a tune type & key, get a starting point

### SLIDE 4 — "What works, what doesn't"
Two columns with checkmarks/warnings. Use green ✓ and amber ⚠.
- **Works**
  - ✓ Stays in the right key and meter
  - ✓ Phrase length and bar structure feel right
  - ✓ Standard cadences (turning on the dominant, etc.)
- **Still fumbles**
  - ⚠ Long-range coherence — B part can wander
  - ⚠ Occasional awkward fingering for the instrument
  - ⚠ No human-listener evaluation yet

### SLIDE 5 — Wrap
- **ceol-gpt**
- GitHub URL (full link, big enough to read)
- "Thanks for listening."

---

## Technical walkthrough slides

### SLIDE T1 — Problem framing
One sentence, large:
> A 47M-parameter decoder-only transformer trained from scratch to generate Irish folk tunes in ABC notation, conditioned on tune type and key.

Below, three small stats badges:
- 54,246 tunes
- 12 tune types · 23 key/modes
- ~930 token vocab

### SLIDE T2 — Architecture diagram

Easiest way to make this: open Excalidraw or Figma, draw boxes top-to-bottom. Suggested structure:

```
                  ┌──────────────────────────────────┐
   Input tokens →│ <TYPE:reel> <KEY:Dmajor>          │
                  │ <METER:4/4> <BOS> | G2 BG ...     │
                  └────────────────┬─────────────────┘
                                   ▼
                       Token embedding (930 → 512)
                                  +
                       Learned positional embedding
                                   ▼
                  ┌──────────────────────────────────┐
                  │   Transformer block × 12          │
                  │   ┌──────────────────────────┐    │
                  │   │ LayerNorm → Causal MHA    │    │
                  │   │       (8 heads)           │    │
                  │   │ + residual                │    │
                  │   │ LayerNorm → FFN (GELU,    │    │
                  │   │       512→2048→512)       │    │
                  │   │ + residual                │    │
                  │   └──────────────────────────┘    │
                  └────────────────┬─────────────────┘
                                   ▼
                          Final LayerNorm
                                   ▼
                     LM head (tied to embedding)
                                   ▼
                       Logits over 930-token vocab
```

Annotate the side with: **pre-norm** · **weight tying** · **causal mask** · **conditioning via prepended tokens (no separate cross-attention)**.

### SLIDE T3 — Training curves
Use the existing `models/large/training_curves.png` directly. If you want it cleaner:
- Re-plot from `models/large/train_log.jsonl` in matplotlib
- Two lines: train loss (solid) and val loss (dashed)
- Vertical dotted line marking the early-stopping epoch (best.pt)
- Annotate final values: "train ≈ 0.6, val ≈ 1.0"

A quick re-plot script (run once, save as `videos/curves.png`):

```python
import json, matplotlib.pyplot as plt
rows = [json.loads(l) for l in open("models/large/train_log.jsonl")]
ep   = [r["epoch"] for r in rows]
plt.plot(ep, [r["train_loss"] for r in rows], label="train")
plt.plot(ep, [r["val_loss"]   for r in rows], "--", label="val")
plt.xlabel("epoch"); plt.ylabel("cross-entropy loss")
plt.legend(); plt.tight_layout()
plt.savefig("videos/curves.png", dpi=150)
```

### SLIDE T4 (optional) — "What was hard"
Three bullets, used as the closing card:
- Tokenizer iteration — ABC edge cases (chords, grace notes, endings)
- Padding-mask NaN bug in attention softmax
- Defining "good" for a generative model in a creative domain

---

## Other prep
- **Pre-generate 2–3 candidate tunes** for the demo and pick the most playable. Save the ABC text so you can re-render the same tune for both the screen-cap demo and the printed sheet you read off.
- **Print the chosen tune** at large size (or load on a tablet) for the fiddle performance shot.
- **Render the architecture diagram once at high res** (export at 2× from your tool) so it doesn't pixelate when you record at 1080p.
