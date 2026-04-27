# Attribution

## AI Tools Used

### Claude (Anthropic) — Claude Code CLI + claude.ai

Used throughout the project for code generation, debugging, and presentation materials. Below is a breakdown by area.

---

#### Code Generated or Co-written with AI

**`src/tokenizer.py`** — I described the ABC notation format and the token vocabulary I wanted (one note per token, regex-based word splitting) and asked Claude to draft the tokenizer. It generated a working initial version that I then rewrote substantially. The original output used a naive character-level split that broke on multi-character tokens like `^F2` or `_B/2`. I had to redesign the regex pattern myself after reading through the ABC 2.1 standard — Claude's pattern was matching accidentals and duration markers separately instead of grouping them. The final `tokenize()` and `build_vocab()` functions are my own; I kept Claude's `encode()` / `decode()` scaffolding and modified it.

**`src/dataset.py`** — AI generated the skeleton `ABCDataset` class and the `make_dataloader()` helper. The initial version didn't handle variable-length sequences or PAD masking correctly. I added the attention mask logic and the conditioning token prepend (`<TYPE:...> <KEY:...>`) myself after realizing the generated code was just truncating sequences without padding.

**`src/model.py`** — I asked Claude to implement a decoder-only transformer following a GPT-2-style architecture. The generated code was mostly correct structurally but used `nn.Transformer` (which is encoder-decoder) instead of a decoder-only stack. I replaced the attention module with a manually masked `nn.MultiheadAttention` block and added the causal mask. The positional encoding and embedding layers are largely AI-generated with light edits.

**`src/train.py`** — AI wrote the AdamW optimizer setup, cosine LR scheduler with warmup, and the basic training loop. I added early stopping, gradient clipping, and the checkpoint saving logic myself. The mixed precision (`torch.cuda.amp`) scaffolding was also AI-generated but required debugging on Colab — the `GradScaler` wasn't being initialized inside the right context and was silently producing NaNs in the loss. Took a while to track that down.

**`src/generate.py`** — Temperature sampling and top-k filtering were AI-generated. The nucleus (top-p) sampling I wrote myself by adapting a reference implementation I found in the Hugging Face source.

**`configs/default.yaml`** — AI suggested reasonable hyperparameter defaults; I tuned learning rate, dropout, and batch size based on actual training runs.

---

#### Slide and Video Materials

**Demo and technical walkthrough slide decks** (in `videos/`) were generated with AI assistance. I gave Claude an outline of the project and asked it to draft slide content and structure for both presentations. It produced a full script and bullet-point layout which I used as a starting point. I edited the framing significantly — particularly the technical walkthrough.

---

#### Debugging and Iteration

Several issues required substantial manual debugging after AI-generated code failed:

- **Tokenizer regex**: Claude's initial regex didn't correctly handle octave markers (`'` and `,`), broken rhythm operators (`>` and `<`), or inline chord syntax. I rewrote the pattern after reading the ABC spec directly.
- **NaN loss during training**: Traced to a `GradScaler` initialization bug in the mixed-precision training loop.
- **Conditioning token alignment**: The model was initially not learning from the conditioning tokens because they were being masked out in the loss computation. I identified this by inspecting the token IDs manually and fixed the loss mask.

---

## Data Sources

- **TheSession.org**: Tune data sourced from [TheSession-data](https://github.com/adactio/TheSession-data), a community-contributed dataset of Irish traditional music in ABC notation. Used under the Creative Commons Attribution-NonCommercial license.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Sheng et al., "TunesFormer: Forming Irish Tunes with Control Codes by Bar-level Transformer" (2023) — `docs/tunesformer.pdf`
- ABC Notation Standard v2.1 — `docs/abc_standard_v2.1.html`
- Andrej Karpathy, *minGPT* / *nanoGPT* — reference for decoder-only architecture implementation
