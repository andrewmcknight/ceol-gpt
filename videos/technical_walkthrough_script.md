# Technical Walkthrough Script (5–10 min)

**Audience:** a fellow ML engineer / the grader. They want to see *how* it works and where the ML lives.
**Recording setup:** screen capture. Mostly your editor (VS Code / similar) + a few slides + the running app at the end. Talking-head insert is optional.
**Target length:** ~8:30. Hard cap 10:00.

---

## [0:00–0:30] Frame the problem (~30s)
**On screen:** SLIDE T1 — one-sentence problem statement + dataset stat.

> "ceol-gpt is a decoder-only transformer trained from scratch to generate Irish folk tunes in ABC notation, conditioned on tune type and key. About 47 million parameters, trained on 54,000 tunes from TheSession.org. The reason this is interesting as an ML project — and not just a music project — is that ABC notation is a structured symbolic language with a small vocabulary and strong long-range constraints, which makes it a clean testbed for the things a transformer is supposed to be good at."

---

## [0:30–1:30] Data and tokenization (~60s)
**On screen:** open `data/tunes.json` briefly to show one record (just scroll, don't dwell). Then open `src/tokenizer.py` and scroll to the regex (around line 24).

**Beat 1:** Dataset.
> "The dataset is a JSON dump of 54K tune settings — each has a type, a meter, a key, and the ABC body. 12 tune types, 23 key/mode combinations."

**Beat 2:** Tokenizer choice.
> "The first real design decision was tokenization. ABC has character-level tokenizers in prior work, but characters lose musical structure — the note `^F2` is a single musical event, not three independent characters. So I wrote a **word-based regex tokenizer**: each musical 'word' — a note with its accidental, octave, and duration; a bar line; a rest; a decoration — becomes one token."

**Beat 3:** Show the regex and conditioning tokens.
> "Here's the regex — it tries patterns longest-first. And critically, conditioning is done by **prepending special tokens**: `<TYPE:reel>`, `<KEY:Gmajor>`, `<METER:4/4>`, then `<BOS>`, then the music. The model learns to attend to those header tokens, and at inference I just supply them and let it generate from there. Vocab ends up around 930 tokens."

---

## [1:30–3:30] Model architecture (~2:00)
**On screen:** SLIDE T2 — architecture diagram (see `visuals.md`).
Then `src/model.py`.

**Beat 1 (slide):**
> "The architecture is a standard GPT-2-style decoder-only transformer, pre-norm. 12 layers, 8 heads, 512-dim model, 2048-dim feed-forward, max sequence length 512."

**Beat 2 (`model.py`, scroll to `CausalSelfAttention`):**
> "I built this from scratch in PyTorch rather than using HuggingFace, because for a project like this I wanted to actually own the implementation. The attention block is multi-head causal self-attention. The causal mask is registered as a buffer once at init time. Note the padding-mask handling — because tunes vary in length, I had to mask out padding keys explicitly, and also nan-to-num after softmax to handle rows where everything gets masked, which was a bug I hit early on."

**Beat 3 (scroll to `CeolGPT.__init__`):**
> "A couple of things worth pointing out. First, **weight tying** — the LM head shares weights with the token embedding. With a 930-token vocab and a 512-dim model that's about half a million parameters saved, and it usually helps perplexity on small vocabularies. Second, GPT-2-style init: linear and embedding weights are normal(0, 0.02), and residual projection weights get scaled down by `1/sqrt(2 * n_layers)` to keep activations stable as the network deepens."

**Beat 4 (briefly mention conditioning mechanically):**
> "There's no separate conditioning head. The TYPE/KEY/METER tokens go through the same embedding table as everything else, and the causal attention lets every later token attend to them. That's all you need."

---

## [3:30–5:00] Training (~1:30)
**On screen:** `src/train.py` (scroll through the training loop), then SLIDE T3 — training curves (use `models/large/training_curves.png`).

**Beat 1:** Training setup.
> "Trained on Colab Pro on an A100. AdamW, cosine learning rate schedule with linear warmup over 1000 steps, weight decay 0.01, gradient clipping at 1.0, mixed-precision fp16. Batch size 64, max sequence length 512. The 85/10/5 train/val/test split is at the **tune level**, not the token level, so the model never sees any setting of a held-out tune."

**Beat 2:** Regularization.
> "Three regularization mechanisms: dropout at 0.1 throughout, weight decay through AdamW, and **early stopping with patience 10** on validation loss. The training-curve image is from the run that produced the checkpoint shipped in the repo."

**Beat 3 (show curves):**
> "Train loss settles around 0.6 and val loss around 1.0 — there's a gap, which is the model memorizing some surface patterns from the training set, but val loss kept improving for a long time before plateauing, so the gap isn't from overfitting in the harmful sense. I also ran a separate **memorization check** — see the notebook — where I verified that generated tunes don't reproduce training tunes verbatim."

---

## [5:00–6:30] Generation (~1:30)
**On screen:** `src/generate.py`, scroll to `_top_k_top_p_filter` and `generate_stream`.

**Beat 1:** Sampling.
> "Generation is autoregressive sampling with three knobs: temperature, top-k, and top-p (nucleus). Top-k clamps to the k highest-logit tokens; top-p keeps the smallest set of tokens whose cumulative probability exceeds p. I apply both — top-k first, then nucleus on the survivors — which gives clean output without ever degenerating into the model's safest, most boring next-token prediction."

**Beat 2:** Conditioning at inference.
> "At inference, I encode the prompt as `<TYPE:x> <KEY:y> <METER:z> <BOS>` and let the model continue. It usually emits an `<EOS>` on its own when the tune feels complete, but I also cap at 512 new tokens as a safety stop."

**Beat 3 (open `app.py` briefly, scroll to the streaming callback):**
> "The Gradio app wraps this in a streaming generator so you see tokens appear in real time, and uses **abcjs** in an iframe to render sheet music and play synthesized audio in the browser. The model itself runs on whatever device is available — CUDA if present, otherwise CPU."

---

## [6:30–7:45] Evaluation (~1:15)
**On screen:** `src/evaluate.py` and the `notebooks/memorization_check.ipynb`.

> "Evaluation has three pieces.
>
> **One**, held-out cross-entropy / perplexity on the test split — the standard language-modeling metric. That tells me the model can predict next tokens in real tunes it's never seen.
>
> **Two**, structural validity checks — does the generated ABC parse? Does it have balanced bar counts? Does the key signature match the notes the model produced? These catch the failure mode where loss looks fine but output is musical garbage.
>
> **Three**, the memorization check in the notebook — for a sample of generated tunes, find the longest n-gram overlap with anything in the training set. If the model were just regurgitating, this would be high; in practice the longest overlaps are short common phrases, which is what you'd expect for a stylistically faithful generator.
>
> What's missing — and I'll be honest about this — is human evaluation at scale. I can play a generated tune myself and tell you whether it's idiomatic, but I haven't done a blind listening study with other musicians. That's the natural next step."

---

## [7:45–8:30] What was hard + wrap (~45s)
**On screen:** brief talking-head insert, or hold on a final slide listing the three challenges.

> "Three things that were genuinely hard.
>
> **One**, the tokenizer. ABC has a lot of edge cases — chord symbols, grace notes, alternate endings, weird decorations — and I went through several iterations getting the regex right without losing musical information. There are commits titled 'Fix tokenizer', 'Update tokenizer', 'Revise tokenizer' for a reason.
>
> **Two**, padding-mask handling in attention. When a row of the attention matrix is fully masked, softmax produces NaNs, which then corrupt the rest of the batch. The `nan_to_num` after softmax was the fix.
>
> **Three**, deciding what 'good' means. Loss curves only get you so far for a generative model in a creative domain. The most useful evaluation tool I have is a fiddle.
>
> Code, the trained checkpoint, and the demo are all in the repo. Thanks."

---

## Production notes
- **Pacing target:** ~8:30. Sections are sized so cutting Eval down to 45s drops you to ~8:00 if needed.
- **Editor zoom:** bump font size to ~16pt before recording so code is readable at 1080p.
- **Don't read code line-by-line.** Scroll to the relevant block, point to the 2–3 things that matter, move on.
- **Slides vs. code:** open in an editor for the architecture, training, and generation sections. Use slides only for the title, the architecture diagram, the training curves, and the optional final-challenges card.
- **One take per section** is fine — you can stitch in a video editor. Don't try to nail the whole thing in one pass.
