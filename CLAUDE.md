# ceol-gpt

GPT-style autoregressive transformer for generating Irish folk music in ABC notation, conditioned on tune type and key signature.

## Project Structure

```
ceol-gpt/
├── src/                  # Core Python modules
│   ├── tokenizer.py      # ABC notation tokenizer (word-based, regex)
│   ├── dataset.py        # PyTorch Dataset + DataLoader utilities
│   ├── model.py          # Transformer architecture
│   ├── train.py          # Training loop
│   ├── generate.py       # Inference / sampling
│   └── evaluate.py       # Evaluation metrics
├── configs/              # Hyperparameter configs (YAML)
├── notebooks/            # EDA and experiment notebooks (run on Colab)
├── data/                 # Dataset (tunes.json, 54K settings from TheSession)
├── models/               # Trained model checkpoints and configs
├── videos/               # Demo and technical walkthrough videos
├── docs/                 # ABC notation reference, project handout, papers
├── SETUP.md              # Installation and setup instructions
├── ATTRIBUTION.md        # AI tool usage attribution (required for class)
└── requirements.txt      # Python dependencies
```

## Dataset

- `data/tunes.json` — 54,246 tune settings from TheSession.org
- Fields: tune_id, setting_id, name, type, meter, mode, abc, date, username, composer
- 12 tune types: barndance, hornpipe, jig, march, mazurka, polka, reel, slide, slip jig, strathspey, three-two, waltz
- 23 key/mode combinations (e.g., Gmajor, Adorian, Eminor)
- 7 meters: 12/8, 2/4, 3/2, 3/4, 4/4, 6/8, 9/8

## Architecture

Decoder-only transformer (GPT-2 style), trained from scratch. Small model (~2-10M params) suitable for Colab Pro GPUs.

Conditioning: tune type and key/mode are prepended as special tokens, e.g., `<TYPE:reel> <KEY:Gmajor> |: G2 BG ...`

## Tokenizer

Word-based regex tokenizer. Each musical "word" is one token:
- Notes with accidentals + duration: `^F2`, `_B/2`, `c'4`
- Bar lines: `|`, `||`, `|:`, `:|`, `|]`
- Rests: `z`, `z2`
- Decorations: `~`, `.`, ties `-`
- Tuplets: `(3`
- Broken rhythm operators: `>`, `<`
- Special conditioning tokens: `<TYPE:reel>`, `<KEY:Gmajor>`, `<METER:4/4>`, `<BOS>`, `<EOS>`, `<PAD>`

## Training

- Platform: Google Colab Pro (GPU)
- Framework: PyTorch
- Optimizer: AdamW with cosine LR scheduling + warmup
- Regularization: dropout + early stopping
- Mixed precision (fp16) for efficiency

## Commands

```bash
# Training (designed for Colab, but runnable locally)
python -m src.train --config configs/default.yaml

# Generation
python -m src.generate --type reel --key Gmajor --temperature 0.8

# Evaluation
python -m src.evaluate --checkpoint checkpoints/best.pt
```

## Key References

- `docs/abc_quick_guide.md` — ABC notation quick reference
- `docs/abc_standard_v2.1.html` — Full ABC standard
- `docs/final_project_handout.html` — Course rubric and requirements
- `docs/tunesformer.pdf` — Related work on transformer-based tune generation
