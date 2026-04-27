---
title: ceol-gpt
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
---

# ceol-gpt

Ceol (pronounced "kyohl" or "ke-yole") is the Irish word for music. This project is a GPT-style autoregressive transformer trained from scratch on ABC notation Irish folk music that generates novel tunes conditioned on tune type and key signature.

## What it Does

Ceol-GPT learns the structure and patterns of Irish traditional music from over 54,000 tune settings sourced from [TheSession.org](https://thesession.org), a community website dedicated to Irish traditional music. Given a tune type (e.g., reel, jig, hornpipe) and key signature (e.g., G major, D dorian), the model generates new tunes in ABC notation that follow the conventions of that style. The system includes a custom word-based tokenizer for ABC notation, a decoder-only transformer architecture, and sampling-based generation with temperature and nucleus sampling controls.

## Quick Start

See [SETUP.md](SETUP.md) for full installation instructions.

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train --config configs/default.yaml

# Generate a tune
python -m src.generate --type reel --key Gmajor --temperature 0.8
```

## Video Links

<!-- TODO: Add video links before submission -->
- Demo video: [link](https://youtu.be/YzsUfMwSh00)
- Technical walkthrough: [link]

## Evaluation

### Model comparison

Three model sizes were trained on the same 85/10/5 train/val/test split. All other settings (dropout, weight decay, grad clip, sampling parameters) were held constant. See [`notebooks/hyperparameter_comparison.ipynb`](notebooks/hyperparameter_comparison.ipynb) for full learning curves.

| Config | Params | Best val loss | Val perplexity | Best epoch | Stopped by |
|--------|-------:|--------------|---------------|-----------|------------|
| small  | 2.1M   | 1.0897       | 2.97          | 95/100    | full schedule |
| medium | 12.0M  | 0.9508       | **2.59**      | 46/56     | early stopping |
| large  | 46.8M  | 0.9547       | 2.60          | 17/27     | early stopping |

The medium and large configs converge to near-identical perplexity (~2.59–2.60), while small plateaus significantly higher (2.97) even after running the full schedule. The large model is used for generation and the live demo.

### Novelty analysis

To verify that the model generalises rather than memorises, 20 tunes were generated and checked against the full training corpus using exact-match and 10-gram coverage (see [`notebooks/memorization_check.ipynb`](notebooks/memorization_check.ipynb) and [`src/evaluate.py`](src/evaluate.py)).

| Metric | Result |
|--------|--------|
| Exact matches (out of 20) | 0 (0%) |
| Avg 10-gram coverage | 22.3% |
| Max 10-gram coverage | 45.3% |

All 20 tunes were novel — none reproduced a training example verbatim. Average 10-gram coverage of 22.3% is well below the 40% threshold for "significant phrase reuse", which is expected given that Irish folk music shares common melodic phrases across tunes even in human compositions.

### Qualitative outcomes

Generated tunes are syntactically valid ABC notation in all tested cases and render correctly as sheet music and playable audio in the Gradio app. Conditioning on tune type and key signature is effective: reels in D major sound stylistically distinct from jigs in G dorian, with appropriate rhythmic patterns, phrase lengths, and repeat structures for each type.

## Repository Structure

```
ceol-gpt/
├── src/           # Source code (tokenizer, model, training, generation, evaluation)
├── data/          # Dataset (tunes.json from TheSession.org)
├── models/        # Trained model checkpoints and configs
├── notebooks/     # EDA and experiment notebooks
├── videos/        # Demo and technical walkthrough videos
├── configs/       # Hyperparameter configuration files
├── docs/          # ABC notation reference, project handout, related papers
├── SETUP.md       # Installation and setup instructions
├── ATTRIBUTION.md # AI tools, data sources, and references
└── requirements.txt
```
