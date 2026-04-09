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
- Demo video: [link]
- Technical walkthrough: [link]

## Evaluation

<!-- TODO: Fill in after training and evaluation -->

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
