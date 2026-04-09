# Setup

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support (for GPU training)
- Google Colab Pro recommended for training

## Installation

```bash
# Clone the repository
git clone https://github.com/<username>/ceol-gpt.git
cd ceol-gpt

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data

The dataset (`data/tunes.json`) is included in the repository. It contains 54,246 tune settings from [TheSession.org](https://thesession.org) via the [TheSession-data](https://github.com/adactio/TheSession-data) repository.

No API keys or external services are required.

## Training

Training is designed for Google Colab Pro with GPU acceleration. To train locally with a GPU:

```bash
python -m src.train --config configs/default.yaml
```

To train on Colab, open the training notebook in `notebooks/` and follow the instructions there.

## Generation

After training, generate tunes with:

```bash
python -m src.generate --type reel --key Gmajor --temperature 0.8
```

See `python -m src.generate --help` for all options.
