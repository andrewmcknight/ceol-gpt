# Setup

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Google Colab Pro recommended for training (A100 or H100 GPU)

## Local Installation

```bash
git clone https://github.com/<username>/ceol-gpt.git
cd ceol-gpt

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Data

`data/tunes.json` is included in the repository — 54,246 tune settings from [TheSession.org](https://thesession.org). No API keys or external services are required.

---

## Training on Google Colab (recommended)

### Step 1 — Select a GPU

In Colab: `Runtime → Change runtime type` → choose **A100 GPU** or **H100 GPU**.

| GPU | Config to use | ~Params | Expected time/epoch |
|-----|--------------|---------|---------------------|
| H100 / A100 | `configs/large.yaml` | 47M | ~3 min |
| L4 | `configs/medium.yaml` | 12M | ~5 min |
| T4 | `configs/small.yaml` | 2M | ~8 min |

### Step 2 — Open the training notebook

Open `notebooks/train_colab.ipynb` in Colab. It will:

1. Mount Google Drive and clone/pull the repo
2. Install dependencies
3. Build (or load) the tokenizer
4. Run training with automatic resume on session disconnect
5. Plot training curves
6. Run a quick generation sanity-check

### Step 3 — Configure the notebook

Edit the two variables at the top of the training cell:

```python
CONFIG   = 'configs/large.yaml'   # match your GPU from the table above
RUN_NAME = 'large'                # checkpoints saved to models/{RUN_NAME}/
```

### Resuming after a disconnect

Colab sessions disconnect after ~12 hours or on inactivity. The training cell uses `--resume`, so simply **re-run all cells** and training picks up from the last completed epoch automatically.

Checkpoints are saved to `models/{RUN_NAME}/` on your Drive:
- `best.pt` — best validation loss
- `latest.pt` — end of most recent epoch (used for resuming)
- `tokenizer.pkl` — vocabulary (required for generation)
- `train_log.jsonl` — epoch-by-epoch loss + LR log

---

## Local Training (GPU required)

```bash
# Uses default.yaml (large config, requires A100/H100)
python -m src.train --config configs/default.yaml

# Or pick a smaller config for your hardware
python -m src.train --config configs/medium.yaml --run-name medium
```

To resume a previous run:

```bash
python -m src.train --config configs/large.yaml --run-name large --resume
```

---

## Generation

```bash
python -m src.generate --checkpoint models/large/best.pt --type reel --key Gmajor --meter 4/4

# All options
python -m src.generate --help
```
