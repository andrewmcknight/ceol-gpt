"""
Training script for ceol-gpt.

Features:
  - AdamW optimiser with cosine LR schedule + linear warmup
  - Mixed-precision training (bf16 on A100/H100, fp16 fallback)
  - Gradient clipping
  - Early stopping on validation loss
  - Best-model and latest-epoch checkpointing
  - JSON training log (loss + LR per epoch) for plotting

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/small.yaml --run-name ablation_small
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import make_dataloaders, stratified_split
from src.model import CeolGPT, ModelConfig, build_model
from src.tokenizer import ABCTokenizer, GRACE_START_TOKEN, CHORD_START_TOKEN

import json as _json


# ---------------------------------------------------------------------------
# Tokenizer staleness check
# ---------------------------------------------------------------------------

def _tokenizer_is_current(tokenizer: ABCTokenizer) -> bool:
    """Return True if the tokenizer was built with the current tokenizer.py.

    Detects tokenizers built before GRACE_START / CHORD_START were added to
    _SPECIAL_TOKENS (vocab sizes 932 or 939 from earlier training runs).
    """
    vocab = tokenizer.vocab.token_to_id
    return GRACE_START_TOKEN in vocab and CHORD_START_TOKEN in vocab


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: CeolGPT,
    loader,
    optimiser: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    pad_id: int,
    grad_clip: float,
    amp_dtype: torch.dtype,
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
) -> tuple[float, int]:
    """Run one full pass over `loader`. If optimiser is None, runs validation only."""
    training = optimiser is not None
    model.train(training)

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(input_ids, attn_mask)           # (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
            )

        if training:
            # Update LR before the step
            lr = get_lr(step, warmup_steps, total_steps, max_lr)
            for pg in optimiser.param_groups:
                pg["lr"] = lr

            optimiser.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()
            step += 1

        # Accumulate token-weighted loss
        n_real = (targets != pad_id).sum().item()
        total_loss += loss.item() * n_real
        total_tokens += n_real

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, step


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: CeolGPT, optimiser, epoch: int, val_loss: float, cfg: dict):
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "vocab_size": model.cfg.vocab_size,   # saved explicitly so resume can validate
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "model_config": cfg,
    }, path)


def load_checkpoint(path: Path, model: CeolGPT, optimiser=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Determine the vocab_size the checkpoint was trained with.
    # New checkpoints store it explicitly; fall back to the embedding weight shape
    # for checkpoints saved before this field was added.
    ckpt_vocab_size = ckpt.get("vocab_size") or ckpt["model_state"]["tok_emb.weight"].shape[0]

    if ckpt_vocab_size != model.cfg.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: checkpoint={ckpt_vocab_size}, "
            f"current tokenizer={model.cfg.vocab_size}. "
            "The tokenizer was rebuilt since this checkpoint was saved. "
            "Either delete the checkpoint and run without --resume, "
            "or restore the matching tokenizer.pkl."
        )

    model.load_state_dict(ckpt["model_state"])
    if optimiser is not None and "optimiser_state" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser_state"])
    return ckpt["epoch"], ckpt["val_loss"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, run_name: str, resume: bool):
    # --- Paths ---
    models_dir = Path("models") / run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = models_dir / "best.pt"
    latest_ckpt = models_dir / "latest.pt"
    log_path = models_dir / "train_log.jsonl"
    tok_path = models_dir / "tokenizer.pkl"

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Prefer bf16 on Ampere+ (A100/H100), fall back to fp16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.float32
    print(f"AMP dtype: {amp_dtype}")

    # --- Tokenizer ---
    # Always load the full tunes list here; it's also needed to rebuild if stale.
    print(f"Loading data from {cfg['data_path']}...")
    with open(cfg["data_path"]) as f:
        tunes = _json.load(f)

    if tok_path.exists():
        tokenizer = ABCTokenizer.load(tok_path)
        if not _tokenizer_is_current(tokenizer):
            print(
                f"  ⚠  Tokenizer at {tok_path} is stale "
                f"(vocab={len(tokenizer):,}, missing GRACE/CHORD tokens). Rebuilding …"
            )
            tokenizer = ABCTokenizer.from_tunes(tunes, min_freq=2)
            tokenizer.save(tok_path)
            print(f"  ✓  Tokenizer rebuilt: {len(tokenizer):,} tokens")
        else:
            print(f"Loading tokenizer from {tok_path}")
    else:
        print(f"Building tokenizer from {cfg['data_path']}...")
        tokenizer = ABCTokenizer.from_tunes(tunes, min_freq=2)
        tokenizer.save(tok_path)
    print(f"Vocab size: {len(tokenizer):,}")

    # --- Data ---
    t_cfg = cfg["training"]
    train_dl, val_dl, test_dl = make_dataloaders(
        data_path=cfg["data_path"],
        tokenizer=tokenizer,
        batch_size=t_cfg["batch_size"],
        max_seq_len=cfg["max_seq_len"],
        train_frac=cfg["train_split"],
        val_frac=cfg["val_split"],
        num_workers=int(os.environ.get("NUM_WORKERS", 2)),
        tunes=tunes,   # already loaded during tokenizer step — skip second file read
    )
    print(f"Batches — train: {len(train_dl):,}  val: {len(val_dl):,}  test: {len(test_dl):,}")

    # --- Model ---
    model = build_model(cfg, vocab_size=len(tokenizer)).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    # --- Optimiser ---
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = GradScaler("cuda") if (device.type == "cuda" and amp_dtype == torch.float16) else None

    # --- LR schedule params ---
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * t_cfg["max_epochs"]
    warmup_steps = t_cfg["warmup_steps"]

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    step = 0

    if resume and latest_ckpt.exists():
        try:
            start_epoch, best_val_loss = load_checkpoint(latest_ckpt, model, optimiser)
            start_epoch += 1
            step = start_epoch * steps_per_epoch
            print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")
        except ValueError as exc:
            print(f"\n  ⚠  Cannot resume: {exc}")
            print("  Starting fresh training with the current tokenizer.\n")

    # --- Training loop ---
    log_entries = []

    for epoch in range(start_epoch, t_cfg["max_epochs"]):
        t0 = time.time()

        train_loss, step = run_epoch(
            model, train_dl, optimiser, scaler, device,
            pad_id=tokenizer.pad_id,
            grad_clip=t_cfg["grad_clip"],
            amp_dtype=amp_dtype,
            step=step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=t_cfg["learning_rate"],
        )

        with torch.no_grad():
            val_loss, _ = run_epoch(
                model, val_dl, None, None, device,
                pad_id=tokenizer.pad_id,
                grad_clip=t_cfg["grad_clip"],
                amp_dtype=amp_dtype,
                step=step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                max_lr=t_cfg["learning_rate"],
            )

        elapsed = time.time() - t0
        current_lr = get_lr(step, warmup_steps, total_steps, t_cfg["learning_rate"])

        print(
            f"Epoch {epoch+1:3d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.0f}s"
        )

        # Log
        entry = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr}
        log_entries.append(entry)
        with open(log_path, "a") as f:
            f.write(_json.dumps(entry) + "\n")

        # Checkpoint latest
        save_checkpoint(latest_ckpt, model, optimiser, epoch, val_loss, cfg)

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(best_ckpt, model, optimiser, epoch, val_loss, cfg)
            print(f"  ✓ new best val loss {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= t_cfg["patience"]:
                print(f"Early stopping: val loss has not improved for {t_cfg['patience']} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {models_dir}")
    return best_val_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ceol-gpt")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--run-name", default=None, help="Name for this run (default: config stem)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or Path(args.config).stem
    train(cfg, run_name=run_name, resume=args.resume)


if __name__ == "__main__":
    main()
