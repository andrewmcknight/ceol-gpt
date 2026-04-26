"""
Inference / autoregressive sampling for ceol-gpt.

Usage (CLI):
    python -m src.generate --type reel --key Dmajor --temperature 0.8

Usage (API):
    from src.generate import load_model, generate
    model, tokenizer = load_model("models/large/best.pt", "models/large/tokenizer.pkl")
    abc_body = generate(model, tokenizer, tune_type="reel", key="Dmajor", meter="4/4")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import CeolGPT, build_model
from src.tokenizer import ABCTokenizer, BOS_TOKEN


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to a 1-D logits tensor."""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens whose *cumulative* probability exceeds top_p,
        # but keep at least the top token.
        to_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    return logits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CeolGPT, ABCTokenizer]:
    """Load a trained CeolGPT and its tokenizer from disk.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by src.train.
        tokenizer_path:  Path to the tokenizer .pkl file.
        device:          Torch device string or object.

    Returns:
        (model, tokenizer) — model is in eval mode.

    Raises:
        ValueError: if the tokenizer's vocab_size doesn't match the checkpoint.
    """
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = ABCTokenizer.load(tokenizer_path)

    # Infer the vocab_size the checkpoint was trained with.
    # New checkpoints store it explicitly; older ones don't, so fall back to the
    # embedding weight shape.
    ckpt_vocab_size = (
        ckpt.get("vocab_size")
        or ckpt["model_state"]["tok_emb.weight"].shape[0]
    )

    if ckpt_vocab_size != len(tokenizer):
        raise ValueError(
            f"Tokenizer/checkpoint vocab_size mismatch: "
            f"tokenizer has {len(tokenizer):,} tokens but checkpoint was trained with "
            f"{ckpt_vocab_size:,}. Make sure you're using the matching tokenizer.pkl "
            f"for this checkpoint (usually in the same models/<run>/ directory)."
        )

    cfg = ckpt["model_config"]
    model = build_model(cfg, vocab_size=ckpt_vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_stream(
    model: CeolGPT,
    tokenizer: ABCTokenizer,
    tune_type: str,
    key: str,
    meter: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str | torch.device = "cpu",
):
    """Autoregressively generate one tune, yielding one token string per step.

    Yields each raw token string (e.g. ``"G2"``, ``"|"``) as it is sampled.
    Control/conditioning tokens (``<BOS>``, ``<EOS>``, ``<TYPE:...>``, etc.)
    are not yielded.  Generation stops at ``<EOS>`` or ``max_new_tokens``.
    """
    device = torch.device(device)
    model = model.to(device)

    prefix_ids = tokenizer.encode(
        abc="",
        tune_type=tune_type,
        key=key,
        meter=meter,
        add_bos=True,
        add_eos=False,
    )
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_id
    id_to_token = tokenizer.vocab.id_to_token
    max_ctx = model.cfg.max_seq_len

    _skip_prefixes = ("<TYPE:", "<KEY:", "<METER:")
    _skip_tokens = {tokenizer.vocab.token_to_id.get(t) for t in
                    ("<PAD>", "<BOS>", "<EOS>", "<UNK>")} - {None}

    for _ in range(max_new_tokens):
        ctx = input_ids[:, -max_ctx:]
        logits = model(ctx)
        logits = logits[0, -1, :]

        if temperature != 1.0:
            logits = logits / temperature

        logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        token_id = next_id.item()
        if token_id == eos_id:
            break

        token_str = id_to_token.get(token_id, "<UNK>")
        if token_id not in _skip_tokens and not any(token_str.startswith(p) for p in _skip_prefixes):
            yield token_str


@torch.no_grad()
def generate(
    model: CeolGPT,
    tokenizer: ABCTokenizer,
    tune_type: str,
    key: str,
    meter: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str | torch.device = "cpu",
) -> str:
    """Autoregressively generate one tune and return its ABC music body.

    The conditioning prefix ``<TYPE:...> <KEY:...> <METER:...> <BOS>`` is
    prepended automatically.  Generation stops at ``<EOS>`` or when
    ``max_new_tokens`` is reached.

    Returns:
        Space-separated ABC music body string (no headers), e.g.
        ``"|: G2 BG A2 GA | B4 A4 :|"``
    """
    tokens = list(generate_stream(
        model, tokenizer,
        tune_type=tune_type,
        key=key,
        meter=meter,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    ))
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a tune with ceol-gpt")
    parser.add_argument("--checkpoint", default="models/large/best.pt")
    parser.add_argument("--tokenizer", default="models/large/tokenizer.pkl")
    parser.add_argument("--type", dest="tune_type", default="reel")
    parser.add_argument("--key", default="Dmajor")
    parser.add_argument("--meter", default="4/4")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device=args.device)
    print(f"Vocab size: {len(tokenizer):,}  |  Device: {args.device}")

    print(f"\nGenerating {args.tune_type} in {args.key} ({args.meter}) ...\n")
    print(
        f"X:1\n"
        f"T:Generated {args.tune_type.title()}\n"
        f"M:{args.meter}\n"
        f"L:{'1/4' if args.meter == '3/2' else '1/8'}\n"
        f"K:{args.key}"
    )
    for token in generate_stream(
        model, tokenizer,
        tune_type=args.tune_type,
        key=args.key,
        meter=args.meter,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    ):
        print(token, end=" ", flush=True)
    print()


if __name__ == "__main__":
    main()
