"""
Memorization and novelty evaluation for ceol-gpt.

Checks whether generated tunes are reproduced verbatim from training data,
or merely share short common phrases (expected for a well-generalizing model).

Two metrics
-----------
1. Exact match   — is the generated token sequence identical to any training tune?
2. Longest match — what is the longest contiguous run of tokens shared between
                   the generated tune and ANY training tune?

Interpretation guide (rough thresholds for ~100-300 token tunes):
  longest_match >= 80% of generated length → likely memorized verbatim
  longest_match 30–80%                     → heavy copying, investigate
  longest_match < 30%                      → sharing common phrases, normal

Usage
-----
    python -m src.evaluate \\
        --data data/tunes.json \\
        --tokenizer models/small/tokenizer.pkl \\
        --generated outputs/generated.txt \\
        --n 20

    # Or import and call directly:
    from src.evaluate import memorization_report
    report = memorization_report(generated_abc_list, train_tunes, tokenizer)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.tokenizer import ABCTokenizer, tokenize_abc


# ---------------------------------------------------------------------------
# Core metric: longest contiguous matching n-gram
# ---------------------------------------------------------------------------

def longest_common_substring_len(a: list, b: list) -> int:
    """Return the length of the longest contiguous subsequence shared by a and b.

    Uses a rolling-hash / DP approach that is O(len(a) * len(b)) but fast
    enough for tune-length sequences (~100-400 tokens each).
    """
    if not a or not b:
        return 0

    # DP: dp[j] = length of longest common suffix of a[:i] and b[:j]
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


# ---------------------------------------------------------------------------
# Build an index of training token sequences for fast lookup
# ---------------------------------------------------------------------------

def build_train_index(tunes: list[dict]) -> list[list[str]]:
    """Tokenize all training tunes and return as a list of token lists."""
    return [tokenize_abc(t["abc"]) for t in tunes]


# ---------------------------------------------------------------------------
# Per-tune analysis
# ---------------------------------------------------------------------------

def analyze_one(
    generated_tokens: list[str],
    train_token_seqs: list[list[str]],
) -> dict:
    """Compute novelty metrics for one generated tune against all training tunes.

    Returns a dict with:
      - exact_match (bool)
      - longest_match_tokens (int)
      - longest_match_frac (float)  — fraction of generated tune length
      - closest_train_idx (int)     — index of the most similar training tune
    """
    if not generated_tokens:
        return {
            "exact_match": False,
            "longest_match_tokens": 0,
            "longest_match_frac": 0.0,
            "closest_train_idx": -1,
        }

    gen_len = len(generated_tokens)
    best_len = 0
    best_idx = -1
    exact = False

    for idx, train_tokens in enumerate(train_token_seqs):
        if train_tokens == generated_tokens:
            exact = True
            best_len = gen_len
            best_idx = idx
            break  # can't do better than exact
        lcs = longest_common_substring_len(generated_tokens, train_tokens)
        if lcs > best_len:
            best_len = lcs
            best_idx = idx

    return {
        "exact_match": exact,
        "longest_match_tokens": best_len,
        "longest_match_frac": best_len / gen_len if gen_len else 0.0,
        "closest_train_idx": best_idx,
    }


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------

def memorization_report(
    generated_abc_list: Sequence[str],
    train_tunes: list[dict],
    tokenizer: ABCTokenizer | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Analyze a list of generated ABC strings against the training corpus.

    Parameters
    ----------
    generated_abc_list : sequence of raw ABC music strings (no conditioning tokens)
    train_tunes        : list of tune dicts from tunes.json
    tokenizer          : optional, used only for its tokenize_abc — can pass None
    verbose            : print a summary table

    Returns
    -------
    List of per-tune result dicts (same fields as analyze_one, plus 'generated_tokens').
    """
    print(f"Indexing {len(train_tunes):,} training tunes...")
    train_seqs = build_train_index(train_tunes)

    results = []
    for i, abc in enumerate(generated_abc_list):
        gen_tokens = tokenize_abc(abc)
        result = analyze_one(gen_tokens, train_seqs)
        result["tune_index"] = i
        result["generated_tokens"] = gen_tokens
        results.append(result)

    if verbose:
        _print_report(results, train_tunes)

    return results


def _print_report(results: list[dict], train_tunes: list[dict]) -> None:
    n = len(results)
    n_exact = sum(r["exact_match"] for r in results)
    fracs = [r["longest_match_frac"] for r in results]
    avg_frac = sum(fracs) / n if n else 0.0
    max_frac = max(fracs) if fracs else 0.0

    print()
    print("=" * 60)
    print("  Memorization Report")
    print("=" * 60)
    print(f"  Generated tunes analyzed : {n}")
    print(f"  Exact matches            : {n_exact} ({100*n_exact/n:.1f}%)")
    print(f"  Avg longest match frac   : {avg_frac:.2%}")
    print(f"  Max longest match frac   : {max_frac:.2%}")
    print()

    print(f"  {'#':>3}  {'len':>5}  {'longest':>9}  {'frac':>6}  exact  closest training tune")
    print(f"  {'-'*3}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*5}  {'-'*30}")
    for r in results:
        gen_len = len(r["generated_tokens"])
        closest_name = ""
        if r["closest_train_idx"] >= 0:
            closest_name = train_tunes[r["closest_train_idx"]].get("name", "?")[:30]
        print(
            f"  {r['tune_index']:>3}  {gen_len:>5}  "
            f"{r['longest_match_tokens']:>9}  {r['longest_match_frac']:>5.1%}  "
            f"{'YES' if r['exact_match'] else 'no':>5}  {closest_name}"
        )

    print()
    print("  Interpretation:")
    print("    frac >= 80%  → likely memorized verbatim")
    print("    frac 30–80%  → heavy copying, investigate")
    print("    frac  < 30%  → sharing common phrases (normal)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check generated tunes for memorization")
    parser.add_argument("--data", default="data/tunes.json", help="Path to tunes.json")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.pkl (optional)")
    parser.add_argument(
        "--generated", required=True,
        help="Text file with generated ABC tunes, one per line (or '---' separated blocks)"
    )
    parser.add_argument(
        "--split", default="---",
        help="Separator between tunes in --generated file (default: '---')"
    )
    parser.add_argument(
        "--train-split", default="train",
        choices=["train", "all"],
        help="Check against 'train' split only or 'all' tunes (default: train)"
    )
    args = parser.parse_args()

    # Load training data
    print(f"Loading {args.data}...")
    with open(args.data) as f:
        tunes = json.load(f)

    # Load tokenizer (optional — only needed if you want to reuse it elsewhere)
    tokenizer = None
    if args.tokenizer:
        tokenizer = ABCTokenizer.load(args.tokenizer)

    # Load generated tunes
    raw = Path(args.generated).read_text()
    if args.split in raw:
        generated = [block.strip() for block in raw.split(args.split) if block.strip()]
    else:
        generated = [line.strip() for line in raw.splitlines() if line.strip()]

    print(f"Loaded {len(generated)} generated tunes.")

    memorization_report(generated, tunes, tokenizer=tokenizer, verbose=True)


if __name__ == "__main__":
    main()
