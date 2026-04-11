"""
Memorization and novelty evaluation for ceol-gpt.

Checks whether generated tunes are reproduced verbatim from training data,
or merely share short common phrases (expected for a well-generalizing model).

Two metrics
-----------
1. Exact match    — is the generated token sequence identical to any training tune?
2. K-gram coverage — what fraction of the generated tune's k-grams (contiguous
                    token windows of length k) appear anywhere in training data?

Why k-gram coverage instead of "longest match":
  The longest-match approach requires building k-gram sets at many different
  lengths (via binary search), blowing up RAM on 54K tunes. Coverage uses a
  single set of fixed-length hashes — fast to build and memory-efficient.

Interpretation guide (k=10, ~100-300 token tunes):
  coverage >= 70%  → heavily derived from training, likely memorized
  coverage 40–70%  → significant phrase reuse, investigate
  coverage  < 40%  → mostly novel phrases (normal for a well-trained model)

  Note: Irish folk music has repetitive structure, so some coverage is expected
  even from a well-generalizing model. Calibrate against your val loss.

Usage
-----
    python -m src.evaluate \\
        --data data/tunes.json \\
        --generated outputs/generated.txt

    # Or import and call directly:
    from src.evaluate import memorization_report
    results = memorization_report(generated_abc_list, train_tunes)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.tokenizer import ABCTokenizer, tokenize_abc


# ---------------------------------------------------------------------------
# K-gram index: one fixed-length hash set, built once
# ---------------------------------------------------------------------------

KGRAM_K = 10  # window size; increase to be more strict, decrease to be more sensitive


class KGramIndex:
    """Set of hashed k-grams from all training tunes, plus an exact-match lookup.

    Memory: stores Python ints (hashed k-grams) and full-sequence tuples for
    exact matching. On 54K tunes with avg 200 tokens, the k-gram set holds
    ~2-4M unique hashes — roughly 200-400 MB, well within Colab limits.
    """

    def __init__(self, seqs: list[list[str]], k: int = KGRAM_K):
        self.k = k
        # Exact match: map full token sequence → training index
        self._exact: dict[tuple, int] = {}
        # K-gram coverage: set of hashed k-grams from entire corpus
        self._kgram_hashes: set[int] = set()

        for idx, seq in enumerate(seqs):
            key = tuple(seq)
            if key not in self._exact:
                self._exact[key] = idx
            for i in range(len(seq) - k + 1):
                self._kgram_hashes.add(hash(tuple(seq[i:i + k])))

        print(f"  Unique {k}-gram hashes: {len(self._kgram_hashes):,}")

    def exact_match_index(self, seq: list[str]) -> int:
        """Return training index of exact match, or -1."""
        return self._exact.get(tuple(seq), -1)

    def coverage(self, gen: list[str]) -> float:
        """Fraction of generated k-grams that appear anywhere in training."""
        if len(gen) < self.k:
            return 0.0
        total = len(gen) - self.k + 1
        hits = sum(
            1 for i in range(total)
            if hash(tuple(gen[i:i + self.k])) in self._kgram_hashes
        )
        return hits / total


# ---------------------------------------------------------------------------
# Build index from training tunes
# ---------------------------------------------------------------------------

def build_train_index(tunes: list[dict], k: int = KGRAM_K) -> KGramIndex:
    """Tokenize all training tunes and build a KGramIndex."""
    print(f"Tokenizing {len(tunes):,} training tunes (k={k})...")
    seqs = [tokenize_abc(t["abc"]) for t in tunes]
    return KGramIndex(seqs, k=k)


# ---------------------------------------------------------------------------
# Per-tune analysis
# ---------------------------------------------------------------------------

def analyze_one(generated_tokens: list[str], index: KGramIndex) -> dict:
    """Compute novelty metrics for one generated tune.

    Returns a dict with:
      - exact_match (bool)
      - kgram_coverage (float)   — fraction of k-grams seen in training
      - closest_train_idx (int)  — index of exact match, or -1
    """
    if not generated_tokens:
        return {
            "exact_match": False,
            "kgram_coverage": 0.0,
            "closest_train_idx": -1,
        }

    exact_idx = index.exact_match_index(generated_tokens)
    cov = 1.0 if exact_idx >= 0 else index.coverage(generated_tokens)

    return {
        "exact_match": exact_idx >= 0,
        "kgram_coverage": cov,
        "closest_train_idx": exact_idx,
    }


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------

def memorization_report(
    generated_abc_list: Sequence[str],
    train_tunes: list[dict],
    tokenizer: ABCTokenizer | None = None,
    k: int = KGRAM_K,
    verbose: bool = True,
) -> list[dict]:
    """Analyze a list of generated ABC strings against the training corpus.

    Parameters
    ----------
    generated_abc_list : sequence of raw ABC music strings (no conditioning tokens)
    train_tunes        : list of tune dicts from tunes.json
    tokenizer          : unused, kept for API compatibility
    k                  : k-gram window size (default 10)
    verbose            : print a summary table

    Returns
    -------
    List of per-tune result dicts (same fields as analyze_one, plus 'generated_tokens').
    """
    index = build_train_index(train_tunes, k=k)

    results = []
    for i, abc in enumerate(generated_abc_list):
        gen_tokens = tokenize_abc(abc)
        result = analyze_one(gen_tokens, index)
        result["tune_index"] = i
        result["generated_tokens"] = gen_tokens
        results.append(result)

    if verbose:
        _print_report(results, train_tunes, k=k)

    return results


def _print_report(results: list[dict], train_tunes: list[dict], k: int = KGRAM_K) -> None:
    n = len(results)
    n_exact = sum(r["exact_match"] for r in results)
    covs = [r["kgram_coverage"] for r in results]
    avg_cov = sum(covs) / n if n else 0.0
    max_cov = max(covs) if covs else 0.0

    print()
    print("=" * 62)
    print("  Memorization Report")
    print("=" * 62)
    print(f"  Generated tunes analyzed : {n}")
    print(f"  K-gram size              : {k}")
    print(f"  Exact matches            : {n_exact} ({100*n_exact/n:.1f}%)")
    print(f"  Avg {k}-gram coverage      : {avg_cov:.1%}")
    print(f"  Max {k}-gram coverage      : {max_cov:.1%}")
    print()

    print(f"  {'#':>3}  {'tokens':>6}  {'coverage':>8}  exact")
    print(f"  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*5}")
    for r in results:
        gen_len = len(r["generated_tokens"])
        print(
            f"  {r['tune_index']:>3}  {gen_len:>6}  "
            f"{r['kgram_coverage']:>7.1%}  "
            f"{'YES' if r['exact_match'] else 'no':>5}"
        )

    print()
    print(f"  Interpretation (k={k}):")
    print("    coverage >= 70%  → heavily derived from training")
    print("    coverage 40–70%  → significant phrase reuse")
    print("    coverage  < 40%  → mostly novel (expected)")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check generated tunes for memorization")
    parser.add_argument("--data", default="data/tunes.json")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--generated", required=True,
                        help="Text file of generated ABC tunes, '---' separated")
    parser.add_argument("--split", default="---")
    parser.add_argument("--k", type=int, default=KGRAM_K, help="K-gram size (default 10)")
    args = parser.parse_args()

    with open(args.data) as f:
        tunes = json.load(f)

    tokenizer = ABCTokenizer.load(args.tokenizer) if args.tokenizer else None

    raw = Path(args.generated).read_text()
    if args.split in raw:
        generated = [b.strip() for b in raw.split(args.split) if b.strip()]
    else:
        generated = [l.strip() for l in raw.splitlines() if l.strip()]

    print(f"Loaded {len(generated)} generated tunes.")
    memorization_report(generated, tunes, tokenizer=tokenizer, k=args.k, verbose=True)


if __name__ == "__main__":
    main()
