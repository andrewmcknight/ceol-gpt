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
# N-gram index: build once, query many times
# ---------------------------------------------------------------------------

class NGramIndex:
    """Precomputed set of all k-grams from a corpus, for fast longest-match queries.

    Building the index is O(N * avg_len).
    Querying for the longest match length is O(log(max_len) * gen_len).
    Both are vastly faster than the naive O(n*m) pairwise DP.
    """

    def __init__(self, seqs: list[list[str]]):
        # exact match: map each full sequence (as tuple) → its index
        self._exact: dict[tuple, int] = {}
        # kgram sets keyed by length, built lazily as needed
        self._kgram_cache: dict[int, set[tuple]] = {}
        self._seqs = seqs

        for idx, seq in enumerate(seqs):
            key = tuple(seq)
            if key not in self._exact:
                self._exact[key] = idx

    def _kgrams_of_length(self, k: int) -> set[tuple]:
        if k not in self._kgram_cache:
            s: set[tuple] = set()
            for seq in self._seqs:
                if len(seq) >= k:
                    for i in range(len(seq) - k + 1):
                        s.add(tuple(seq[i:i + k]))
            self._kgram_cache[k] = s
        return self._kgram_cache[k]

    def exact_match_index(self, seq: list[str]) -> int:
        """Return training index of exact match, or -1."""
        return self._exact.get(tuple(seq), -1)

    def longest_match_len(self, gen: list[str]) -> int:
        """Binary search for the longest k such that some k-gram of gen appears
        in the training corpus. Returns that length (0 if no match at all)."""
        if not gen:
            return 0

        lo, hi = 1, len(gen)
        result = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            kgrams = self._kgrams_of_length(mid)
            found = any(tuple(gen[i:i + mid]) in kgrams for i in range(len(gen) - mid + 1))
            if found:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return result


# ---------------------------------------------------------------------------
# Build index from training tunes
# ---------------------------------------------------------------------------

def build_train_index(tunes: list[dict]) -> NGramIndex:
    """Tokenize all training tunes and build an NGramIndex."""
    seqs = [tokenize_abc(t["abc"]) for t in tunes]
    return NGramIndex(seqs)


# ---------------------------------------------------------------------------
# Per-tune analysis
# ---------------------------------------------------------------------------

def analyze_one(
    generated_tokens: list[str],
    index: NGramIndex,
) -> dict:
    """Compute novelty metrics for one generated tune.

    Returns a dict with:
      - exact_match (bool)
      - longest_match_tokens (int)
      - longest_match_frac (float)  — fraction of generated tune length
      - closest_train_idx (int)     — index of exact match (-1 if none)
    """
    if not generated_tokens:
        return {
            "exact_match": False,
            "longest_match_tokens": 0,
            "longest_match_frac": 0.0,
            "closest_train_idx": -1,
        }

    exact_idx = index.exact_match_index(generated_tokens)
    exact = exact_idx >= 0

    if exact:
        best_len = len(generated_tokens)
    else:
        best_len = index.longest_match_len(generated_tokens)

    gen_len = len(generated_tokens)
    return {
        "exact_match": exact,
        "longest_match_tokens": best_len,
        "longest_match_frac": best_len / gen_len if gen_len else 0.0,
        "closest_train_idx": exact_idx,
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
    tokenizer          : optional, unused — kept for API compatibility
    verbose            : print a summary table

    Returns
    -------
    List of per-tune result dicts (same fields as analyze_one, plus 'generated_tokens').
    """
    print(f"Indexing {len(train_tunes):,} training tunes...")
    index = build_train_index(train_tunes)
    print("Index built.")

    results = []
    for i, abc in enumerate(generated_abc_list):
        gen_tokens = tokenize_abc(abc)
        result = analyze_one(gen_tokens, index)
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
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data) as f:
        tunes = json.load(f)

    tokenizer = None
    if args.tokenizer:
        tokenizer = ABCTokenizer.load(args.tokenizer)

    raw = Path(args.generated).read_text()
    if args.split in raw:
        generated = [block.strip() for block in raw.split(args.split) if block.strip()]
    else:
        generated = [line.strip() for line in raw.splitlines() if line.strip()]

    print(f"Loaded {len(generated)} generated tunes.")
    memorization_report(generated, tunes, tokenizer=tokenizer, verbose=True)


if __name__ == "__main__":
    main()
