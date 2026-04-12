"""
Word-based ABC notation tokenizer for ceol-gpt.

Each musical "word" becomes one token: a note (with accidental + octave + duration),
a bar line, a rest, a decoration, a tuplet marker, etc.

Special conditioning tokens are prepended at encode time:
    <TYPE:reel> <KEY:Gmajor> <BOS> ... music tokens ... <EOS>
"""

from __future__ import annotations

import re
import json
import pickle
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Regex patterns (order matters — first match wins)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""(?x)
    # --- Strip: grace notes, chord symbols, bang-decorations, comments ---
    \{[^}]*\}                        # grace notes {abc}
  | "[^"]*"                          # chord symbols "G", "Am7"
  | ![^!]*!                          # long decorations !trill! !f!
  | %[^\n]*                          # comments % ...

    # --- Inline chords [CEG]: captured whole, first note extracted in caller ---
  | \[(?=[A-Ga-g\^_=])[^\]]*\]      # inline chords [CEG] [FA] [G2D2B2] etc.

    # --- Ending markers ([1, [2, |1, |2 etc.) ---
  | \[\d                             # [1  [2  [3
  | \|[\d]                           # |1  |2  (alternate ending syntax)

    # --- Bar lines (longest first) ---
  | ::                               # double repeat ::
  | \|:                              # start repeat |:
  | :\|                              # end repeat :|
  | \|\|                             # double bar ||
  | \|\]                             # final bar |]
  | \[\|                             # thick-thin bar [|
  | \|                               # simple bar |

    # --- Tuplets ---
  | \(\d                             # (3  (2  (4  (5  (6  (7

    # --- Single-character decorations (kept as own tokens) ---
  | [~.THuvJLMPS](?=[A-Ga-g\^_=])   # decoration immediately before a note

    # --- Notes and rests ---
    # Accidental + pitch + octave modifiers + duration
  | (?:\^\^|\^|__|_|=)?              # optional accidental
    [A-Ga-gzxZ]                      # pitch letter (z/x = rest, Z = full-bar rest)
    [,']*                            # octave modifiers , and '
    (?:\d+\/\d+|\d+\/|\/\d+|\/+|\d+)?  # duration: 3/2  2/  /2  /  //  2  (or nothing)

    # --- Broken rhythm ---
  | >{1,3}                           # >  >>  >>>
  | <{1,3}                           # <  <<  <<<

    # --- Tie ---
  | -

    # --- Slur delimiters (phrasing only — discard content via skip-in-caller) ---
  | [()]
""")

# Characters/sequences to treat as whitespace and discard
_STRIP_RE = re.compile(r"[\r\n\t \\]+")

# Matches a single note at the start of a string (used to extract from inline chords)
_FIRST_NOTE_RE = re.compile(
    r"(?:\^\^|\^|__|_|=)?[A-Ga-gzxZ][,']*(?:\d+\/\d+|\d+\/|\/\d+|\/+|\d+)?"
)

# Matches all notes inside a grace note group (grace notes rarely have durations,
# but the pattern is permissive to handle any that do)
_GRACE_NOTE_RE = re.compile(
    r"(?:\^\^|\^|__|_|=)?[A-Ga-g][,']*(?:\d+\/\d+|\d+\/|\/\d+|\/+|\d+)?"
)


def _first_note_from_chord(chord: str) -> str | None:
    """Extract the first note from an inline chord string like '[G2D2B2]' → 'G2'."""
    inner = chord[1:-1]  # strip [ and ]
    m = _FIRST_NOTE_RE.match(inner)
    return m.group() if m else None


def _tokenize_grace(grace: str) -> list[str]:
    """Expand a grace note group like '{GBd}' into individual note tokens."""
    inner = grace[1:-1]  # strip { and }
    return _GRACE_NOTE_RE.findall(inner)


def _tokenize_chord(chord: str) -> list[str]:
    """Expand an inline chord like '[G2D2B2]' into individual note tokens."""
    inner = chord[1:-1]  # strip [ and ]
    return _GRACE_NOTE_RE.findall(inner)


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
GRACE_START_TOKEN = "<GRACE_START>"
GRACE_END_TOKEN = "<GRACE_END>"
CHORD_START_TOKEN = "<CHORD_START>"
CHORD_END_TOKEN = "<CHORD_END>"

# All tokens that get reserved IDs at the front of the vocab
_SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
                   GRACE_START_TOKEN, GRACE_END_TOKEN,
                   CHORD_START_TOKEN, CHORD_END_TOKEN]

# Tokens that are stripped from ABC output during decode (conditioning + padding)
_CONTROL_TOKENS = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}


def _type_token(tune_type: str) -> str:
    return f"<TYPE:{tune_type}>"


def _key_token(key: str) -> str:
    return f"<KEY:{key}>"


def _meter_token(meter: str) -> str:
    return f"<METER:{meter}>"


# ---------------------------------------------------------------------------
# Tokenize a single ABC music string → list[str]
# ---------------------------------------------------------------------------

# Tokens that are purely syntactic and we discard
_DISCARD = {"(", ")"}


def tokenize_abc(abc: str) -> list[str]:
    """Tokenize one ABC music body string into a list of string tokens.

    Grace notes are expanded: {GBd} → <GRACE_START> G B d <GRACE_END>.
    Chord symbols are kept as atomic tokens: "Am7" → "Am7".
    Bang decorations, comments, slur markers, whitespace discarded.
    """
    tokens = []
    for m in _TOKEN_RE.finditer(abc):
        tok = m.group()
        if not tok:
            continue
        first = tok[0]

        # Grace notes: expand into framing tokens + individual notes
        if first == '{':
            inner = _tokenize_grace(tok)
            if inner:
                tokens.append(GRACE_START_TOKEN)
                tokens.extend(inner)
                tokens.append(GRACE_END_TOKEN)
            continue
        # Chord symbols: keep as atomic token
        if first == '"':
            tokens.append(tok)
            continue
        # Discard bang decorations and comments
        if first in ('!', '%'):
            continue
        # Inline chords ([CEG], [G2D2B2]) — expand with framing tokens
        if first == '[' and len(tok) > 1 and not tok[1].isdigit():
            notes = _tokenize_chord(tok)
            if notes:
                tokens.append(CHORD_START_TOKEN)
                tokens.extend(notes)
                tokens.append(CHORD_END_TOKEN)
            continue
        # Discard slur markers
        if tok in _DISCARD:
            continue
        # Discard whitespace fragments that sneak through
        if _STRIP_RE.fullmatch(tok):
            continue
        # Discard lone digits (leftover from |1 ending patterns already captured)
        if tok.isdigit():
            continue

        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class ABCVocab:
    """Mapping between string tokens and integer IDs."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.unk_id
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.token_to_id, f)

    @classmethod
    def load(cls, path: str | Path) -> "ABCVocab":
        with open(path, "rb") as f:
            token_to_id = pickle.load(f)
        return cls(token_to_id)


def build_vocab(
    tunes: list[dict],
    min_freq: int = 2,
) -> ABCVocab:
    """Build vocabulary by tokenizing all tunes and counting token frequencies.

    Special conditioning tokens for every (type, key) pair are added automatically.
    Tokens appearing fewer than `min_freq` times are replaced with <UNK>.
    """
    counter: Counter = Counter()

    # Collect conditioning token strings
    type_tokens = {_type_token(t["type"]) for t in tunes}
    key_tokens = {_key_token(t["mode"]) for t in tunes}
    meter_tokens = {_meter_token(t["meter"]) for t in tunes}

    # Tokenize all ABC bodies
    for tune in tunes:
        tokens = tokenize_abc(tune["abc"])
        counter.update(tokens)

    # Build ordered vocab: specials first, then conditioning, then music tokens
    vocab_tokens = list(_SPECIAL_TOKENS)
    vocab_tokens.extend(sorted(type_tokens))
    vocab_tokens.extend(sorted(key_tokens))
    vocab_tokens.extend(sorted(meter_tokens))
    # Music tokens filtered by min_freq, sorted for determinism
    music_tokens = sorted(tok for tok, cnt in counter.items() if cnt >= min_freq)
    vocab_tokens.extend(music_tokens)

    token_to_id = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    return ABCVocab(token_to_id)


# ---------------------------------------------------------------------------
# Full encode / decode with conditioning tokens
# ---------------------------------------------------------------------------

class ABCTokenizer:
    """High-level tokenizer: tokenizes ABC music and manages conditioning tokens.

    Usage::

        tokenizer = ABCTokenizer.from_tunes(tunes)
        ids = tokenizer.encode(abc, tune_type="reel", key="Gmajor")
        back = tokenizer.decode_to_abc(ids)
    """

    def __init__(self, vocab: ABCVocab):
        self.vocab = vocab

    # -- construction -------------------------------------------------------

    @classmethod
    def from_tunes(cls, tunes: list[dict], min_freq: int = 2) -> "ABCTokenizer":
        """Build tokenizer from a list of tune dicts (as loaded from tunes.json)."""
        vocab = build_vocab(tunes, min_freq=min_freq)
        return cls(vocab)

    @classmethod
    def load(cls, path: str | Path) -> "ABCTokenizer":
        vocab = ABCVocab.load(path)
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        self.vocab.save(path)

    # -- properties ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab.pad_id

    @property
    def bos_id(self) -> int:
        return self.vocab.bos_id

    @property
    def eos_id(self) -> int:
        return self.vocab.eos_id

    # -- encode / decode ----------------------------------------------------

    def encode(
        self,
        abc: str,
        tune_type: str,
        key: str,
        meter: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode one tune to a list of token IDs.

        Output format:
            <TYPE:reel> <KEY:Gmajor> <METER:4/4> <BOS> tok tok tok ... <EOS>
        """
        prefix = [_type_token(tune_type), _key_token(key), _meter_token(meter)]
        if add_bos:
            prefix.append(BOS_TOKEN)
        music_tokens = tokenize_abc(abc)
        suffix = [EOS_TOKEN] if add_eos else []
        return self.vocab.encode(prefix + music_tokens + suffix)

    def decode_to_tokens(self, ids: list[int]) -> list[str]:
        """Decode IDs back to string tokens (includes special tokens)."""
        return self.vocab.decode(ids)

    def decode_to_abc(self, ids: list[int]) -> str:
        """Decode IDs back to an ABC music string, stripping control/conditioning tokens
        and reconstructing grace note groups as {notes}."""
        tokens = self.vocab.decode(ids)
        music = [
            t for t in tokens
            if t not in _CONTROL_TOKENS
            and not t.startswith("<TYPE:")
            and not t.startswith("<KEY:")
            and not t.startswith("<METER:")
        ]
        # Reconstruct grace note and inline chord groups from framing tokens
        result = []
        i = 0
        while i < len(music):
            if music[i] == GRACE_START_TOKEN:
                notes = []
                i += 1
                while i < len(music) and music[i] != GRACE_END_TOKEN:
                    notes.append(music[i])
                    i += 1
                result.append("{" + "".join(notes) + "}")
                i += 1  # skip GRACE_END_TOKEN
            elif music[i] == CHORD_START_TOKEN:
                notes = []
                i += 1
                while i < len(music) and music[i] != CHORD_END_TOKEN:
                    notes.append(music[i])
                    i += 1
                result.append("[" + "".join(notes) + "]")
                i += 1  # skip CHORD_END_TOKEN
            else:
                result.append(music[i])
                i += 1
        return " ".join(result)


# ---------------------------------------------------------------------------
# CLI: build and save vocab from tunes.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ABC tokenizer vocabulary")
    parser.add_argument("--data", default="data/tunes.json")
    parser.add_argument("--out", default="models/tokenizer.pkl")
    parser.add_argument("--min-freq", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data) as f:
        tunes = json.load(f)

    print(f"Building vocab from {len(tunes):,} tunes (min_freq={args.min_freq})...")
    tokenizer = ABCTokenizer.from_tunes(tunes, min_freq=args.min_freq)

    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Sample tokens: {list(tokenizer.vocab.token_to_id.keys())[4:24]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"Saved to {args.out}")
