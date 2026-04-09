# ABC Notation — Quick Reference

ABC is a plain-text music notation system. Notes, bar lines, and musical symbols are written as ASCII characters; metadata goes in header fields above the music.

---

## Anatomy of a Tune

```
%abc-2.1              ← version declaration (optional but recommended)
X:1                   ← reference number (required)
T:Scarborough Fair    ← title
C:Trad.               ← composer
M:3/4                 ← meter (time signature)
L:1/8                 ← unit note length — what a bare letter equals
Q:1/4=100             ← tempo (quarter note = 100 bpm)
K:Amin                ← key — also marks the END of the header
A2 AG E2|c3 B AG|A4 GE|D6|
w:Are you go-ing to Scar-bor-ough Fair?
```

The header ends at `K:`. Everything after is music code.

---

## Pitch

| Code | Note |
|---|---|
| `C D E F G A B` | Octave below middle C (uppercase) |
| `c d e f g a b` | Octave at/above middle C (lowercase) |
| `c'` `c''` | Each `'` raises one octave |
| `C,` `C,,` | Each `,` lowers one octave |

---

## Accidentals

Placed immediately **before** the note letter. They apply for the rest of the bar.

| Code | Meaning |
|---|---|
| `^F` | F sharp |
| `^^F` | F double sharp |
| `_B` | B flat |
| `__B` | B double flat |
| `=C` | C natural (cancel) |

---

## Note Lengths

Set `L:1/8` (eighth note as the unit) — most common for folk and traditional music. A multiplier or divisor after the letter scales from there.

| Code | Duration (with `L:1/8`) |
|---|---|
| `A` | Eighth note |
| `A2` | Quarter note |
| `A3` | Dotted quarter note |
| `A4` | Half note |
| `A8` | Whole note |
| `A/2` or `A/` | Sixteenth note |
| `A3/2` | Dotted eighth note |

> **Default when `L:` is omitted:** if the meter as a decimal is less than 0.75, the default is `1/16`; if 0.75 or greater, it is `1/8`. For `M:C`, `M:C|`, and `M:none`, the default is `1/8`.

---

## Rests

| Code | Meaning |
|---|---|
| `z` | Visible rest (same length rules as notes) |
| `x` | Invisible rest |
| `z2` `z4` | Longer rests via multiplier |
| `Z4` | Four full measures of rest |

---

## Bar Lines & Repeats

| Code | Meaning |
|---|---|
| `\|` | Bar line |
| `\|\|` | Double bar line |
| `\|]` | Final bar line (end of piece) |
| `\|:` | Start repeat |
| `:\|` | End repeat |
| `::` | End + start repeat |
| `[1 … :\|[2 … \|]` | First and second endings |

---

## Broken Rhythm (Dots)

| Code | Meaning |
|---|---|
| `A>B` | A dotted, B halved (long–short) |
| `A<B` | A halved, B dotted (short–long) |
| `A>>B` | A double-dotted, B quartered |

`a>b c<d` is equivalent to `a3/2b/2 c/2d3/2`. Common in hornpipes and strathspeys.

---

## Ties, Slurs & Grace Notes

| Code | Meaning |
|---|---|
| `c4-c2` | Tie — same pitch, played as one note |
| `(CDEG)` | Slur — phrasing mark over any notes |
| `{g}A` | Grace note `g` before `A` |
| `{/g}A` | Acciaccatura (crushed grace note) |

> **Tie vs. slur:** a tie connects two notes of the *same pitch* into one sustained note; a slur is a phrasing mark and does not merge notes.

---

## Common Decorations

Placed immediately before the note they decorate.

| Code | Meaning |
|---|---|
| `.A` | Staccato |
| `~A` | Irish roll |
| `TA` | Trill |
| `HA` | Fermata (hold) |
| `uA` / `vA` | Up-bow / down-bow |
| `!p!` `!f!` `!mf!` | Dynamics (pp, p, mp, mf, f, ff, fff…) |
| `!trill!` `!mordent!` | Full decoration syntax for any symbol |

---

## Chords, Tuplets & Beams

| Code | Meaning |
|---|---|
| `[CEG]` | Simultaneous notes (chord) |
| `(3abc` | Triplet — 3 notes in the time of 2 |
| `(2ab` | Duplet — 2 notes in the time of 3 |
| `CDEF` | Beamed together (no spaces between) |
| `C D E F` | Not beamed (spaces separate) |
| `"G"G4` | Chord symbol printed above the note |

---

## Key Signatures

```
K:G        G major
K:Dm       D minor
K:Bb       B-flat major
K:Ador     A Dorian
K:Cmix     C Mixolydian
K:C        C major (no sharps or flats)
```

Mode suffixes: `min` `mix` `dor` `phr` `lyd` `loc`

---

## Comments & File Structure

| Code | Meaning |
|---|---|
| `%` | Everything after this on the line is a comment |
| `[r:text]` | Inline remark inside music code |
| `\` at end of line | Continue music onto next line (no score line-break) |
| Empty line | Separates tunes from each other |
| `+:` | Continues an information field onto the next line |

---

## Minimal Working Tune

```
%abc-2.1
X:1
T:Untitled
M:4/4
L:1/8
K:G
GABG DEFD|GABG G4|ABAG EFGE|ABAG A4:|
```