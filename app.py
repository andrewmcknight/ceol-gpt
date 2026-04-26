"""
ceol-gpt Gradio demo — generate Irish folk tunes in ABC notation,
render sheet music, and play back audio via abcjs.

Local:
    python app.py

HuggingFace Spaces:
    Set HF_MODEL_REPO to your model repo (e.g. "yourname/ceol-gpt-weights").
    The app downloads best.pt and tokenizer.pkl from that repo on first run.
    MODEL_PATH / TOKENIZER_PATH env vars override the download if set.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import gradio as gr
import torch

from src.generate import generate_stream, load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model / tokenizer paths — env vars let HF Spaces override without code changes.
_MODEL_PATH     = os.environ.get("MODEL_PATH",     "models/large/best.pt")
_TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "models/large/tokenizer.pkl")
_HF_MODEL_REPO  = os.environ.get("HF_MODEL_REPO",  None)   # e.g. "yourname/ceol-gpt-weights"

# Tune types and default meters — mirrors the 12 types in the dataset.
_DEFAULT_METER: dict[str, str] = {
    "barndance":  "4/4",
    "hornpipe":   "4/4",
    "jig":        "6/8",
    "march":      "4/4",
    "mazurka":    "3/4",
    "polka":      "2/4",
    "reel":       "4/4",
    "slide":      "12/8",
    "slip jig":   "9/8",
    "strathspey": "4/4",
    "three-two":  "3/2",
    "waltz":      "3/4",
}

_ALL_TUNE_TYPES = sorted(_DEFAULT_METER.keys())

# Valid meters for each tune type — most types are tied to exactly one meter.
# March and a few others legitimately appear in multiple meters in the dataset.
_VALID_METERS: dict[str, list[str]] = {
    "barndance":  ["4/4"],
    "hornpipe":   ["4/4"],
    "jig":        ["6/8"],
    "march":      ["4/4", "2/4", "6/8"],
    "mazurka":    ["3/4"],
    "polka":      ["2/4"],
    "reel":       ["4/4"],
    "slide":      ["12/8"],
    "slip jig":   ["9/8"],
    "strathspey": ["4/4"],
    "three-two":  ["3/2"],
    "waltz":      ["3/4"],
}

_ALL_KEYS = [
    # Major — most common first
    "Dmajor", "Gmajor", "Amajor", "Emajor", "Bmajor", "Cmajor", "Fmajor",
    # Dorian
    "Adorian", "Ddorian", "Gdorian", "Edorian", "Bdorian",
    # Mixolydian
    "Amixolydian", "Dmixolydian", "Gmixolydian", "Emixolydian",
    # Minor
    "Aminor", "Dminor", "Eminor", "Bminor",
]

_ALL_METERS = ["4/4", "6/8", "3/4", "2/4", "9/8", "12/8", "3/2"]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_paths() -> tuple[Path, Path]:
    """Return (checkpoint_path, tokenizer_path), downloading from HF Hub if needed."""
    ckpt_path = Path(_MODEL_PATH)
    tok_path  = Path(_TOKENIZER_PATH)

    if not ckpt_path.exists() or not tok_path.exists():
        if _HF_MODEL_REPO:
            from huggingface_hub import hf_hub_download
            print(f"Downloading model from {_HF_MODEL_REPO} ...")
            ckpt_path = Path(hf_hub_download(_HF_MODEL_REPO, "best.pt"))
            tok_path  = Path(hf_hub_download(_HF_MODEL_REPO, "tokenizer.pkl"))
        else:
            raise FileNotFoundError(
                f"Model not found at {ckpt_path}. "
                "Set HF_MODEL_REPO env var or copy weights to models/large/."
            )

    return ckpt_path, tok_path


print("Loading model …")
_ckpt_path, _tok_path = _resolve_paths()
_model, _tokenizer = load_model(_ckpt_path, _tok_path, device=DEVICE)
print(f"Model loaded  |  vocab: {len(_tokenizer):,}  |  device: {DEVICE}")

# Filter displayed options to keys that are actually in the vocabulary.
TUNE_TYPES = [t for t in _ALL_TUNE_TYPES if f"<TYPE:{t}>"  in _tokenizer.vocab.token_to_id]
KEYS       = [k for k in _ALL_KEYS       if f"<KEY:{k}>"   in _tokenizer.vocab.token_to_id]
METERS     = [m for m in _ALL_METERS     if f"<METER:{m}>" in _tokenizer.vocab.token_to_id]

# ---------------------------------------------------------------------------
# Key distribution per tune type (from training data)
# ---------------------------------------------------------------------------

def _build_key_dist(data_path: str = "data/tunes.json") -> dict[str, dict[str, int]]:
    """Return {tune_type: {key: count}} from the dataset, filtered to vocab keys."""
    valid_keys = set(KEYS)
    dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    try:
        with open(data_path) as f:
            tunes = json.load(f)
        for tune in tunes:
            t, k = tune.get("type"), tune.get("mode")
            if t and k and k in valid_keys:
                dist[t][k] += 1
    except Exception:
        pass
    return {t: dict(counts) for t, counts in dist.items()}


_KEY_DIST = _build_key_dist()


def _sample_key(tune_type: str) -> str:
    """Sample a key weighted by its frequency for the given tune type.

    Falls back to uniform sampling over all KEYS if no distribution data exists.
    """
    counts = _KEY_DIST.get(tune_type, {})
    if not counts:
        return random.choice(KEYS)
    keys, weights = zip(*counts.items())
    return random.choices(keys, weights=weights, k=1)[0]


KEYS_WITH_RANDOM = ["Random"] + KEYS

# ---------------------------------------------------------------------------
# ABC helpers
# ---------------------------------------------------------------------------

def _format_key_for_abc(key: str) -> str:
    """Convert vocab key token (e.g. 'Gmajor') to ABC K: field value (e.g. 'G')."""
    for mode, abc_suffix in [
        ("mixolydian", "mix"),
        ("dorian",     "dor"),
        ("lydian",     "lyd"),
        ("phrygian",   "phr"),
        ("locrian",    "loc"),
        ("minor",      "m"),
        ("major",      ""),
    ]:
        if key.lower().endswith(mode):
            return key[: -len(mode)] + abc_suffix
    return key


def _build_headers(tune_type: str, key: str, meter: str) -> str:
    unit = "1/4" if meter == "3/2" else "1/8"
    return (
        f"X:1\n"
        f"T:Generated {tune_type.title()}\n"
        f"M:{meter}\n"
        f"L:{unit}\n"
        f"K:{_format_key_for_abc(key)}"
    )


_BAR_TOKENS = {'|', '||', '|:', ':|', '::', '|]', '[|'}

def _wrap_bars(music_body: str, bars_per_line: int = 4) -> str:
    """Insert newlines every `bars_per_line` bars for readable ABC output.

    This is purely a display post-process — the tokenizer and model are
    unchanged.  abcjs uses newlines to decide where to wrap staff lines.
    """
    tokens = music_body.split()
    lines: list[list[str]] = []
    current: list[str] = []
    bar_count = 0

    for tok in tokens:
        current.append(tok)
        if tok in _BAR_TOKENS:
            bar_count += 1
            if bar_count % bars_per_line == 0:
                lines.append(current)
                current = []

    if current:
        lines.append(current)

    return "\n".join(" ".join(line) for line in lines)


def _build_full_abc(music_body: str, tune_type: str, key: str, meter: str) -> str:
    return _build_headers(tune_type, key, meter) + "\n" + _wrap_bars(music_body)


# ---------------------------------------------------------------------------
# abcjs HTML renderer
# ---------------------------------------------------------------------------
# Gradio 6 sanitises <script> tags in gr.HTML(), so we use an iframe with
# srcdoc.  The iframe is its own browsing context — scripts run normally.

_ABCJS_VERSION = "6.4.4"


def _render_html(full_abc: str) -> str:
    """Return an iframe srcdoc string that renders full_abc with abcjs."""
    abc_js = json.dumps(full_abc)   # safely escaped JS string literal

    inner = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/abcjs@{_ABCJS_VERSION}/abcjs-audio.css">
<style>
  body  {{ margin:0; padding:12px 16px; background:#fff; font-family:sans-serif; }}
  #sheet svg {{ max-width:100%; height:auto; }}
  .abcjs-inline-audio {{ width:100%; margin-top:12px; }}
</style>
</head>
<body>
<div id="sheet"></div>
<div id="audio"></div>
<script src="https://cdn.jsdelivr.net/npm/abcjs@{_ABCJS_VERSION}/dist/abcjs-basic-min.js"></script>
<script>
(function() {{
  var visual = ABCJS.renderAbc("sheet", {abc_js}, {{responsive:"resize", add_classes:true}});
  if (ABCJS.synth.supportsAudio()) {{
    var ctrl = new ABCJS.synth.SynthController();
    ctrl.load("#audio", null, {{displayLoop:true, displayRestart:true,
                               displayPlay:true, displayProgress:true, displayWarp:false}});
    ctrl.setTune(visual[0], false).catch(function(e){{ console.warn("synth:", e); }});
  }}
}})();
</script>
</body>
</html>"""

    # Encode for srcdoc double-quoted attribute: escape & and " only.
    srcdoc = inner.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:460px;border:none;border-radius:8px;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


_PLACEHOLDER_HTML = (
    "<div style='padding:40px; text-align:center; color:#94a3b8; "
    "border:2px dashed #e2e8f0; border-radius:8px; font-family:sans-serif;'>"
    "Sheet music will appear here after you generate a tune."
    "</div>"
)

# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def cb_set_meter(tune_type: str):
    """Update meter dropdown choices and default when tune type changes."""
    valid   = [m for m in _VALID_METERS.get(tune_type, _ALL_METERS) if m in METERS]
    default = _DEFAULT_METER.get(tune_type, valid[0] if valid else "4/4")
    return gr.update(choices=valid, value=default)


def cb_generate_stream(tune_type: str, key: str, meter: str, temperature: float):
    """Stream tokens into the textbox as they are generated, then render."""
    if key == "Random":
        key = _sample_key(tune_type)
        key_note = f"Randomly selected key: **{key}**  |  Generating…"
    else:
        key_note = "Generating…"

    headers = _build_headers(tune_type, key, meter)
    # Show the header block immediately so the textbox isn't empty while waiting
    yield headers + "\n", _PLACEHOLDER_HTML, key_note

    music_tokens: list[str] = []
    try:
        for token in generate_stream(
            _model, _tokenizer,
            tune_type=tune_type,
            key=key,
            meter=meter,
            temperature=temperature,
            max_new_tokens=512,
            top_k=50,
            top_p=0.95,
            device=DEVICE,
        ):
            music_tokens.append(token)
            # Yield every 4 tokens to keep UI responsive without flooding
            if len(music_tokens) % 4 == 0:
                partial = headers + "\n" + _wrap_bars(" ".join(music_tokens))
                yield partial, _PLACEHOLDER_HTML, key_note

        full_abc = _build_full_abc(" ".join(music_tokens), tune_type, key, meter)
        yield full_abc, _render_html(full_abc), ""

    except Exception as exc:
        partial = headers + "\n" + _wrap_bars(" ".join(music_tokens))
        yield partial, _PLACEHOLDER_HTML, f"Error: {exc}"


def cb_render(abc_text: str):
    """Re-render edited ABC text."""
    if not abc_text.strip():
        return _PLACEHOLDER_HTML, ""
    try:
        return _render_html(abc_text), ""
    except Exception as exc:
        return _PLACEHOLDER_HTML, f"Render error: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500&display=swap');

body, .gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
h1, h2, h3, #title { font-family: 'Lora', Georgia, serif !important; }

#title { text-align: center; font-size: 2rem; font-weight: 600;
         margin-bottom: 2px; color: #1e293b; }
#subtitle { text-align: center; font-family: 'Inter', sans-serif !important;
            color: #64748b; margin-top: 0; margin-bottom: 24px; font-size: 0.95rem; }
#status { color: #dc2626; min-height: 20px; font-size: 0.82rem; }
button.primary { background: #16a34a !important; border-color: #15803d !important; }

/* Shrink render button to hug its text */
#render-btn { width: fit-content !important; min-width: 0 !important; }
"""

with gr.Blocks(title="ceol-gpt") as demo:

    gr.HTML("<h1 id='title'>🎵 ceol-gpt</h1>")
    gr.HTML("<p id='subtitle'>Irish folk tune generator - GPT-style transformer trained on 54K tunes</p>")

    with gr.Row():
        # ---- Controls -------------------------------------------------------
        with gr.Column(scale=1, min_width=180):
            tune_type_dd = gr.Dropdown(
                label="Tune type",
                choices=TUNE_TYPES,
                value="reel",
            )
            key_dd = gr.Dropdown(
                label="Key / mode",
                choices=KEYS_WITH_RANDOM,
                value="Dmajor",
            )
            meter_dd = gr.Dropdown(
                label="Meter",
                choices=["4/4"],   # updated dynamically when tune type changes
                value="4/4",
            )
            temperature_sl = gr.Slider(
                label="Temperature",
                minimum=0.5,
                maximum=1.2,
                step=0.05,
                value=0.8,
            )
            generate_btn = gr.Button("Generate", variant="primary")
            status_txt = gr.Markdown("", elem_id="status")

        # ---- Output ---------------------------------------------------------
        with gr.Column(scale=3):
            abc_box = gr.Textbox(
                label="ABC notation (editable)",
                lines=10,
                placeholder="Generated ABC will appear here …",
            )
            with gr.Row(elem_id="abc-btn-row"):
                render_btn = gr.Button("Re-render sheet music", size="sm", elem_id="render-btn")
            sheet_html = gr.HTML(_PLACEHOLDER_HTML, label="Sheet music & playback")

    # ---- Wiring -------------------------------------------------------------
    tune_type_dd.change(
        fn=cb_set_meter,
        inputs=tune_type_dd,
        outputs=meter_dd,
    )

    generate_btn.click(
        fn=cb_generate_stream,
        inputs=[tune_type_dd, key_dd, meter_dd, temperature_sl],
        outputs=[abc_box, sheet_html, status_txt],
    )

    render_btn.click(
        fn=cb_render,
        inputs=abc_box,
        outputs=[sheet_html, status_txt],
    )

    # Auto re-render when the user finishes editing (on blur / submit)
    abc_box.submit(
        fn=cb_render,
        inputs=abc_box,
        outputs=[sheet_html, status_txt],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # needed for containers / HF Spaces
        share=False,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
