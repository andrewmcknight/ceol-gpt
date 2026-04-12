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
import uuid
from pathlib import Path

import gradio as gr
import torch

from src.generate import generate, load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model / tokenizer paths — env vars let HF Spaces override without code changes.
_MODEL_PATH     = os.environ.get("MODEL_PATH",     "models/small/best.pt")
_TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "models/small/tokenizer.pkl")
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


def _build_full_abc(music_body: str, tune_type: str, key: str, meter: str) -> str:
    unit = "1/4" if meter == "3/2" else "1/8"
    return (
        f"X:1\n"
        f"T:Generated {tune_type.title()}\n"
        f"M:{meter}\n"
        f"L:{unit}\n"
        f"K:{_format_key_for_abc(key)}\n"
        f"{music_body}"
    )


# ---------------------------------------------------------------------------
# abcjs HTML renderer
# ---------------------------------------------------------------------------

_ABCJS_VERSION = "6.4.4"

_HTML_TEMPLATE = """\
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/abcjs@{version}/abcjs-audio.css">
<script src="https://cdn.jsdelivr.net/npm/abcjs@{version}/dist/abcjs-basic-min.js"></script>

<div style="background:#fff; padding:16px; border-radius:8px;
            border:1px solid #e2e8f0; font-family:sans-serif;">
  <div id="sheet-{uid}"></div>
  <div id="audio-{uid}" style="margin-top:14px;"></div>
  <p id="no-audio-{uid}" style="display:none; color:#888; font-size:13px;">
    Audio playback is not supported in this browser.
  </p>
</div>

<script>
(function () {{
  var abc = {abc_json};
  var visual = ABCJS.renderAbc("sheet-{uid}", abc, {{
    responsive: "resize",
    add_classes: true,
    staffwidth: 680,
  }});

  if (ABCJS.synth.supportsAudio()) {{
    var ctrl = new ABCJS.synth.SynthController();
    ctrl.load("#audio-{uid}", null, {{
      displayLoop:    true,
      displayRestart: true,
      displayPlay:    true,
      displayProgress: true,
      displayWarp:    false,
    }});
    ctrl.setTune(visual[0], false).catch(function (e) {{
      console.warn("ceol-gpt synth error:", e);
    }});
  }} else {{
    document.getElementById("no-audio-{uid}").style.display = "block";
  }}
}})();
</script>
"""

def _render_html(full_abc: str) -> str:
    """Return an HTML string that renders full_abc with abcjs."""
    return _HTML_TEMPLATE.format(
        version=_ABCJS_VERSION,
        uid=uuid.uuid4().hex[:8],
        abc_json=json.dumps(full_abc),
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

def cb_set_meter(tune_type: str) -> gr.update:
    """Auto-update the meter dropdown when tune type changes."""
    return gr.update(value=_DEFAULT_METER.get(tune_type, "4/4"))


def cb_generate(tune_type: str, key: str, meter: str, temperature: float):
    """Generate a tune and return (abc_text, rendered_html, status)."""
    try:
        music_body = generate(
            _model, _tokenizer,
            tune_type=tune_type,
            key=key,
            meter=meter,
            temperature=temperature,
            max_new_tokens=512,
            top_k=50,
            top_p=0.95,
            device=DEVICE,
        )
        full_abc = _build_full_abc(music_body, tune_type, key, meter)
        return full_abc, _render_html(full_abc), ""
    except Exception as exc:
        return "", _PLACEHOLDER_HTML, f"Generation error: {exc}"


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
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #64748b; margin-top: 0; margin-bottom: 20px; }
#status { color: #dc2626; min-height: 20px; font-size: 13px; }
.gr-button-primary { background: #16a34a !important; border-color: #15803d !important; }
"""

with gr.Blocks(title="ceol-gpt", theme=gr.themes.Soft(), css=_CSS) as demo:

    gr.HTML("<h1 id='title'>🎵 ceol-gpt</h1>")
    gr.HTML("<p id='subtitle'>Irish folk tune generator &mdash; GPT-style transformer trained on 54K tunes</p>")

    with gr.Row():
        # ---- Controls -------------------------------------------------------
        with gr.Column(scale=1, min_width=260):
            tune_type_dd = gr.Dropdown(
                label="Tune type",
                choices=TUNE_TYPES,
                value="reel",
            )
            key_dd = gr.Dropdown(
                label="Key / mode",
                choices=KEYS,
                value="Dmajor",
            )
            meter_dd = gr.Dropdown(
                label="Meter",
                choices=METERS,
                value="4/4",
            )
            temperature_sl = gr.Slider(
                label="Temperature",
                minimum=0.5,
                maximum=1.2,
                step=0.05,
                value=0.8,
                info="Higher = more variety, lower = more predictable",
            )
            generate_btn = gr.Button("Generate", variant="primary")
            status_txt = gr.Markdown("", elem_id="status")

        # ---- Output ---------------------------------------------------------
        with gr.Column(scale=2):
            abc_box = gr.Textbox(
                label="ABC notation (editable)",
                lines=10,
                placeholder="Generated ABC will appear here …",
                show_copy_button=True,
            )
            render_btn = gr.Button("Re-render sheet music", size="sm")
            sheet_html = gr.HTML(_PLACEHOLDER_HTML, label="Sheet music & playback")

    # ---- Wiring -------------------------------------------------------------
    tune_type_dd.change(
        fn=cb_set_meter,
        inputs=tune_type_dd,
        outputs=meter_dd,
    )

    generate_btn.click(
        fn=cb_generate,
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
    )
