// Build the two slide decks for ceol-gpt videos.
//   node videos/build_slides.js
// Outputs:
//   videos/demo_slides.pptx
//   videos/technical_walkthrough_slides.pptx

const pptxgen = require("pptxgenjs");
const path = require("path");

// ---- Palette: Forest & Moss (Irish music feel) ----
const FOREST = "1F4D2C";   // deep forest green (primary)
const MOSS   = "5C8A4A";   // moss (secondary)
const CREAM  = "F5F2E8";   // warm cream (light bg)
const INK    = "1A1F1A";   // near-black for text on light
const MUTED  = "6B7368";   // gray for captions
const ACCENT = "C2410C";   // burnt orange (sparingly, for warnings)
const GOLD   = "B8893A";   // amber (for ✓ ticks)

const HEADER_FONT = "Georgia";
const BODY_FONT   = "Calibri";

// =====================================================================
//  DEMO DECK
// =====================================================================
function buildDemo() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Andrew McKnight";
  pres.title  = "ceol-gpt — demo";

  // ---- Slide 1: Title (dark forest bg) ----
  {
    const s = pres.addSlide();
    s.background = { color: FOREST };

    // Decorative motif: a subtle vertical accent stripe on the left
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: MOSS }, line: { type: "none" },
    });

    s.addText("ceol-gpt", {
      x: 0.8, y: 1.7, w: 8.4, h: 1.4,
      fontFace: HEADER_FONT, fontSize: 88, bold: true, color: CREAM,
      align: "left", valign: "middle", margin: 0,
    });

    s.addText("A transformer that writes Irish folk tunes", {
      x: 0.8, y: 3.05, w: 8.4, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 22, italic: true, color: "D7E4C8",
      align: "left", margin: 0,
    });

    s.addText("Andrew McKnight  ·  CS 372  ·  Spring 2026", {
      x: 0.8, y: 4.85, w: 8.4, h: 0.4,
      fontFace: BODY_FONT, fontSize: 14, color: "9FB18C",
      align: "left", charSpacing: 2, margin: 0,
    });
  }

  // ---- Slide 2: What is ABC notation (two-column) ----
  {
    const s = pres.addSlide();
    s.background = { color: CREAM };

    s.addText("ABC notation: text in, sheet music out", {
      x: 0.5, y: 0.35, w: 9, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: FOREST,
      align: "left", margin: 0,
    });

    // Left column — ABC text in a "code card"
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.15, w: 4.4, h: 3.6,
      fill: { color: "FFFFFF" }, line: { color: "E2DFD2", width: 1 },
    });
    s.addText("ABC NOTATION (TEXT)", {
      x: 0.7, y: 1.3, w: 4.0, h: 0.35,
      fontFace: BODY_FONT, fontSize: 10, bold: true, color: MOSS,
      charSpacing: 3, margin: 0,
    });
    s.addText(
      "X:1\nT:The Kesh\nM:6/8\nL:1/8\nK:Gmaj\n|:G3 GAB|A3 ABd|\nedd gdd|edB dBA|\nG3 GAB|A3 ABd|\nedB dBA|G3 G3:|",
      {
        x: 0.7, y: 1.7, w: 4.0, h: 2.95,
        fontFace: "Consolas", fontSize: 14, color: INK,
        valign: "top", margin: 0,
      }
    );

    // Arrow between columns
    s.addText("→", {
      x: 4.95, y: 2.65, w: 0.6, h: 0.6,
      fontFace: HEADER_FONT, fontSize: 44, color: MOSS,
      align: "center", valign: "middle", margin: 0,
    });

    // Right column — placeholder for sheet music screenshot
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.55, y: 1.15, w: 3.95, h: 3.6,
      fill: { color: "FFFFFF" }, line: { color: "E2DFD2", width: 1 },
    });
    s.addText("RENDERED SHEET MUSIC", {
      x: 5.75, y: 1.3, w: 3.55, h: 0.35,
      fontFace: BODY_FONT, fontSize: 10, bold: true, color: MOSS,
      charSpacing: 3, margin: 0,
    });
    // Placeholder — replace with real screenshot from abcjs.net editor.
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.75, y: 1.8, w: 3.55, h: 2.8,
      fill: { color: "F5F2E8" }, line: { color: "C9C4B0", width: 1, dashType: "dash" },
    });
    s.addText("[ Paste screenshot of the rendered\nstaff here — abcjs.net/abcjs-editor ]", {
      x: 5.75, y: 1.8, w: 3.55, h: 2.8,
      fontFace: BODY_FONT, fontSize: 12, italic: true, color: MUTED,
      align: "center", valign: "middle", margin: 0,
    });

    s.addText("Same tune, two representations. The model reads and writes the left.", {
      x: 0.5, y: 4.95, w: 9, h: 0.4,
      fontFace: HEADER_FONT, fontSize: 14, italic: true, color: MUTED,
      align: "center", margin: 0,
    });
  }

  // ---- Slide 3: Why this matters (icon+text rows) ----
  {
    const s = pres.addSlide();
    s.background = { color: CREAM };

    s.addText("Why this is interesting", {
      x: 0.5, y: 0.35, w: 9, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: FOREST, margin: 0,
    });

    const rows = [
      ["1", "Generative models for structured creative domains",
       "Not just chat or images — symbolic music has its own grammar to learn."],
      ["2", "A living oral tradition",
       "Composition has always been part of Irish music. New tunes enter the repertoire constantly."],
      ["3", "A tool for musicians",
       "Give it a tune type and a key, get a starting point you can take to the session."],
    ];

    rows.forEach(([num, head, body], i) => {
      const y = 1.25 + i * 1.30;
      // Numbered circle
      s.addShape(pres.shapes.OVAL, {
        x: 0.7, y: y, w: 0.85, h: 0.85,
        fill: { color: FOREST }, line: { type: "none" },
      });
      s.addText(num, {
        x: 0.7, y: y, w: 0.85, h: 0.85,
        fontFace: HEADER_FONT, fontSize: 32, bold: true, color: CREAM,
        align: "center", valign: "middle", margin: 0,
      });
      // Header
      s.addText(head, {
        x: 1.85, y: y - 0.05, w: 7.6, h: 0.5,
        fontFace: HEADER_FONT, fontSize: 19, bold: true, color: INK, margin: 0,
      });
      // Body
      s.addText(body, {
        x: 1.85, y: y + 0.42, w: 7.6, h: 0.6,
        fontFace: BODY_FONT, fontSize: 14, color: MUTED, margin: 0,
      });
    });
  }

  // ---- Slide 4: Works / Fumbles (two-column comparison) ----
  {
    const s = pres.addSlide();
    s.background = { color: CREAM };

    s.addText("What works  ·  what still fumbles", {
      x: 0.5, y: 0.35, w: 9, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: FOREST, margin: 0,
    });

    // Left card — Works
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.2, w: 4.4, h: 3.85,
      fill: { color: "FFFFFF" }, line: { color: "E2DFD2", width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.2, w: 4.4, h: 0.55, fill: { color: MOSS }, line: { type: "none" },
    });
    s.addText("WORKS", {
      x: 0.7, y: 1.2, w: 4.0, h: 0.55,
      fontFace: BODY_FONT, fontSize: 13, bold: true, color: "FFFFFF",
      charSpacing: 4, valign: "middle", margin: 0,
    });
    const works = [
      "Stays in the right key and meter",
      "Phrase length and bar structure feel right",
      "Standard cadences (turning on the dominant)",
      "Outputs are playable on a real instrument",
    ];
    works.forEach((t, i) => {
      const y = 2.0 + i * 0.65;
      s.addText("✓", {
        x: 0.75, y: y, w: 0.4, h: 0.5,
        fontFace: HEADER_FONT, fontSize: 22, bold: true, color: GOLD,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(t, {
        x: 1.15, y: y, w: 3.6, h: 0.5,
        fontFace: BODY_FONT, fontSize: 14, color: INK, valign: "middle", margin: 0,
      });
    });

    // Right card — Fumbles
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.1, y: 1.2, w: 4.4, h: 3.85,
      fill: { color: "FFFFFF" }, line: { color: "E2DFD2", width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.1, y: 1.2, w: 4.4, h: 0.55, fill: { color: ACCENT }, line: { type: "none" },
    });
    s.addText("STILL FUMBLES", {
      x: 5.3, y: 1.2, w: 4.0, h: 0.55,
      fontFace: BODY_FONT, fontSize: 13, bold: true, color: "FFFFFF",
      charSpacing: 4, valign: "middle", margin: 0,
    });
    const fumbles = [
      "Long-range coherence — B parts can wander",
      "Occasionally awkward fingering for the instrument",
      "No human-listener evaluation at scale yet",
      "Sometimes generates an unusual ornament choice",
    ];
    fumbles.forEach((t, i) => {
      const y = 2.0 + i * 0.65;
      s.addText("!", {
        x: 5.35, y: y, w: 0.4, h: 0.5,
        fontFace: HEADER_FONT, fontSize: 22, bold: true, color: ACCENT,
        align: "center", valign: "middle", margin: 0,
      });
      s.addText(t, {
        x: 5.75, y: y, w: 3.6, h: 0.5,
        fontFace: BODY_FONT, fontSize: 14, color: INK, valign: "middle", margin: 0,
      });
    });

    s.addText("Honest take: surprisingly idiomatic for a from-scratch model trained in a single project.", {
      x: 0.5, y: 5.2, w: 9, h: 0.35,
      fontFace: HEADER_FONT, fontSize: 13, italic: true, color: MUTED,
      align: "center", margin: 0,
    });
  }

  // ---- Slide 5: Wrap (dark) ----
  {
    const s = pres.addSlide();
    s.background = { color: FOREST };
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: MOSS }, line: { type: "none" },
    });

    s.addText("ceol-gpt", {
      x: 0.8, y: 1.5, w: 8.4, h: 1.0,
      fontFace: HEADER_FONT, fontSize: 64, bold: true, color: CREAM,
      align: "left", margin: 0,
    });
    s.addText("Code, model weights, and the technical walkthrough:", {
      x: 0.8, y: 2.6, w: 8.4, h: 0.4,
      fontFace: BODY_FONT, fontSize: 16, color: "D7E4C8", margin: 0,
    });
    s.addText("github.com/<your-handle>/ceol-gpt", {
      x: 0.8, y: 3.05, w: 8.4, h: 0.55,
      fontFace: "Consolas", fontSize: 22, bold: true, color: CREAM, margin: 0,
    });
    s.addText("Thanks for listening.", {
      x: 0.8, y: 4.7, w: 8.4, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 22, italic: true, color: "9FB18C",
      align: "left", margin: 0,
    });
  }

  return pres.writeFile({ fileName: path.join(__dirname, "demo_slides.pptx") });
}

// =====================================================================
//  TECHNICAL WALKTHROUGH DECK
// =====================================================================
function buildTechnical() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Andrew McKnight";
  pres.title  = "ceol-gpt — technical walkthrough";

  // ---- T1: Problem framing ----
  {
    const s = pres.addSlide();
    s.background = { color: CREAM };

    s.addText("ceol-gpt", {
      x: 0.5, y: 0.35, w: 9, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: FOREST, margin: 0,
    });
    s.addText("Technical walkthrough", {
      x: 0.5, y: 0.85, w: 9, h: 0.35,
      fontFace: BODY_FONT, fontSize: 14, color: MOSS, charSpacing: 4, margin: 0,
    });

    s.addText(
      "A 47M-parameter decoder-only transformer trained from scratch to generate Irish folk tunes in ABC notation, conditioned on tune type and key.",
      {
        x: 0.6, y: 1.7, w: 8.8, h: 1.7,
        fontFace: HEADER_FONT, fontSize: 24, color: INK,
        valign: "top", margin: 0,
      }
    );

    // Three stat badges
    const stats = [
      ["54,246", "tunes from TheSession.org"],
      ["12 / 23", "tune types  ·  key/modes"],
      ["~930", "token vocabulary"],
    ];
    stats.forEach(([num, label], i) => {
      const x = 0.6 + i * 3.0;
      s.addShape(pres.shapes.RECTANGLE, {
        x: x, y: 4.1, w: 2.8, h: 1.1,
        fill: { color: "FFFFFF" }, line: { color: "E2DFD2", width: 1 },
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x: x, y: 4.1, w: 0.08, h: 1.1, fill: { color: FOREST }, line: { type: "none" },
      });
      s.addText(num, {
        x: x + 0.2, y: 4.18, w: 2.55, h: 0.55,
        fontFace: HEADER_FONT, fontSize: 26, bold: true, color: FOREST, margin: 0,
      });
      s.addText(label, {
        x: x + 0.2, y: 4.7, w: 2.55, h: 0.4,
        fontFace: BODY_FONT, fontSize: 12, color: MUTED, margin: 0,
      });
    });
  }

  // ---- T2: Architecture diagram (drawn with shapes) ----
  {
    const s = pres.addSlide();
    s.background = { color: CREAM };

    s.addText("Architecture", {
      x: 0.5, y: 0.3, w: 9, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: FOREST, margin: 0,
    });
    s.addText("GPT-2 style decoder-only transformer  ·  pre-norm  ·  weight tying", {
      x: 0.5, y: 0.82, w: 9, h: 0.35,
      fontFace: BODY_FONT, fontSize: 13, color: MUTED, margin: 0,
    });

    // Center column box layout
    const cx = 2.0;     // left edge of stack
    const cw = 4.6;     // width of stack
    const boxH = 0.55;
    const gap  = 0.20;
    let y = 1.4;

    function box(label, sub, opts = {}) {
      const fill = opts.fill || "FFFFFF";
      const stroke = opts.stroke || "C9C4B0";
      const txtColor = opts.color || INK;
      const h = opts.h || boxH;
      s.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: y, w: cw, h: h,
        fill: { color: fill }, line: { color: stroke, width: 1 },
      });
      if (sub) {
        s.addText(label, {
          x: cx + 0.15, y: y + 0.04, w: cw - 0.3, h: h * 0.55,
          fontFace: BODY_FONT, fontSize: 13, bold: true, color: txtColor,
          valign: "middle", margin: 0,
        });
        s.addText(sub, {
          x: cx + 0.15, y: y + h * 0.50, w: cw - 0.3, h: h * 0.50,
          fontFace: BODY_FONT, fontSize: 11, color: MUTED,
          valign: "middle", margin: 0,
        });
      } else {
        s.addText(label, {
          x: cx + 0.15, y: y, w: cw - 0.3, h: h,
          fontFace: BODY_FONT, fontSize: 13, bold: true, color: txtColor,
          valign: "middle", margin: 0,
        });
      }
      y += h;
    }
    function arrow() {
      s.addText("▼", {
        x: cx, y: y, w: cw, h: gap,
        fontFace: BODY_FONT, fontSize: 12, color: MOSS,
        align: "center", valign: "middle", margin: 0,
      });
      y += gap;
    }

    // Input
    box("Input tokens", "<TYPE:reel>  <KEY:Dmajor>  <METER:4/4>  <BOS>  G2 BG …",
        { fill: "EFEAD8", h: 0.7 });
    arrow();
    // Embedding
    box("Token embedding  +  positional embedding", "vocab 930 → d_model 512", { h: 0.7 });
    arrow();
    // Transformer stack
    box("Transformer block × 12",
        "LayerNorm → Causal MHA (8 heads) + residual    ·    LayerNorm → FFN (512→2048→512, GELU) + residual",
        { fill: "E8EFE0", stroke: MOSS, h: 1.0 });
    arrow();
    // Final
    box("Final LayerNorm  →  LM head (tied to embedding)  →  logits over 930-token vocab",
        null, { h: 0.7 });

    // Side annotations
    const annot = [
      ["pre-norm", "LayerNorm before each sub-layer for stable training from scratch"],
      ["weight tying", "LM head shares the embedding weights (~½M params saved)"],
      ["causal mask", "registered as a buffer; padding masked separately in attention"],
      ["conditioning", "no cross-attention — header tokens flow through standard self-attn"],
    ];
    annot.forEach(([k, v], i) => {
      const yA = 1.4 + i * 0.95;
      s.addText(k.toUpperCase(), {
        x: 7.0, y: yA, w: 2.6, h: 0.3,
        fontFace: BODY_FONT, fontSize: 10, bold: true, color: FOREST,
        charSpacing: 3, margin: 0,
      });
      s.addText(v, {
        x: 7.0, y: yA + 0.28, w: 2.6, h: 0.65,
        fontFace: BODY_FONT, fontSize: 10, color: MUTED, margin: 0,
      });
    });
  }

  // ---- T3: Training curves ----
  {
    const s = pres.addSlide();
    s.background = { color: CREAM };

    s.addText("Training", {
      x: 0.5, y: 0.3, w: 9, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: FOREST, margin: 0,
    });
    s.addText("AdamW · cosine LR w/ warmup · fp16 · early stopping · 85/10/5 split at the tune level",
      { x: 0.5, y: 0.82, w: 9, h: 0.35,
        fontFace: BODY_FONT, fontSize: 13, color: MUTED, margin: 0 });

    // Built-in chart — re-plot of the training log (more reliable than embedding the PNG)
    const epochs = Array.from({ length: 27 }, (_, i) => String(i + 1));
    const trainLoss = [3.2150,2.0473,1.5190,1.3321,1.2328,1.1660,1.1154,1.0742,1.0389,1.0077,0.9792,0.9527,0.9267,0.9014,0.8767,0.8519,0.8274,0.8032,0.7788,0.7546,0.7307,0.7077,0.6843,0.6621,0.6405,0.6195,0.5989];
    const valLoss   = [2.3133,1.5654,1.3106,1.2112,1.1467,1.1044,1.0700,1.0448,1.0232,1.0114,0.9918,0.9830,0.9740,0.9664,0.9623,0.9578,0.9547,0.9627,0.9630,0.9609,0.9659,0.9721,0.9803,0.9922,0.9908,1.0152,1.0165];

    s.addChart(pres.charts.LINE,
      [
        { name: "train loss", labels: epochs, values: trainLoss },
        { name: "val loss",   labels: epochs, values: valLoss },
      ],
      {
        x: 0.6, y: 1.4, w: 5.7, h: 3.6,
        chartColors: [FOREST, ACCENT],
        chartArea: { fill: { color: "FFFFFF" }, roundedCorners: false },
        catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
        catAxisLabelFontSize: 9, valAxisLabelFontSize: 9,
        valGridLine: { color: "E8E4D6", size: 0.5 },
        catGridLine: { style: "none" },
        lineSize: 2, lineSmooth: true,
        showLegend: true, legendPos: "b", legendFontSize: 11,
        showTitle: true, title: "Loss vs. epoch",
        titleFontSize: 12, titleColor: INK,
        catAxisTitle: "epoch", showCatAxisTitle: true,
        catAxisTitleFontSize: 10, catAxisTitleColor: MUTED,
      }
    );

    // Right column: takeaways
    const items = [
      ["Final", "train ≈ 0.60   ·   val ≈ 1.02   (best val: 0.955)"],
      ["Train/val gap", "Real but stable — not blowing up; checked separately for memorization."],
      ["Memorization", "Longest n-gram overlap with training set is short, common phrases only."],
      ["Regularization", "Dropout 0.1 + weight decay + early stopping (patience 10)."],
    ];
    items.forEach(([k, v], i) => {
      const yI = 1.5 + i * 0.85;
      s.addShape(pres.shapes.RECTANGLE, {
        x: 6.6, y: yI, w: 0.06, h: 0.72,
        fill: { color: FOREST }, line: { type: "none" },
      });
      s.addText(k, {
        x: 6.78, y: yI, w: 2.7, h: 0.3,
        fontFace: BODY_FONT, fontSize: 11, bold: true, color: FOREST,
        charSpacing: 2, margin: 0,
      });
      s.addText(v, {
        x: 6.78, y: yI + 0.28, w: 2.7, h: 0.5,
        fontFace: BODY_FONT, fontSize: 11, color: INK, margin: 0,
      });
    });
  }

  // ---- T4: What was hard ----
  {
    const s = pres.addSlide();
    s.background = { color: FOREST };
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: MOSS }, line: { type: "none" },
    });

    s.addText("What was hard", {
      x: 0.7, y: 0.5, w: 9, h: 0.65,
      fontFace: HEADER_FONT, fontSize: 32, bold: true, color: CREAM, margin: 0,
    });

    const items = [
      ["01", "Tokenizer iteration",
       "ABC has many edge cases — chord symbols, grace notes, alt endings, decorations. Multiple regex revisions to capture musical structure without dropping information."],
      ["02", "Padding-mask NaN bug",
       "Fully-masked rows in attention softmax produce NaNs that corrupt the rest of the batch. Fix: nan_to_num after softmax + explicit padding-key masking."],
      ["03", "Defining ‘good’ for a generative model",
       "Loss curves only go so far in a creative domain. Combined held-out perplexity, structural validity checks, and a memorization audit — plus a fiddle."],
    ];

    items.forEach(([num, head, body], i) => {
      const y = 1.55 + i * 1.15;
      s.addText(num, {
        x: 0.7, y: y, w: 0.9, h: 0.9,
        fontFace: HEADER_FONT, fontSize: 36, bold: true, color: MOSS,
        valign: "top", margin: 0,
      });
      s.addText(head, {
        x: 1.7, y: y, w: 7.7, h: 0.4,
        fontFace: HEADER_FONT, fontSize: 18, bold: true, color: CREAM, margin: 0,
      });
      s.addText(body, {
        x: 1.7, y: y + 0.35, w: 7.7, h: 0.7,
        fontFace: BODY_FONT, fontSize: 12, color: "C8D4BC", margin: 0,
      });
    });
  }

  return pres.writeFile({ fileName: path.join(__dirname, "technical_walkthrough_slides.pptx") });
}

(async () => {
  await buildDemo();
  await buildTechnical();
  console.log("Wrote videos/demo_slides.pptx and videos/technical_walkthrough_slides.pptx");
})();
