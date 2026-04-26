# Demo Video Script (3–5 min)

**Audience:** non-specialist. **Rule:** no code on screen.
**Recording setup:** screen capture of the Gradio app + slides; cut to a phone/camera shot of you with the fiddle for the live performance.

---

## [0:00–0:25] Cold open — fiddle hook (~25s)
**On screen:** YOU on camera, fiddle in hand. Brief, warm intro shot — no slides yet.

> "Irish traditional music has been passed down by ear for hundreds of years — a tune you hear tonight in a pub in Dublin might be three centuries old, or it might have been written last week. I'm a fiddle player, and I built an AI that learned this tradition from scratch and writes brand-new tunes in it. I'm going to ask it for a tune right now, and then I'm going to play it for you."

---

## [0:25–1:00] What the project is — the pitch (~35s)
**On screen:** SLIDE 1 — Title slide ("ceol-gpt: a transformer that writes Irish folk tunes").
Then SLIDE 2 — the "what is ABC" slide (text on left, rendered sheet music on right).

> "The project is called ceol-gpt — *ceol* is the Irish word for music. It's a small GPT-style language model, the same kind of architecture behind ChatGPT, but instead of English it speaks **ABC notation** — a compact text format musicians use to write down tunes. So a tune looks like this *(point to ABC text)* and renders as sheet music like this *(point to staff)*. The model learned this language by reading 54,000 tunes from TheSession.org, an online community archive of Irish traditional music."

---

## [1:00–1:40] Why this matters (~40s)
**On screen:** SLIDE 3 — "Why this is interesting" — three bullets fade in:
- Generative models for **structured creative domains** (not just text or images)
- A **living oral tradition** — composition has always been part of it
- A **tool for musicians**: prompt by tune type & key, get a starting point

> "Why bother? Two reasons. First, it's a fun test of whether a transformer can learn the *grammar* of a musical style — the phrase shapes, the cadences, the way a reel resolves differently from a jig. Second, composing new tunes has always been part of this tradition. Most great session tunes were written by someone, often anonymously, and the tradition keeps growing. So this is a tool a musician could actually use as a starting point — give it a tune type and a key, get an idea, take what works, throw out the rest."

---

## [1:40–3:00] Live demo of the app (~80s)
**On screen:** the Gradio app (`python app.py` running locally, full browser window).

**Beat 1 (10s):** Show the controls. Hover/point.
> "Here's the app. I pick a tune type — let's do a **reel** — a key — let's do **D major**, the classic fiddle key — and a temperature, which controls how adventurous the model is. I'll leave it at 0.8."

**Beat 2 (15s):** Click Generate. Tokens stream in.
> "It generates token by token, one musical 'word' at a time — a note, a bar line, an ornament. Each one is a prediction over the model's vocabulary of about 900 musical symbols."

**Beat 3 (20s):** Sheet music renders, audio playback appears.
> "Now we have notation, and we can hear what the model wrote." *(Click play — let 8–10 seconds of synth playback run.)* "That's a synthesized version. But this is meant to be played by a person, on a real instrument. So let me try."

**Beat 4 (35s):** Cut to YOU on camera, sheet music visible on a stand or tablet beside you.
> "Reading it cold, first time through." *(Play the A part once — roughly 15–20 seconds. If it has a B part and you can sight-read it, play that too. If the tune has a rough spot, smile and keep going — that's part of the honesty of the demo.)*

---

## [3:00–3:45] What worked, what didn't (~45s)
**On screen:** SLIDE 4 — a side-by-side: "Model gets right" / "Model still fumbles."
Suggested bullets:
- ✅ Stays in the right key and meter
- ✅ Phrase length and bar structure feel like a reel
- ✅ Standard cadences (e.g., turning on the dominant)
- ⚠ Sometimes wanders melodically in the B part
- ⚠ Occasionally produces a phrase that's awkward to finger

> "Honest take: the model has clearly learned the *shape* of a reel — the phrase length, the bar structure, the way a part resolves. Where it still struggles is longer-range coherence — the second half of a tune sometimes wanders, and once in a while it writes something physically awkward to play. But for a model trained from scratch on a single laptop-sized GPU, in a single project, getting a fiddler-playable tune on the first try is genuinely surprising."

---

## [3:45–4:15] Wrap (~30s)
**On screen:** SLIDE 5 — final slide. Project name, GitHub URL, "thanks for listening."

> "So that's ceol-gpt — a small transformer that learned a centuries-old musical tradition from text, well enough to hand a fiddler something they can actually play. Code, the trained model, and a longer technical walkthrough are all in the repo. Thanks for listening."

---

## Production notes
- **Total target:** 4:00–4:30. You have ~30s of slack to land at 5:00 if the live performance runs long.
- **Don't show terminal, code, or the IDE.** The handout explicitly says "no reason to show any code."
- **Tune choice:** generate **two or three** candidate tunes before recording and pick the most playable one. The demo doesn't lose authenticity if you pre-screen — it just respects the viewer's time. Mention you generated it "right before recording" rather than implying it's the literal first one ever.
- **Audio:** record fiddle with a real mic if you have one. Phone mic is fine but get close.
- **Sheet music for performance:** render the generated tune in MuseScore or print directly from the abcjs view in the app for a clean stand-readable copy.
