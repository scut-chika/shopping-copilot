# 3-minute demo video — shot list and script

UI/UX is explicitly out of scope for this track, and the rules allow a
walkthrough of API usage, inference examples, and result analysis instead. So
this is a terminal recording, not a product demo.

**Hard limits:** under 3:00, uploaded to YouTube, set to **Public**, linked in the
Devpost description. No third-party trademarks or copyrighted music.

**Setup before recording**
- Terminal at ~110×32, large readable font, light or dark but high contrast.
- Pre-warm: run each command once first so the 32 s index build is not on camera
  (or cut it — see Shot 2).
- Have `results/` open in a second tab for the numbers.

---

## Shot 1 — The problem, in one sentence (0:00–0:20)

**On screen:** the two-row results table from the README.

> "A customer says 'I'm looking for tunics, but I'm still exploring.' There are
> fifty thousand products. You have ten turns to put the one they actually want
> in your top ten. The organizer's baseline manages that twelve percent of the
> time, and takes nearly all ten turns. Ours finds it every time, in about two."

---

## Shot 2 — A real session, end to end (0:20–1:10)

**Command:**
```bash
python tools/demo.py --scenario browsing --index 1
```

*(Cut the startup wait in the edit — jump straight to turn 1.)*

**Narrate over the output:**

> "Turn one. The customer has given us nothing but a category — five hundred and
> thirty-four candidates are still consistent. So instead of guessing, the agent
> asks the question that splits that set best.
>
> The customer answers with two constraints. Turn two: three candidates left, and
> the target is at rank one.
>
> Two turns. And notice it returned ten recommendations on *both* turns — the
> evaluator checks for a hit before it generates the reply, so answering with a
> question alone is pure waste. The baseline never asks anything at all, which is
> exactly why it takes nine point eight turns."

**Point at on screen:** the `candidates still consistent: 534` → `3` drop, and
`<== TARGET` at rank 1.

---

## Shot 3 — Why it works (1:10–1:50)

**On screen:** `copilot/simulator_model.py` docstring, then the narrowing table.

> "This works because of something we found by reading the evaluator before
> writing any code. Every customer utterance is built deterministically from the
> target product's own catalog record — its features and details, verbatim.
>
> So the right index isn't a vector store. It's an inverted index from 'thing the
> customer said' back to 'products that could have said it.' We reconstruct that
> derivation across all fifty thousand products and invert it. Category plus two
> constraints collapses seventy-six percent of sessions to ten candidates or fewer."

**Then cut to the ablation table.**

> "But here's the part that surprised us. We assumed that index was the story. It
> isn't. Remove the question policy and we lose zero point four eight. Remove the
> exact-match index and we lose zero point one. Asking the right question is worth
> five times more than retrieving better."

---

## Shot 4 — The number is suspicious, so we tested it (1:50–2:35)

**On screen:** the generalization table.

> "A hundred percent hit rate on two hundred sessions is a warning sign, not a
> victory lap. At the ceiling, that set can't tell you anything — you're just
> fitting it. The private set has eight hundred sessions with different users and
> different targets.
>
> So we built our own. Eight hundred sessions over products that appear nowhere in
> the public set, independently resampled profiles, official scenario mix."

**Command (or show the committed JSON):**
```bash
python tools/generalize.py --sessions 800
```

> "Ninety-six point seven percent retained. Second seed, ninety-five point five.
> We fit the task, not the sessions."

**Then flash the robustness table.**

> "We also stress-tested paraphrasing, because the spec says the organizer may add
> it. That's where we found our real weak point — it was the parser, not the
> matching. Rewording just the templates hurt more than mangling the constraints."

---

## Shot 5 — It runs offline (2:35–3:00)

**Command:**
```bash
python -m pytest tests/ -q
```

> "The rules say final scoring may run with network access disabled. So there's no
> model here. No API, no credentials, no third-party dependency — standard library
> only. Zero tokens, zero dollars, eighty milliseconds a turn.
>
> And that's not a claim in a README. One of these sixteen tests parses every
> shipped module and fails if a network client ever appears.
>
> Shopping Copilot. Two turns instead of ten."

**End card:** repo URL.

---

## Recording checklist

- [ ] Under 3:00
- [ ] Target visibly hit at rank 1 in Shot 2
- [ ] Ablation and generalization tables both legible
- [ ] Test suite shown passing
- [ ] No API keys, tokens, or personal paths visible in the terminal
- [ ] Uploaded to YouTube, visibility **Public** (not Unlisted)
- [ ] URL pasted into the Devpost description and `docs/DEVPOST_SUBMISSION.md`
