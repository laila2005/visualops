# VisualOps — Live Build Plan (execute during the hackathon)

**Event:** Cerebras × Google DeepMind Gemma 4 24-Hour Hackathon
**Window:** Sun Jun 28 10:00 AM PT → Mon Jun 29 10:00 AM PT (= Sun 8 PM → Mon 8 PM Egypt)
**Model unlock:** ~10:30 AM PT (~8:30 PM Egypt)
**Submitting to:** Track 1 `#g4hackathon-multiverse-agents` · Track 3 `#g4hackathon-enterprise-impact`

> Commit after each step (timestamps document that core work happened during the event).

## Phase 0 — Pre-kickoff (before 8 PM Egypt)
- [ ] Download the 4 sample PNGs from `localhost:3000/samples.html` (keep locally)
- [ ] Set up screen recorder (OBS / Win+G) + open Gemini/AI Studio tab for comparison
- [ ] Rehearse the 60-second voiceover once

## Phase 1 — Unlock & smoke test (~8:30 PM Egypt)
- [ ] `localhost:3000` → **check models** → confirm `gemma-4-31b` ✓  → `git commit -m "Confirm live model access"`
- [ ] Upload `visualops-proposal.png` → Run → all 4 agents reach ✓
- [ ] Note the **tok/s** number and whether timeline bars overlap
- [ ] Verify Structured Outputs path works (no fallback). `git commit -m "Validate structured outputs against live Gemma 4"`

## Phase 2 — Tune against the live model (core work)
- [ ] Run all 4 sample docs; check each agent's fields are sensible
- [ ] Tune any weak prompt (commit each fix separately)
- [ ] Run the **Service Agreement** → confirm Risk agent flags red flags + ESCALATE

## Phase 3 — New in-event feature: live Gemini side-by-side (Speed in Action)
- [ ] Add a real GPU-provider comparison call (Gemini) timed next to Cerebras
- [ ] `git commit -m "Add live Gemini vs Cerebras speed comparison"`

## Phase 4 — Demo recording (≤ 60s)
- [ ] Clean desktop: close personal tabs, hide notifications, no API keys on screen
- [ ] Record: upload proposal → pipeline lights up → tok/s + Nx faster badge → scroll results → download package
- [ ] Capture the side-by-side speed shot
- [ ] Edit to ≤ 60 seconds

## Phase 5 — Submit (before Mon 8 PM Egypt)
- [ ] Post demo video + description in `#g4hackathon-multiverse-agents` (Track 1)
- [ ] Post demo video + description in `#g4hackathon-enterprise-impact` (Track 3)
- [ ] Post video on X tagging @Cerebras and @googlegemma
- [ ] Final `git commit -m "Final submission build"`

## Talking points for the write-up
- 4 agents, 2 running in parallel via `Promise.all()` (shown in the execution timeline)
- Multimodal: Gemma 4 reads the document image directly
- Reasoning mode (`reasoning_effort: "low"`) on the Risk agent
- Strict Structured Outputs (`strict: true` JSON schema) for production-grade reliability
- Real measured tok/s from the API `time_info` — Nx faster than GPU
