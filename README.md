# VisualOps — Setup Guide

## 4-agent B2B onboarding intelligence
**Gemma 4 31B on Cerebras · 1,500 tokens/sec**

---

## Quick start (5 minutes)

```bash
# 1. Install dependencies
npm install

# 2. Start dev server
npm run dev

# 3. Open http://localhost:3000
# 4. Enter your Cerebras API key in the UI
# 5. Upload a business document image
# 6. Click "Run 4-agent pipeline"
```

---

## Architecture overview

```
User uploads image
        │
        ▼
┌──────────────────┐
│  Agent 1: Vision  │  ← Reads image, extracts structured JSON
│  (Gemma 4 31B)   │    (company, contacts, deal value, doc type)
└────────┬─────────┘
         │
    ┌────┴────┐  ← Runs in PARALLEL (Promise.all)
    ▼         ▼
┌────────┐ ┌──────────┐
│ Agent 2│ │ Agent 3  │
│Research│ │  Risk    │  ← reasoning_effort: "low"
└────┬───┘ └────┬─────┘
     └────┬─────┘
          ▼
┌──────────────────┐
│  Agent 4: Output │  ← Synthesizes everything
│  Onboarding plan │    Welcome email + checklist
│  + Email draft   │    + success metrics
└──────────────────┘
```

---

## Agent system prompts

Each agent is in `src/App.jsx`:

| Agent | Function | Key technique |
|-------|----------|---------------|
| `agentVision` | Reads uploaded image | Multimodal: `image_url` in message |
| `agentResearch` | Client intel profile | Structured JSON output |
| `agentRisk` | Red flag detection | `reasoning_effort: "low"` |
| `agentOutput` | Full action package | Synthesizes all 3 previous outputs |

---

## API call structure (Cerebras)

```javascript
// Text + Image (Vision agent)
{
  model: "gemma-4-31b",
  messages: [{
    role: "user",
    content: [
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,..." } },
      { type: "text", text: "your prompt" }
    ]
  }]
}

// Text only with reasoning (Risk agent)
{
  model: "gemma-4-31b",
  reasoning_effort: "low",   // or "medium" / "high"
  messages: [{ role: "user", content: "your prompt" }]
}
```

---

## Demo video tips (60 seconds)

1. **0–5s**: "This is VisualOps — B2B onboarding intelligence in under 5 seconds"
2. **5–15s**: Show split screen — your app vs GPU provider. Upload same image both.
3. **15–30s**: Cerebras finishes. GPU still loading. Show the agent pipeline lighting up.
4. **30–50s**: Walk through the output: exec summary, risk flags, onboarding plan, email.
5. **50–60s**: "Powered by Gemma 4 31B on Cerebras — 15x faster than GPU inference"

---

## Hackathon submission checklist

- [ ] Upload a realistic-looking contract screenshot for the demo
- [ ] Record side-by-side with a GPU provider (use Google AI Studio as comparison)
- [ ] Post to Discord: #g4hackathon-multiverse-agents (Track 1)
- [ ] Post to Discord: #g4hackathon-enterprise-impact (Track 3)
- [ ] Post video on X/Twitter tagging @Cerebras and @googlegemma
- [ ] Submit before Monday June 29, 8:00 PM Egypt time (EET)
