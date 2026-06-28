# VisualOps — 5-Agent B2B Onboarding Intelligence

**Gemma 4 31B on Cerebras WSE-3 · ~1,850 tok/s · Built for the Cerebras × Google DeepMind Gemma 4 Hackathon**

---

## What is VisualOps?

VisualOps transforms any business document image (contracts, proposals, invoices, org charts) into a complete client onboarding package in seconds — powered by 5 AI agents running Gemma 4 31B on Cerebras, including multimodal vision, parallel execution, reasoning mode, live token streaming, and real tool calling.

**Upload a document → 5 agents analyze it → Get a complete onboarding package + a GO/HOLD/ESCALATE deal verdict.**

No manual data entry. No waiting. Just instant, intelligent onboarding.

---

## Architecture: 5 Agents, Parallel Execution + Tool Calling

```
User uploads document image
         │
         ▼
┌──────────────────────┐
│  Agent 1: Vision     │  ← Multimodal: reads image directly
│  (Gemma 4 31B)       │    Extracts company, contacts, deal value,
│  Structured Outputs  │    terms, dates via strict JSON schema
└──────────┬───────────┘
           │
     ┌─────┴─────┐  ← Promise.all() — runs in PARALLEL
     ▼           ▼
┌──────────┐ ┌────────────┐
│ Agent 2  │ │  Agent 3   │
│ Research │ │   Risk     │  ← reasoning_effort: "low" (thinking mode)
│ Intel    │ │  Scanner   │    Detects red flags, compliance issues
└────┬─────┘ └─────┬──────┘
     └──────┬──────┘
            ▼
     ┌──────┴───────┐  ← Promise.all() — Output + Strategist in PARALLEL
     ▼              ▼
┌──────────────┐ ┌──────────────────────┐
│  Agent 4:    │ │  Agent 5: Strategist │  ← Tool calling (strict schemas)
│  Output      │ │  Deal Reconciler     │    calls compute_deal_score()
│  Synthesizes:│ │  Reconciles opp vs   │    and flag_for_human(), then
│  • Exec summ │ │  risk → GO / HOLD /  │    emits a verdict + 0-100 score
│  • Plan      │ │  ESCALATE verdict    │
│  • Email     │ └──────────────────────┘
│  • Checklist │
│  • Metrics   │  + a final streamed Executive Briefing (live token streaming)
└──────────────┘
```

---

## Key Technical Features

### 🤖 Multi-Agent Collaboration
- 5 specialized agents with distinct roles and prompts
- Two parallel stages via `Promise.all()`: Research ∥ Risk, then Output ∥ Strategist
- Execution timeline + live agent communication log prove concurrent operation

### 🛠️ Tool Calling (Agentic Reconciliation)
- Agent 5 (Deal Strategist) runs a real **tool-calling loop** with strict function schemas (`strict: true`)
- Calls `compute_deal_score(opportunity, risk)` and `flag_for_human(reason)` — executed locally, results fed back to the model
- Reconciles the Research (opportunity) and Risk (threat) agents into a GO / HOLD / ESCALATE verdict with a 0-100 deal score
- Every tool call is shown live — autonomous agent ↔ tool coordination, not a prompt chain

### ✍️ Live Token Streaming
- A final Executive Briefing is **streamed token-by-token** (SSE) so Cerebras' ~1,850 tok/s is visible on screen as the text types in

### 🖼️ Multimodal Intelligence
- Agent 1 (Vision) reads document **images directly** using Gemma 4's multimodal capabilities
- Supports contracts, proposals, invoices, org charts, any business document
- Auto-converts any image format to JPEG, downscales to 1600px for optimal processing

### ⚡ Speed in Action
- **~1,850 tok/s** on Cerebras WSE-3 (4 trillion transistors, 900K cores, 44GB SRAM)
- Complete 4-agent pipeline finishes in **~3-5 seconds**
- Real-time `time_info.completion_time` measurement from the API
- **Live side-by-side speed comparison** against OpenAI GPT-4o-mini (NVIDIA GPU)
- Header badges show: elapsed time, tok/s, Nx faster than GPU

### 🏗️ Production-Grade Architecture
- **Strict Structured Outputs** (`strict: true` JSON schemas) — guaranteed valid JSON
- Automatic fallback to prompt-only JSON if schema is rejected
- Retry logic: retry once on 429/5xx, fail fast on 4xx
- Image normalization: auto-converts webp/heic/gif to JPEG, flattens transparency

### 🧠 Reasoning Mode
- Risk agent uses `reasoning_effort: "low"` to enable Gemma 4's thinking mode
- Specifically tuned to detect: auto-renewal clauses, unlimited liability, missing HIPAA BAA, unsigned contracts, no-refund clauses
- Outputs risk score (1-10), individual flags with severity, and proceed/escalate recommendation

---

## Enterprise Use Case: B2B Client Onboarding

### The Problem
Enterprise onboarding teams spend **hours** manually reviewing contracts, researching clients, assessing risks, and creating onboarding plans. This process involves:
- Reading and extracting data from scanned/photographed documents
- Researching client background and industry context
- Identifying contractual risks and compliance issues
- Creating personalized onboarding plans and welcome communications

### The Solution
VisualOps automates the entire workflow in seconds:
1. **Document Intelligence** — AI reads the document image directly (no OCR pipeline needed)
2. **Client Research** — AI builds a company intelligence profile with pain points and tech stack
3. **Risk Assessment** — AI flags red flags with severity levels and recommendations
4. **Action Package** — AI generates a complete onboarding package ready for the team

### Business Impact
- **Hours → Seconds**: Complete onboarding analysis in under 5 seconds
- **Consistency**: Every document gets the same thorough analysis
- **Risk Prevention**: Automated red flag detection catches issues humans miss
- **Scalability**: Process hundreds of documents without additional headcount

---

## Quick Start

```bash
npm install
npm run dev
# Open http://localhost:3000
```

1. Enter your Cerebras API key (`csk-...`)
2. (Optional) Enter your OpenAI API key (`sk-...`) for speed comparison
3. Upload a business document image
4. Click "Run 5-agent pipeline"
5. Review results and download the onboarding package (.md)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 18 + Vite |
| AI Model | Gemma 4 31B on Cerebras |
| API | OpenAI-compatible Chat Completions |
| Structured Output | JSON Schema with `strict: true` |
| Tool Calling | Strict function schemas (`compute_deal_score`, `flag_for_human`) |
| Streaming | SSE token streaming for the live Executive Briefing |
| Reasoning | `reasoning_effort: "low"` on Risk agent |
| Speed Comparison | OpenAI GPT-4o-mini (GPU) + Gemini 2.0 Flash (TPU) |
| Deployment | Single-file architecture (`src/App.jsx`) |

---

## Hackathon Tracks

### Track 1: Multiverse Agents ($2K)
- ✅ 5 agents with distinct roles collaborating on a shared task
- ✅ Multimodal: Vision agent reads document images directly
- ✅ Parallel execution proven via execution timeline (two parallel stages)
- ✅ Real tool calling: Deal Strategist reconciles agents via `compute_deal_score` / `flag_for_human`
- ✅ Live token streaming + side-by-side speed comparison vs GPU & TPU

### Track 3: Enterprise Impact ($1K)
- ✅ Solves real enterprise B2B onboarding challenge
- ✅ Production-ready with structured outputs and error handling
- ✅ Scalable architecture — single API, no external dependencies
- ✅ Cerebras speed enables real-time document processing at scale

---

## Sample Documents

The `assets/` folder contains 4 test documents:
- `proposal.jpeg` — Brightwave Retail Group inventory platform proposal ($144K)
- `2.jpeg` — ApexCloud Services invoice to Northwind Logistics ($28,500)
- `digram.jpeg` — Meridian Software organizational chart
- `WhatsApp Image...jpeg` — Vertex Consulting service agreement (intentionally contains red flags for Risk agent testing)

---

Built with ❤️ for the Cerebras × Google DeepMind Gemma 4 24-Hour Hackathon
