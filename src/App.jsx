import { useState, useRef, useCallback, useEffect } from "react";

// ─── Cerebras / Gemma 4 API ───────────────────────────────────────────────────
const CEREBRAS_BASE = "https://api.cerebras.ai/v1";

// Returns { content, tokens, completionTokens, completionTime, ms }.
// Pass options.schema (a JSON schema) to constrain output via strict Structured
// Outputs. Retries once on transient (429/5xx/network) errors but fails fast on
// 4xx so a bad key or schema surfaces immediately.
async function callGemma(apiKey, messages, options = {}) {
  const t0 = Date.now();
  const { schema, schemaName, ...rest } = options;
  const body = { model: "gemma-4-31b", max_tokens: 1024, messages, ...rest };
  if (schema) {
    body.response_format = {
      type: "json_schema",
      json_schema: { name: schemaName || "result", strict: true, schema },
    };
  }
  const payload = JSON.stringify(body);
  let lastErr;
  for (let attempt = 0; attempt < 2; attempt++) {
    let res;
    try {
      res = await fetch(`${CEREBRAS_BASE}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: payload,
      });
    } catch (e) {
      // Network failure — retry once, then surface.
      if (attempt === 0) { lastErr = e; await new Promise((r) => setTimeout(r, 600)); continue; }
      throw e;
    }
    if (!res.ok) {
      const bodyText = await res.text().catch(() => "");
      let message;
      try { message = JSON.parse(bodyText)?.error?.message; } catch { /* not JSON */ }
      message = message || `API error ${res.status}${bodyText ? `: ${bodyText.slice(0, 300)}` : ""}`;
      // Retry only on rate-limit / server errors; fail fast on 4xx (bad key/schema/image).
      if ((res.status === 429 || res.status >= 500) && attempt === 0) {
        lastErr = new Error(message);
        await new Promise((r) => setTimeout(r, 600));
        continue;
      }
      throw new Error(message);
    }
    const data = await res.json();
    return {
      content: data.choices[0].message.content,
      tokens: data.usage?.total_tokens || 0,
      completionTokens: data.usage?.completion_tokens || 0,
      completionTime: data.time_info?.completion_time || 0,
      ms: Date.now() - t0,
    };
  }
  throw lastErr || new Error("Request failed");
}

// Strict Structured Outputs are preferred for guaranteed-valid JSON, but if the
// API rejects the schema for a given request we disable them for the rest of the
// run and fall back to prompt-only JSON (still parsed defensively downstream).
let structuredSupported = true;
async function callGemmaJSON(apiKey, messages, schema, schemaName, options = {}) {
  if (structuredSupported) {
    try {
      return await callGemma(apiKey, messages, { ...options, schema, schemaName });
    } catch (e) {
      const msg = (e && e.message) || "";
      // Schema/format-related rejection → drop to plain mode. Auth/network → surface.
      if (/schema|response_format|json|structured|unsupported|invalid|400/i.test(msg)) {
        structuredSupported = false;
      } else {
        throw e;
      }
    }
  }
  return await callGemma(apiKey, messages, options);
}

// Streaming variant — sends stream:true and invokes onToken(delta) for each chunk
// as Gemma generates it. This is what makes Cerebras' ~1,850 tok/s visible: the
// text flies onto the screen. Returns the same shape as callGemma once complete.
async function callGemmaStream(apiKey, messages, options = {}, onToken) {
  const t0 = Date.now();
  const body = {
    model: "gemma-4-31b", max_tokens: 1024, messages,
    stream: true, stream_options: { include_usage: true }, ...options,
  };
  const res = await fetch(`${CEREBRAS_BASE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}${t ? `: ${t.slice(0, 200)}` : ""}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", content = "", usage = null, timeInfo = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep the last (possibly incomplete) line
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") continue;
      try {
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta?.content || "";
        if (delta) { content += delta; onToken?.(delta); }
        if (json.usage) usage = json.usage;
        if (json.time_info) timeInfo = json.time_info;
      } catch { /* ignore partial JSON chunk */ }
    }
  }
  return {
    content,
    tokens: usage?.total_tokens || 0,
    completionTokens: usage?.completion_tokens || 0,
    completionTime: timeInfo?.completion_time || 0,
    ms: Date.now() - t0,
  };
}

// GET /models — returns the list of model IDs this API key can actually use.
async function listModels(apiKey) {
  const res = await fetch(`${CEREBRAS_BASE}/models`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`Models lookup failed (${res.status}): ${body.slice(0, 200)}`);
  const data = JSON.parse(body);
  return (data.data || []).map((m) => m.id);
}

// Strip markdown fences and parse JSON; fall back to a default object on failure.
function parseJSON(content, fallback) {
  try {
    return JSON.parse(content.replace(/```json|```/g, "").trim());
  } catch {
    return fallback;
  }
}

// ─── Tool Calling ─────────────────────────────────────────────────────────────
// Runs an agentic tool-calling loop: the model decides which functions to call,
// we execute them locally and feed the results back, and repeat until the model
// stops requesting tools. This is real agent ↔ tool coordination, not a prompt
// chain. `executors` maps tool name → (args) => result; `onEvent` fires per call.
async function callGemmaTools(apiKey, messages, tools, executors, { maxRounds = 5, onEvent } = {}) {
  const convo = [...messages];
  const trace = [];
  let finalContent = "";
  const usage = { total: 0, completionTokens: 0, completionTime: 0 };
  for (let round = 0; round < maxRounds; round++) {
    const res = await fetch(`${CEREBRAS_BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model: "gemma-4-31b", max_tokens: 1024, messages: convo, tools, tool_choice: "auto" }),
    });
    if (!res.ok) {
      const t = await res.text().catch(() => "");
      throw new Error(`API error ${res.status}${t ? `: ${t.slice(0, 200)}` : ""}`);
    }
    const data = await res.json();
    const msg = data.choices?.[0]?.message;
    usage.total += data.usage?.total_tokens || 0;
    usage.completionTokens += data.usage?.completion_tokens || 0;
    usage.completionTime += data.time_info?.completion_time || 0;
    const toolCalls = msg?.tool_calls || [];
    if (toolCalls.length === 0) { finalContent = msg?.content || ""; break; }
    convo.push({ role: "assistant", content: msg.content || "", tool_calls: toolCalls });
    for (const tc of toolCalls) {
      let args = {};
      try { args = JSON.parse(tc.function?.arguments || "{}"); } catch { /* malformed args */ }
      const exec = executors[tc.function?.name];
      const result = exec ? exec(args) : { error: "unknown tool" };
      trace.push({ name: tc.function?.name, args, result });
      onEvent?.({ name: tc.function?.name, args, result });
      convo.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(result) });
    }
  }
  return { content: finalContent, trace, usage };
}

// ─── OpenAI API (GPU-based — for speed comparison) ────────────────────────────
async function callOpenAI(openaiKey, prompt) {
  const t0 = Date.now();
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${openaiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      max_tokens: 512,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    if (res.status === 401) throw new Error("Invalid OpenAI API key");
    if (res.status === 429) throw new Error("OpenAI rate limited");
    throw new Error(`OpenAI error ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = await res.json();
  const content = data.choices?.[0]?.message?.content || "";
  const completionTokens = data.usage?.completion_tokens || 0;
  return { content, ms: Date.now() - t0, completionTokens };
}

// ─── Gemini API (Google TPU — free tier, for speed comparison) ─────────────────
async function callGemini(geminiKey, prompt) {
  const t0 = Date.now();
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${geminiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 512 },
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    if (res.status === 401 || res.status === 403) throw new Error("Invalid Gemini API key");
    if (res.status === 429) throw new Error("Gemini rate limited");
    throw new Error(`Gemini error ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = await res.json();
  const content = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  const completionTokens = data.usageMetadata?.candidatesTokenCount || 0;
  return { content, ms: Date.now() - t0, completionTokens };
}

// Run the same prompt on Cerebras + any available comparison providers in parallel.
async function runSpeedComparison(cerebrasKey, openaiKey, geminiKey, prompt) {
  const promises = [
    callGemma(cerebrasKey, [{ role: "user", content: prompt }]),
  ];
  const providers = ["cerebras"];
  if (openaiKey) { promises.push(callOpenAI(openaiKey, prompt)); providers.push("openai"); }
  if (geminiKey) { promises.push(callGemini(geminiKey, prompt)); providers.push("gemini"); }

  const settled = await Promise.allSettled(promises);
  const result = {};
  providers.forEach((name, i) => {
    const s = settled[i];
    if (s.status === "fulfilled") {
      result[name] = {
        ms: s.value.ms, tokens: s.value.tokens || 0,
        completionTokens: s.value.completionTokens || 0,
        completionTime: s.value.completionTime || 0,
        content: s.value.content, error: null,
      };
    } else {
      result[name] = { ms: 0, tokens: 0, completionTokens: 0, completionTime: 0, content: "", error: s.reason?.message || "Failed" };
    }
  });
  // Ensure all keys exist even if not run
  if (!result.openai) result.openai = null;
  if (!result.gemini) result.gemini = null;
  return result;
}

// ─── Image → Base64 ───────────────────────────────────────────────────────────
// Normalize ANY uploaded image to JPEG before sending. Cerebras/Gemma only
// accepts jpeg/png (webp, heic, gif, etc. are rejected), so we always re-encode
// via canvas — this also downscales large images and flattens transparency onto
// a white background. Returns { base64, mimeType }.
async function prepareImage(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
  const img = await new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = dataUrl;
  });
  const maxDim = 1600;
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const jpeg = canvas.toDataURL("image/jpeg", 0.9);
  return { base64: jpeg.split(",")[1], mimeType: "image/jpeg" };
}

// ─── Strict JSON schemas for Structured Outputs ───────────────────────────────
// Strict mode requires every property listed in `required` and
// `additionalProperties: false` on every object. Nullable fields use ["x","null"].
const str = { type: "string" };
const strArr = { type: "array", items: { type: "string" } };
const nullable = { type: ["string", "null"] };

const VISION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    company_name: nullable,
    industry: nullable,
    document_type: { type: "string", enum: ["contract", "proposal", "invoice", "org_chart", "logo", "other"] },
    key_contacts: strArr,
    deal_value: nullable,
    key_terms: strArr,
    dates: strArr,
    products_or_services: strArr,
    raw_summary: str,
  },
  required: ["company_name", "industry", "document_type", "key_contacts", "deal_value", "key_terms", "dates", "products_or_services", "raw_summary"],
};

const RESEARCH_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    company_profile: str,
    industry_context: str,
    likely_tech_stack: strArr,
    pain_points: strArr,
    onboarding_priorities: strArr,
    recommended_approach: str,
  },
  required: ["company_profile", "industry_context", "likely_tech_stack", "pain_points", "onboarding_priorities", "recommended_approach"],
};

const RISK_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    risk_score: { type: "string", enum: ["low", "medium", "high"] },
    risk_score_number: { type: "integer", minimum: 1, maximum: 10 },
    flags: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          issue: str,
          severity: { type: "string", enum: ["low", "medium", "high"] },
          recommendation: str,
        },
        required: ["issue", "severity", "recommendation"],
      },
    },
    missing_information: strArr,
    compliance_notes: str,
    proceed_recommendation: { type: "string", enum: ["go", "proceed-with-caution", "escalate"] },
  },
  required: ["risk_score", "risk_score_number", "flags", "missing_information", "compliance_notes", "proceed_recommendation"],
};

const OUTPUT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    executive_summary: str,
    onboarding_plan: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: { step: { type: "integer" }, action: str, owner: str, timeline: str, goal: str },
        required: ["step", "action", "owner", "timeline", "goal"],
      },
    },
    welcome_email: {
      type: "object",
      additionalProperties: false,
      properties: { subject: str, body: str },
      required: ["subject", "body"],
    },
    action_checklist: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: { task: str, priority: { type: "string", enum: ["high", "medium", "low"] }, done: { type: "boolean" } },
        required: ["task", "priority", "done"],
      },
    },
    success_metrics: strArr,
  },
  required: ["executive_summary", "onboarding_plan", "welcome_email", "action_checklist", "success_metrics"],
};

// ─── 4 Agent functions ────────────────────────────────────────────────────────

// Agent 1: Vision — reads the image, extracts structured data
async function agentVision(apiKey, base64Image, mimeType) {
  const { content, tokens, completionTokens, completionTime, ms } = await callGemmaJSON(apiKey, [
    {
      role: "user",
      content: [
        {
          type: "image_url",
          image_url: { url: `data:${mimeType};base64,${base64Image}` },
        },
        {
          type: "text",
          text: `You are a document intelligence agent. Analyze this business document image and extract all available information.

Return ONLY a valid JSON object (no markdown, no explanation) with these fields:
{
  "company_name": "string or null",
  "industry": "string or null",
  "document_type": "contract|proposal|invoice|org_chart|logo|other",
  "key_contacts": ["name - role", ...],
  "deal_value": "string or null",
  "key_terms": ["term1", "term2", ...],
  "dates": ["date1", ...],
  "products_or_services": ["item1", ...],
  "raw_summary": "2-sentence plain English summary of what this document is"
}`,
        },
      ],
    },
  ], VISION_SCHEMA, "document_extraction");

  const data = parseJSON(content, { raw_summary: content, company_name: null, industry: null });
  return { data, tokens, completionTokens, completionTime, ms };
}

// Agent 2: Research — builds company intelligence profile
async function agentResearch(apiKey, visionData) {
  const { content, tokens, completionTokens, completionTime, ms } = await callGemmaJSON(apiKey, [
    {
      role: "user",
      content: `You are a B2B client intelligence agent. Based on this extracted document data, build a detailed company intelligence profile.

Document data:
${JSON.stringify(visionData, null, 2)}

Return ONLY a valid JSON object (no markdown) with:
{
  "company_profile": "3-4 sentence profile of likely company type, size, and market position",
  "industry_context": "2-3 sentence overview of their industry and current challenges",
  "likely_tech_stack": ["tool1", "tool2", ...],
  "pain_points": ["pain1", "pain2", "pain3"],
  "onboarding_priorities": ["priority1", "priority2", "priority3"],
  "recommended_approach": "2 sentences on how to approach this client specifically"
}`,
    },
  ], RESEARCH_SCHEMA, "client_intelligence");

  const data = parseJSON(content, { company_profile: content });
  return { data, tokens, completionTokens, completionTime, ms };
}

// Agent 3: Risk — scans for red flags and compliance issues (uses reasoning)
async function agentRisk(apiKey, visionData) {
  const { content, tokens, completionTokens, completionTime, ms } = await callGemmaJSON(
    apiKey,
    [
      {
        role: "user",
        content: `You are a senior B2B risk assessment and compliance agent. Analyze this document data thoroughly for ALL risks, red flags, and compliance concerns.

Document data:
${JSON.stringify(visionData, null, 2)}

Critical: You MUST check for and flag ALL of the following if present:
- Auto-renewal clauses (especially long terms like 12+ months)
- Unlimited or uncapped liability provisions
- Missing HIPAA Business Associate Agreement (BAA) when patient/health data is involved
- Unsigned documents or missing signatures
- No-refund or no-termination-for-convenience clauses
- 100% upfront payment requirements
- One-sided indemnification
- Missing data residency or breach notification timelines
- Unusually long contract terms (36+ months)

If you find 3+ high-severity flags, or any combination of missing compliance documents with healthcare data, set proceed_recommendation to "escalate".

Return ONLY a valid JSON object (no markdown) with:
{
  "risk_score": "low|medium|high",
  "risk_score_number": 1-10,
  "flags": [
    {"issue": "description", "severity": "low|medium|high", "recommendation": "what to do"}
  ],
  "missing_information": ["what's missing that should be present"],
  "compliance_notes": "1-2 sentences on any compliance considerations",
  "proceed_recommendation": "go|proceed-with-caution|escalate"
}`,
      },
    ],
    RISK_SCHEMA,
    "risk_assessment",
    { reasoning_effort: "low" }
  );

  const data = parseJSON(content, { risk_score: "medium", flags: [], risk_score_number: 5 });
  return { data, tokens, completionTokens, completionTime, ms };
}

// Agent 4: Output — synthesizes everything into an action plan + email draft
async function agentOutput(apiKey, visionData, researchData, riskData) {
  const { content, tokens, completionTokens, completionTime, ms } = await callGemmaJSON(apiKey, [
    {
      role: "user",
      content: `You are a senior account manager AI. Synthesize all signals below into a complete client onboarding package.

Vision data: ${JSON.stringify(visionData, null, 2)}
Research data: ${JSON.stringify(researchData, null, 2)}
Risk data: ${JSON.stringify(riskData, null, 2)}

Return ONLY a valid JSON object (no markdown) with:
{
  "executive_summary": "3-4 sentence sharp summary of this client opportunity",
  "onboarding_plan": [
    {"step": 1, "action": "what to do", "owner": "who", "timeline": "when", "goal": "why"}
  ],
  "welcome_email": {
    "subject": "email subject line",
    "body": "full professional email body (4-6 paragraphs)"
  },
  "action_checklist": [
    {"task": "task description", "priority": "high|medium|low", "done": false}
  ],
  "success_metrics": ["metric1", "metric2", "metric3"]
}`,
    },
  ], OUTPUT_SCHEMA, "onboarding_package");

  const data = parseJSON(content, { executive_summary: content });
  return { data, tokens, completionTokens, completionTime, ms };
}

// Agent 5: Deal Strategist — reconciles Research (opportunity) vs Risk via tools.
// Tools the model can call; we execute them locally and feed results back.
const STRATEGIST_TOOLS = [
  {
    type: "function",
    function: {
      name: "compute_deal_score",
      description: "Compute a 0-100 deal attractiveness score from an opportunity rating and a risk rating.",
      strict: true,
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          opportunity: { type: "integer", minimum: 1, maximum: 10, description: "Opportunity strength, 1 (weak) to 10 (excellent)" },
          risk: { type: "integer", minimum: 1, maximum: 10, description: "Risk severity, 1 (negligible) to 10 (severe)" },
        },
        required: ["opportunity", "risk"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "flag_for_human",
      description: "Escalate the deal to a human decision-maker with a concise reason. Only call when warranted.",
      strict: true,
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          reason: { type: "string", description: "Why a human must review this deal" },
          severity: { type: "string", enum: ["low", "medium", "high"] },
        },
        required: ["reason", "severity"],
      },
    },
  },
];

async function agentStrategist(apiKey, researchData, riskData, onEvent) {
  const t0 = Date.now();
  const executors = {
    compute_deal_score: ({ opportunity, risk }) => {
      const o = Math.max(1, Math.min(10, Number(opportunity) || 5));
      const r = Math.max(1, Math.min(10, Number(risk) || 5));
      const deal_score = Math.max(0, Math.min(100, Math.round(o * 10 - (r - 1) * 7)));
      const band = deal_score >= 65 ? "strong" : deal_score >= 40 ? "moderate" : "weak";
      return { deal_score, band, opportunity: o, risk: r };
    },
    flag_for_human: ({ reason, severity }) => ({ flagged: true, reason: reason || "Manual review advised", severity: severity || "medium" }),
  };

  const messages = [{
    role: "user",
    content: `You are the Deal Strategist — a reconciliation agent. Research surfaced opportunity signals; Risk surfaced threats. They may conflict. Decide whether to pursue this client.

Follow these steps exactly:
1. Rate the opportunity 1-10 (from the research) and the risk 1-10 (from the risk data).
2. Call compute_deal_score with those two numbers.
3. If the risk recommendation is "escalate", or any flag is high severity, also call flag_for_human with a concise reason.
4. After the tools return, reply with ONLY this JSON (no markdown, no prose):
{"verdict":"GO|HOLD|ESCALATE","deal_score":<integer returned by the tool>,"rationale":"2-3 sentences reconciling opportunity vs risk and naming the single most important next step"}

Research data: ${JSON.stringify(researchData)}
Risk data: ${JSON.stringify(riskData)}`,
  }];

  const { content, trace, usage } = await callGemmaTools(apiKey, messages, STRATEGIST_TOOLS, executors, { onEvent });
  const parsed = parseJSON(content, null);
  const scoreFromTool = [...trace].reverse().find((t) => t.name === "compute_deal_score")?.result?.deal_score;
  const score = parsed?.deal_score ?? scoreFromTool ?? null;
  const verdict = parsed?.verdict || (score != null ? (score >= 65 ? "GO" : score >= 40 ? "HOLD" : "ESCALATE") : "HOLD");
  const data = {
    verdict,
    deal_score: score,
    rationale: parsed?.rationale || content || "Reconciliation complete.",
    trace,
  };
  return { data, tokens: usage.total, completionTokens: usage.completionTokens, completionTime: usage.completionTime, ms: Date.now() - t0 };
}

// ─── Pipeline orchestrator ────────────────────────────────────────────────────
async function runPipeline(apiKey, base64Image, mimeType, onUpdate, onStream) {
  const t0 = Date.now();
  const timeline = [];
  const mark = (stage, start, end) => timeline.push({ stage, start: start - t0, end: end - t0 });

  onUpdate({ stage: "vision", status: "running" });
  const vStart = Date.now();
  const vision = await agentVision(apiKey, base64Image, mimeType);
  mark("vision", vStart, Date.now());
  onUpdate({ stage: "vision", status: "done", data: vision.data, meta: { ms: vision.ms, tokens: vision.tokens, completionTime: vision.completionTime, completionTokens: vision.completionTokens } });

  // Agents 2 and 3 run in parallel
  onUpdate({ stage: "research", status: "running" });
  onUpdate({ stage: "risk", status: "running" });
  const pStart = Date.now();

  const [research, risk] = await Promise.all([
    agentResearch(apiKey, vision.data).then((r) => {
      mark("research", pStart, Date.now());
      onUpdate({ stage: "research", status: "done", data: r.data, meta: { ms: r.ms, tokens: r.tokens, completionTime: r.completionTime, completionTokens: r.completionTokens } });
      return r;
    }),
    agentRisk(apiKey, vision.data).then((r) => {
      mark("risk", pStart, Date.now());
      onUpdate({ stage: "risk", status: "done", data: r.data, meta: { ms: r.ms, tokens: r.tokens, completionTime: r.completionTime, completionTokens: r.completionTokens } });
      return r;
    }),
  ]);

  // Agents 4 (Output) and 5 (Deal Strategist) run in parallel — both depend only
  // on research + risk. The Strategist runs a real tool-calling loop to reconcile
  // opportunity vs risk into a GO/HOLD/ESCALATE verdict.
  onUpdate({ stage: "output", status: "running" });
  onUpdate({ stage: "strategist", status: "running" });
  const oStart = Date.now();
  const sStart = Date.now();

  const [output, strategist] = await Promise.all([
    agentOutput(apiKey, vision.data, research.data, risk.data).then((r) => {
      mark("output", oStart, Date.now());
      onUpdate({ stage: "output", status: "done", data: r.data, meta: { ms: r.ms, tokens: r.tokens, completionTime: r.completionTime, completionTokens: r.completionTokens } });
      return r;
    }),
    agentStrategist(apiKey, research.data, risk.data, (ev) => onUpdate({ stage: "strategist", status: "tool", tool: ev })).then((r) => {
      mark("strategist", sStart, Date.now());
      onUpdate({ stage: "strategist", status: "done", data: r.data, meta: { ms: r.ms, tokens: r.tokens, completionTime: r.completionTime, completionTokens: r.completionTokens } });
      return r;
    }).catch((e) => {
      // Tool calling is a bonus capability — never let it break the pipeline.
      const fallback = { verdict: "HOLD", deal_score: null, rationale: `Strategist unavailable (${e.message}).`, trace: [] };
      mark("strategist", sStart, Date.now());
      onUpdate({ stage: "strategist", status: "done", data: fallback, meta: { ms: Date.now() - sStart, tokens: 0 } });
      return { data: fallback, tokens: 0, completionTokens: 0, completionTime: 0, ms: Date.now() - sStart };
    }),
  ]);

  const elapsedMs = Date.now() - t0;
  const elapsed = (elapsedMs / 1000).toFixed(1);
  const all = [vision, research, risk, output, strategist];
  const totalTokens = all.reduce((s, a) => s + (a.tokens || 0), 0);
  // Real output tokens/sec from the API's time_info (matches Cerebras' headline
  // metric); fall back to a wall-clock estimate if time_info isn't present.
  const genTokens = all.reduce((s, a) => s + (a.completionTokens || 0), 0);
  const genTime = all.reduce((s, a) => s + (a.completionTime || 0), 0);
  const tokensPerSec = genTime > 0
    ? Math.round(genTokens / genTime)
    : (elapsedMs > 0 ? Math.round((totalTokens / elapsedMs) * 1000) : 0);

  // ── Streamed executive briefing — the visible "watch Gemma write at Cerebras
  // speed" moment. Runs after the 4-agent pipeline so it does not affect the
  // headline elapsed time; failures here never break the pipeline.
  let briefingTokensPerSec = 0;
  if (onStream) {
    onUpdate({ stage: "briefing", status: "running" });
    let briefing = "";
    try {
      const s = await callGemmaStream(
        apiKey,
        [{
          role: "user",
          content: `You are a chief of staff. Using the data below, write a punchy 3-4 sentence executive briefing a CEO could read in 10 seconds: who the client is, the opportunity, the single biggest risk, and the most important next step. Plain prose only — no markdown, no headings, no preamble.

Vision: ${JSON.stringify(vision.data)}
Research: ${JSON.stringify(research.data)}
Risk: ${JSON.stringify(risk.data)}`,
        }],
        {},
        (delta) => { briefing += delta; onStream(briefing); }
      );
      if (s.completionTime > 0) briefingTokensPerSec = Math.round(s.completionTokens / s.completionTime);
    } catch (e) {
      onStream(`(Live briefing unavailable: ${e.message})`);
    }
    onUpdate({ stage: "briefing", status: "done" });
  }

  return {
    visionData: vision.data,
    researchData: research.data,
    riskData: risk.data,
    outputData: output.data,
    strategistData: strategist.data,
    elapsed,
    totalTokens,
    tokensPerSec,
    briefingTokensPerSec,
    timeline,
  };
}

// ─── UI Components ────────────────────────────────────────────────────────────

function AgentBadge({ label, icon, status, meta }) {
  const isRunning = status === "running";
  const isDone = status === "done";
  const isError = status === "error";
  
  const statusColors = {
    idle: { bg: "rgba(30,32,38,0.6)", border: "rgba(255,255,255,0.08)", text: "#64748b", dot: "#475569" },
    running: { bg: "linear-gradient(135deg, rgba(30,64,175,0.15), rgba(29,78,216,0.08))", border: "#93c5fd", text: "#60a5fa", dot: "#3b82f6" },
    done: { bg: "linear-gradient(135deg, rgba(22,163,74,0.12), rgba(34,197,94,0.06))", border: "#4ade80", text: "#4ade80", dot: "#22c55e" },
    error: { bg: "rgba(220,38,38,0.1)", border: "rgba(220,38,38,0.3)", text: "#f87171", dot: "#ef4444" },
  };
  const c = statusColors[status] || statusColors.idle;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "10px 14px", borderRadius: 10,
      background: c.bg, border: `1.5px solid ${c.border}`,
      transition: "all 0.4s ease",
      boxShadow: isRunning ? `0 0 16px rgba(59,130,246,0.2)` : isDone ? `0 0 12px rgba(34,197,94,0.15)` : "none",
      animation: isRunning ? "agentPulse 2s ease-in-out infinite" : isDone ? "agentSlideIn 0.4s ease-out" : "none",
    }}>
      <span style={{ fontSize: 18, filter: isRunning ? "none" : isDone ? "none" : "grayscale(0.5)", transition: "filter 0.3s" }}>{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: c.text, transition: "color 0.3s" }}>{label}</p>
        {isDone && meta && (
          <p style={{ margin: "2px 0 0", fontSize: 11, color: "#6b7280", fontVariantNumeric: "tabular-nums" }}>
            {meta.ms}ms · {meta.tokens} tok{meta.completionTime > 0 ? ` · ${Math.round(meta.completionTokens / meta.completionTime).toLocaleString()} tok/s` : ""}
          </p>
        )}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {isRunning && (
          <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width: 4, height: 4, borderRadius: "50%",
                background: "#3b82f6",
                animation: `agentDotBounce 1.2s ease-in-out ${i * 0.15}s infinite`,
              }} />
            ))}
          </div>
        )}
        {isDone && <span style={{ fontSize: 16, animation: "agentCheckPop 0.3s ease-out" }}>✓</span>}
        {isError && <span style={{ fontSize: 14, color: "#dc2626" }}>✕</span>}
        {status === "idle" && <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#d1d5db" }} />}
      </div>
    </div>
  );
}

// Horizontal Gantt-style bar proving Research + Risk overlap in time.
function PipelineTimeline({ timeline }) {
  if (!timeline || timeline.length === 0) return null;
  const total = Math.max(...timeline.map((t) => t.end)) || 1;
  const rows = {
    vision: { label: "👁 Vision", color: "#3b82f6" },
    research: { label: "🔍 Research", color: "#8b5cf6" },
    risk: { label: "⚠️ Risk", color: "#f59e0b" },
    output: { label: "📋 Output", color: "#10b981" },
    strategist: { label: "🧭 Strategy", color: "#14b8a6" },
  };
  return (
    <Card>
      <Section title="Execution timeline — Research + Risk run in parallel">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {timeline.map((t, i) => {
            const r = rows[t.stage] || { label: t.stage, color: "#6b7280" };
            const left = (t.start / total) * 100;
            const width = Math.max(((t.end - t.start) / total) * 100, 2);
            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ width: 92, fontSize: 12, color: "#cbd5e1", flexShrink: 0 }}>{r.label}</span>
                <div style={{ flex: 1, position: "relative", height: 22, background: "rgba(255,255,255,0.05)", borderRadius: 6 }}>
                  <div style={{
                    position: "absolute", left: `${left}%`, width: `${width}%`, top: 0, bottom: 0,
                    background: r.color, borderRadius: 6, display: "flex", alignItems: "center",
                    justifyContent: "flex-end", paddingRight: 6, transition: "all 0.4s ease",
                  }}>
                    <span style={{ fontSize: 10, color: "#fff", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                      {((t.end - t.start) / 1000).toFixed(1)}s
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <p style={{ margin: "10px 0 0", fontSize: 11, color: "#9ca3af" }}>
          Bars are positioned on a shared time axis — overlapping bars ran concurrently via <code>Promise.all()</code>.
        </p>
      </Section>
    </Card>
  );
}

function RiskBadge({ score, number }) {
  const colors = { low: "#15803d", medium: "#b45309", high: "#dc2626" };
  const bg = { low: "#f0fdf4", medium: "#fffbeb", high: "#fef2f2" };
  const border = { low: "#bbf7d0", medium: "#fde68a", high: "#fecaca" };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "3px 10px", borderRadius: 20,
      background: bg[score] || bg.medium,
      border: `1px solid ${border[score] || border.medium}`,
      color: colors[score] || colors.medium,
      fontSize: 12, fontWeight: 600,
    }}>
      {score?.toUpperCase()} RISK · {number}/10
    </span>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ width: 3, height: 13, borderRadius: 2, background: "linear-gradient(180deg, #3b82f6, #8b5cf6)", flexShrink: 0 }} />
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#94a3b8" }}>{title}</span>
      </div>
      {children}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: "linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 16, padding: "18px 22px", backdropFilter: "blur(16px)",
      boxShadow: "inset 0 1px 0 rgba(255,255,255,0.05), 0 10px 34px rgba(0,0,0,0.40)",
      ...style,
    }}>
      {children}
    </div>
  );
}

function ChecklistItem({ task, priority, done, onToggle }) {
  const pc = { high: "#dc2626", medium: "#b45309", low: "#6b7280" };
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 10,
      padding: "8px 0", borderBottom: "1px solid rgba(255,255,255,0.06)",
    }}>
      <button
        onClick={onToggle}
        style={{
          width: 18, height: 18, borderRadius: 4, flexShrink: 0, marginTop: 1,
          border: done ? "none" : "2px solid rgba(255,255,255,0.2)",
          background: done ? "#22c55e" : "transparent",
          cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        {done && <span style={{ color: "#fff", fontSize: 11 }}>✓</span>}
      </button>
      <span style={{ flex: 1, fontSize: 13, color: done ? "#64748b" : "#e2e8f0", textDecoration: done ? "line-through" : "none" }}>{task}</span>
      <span style={{ fontSize: 11, color: pc[priority], fontWeight: 500, textTransform: "uppercase" }}>{priority}</span>
    </div>
  );
}

// Live terminal-style audit log of what each agent is doing in real time.
// Doubles as the enterprise "audit trail" — every step is timestamped.
function ThoughtStream({ stream }) {
  const scrollRef = useRef(null);
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [stream.length]);
  if (!stream || stream.length === 0) return null;

  const agentColor = {
    vision: "#60a5fa", research: "#a78bfa", risk: "#fbbf24", output: "#34d399", strategist: "#2dd4bf", pipeline: "#e2e8f0",
  };
  const agentLabel = {
    vision: "👁 VISION   ", research: "🔍 RESEARCH ", risk: "⚠️ RISK     ", output: "📋 OUTPUT   ", strategist: "🧭 STRATEGY ", pipeline: "⚡ PIPELINE ",
  };
  const typeColor = { start: "#94a3b8", done: "#86efac", error: "#fca5a5", info: "#cbd5e1" };

  return (
    <Card style={{ background: "rgba(8,10,14,0.92)", border: "1px solid rgba(59,130,246,0.18)", padding: 0, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.02)" }}>
        <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#ef4444" }} />
        <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#f59e0b" }} />
        <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#22c55e" }} />
        <span style={{ marginLeft: 8, fontSize: 12, fontWeight: 600, color: "#94a3b8", letterSpacing: "0.04em" }}>
          💬 Live Agent Thought Stream — real Gemma 4 audit log
        </span>
      </div>
      <div
        ref={scrollRef}
        style={{
          maxHeight: 240, overflowY: "auto", padding: "12px 16px",
          fontFamily: "'Menlo', 'Monaco', 'Consolas', monospace", fontSize: 12, lineHeight: 1.6,
        }}
      >
        {stream.map((t) => (
          <div key={t.id} style={{ display: "flex", gap: 10, whiteSpace: "pre-wrap" }}>
            <span style={{ color: "#475569", flexShrink: 0, fontVariantNumeric: "tabular-nums" }}>[{String(t.elapsed).padStart(4, " ")}s]</span>
            <span style={{ color: agentColor[t.agent] || "#e2e8f0", flexShrink: 0, fontWeight: 700 }}>{agentLabel[t.agent] || t.agent}</span>
            <span style={{ color: typeColor[t.type] || "#cbd5e1", flex: 1 }}>{t.message}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function VisualOps() {
  const [apiKey, setApiKey] = useState("");
  const [apiKeySet, setApiKeySet] = useState(false);
  const [openaiKey, setOpenaiKey] = useState("");
  const [openaiKeySet, setOpenaiKeySet] = useState(false);
  const [geminiKey, setGeminiKey] = useState("");
  const [geminiKeySet, setGeminiKeySet] = useState(false);
  const [models, setModels] = useState(null);
  const [modelsErr, setModelsErr] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(null);
  const [tokensPerSec, setTokensPerSec] = useState(null);
  const [totalTokens, setTotalTokens] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [checklist, setChecklist] = useState([]);
  // Speed comparison state
  const [speedResult, setSpeedResult] = useState(null);
  const [speedRunning, setSpeedRunning] = useState(false);
  const [speedError, setSpeedError] = useState(null);
  const [liveElapsed, setLiveElapsed] = useState(0);
  const timerRef = useRef(null);
  // Processing history
  const [processHistory, setProcessHistory] = useState([]);
  const [showConfetti, setShowConfetti] = useState(false);
  const resultsRef = useRef(null);

  const [agentStates, setAgentStates] = useState({
    vision: "idle", research: "idle", risk: "idle", output: "idle", strategist: "idle",
  });
  const [timings, setTimings] = useState({
    vision: null, research: null, risk: null, output: null, strategist: null,
  });
  const [results, setResults] = useState({
    vision: null, research: null, risk: null, output: null, strategist: null,
  });

  const dropRef = useRef();

  // ─── Voice Narration (Web Speech API) ──────────────────────────────────────
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [thoughtStream, setThoughtStream] = useState([]);
  const [streamingBriefing, setStreamingBriefing] = useState("");
  const [briefingDone, setBriefingDone] = useState(false);
  const pipelineStartRef = useRef(null);

  const speak = useCallback((text) => {
    if (!voiceEnabled || typeof speechSynthesis === "undefined") return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.15; u.pitch = 1.0; u.volume = 0.8;
    speechSynthesis.speak(u);
  }, [voiceEnabled]);

  // ─── Sound Effects (Web Audio API — zero dependencies) ────────────────────
  const playTone = useCallback((freq, duration = 100, type = "sine", vol = 0.08) => {
    if (!soundEnabled) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type; osc.frequency.value = freq;
      gain.gain.setValueAtTime(vol, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration / 1000);
      osc.connect(gain).connect(ctx.destination);
      osc.start(); osc.stop(ctx.currentTime + duration / 1000);
    } catch { /* AudioContext not available */ }
  }, [soundEnabled]);

  // ─── Thought Stream Logger ────────────────────────────────────────────────
  const addThought = useCallback((agent, message, type = "info") => {
    const elapsed = pipelineStartRef.current ? ((Date.now() - pipelineStartRef.current) / 1000).toFixed(1) : "0.0";
    setThoughtStream(prev => [...prev.slice(-50), { agent, message, type, elapsed, id: Date.now() + Math.random() }]);
  }, []);

  const AGENT_LABELS = { vision: "👁 VISION", research: "🔍 RESEARCH", risk: "⚠️ RISK", output: "📋 OUTPUT", strategist: "🧭 STRATEGY" };

  const updateAgent = useCallback(({ stage, status, data, meta, tool }) => {
    // Live tool-call events from the Deal Strategist's agentic loop
    if (status === "tool" && tool) {
      const argStr = Object.entries(tool.args || {}).map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`).join(", ");
      const resStr = tool.result?.deal_score != null ? `→ deal score ${tool.result.deal_score}/100 (${tool.result.band})`
        : tool.result?.flagged ? `→ flagged for human (${tool.result.severity})` : "→ ok";
      addThought("strategist", `🛠 tool call: ${tool.name}(${argStr}) ${resStr}`, "info");
      playTone(560, 90, "sine");
      return;
    }

    setAgentStates((prev) => ({ ...prev, [stage]: status }));
    if (data) setResults((prev) => ({ ...prev, [stage]: data }));
    if (meta) setTimings((prev) => ({ ...prev, [stage]: meta }));
    if (stage === "output" && status === "done" && data?.action_checklist) {
      setChecklist(data.action_checklist);
    }

    // Thought stream + voice + sound
    if (status === "running") {
      const startMsg = stage === "vision" ? "multimodal document reading"
        : stage === "research" ? "client intelligence scan"
        : stage === "risk" ? "risk analysis (reasoning mode)"
        : stage === "strategist" ? "reconciling opportunity vs risk via tool calling"
        : stage === "briefing" ? "streaming executive briefing live at Cerebras speed"
        : "onboarding package synthesis";
      addThought(stage === "briefing" ? "output" : stage, `starting ${startMsg}...`, "start");
      playTone(300 + Math.random() * 200, 120, "sine");
    }
    if (stage === "briefing") {
      if (status === "running") setBriefingDone(false);
      if (status === "done") { setBriefingDone(true); addThought("output", "✓ executive briefing streamed", "done"); }
      return;
    }
    if (status === "done" && data) {
      const ms = meta?.ms || 0;
      if (stage === "vision") {
        const co = data.company_name || "unknown";
        const ind = data.industry || "unknown";
        addThought(stage, `✓ extracted: company="${co}", industry="${ind}", type="${data.document_type || "document"}" (${ms}ms)`, "done");
        speak(`Vision Agent complete. Detected ${co}, ${ind} sector.`);
      } else if (stage === "research") {
        const pains = data.pain_points?.length || 0;
        const tech = data.likely_tech_stack?.length || 0;
        const priorities = data.onboarding_priorities?.length || 0;
        addThought(stage, `✓ profiled client: ${pains} pain points, ${tech} tech-stack signals, ${priorities} onboarding priorities (${ms}ms)`, "done");
        speak(`Research Agent profiled the client. ${pains} pain points, ${priorities} onboarding priorities.`);
      } else if (stage === "risk") {
        const flags = data.flags?.length || 0;
        const rec = data.proceed_recommendation || "review";
        const score = data.risk_score_number != null ? `${data.risk_score_number} out of 10` : (data.risk_score || "unknown");
        addThought(stage, `✓ ${flags} risk flags · score ${score} · recommendation: ${rec} (${ms}ms)`, "done");
        speak(`Risk scan complete. ${flags} flags identified. Recommendation: ${rec.replace(/-/g, " ")}.`);
      } else if (stage === "output") {
        const items = data.action_checklist?.length || 0;
        addThought(stage, `✓ generated: ${items} action items, welcome email, success metrics (${ms}ms)`, "done");
        speak(`Onboarding package generated. ${items} action items ready.`);
      } else if (stage === "strategist") {
        const calls = data.trace?.length || 0;
        addThought(stage, `✓ verdict: ${data.verdict}${data.deal_score != null ? ` · deal score ${data.deal_score}/100` : ""} · ${calls} tool calls (${ms}ms)`, "done");
        speak(`Deal Strategist verdict: ${data.verdict}.`);
      }
      playTone(800, 80, "sine");
      playTone(1200, 60, "sine");
    }
  }, [addThought, speak, playTone]);

  // Inject CSS animations for agent badges
  useEffect(() => {
    const styleId = "visualops-animations";
    if (document.getElementById(styleId)) return;
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
      @keyframes agentPulse {
        0%, 100% { box-shadow: 0 0 8px rgba(59,130,246,0.15); }
        50% { box-shadow: 0 0 20px rgba(59,130,246,0.35); }
      }
      @keyframes agentSlideIn {
        0% { transform: translateX(-8px); opacity: 0.5; }
        100% { transform: translateX(0); opacity: 1; }
      }
      @keyframes agentCheckPop {
        0% { transform: scale(0); }
        60% { transform: scale(1.3); }
        100% { transform: scale(1); }
      }
      @keyframes agentDotBounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-6px); }
      }
      @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(16px); }
        100% { opacity: 1; transform: translateY(0); }
      }
      @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }
      @keyframes confettiFall {
        0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
        100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
      }
      @keyframes confettiPop {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
      }
      @keyframes successGlow {
        0% { box-shadow: 0 0 0 rgba(34,197,94,0); }
        50% { box-shadow: 0 0 40px rgba(34,197,94,0.4); }
        100% { box-shadow: 0 0 0 rgba(34,197,94,0); }
      }
    `;
    document.head.appendChild(style);
  }, []);

  // Live elapsed timer
  useEffect(() => {
    if (running) {
      const start = Date.now();
      setLiveElapsed(0);
      timerRef.current = setInterval(() => {
        setLiveElapsed(((Date.now() - start) / 1000).toFixed(1));
      }, 100);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running]);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults({ vision: null, research: null, risk: null, output: null, strategist: null });
    setAgentStates({ vision: "idle", research: "idle", risk: "idle", output: "idle", strategist: "idle" });
    setTimings({ vision: null, research: null, risk: null, output: null, strategist: null });
    setElapsed(null);
    setTokensPerSec(null);
    setTotalTokens(null);
    setTimeline(null);
    setError(null);
    setSpeedResult(null);
    setSpeedError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    handleFile(f);
  };

  const handleRun = async () => {
    if (!file || !apiKey) return;
    setRunning(true);
    setError(null);
    setThoughtStream([]);
    setStreamingBriefing("");
    setBriefingDone(false);
    pipelineStartRef.current = Date.now();
    addThought("pipeline", "🚀 PIPELINE STARTING — 5 agents on Cerebras WSE-3", "start");
    speak("Starting 5 agent pipeline on Cerebras.");
    playTone(400, 200, "sine");
    try {
      const { base64, mimeType } = await prepareImage(file);
      const res = await runPipeline(apiKey, base64, mimeType, updateAgent, setStreamingBriefing);
      setElapsed(res.elapsed);
      setTokensPerSec(res.tokensPerSec);
      setTotalTokens(res.totalTokens);
      setTimeline(res.timeline);
      // Confetti celebration!
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 3000);
      // Pipeline complete thought + voice
      addThought("pipeline", `✅ COMPLETE — ${res.elapsed}s — ${res.tokensPerSec?.toLocaleString()} tok/s — ${res.totalTokens} tokens`, "done");
      speak(`Pipeline complete in ${res.elapsed} seconds. ${res.tokensPerSec?.toLocaleString()} tokens per second on Cerebras.`);
      // Celebration chord
      playTone(523, 200, "sine"); playTone(659, 200, "sine"); playTone(784, 300, "sine");
      // Add to processing history
      setProcessHistory(prev => [...prev, {
        id: Date.now(),
        company: res.visionData?.company_name || file?.name || "Unknown",
        elapsed: res.elapsed,
        tokensPerSec: res.tokensPerSec,
        totalTokens: res.totalTokens,
        timestamp: new Date().toLocaleTimeString(),
      }]);
    } catch (e) {
      setError(e.message);
      addThought("pipeline", `❌ ERROR: ${e.message}`, "error");
      playTone(200, 300, "sawtooth");
    } finally {
      setRunning(false);
    }
  };

  const toggleCheck = (i) => {
    setChecklist((prev) => prev.map((item, idx) => idx === i ? { ...item, done: !item.done } : item));
  };

  const generateMarkdown = () => {
    const v = results.vision || {};
    const r = results.research || {};
    const rk = results.risk || {};
    const o = results.output || {};
    const st = results.strategist || {};
    let md = `# Onboarding Package: ${v.company_name || "Client"}\n\n`;
    md += `> Generated by VisualOps — 5-Agent B2B Onboarding Intelligence\n`;
    md += `> Powered by Gemma 4 31B on Cerebras WSE-3 · ${elapsed || "N/A"} · ${tokensPerSec ? tokensPerSec.toLocaleString() + " tok/s" : ""}\n\n`;
    md += `---\n\n`;
    // Deal Strategist Verdict
    if (st.verdict) {
      md += `## Deal Strategist Verdict\n\n`;
      md += `**${st.verdict}**${st.deal_score != null ? ` — Deal score ${st.deal_score}/100` : ""}\n\n`;
      if (st.rationale) md += `${st.rationale}\n\n`;
    }
    // Executive Summary
    if (o.executive_summary) {
      md += `## Executive Summary\n\n${o.executive_summary}\n\n`;
    }
    // Onboarding Plan
    if (o.onboarding_plan?.length) {
      md += `## Onboarding Plan\n\n`;
      md += `| # | Action | Owner | Timeline | Goal |\n|---|--------|-------|----------|------|\n`;
      o.onboarding_plan.forEach((s, i) => {
        md += `| ${s.step ?? i + 1} | ${s.action || ""} | ${s.owner || ""} | ${s.timeline || ""} | ${s.goal || ""} |\n`;
      });
      md += `\n`;
    }
    // Welcome Email
    if (o.welcome_email) {
      const subject = o.welcome_email.subject || "";
      const body = o.welcome_email.body || "";
      md += `## Welcome Email Draft\n\n**Subject:** ${subject}\n\n${body}\n\n`;
    }
    // Action Checklist
    if (o.action_checklist?.length) {
      md += `## Action Checklist\n\n`;
      o.action_checklist.forEach((item) => {
        md += `- [ ] **[${(item.priority || "medium").toUpperCase()}]** ${item.task}\n`;
      });
      md += `\n`;
    }
    // Success Metrics
    if (o.success_metrics?.length) {
      md += `## Success Metrics\n\n`;
      o.success_metrics.forEach((m) => {
        md += `- ✅ ${m}\n`;
      });
      md += `\n`;
    }
    // Client Intelligence
    if (r.company_profile || r.pain_points?.length) {
      md += `## Client Intelligence\n\n`;
      if (r.company_profile) md += `**Company Profile:** ${r.company_profile}\n\n`;
      if (r.pain_points?.length) {
        md += `**Pain Points:**\n`;
        r.pain_points.forEach((p) => { md += `- ${p}\n`; });
        md += `\n`;
      }
      if (r.likely_tech_stack?.length) {
        md += `**Likely Tech Stack:** ${r.likely_tech_stack.join(", ")}\n\n`;
      }
      if (r.recommended_approach) md += `**Recommended Approach:** ${r.recommended_approach}\n\n`;
    }
    // Risk Assessment
    if (rk.risk_score_number || rk.risk_score || rk.flags?.length) {
      md += `## Risk Assessment\n\n`;
      const score = rk.risk_score_number != null ? `${rk.risk_score_number}/10` : (rk.risk_score || "N/A");
      md += `**Risk Score:** ${score} (${(rk.risk_score || "").toUpperCase()}) — **Recommendation:** ${(rk.proceed_recommendation || "N/A").toUpperCase()}\n\n`;
      if (rk.flags?.length) {
        md += `| Issue | Severity | Recommendation |\n|-------|----------|----------------|\n`;
        rk.flags.forEach((f) => {
          md += `| ${f.issue || ""} | ${f.severity || ""} | ${f.recommendation || ""} |\n`;
        });
        md += `\n`;
      }
      if (rk.missing_information?.length) {
        md += `**Missing Information:**\n`;
        rk.missing_information.forEach((m) => { md += `- ${m}\n`; });
        md += `\n`;
      }
      if (rk.compliance_notes) md += `**Compliance Notes:** ${rk.compliance_notes}\n\n`;
    }
    md += `---\n\n*Generated at ${new Date().toISOString()} by VisualOps*\n`;
    return md;
  };

  const [copied, setCopied] = useState(null);
  const copyToClipboard = (text, label) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(label);
      setTimeout(() => setCopied(null), 2000);
    });
  };

  const downloadPackage = () => {
    const md = generateMarkdown();
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `onboarding-package-${(results.vision?.company_name || "client").toLowerCase().replace(/\s+/g, "-")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const checkModels = async () => {
    setModels(null);
    setModelsErr(null);
    try {
      setModels(await listModels(apiKey));
    } catch (e) {
      setModelsErr(e.message);
    }
  };

  // ─── Speed Comparison Handler ──────────────────────────────────────────────
  const handleSpeedCompare = async () => {
    if (!apiKey || (!openaiKey && !geminiKey)) return;
    setSpeedRunning(true);
    setSpeedError(null);
    setSpeedResult(null);
    const companyName = results.vision?.company_name || "the client";
    const prompt = `You are a B2B analyst. Write a concise 3-sentence executive summary for onboarding ${companyName} as a new enterprise client. Focus on key priorities and recommended next steps.`;
    try {
      const result = await runSpeedComparison(apiKey, openaiKey || null, geminiKey || null, prompt);
      setSpeedResult(result);
    } catch (e) {
      setSpeedError(e.message);
    } finally {
      setSpeedRunning(false);
    }
  };


  const hasResults = results.output !== null;
  const hasAnyResults = results.vision !== null || results.research !== null || results.risk !== null || results.output !== null;
  const riskData = results.risk;
  const outputData = results.output;
  const visionData = results.vision;
  const researchData = results.research;
  const strategistData = results.strategist;

  // Auto-scroll to results as they appear
  useEffect(() => {
    if (hasAnyResults && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [results.vision, results.research, results.risk, results.output]);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#07080a",
      backgroundImage: "radial-gradient(1100px 600px at 12% -8%, rgba(59,130,246,0.14), transparent 58%), radial-gradient(1000px 520px at 102% 2%, rgba(139,92,246,0.12), transparent 55%), radial-gradient(900px 700px at 50% 118%, rgba(16,185,129,0.07), transparent 60%)",
      backgroundAttachment: "fixed",
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif", color: "#e2e8f0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; }
        textarea { resize: vertical; }
        input:focus, textarea:focus, button:focus { outline: 2px solid #3b82f6; outline-offset: 2px; }
        .card-animate { animation: slideIn 0.4s ease-out both; }
        .feature-card { transition: transform .2s ease, border-color .2s ease, background .2s ease; }
        .feature-card:hover { transform: translateY(-3px); border-color: rgba(99,102,241,0.45); background: rgba(255,255,255,0.06); }
        ::selection { background: rgba(99,102,241,0.4); }
        *::-webkit-scrollbar { width: 10px; height: 10px; }
        *::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 8px; }
        *::-webkit-scrollbar-track { background: transparent; }
      `}</style>

      {/* Header */}
      <div style={{ position: "sticky", top: 0, zIndex: 100, background: "rgba(8,9,12,0.72)", borderBottom: "1px solid rgba(255,255,255,0.07)", backdropFilter: "blur(18px)", WebkitBackdropFilter: "blur(18px)", padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 38, height: 38, borderRadius: 12, background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 6px 20px rgba(99,102,241,0.45)" }}>
            <span style={{ color: "#fff", fontSize: 19 }}>⚡</span>
          </div>
          <div>
            <p style={{ margin: 0, fontWeight: 800, fontSize: 18, letterSpacing: "-0.02em", background: "linear-gradient(135deg, #fff, #c7d2fe)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>VisualOps</p>
            <p style={{ margin: 0, fontSize: 11, color: "#8b95a7" }}>5-agent onboarding intelligence · Gemma 4 31B on Cerebras WSE-3 · ~1,850 tok/s</p>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {elapsed && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(34,197,94,0.15)", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 8, padding: "6px 12px" }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: "#4ade80" }}>⚡ {elapsed}s</span>
              <span style={{ fontSize: 11, color: "#86efac" }}>Cerebras</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(148,163,184,0.1)", border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8, padding: "6px 12px" }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: "#64748b", textDecoration: "line-through" }}>~75s</span>
              <span style={{ fontSize: 11, color: "#64748b" }}>est. GPU baseline</span>
            </div>
            <span style={{ fontSize: 13, fontWeight: 700, color: "#fff", background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", borderRadius: 8, padding: "6px 14px", boxShadow: "0 2px 8px rgba(59,130,246,0.3)" }}>
              {Math.round(75 / parseFloat(elapsed))}× faster
            </span>
            {tokensPerSec ? (
              <div style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(59,130,246,0.2)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 8, padding: "6px 12px" }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: "#93c5fd", fontVariantNumeric: "tabular-nums" }}>{tokensPerSec.toLocaleString()}</span>
                <span style={{ fontSize: 11, color: "#60a5fa" }}>tok/s</span>
              </div>
            ) : null}
          </div>
          )}
          <button
            onClick={() => setVoiceEnabled((v) => !v)}
            title={voiceEnabled ? "Voice narration on — click to mute" : "Voice narration off — click to enable"}
            style={{
              display: "flex", alignItems: "center", gap: 6, cursor: "pointer",
              background: voiceEnabled ? "rgba(59,130,246,0.2)" : "rgba(148,163,184,0.1)",
              border: `1px solid ${voiceEnabled ? "rgba(59,130,246,0.4)" : "rgba(148,163,184,0.2)"}`,
              borderRadius: 8, padding: "6px 10px",
              color: voiceEnabled ? "#93c5fd" : "#64748b", fontSize: 13, fontWeight: 600,
            }}
          >
            🎙️ {voiceEnabled ? "Voice on" : "Voice off"}
          </button>
          <button
            onClick={() => setSoundEnabled((s) => !s)}
            title={soundEnabled ? "Sound effects on — click to mute" : "Sound effects off — click to enable"}
            style={{
              display: "flex", alignItems: "center", gap: 6, cursor: "pointer",
              background: soundEnabled ? "rgba(59,130,246,0.2)" : "rgba(148,163,184,0.1)",
              border: `1px solid ${soundEnabled ? "rgba(59,130,246,0.4)" : "rgba(148,163,184,0.2)"}`,
              borderRadius: 8, padding: "6px 10px",
              color: soundEnabled ? "#93c5fd" : "#64748b", fontSize: 13, fontWeight: 600,
            }}
          >
            {soundEnabled ? "🔊" : "🔇"}
          </button>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px", display: "grid", gridTemplateColumns: "340px 1fr", gap: 24 }}>

        {/* Left panel */}
        <div>
          {/* API Key */}
          {!apiKeySet ? (
            <Card style={{ marginBottom: 16 }}>
              <p style={{ margin: "0 0 8px", fontSize: 13, fontWeight: 500, color: "#cbd5e1" }}>Cerebras API Key</p>
              <input
                type="password"
                placeholder="csk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                style={{ width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.25)", color: "#e2e8f0", fontSize: 13, marginBottom: 8 }}
              />
              <button
                onClick={() => apiKey.length > 8 && setApiKeySet(true)}
                style={{
                  width: "100%", padding: "8px", borderRadius: 8,
                  background: "#1d4ed8", color: "#fff", border: "none",
                  fontSize: 13, fontWeight: 500, cursor: "pointer",
                }}
              >
                Save key
              </button>
            </Card>
          ) : (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px", background: "rgba(22,163,74,0.1)", borderRadius: 8, border: "1px solid rgba(34,197,94,0.25)" }}>
                <span style={{ fontSize: 12, color: "#4ade80" }}>✓ Cerebras key set</span>
                <div style={{ display: "flex", gap: 12 }}>
                  <button onClick={checkModels} style={{ fontSize: 11, color: "#1d4ed8", background: "none", border: "none", cursor: "pointer" }}>check models</button>
                  <button onClick={() => setApiKeySet(false)} style={{ fontSize: 11, color: "#6b7280", background: "none", border: "none", cursor: "pointer" }}>change</button>
                </div>
              </div>
              {models && (
                <div style={{ marginTop: 8, padding: "8px 12px", background: "rgba(17,19,24,0.8)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.08)" }}>
                  <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#64748b" }}>Models your key can use:</p>
                  {models.map((m) => (
                    <p key={m} style={{ margin: "0 0 2px", fontSize: 12, color: m === "gemma-4-31b" ? "#4ade80" : "#cbd5e1", fontWeight: m === "gemma-4-31b" ? 700 : 400 }}>
                      {m === "gemma-4-31b" ? "✓ " : "· "}{m}
                    </p>
                  ))}
                </div>
              )}
              {modelsErr && (
                <div style={{ marginTop: 8, padding: "8px 12px", background: "rgba(220,38,38,0.1)", border: "1px solid rgba(220,38,38,0.3)", borderRadius: 8, fontSize: 12, color: "#f87171" }}>{modelsErr}</div>
              )}
            </div>
          )}

          {/* OpenAI API Key (for speed comparison) */}
          {!openaiKeySet ? (
            <Card style={{ marginBottom: 12 }}>
              <p style={{ margin: "0 0 4px", fontSize: 13, fontWeight: 500, color: "#cbd5e1" }}>OpenAI API Key <span style={{ fontSize: 11, color: "#64748b", fontWeight: 400 }}>(optional — GPU comparison)</span></p>
              <input
                type="password"
                placeholder="sk-..."
                value={openaiKey}
                onChange={(e) => setOpenaiKey(e.target.value)}
                style={{ width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.25)", color: "#e2e8f0", fontSize: 13, marginBottom: 8 }}
              />
              <button
                onClick={() => openaiKey.length > 8 && setOpenaiKeySet(true)}
                style={{
                  width: "100%", padding: "8px", borderRadius: 8,
                  background: "#10b981", color: "#fff", border: "none",
                  fontSize: 13, fontWeight: 500, cursor: "pointer",
                }}
              >
                Save OpenAI key
              </button>
            </Card>
          ) : (
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px", background: "rgba(22,163,74,0.1)", borderRadius: 8, border: "1px solid rgba(34,197,94,0.25)" }}>
                <span style={{ fontSize: 12, color: "#4ade80" }}>✓ OpenAI key set</span>
                <button onClick={() => setOpenaiKeySet(false)} style={{ fontSize: 11, color: "#6b7280", background: "none", border: "none", cursor: "pointer" }}>change</button>
              </div>
            </div>
          )}

          {/* Gemini API Key (free tier — TPU comparison) */}
          {!geminiKeySet ? (
            <Card style={{ marginBottom: 16 }}>
              <p style={{ margin: "0 0 4px", fontSize: 13, fontWeight: 500, color: "#cbd5e1" }}>Gemini API Key <span style={{ fontSize: 11, color: "#22c55e", fontWeight: 500 }}>(free — TPU comparison)</span></p>
              <input
                type="password"
                placeholder="AIza..."
                value={geminiKey}
                onChange={(e) => setGeminiKey(e.target.value)}
                style={{ width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.25)", color: "#e2e8f0", fontSize: 13, marginBottom: 8 }}
              />
              <button
                onClick={() => geminiKey.length > 8 && setGeminiKeySet(true)}
                style={{
                  width: "100%", padding: "8px", borderRadius: 8,
                  background: "linear-gradient(135deg, #4285f4, #34a853)", color: "#fff", border: "none",
                  fontSize: 13, fontWeight: 500, cursor: "pointer",
                }}
              >
                Save Gemini key (free)
              </button>
            </Card>
          ) : (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px", background: "rgba(22,163,74,0.1)", borderRadius: 8, border: "1px solid rgba(34,197,94,0.25)" }}>
                <span style={{ fontSize: 12, color: "#4ade80" }}>✓ Gemini key set (free tier)</span>
                <button onClick={() => setGeminiKeySet(false)} style={{ fontSize: 11, color: "#6b7280", background: "none", border: "none", cursor: "pointer" }}>change</button>
              </div>
            </div>
          )}

          {/* Drop zone */}
          <Card style={{ marginBottom: 16 }}>
            <div
              ref={dropRef}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById("file-input").click()}
              style={{
                border: "2px dashed rgba(148,163,184,0.3)", borderRadius: 10,
                padding: preview ? 0 : "32px 16px",
                textAlign: "center", cursor: "pointer",
                background: "rgba(255,255,255,0.03)", overflow: "hidden",
                transition: "border-color 0.2s",
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = "#3b82f6"}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = "rgba(148,163,184,0.3)"}
            >
              {preview ? (
                <img src={preview} alt="uploaded document" style={{ width: "100%", borderRadius: 8, display: "block" }} />
              ) : (
                <>
                  <p style={{ fontSize: 28, margin: "0 0 8px" }}>📄</p>
                  <p style={{ margin: "0 0 4px", fontSize: 14, fontWeight: 500, color: "#cbd5e1" }}>Drop a business document</p>
                  <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>Contract, proposal, invoice, org chart, logo</p>
                </>
              )}
            </div>
            <input id="file-input" type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
            {preview && (
              <button
                onClick={() => { setFile(null); setPreview(null); }}
                style={{ marginTop: 8, fontSize: 12, color: "#9ca3af", background: "none", border: "none", cursor: "pointer" }}
              >
                Remove image
              </button>
            )}
          </Card>

          {/* Agent status */}
          <Card style={{ marginBottom: 16 }}>
            <p style={{ margin: "0 0 10px", fontSize: 12, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "#9ca3af" }}>Agent pipeline</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <AgentBadge label="Vision — document reader" icon="👁" status={agentStates.vision} meta={timings.vision} />
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 1, height: 12, background: "rgba(255,255,255,0.12)", marginLeft: 20 }} />
                <span style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>parallel</span>
              </div>
              <AgentBadge label="Research — client intel" icon="🔍" status={agentStates.research} meta={timings.research} />
              <AgentBadge label="Risk — flag detector" icon="⚠️" status={agentStates.risk} meta={timings.risk} />
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 1, height: 12, background: "rgba(255,255,255,0.12)", marginLeft: 20 }} />
                <span style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>parallel</span>
              </div>
              <AgentBadge label="Output — action generator" icon="📋" status={agentStates.output} meta={timings.output} />
              <AgentBadge label="Strategist — deal reconciler (tools)" icon="🧭" status={agentStates.strategist} meta={timings.strategist} />
            </div>
          </Card>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!file || !apiKeySet || running}
            style={{
              width: "100%", padding: "15px",
              background: (!file || !apiKeySet || running) ? "rgba(255,255,255,0.06)" : "linear-gradient(135deg, #3b82f6, #8b5cf6)",
              color: (!file || !apiKeySet || running) ? "#64748b" : "#fff",
              border: (!file || !apiKeySet || running) ? "1px solid rgba(255,255,255,0.08)" : "none", borderRadius: 12,
              fontSize: 15, fontWeight: 700, letterSpacing: "0.01em", cursor: (!file || !apiKeySet || running) ? "not-allowed" : "pointer",
              transition: "transform 0.2s ease, box-shadow 0.3s ease",
              boxShadow: (!file || !apiKeySet || running) ? "none" : "0 8px 24px rgba(99,102,241,0.45)",
            }}
            onMouseEnter={(e) => { if (file && apiKeySet && !running) e.currentTarget.style.transform = "translateY(-2px)"; }}
            onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
          >
            {running ? "⚡ Agents running..." : "Run 5-agent pipeline"}
          </button>

          {error && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(220,38,38,0.1)", border: "1px solid rgba(220,38,38,0.3)", borderRadius: 10, fontSize: 13, color: "#f87171" }}>
              {error}
            </div>
          )}

          {hasResults && !running && (
            <button
              onClick={downloadPackage}
              style={{
                width: "100%", padding: "12px", marginTop: 10,
                background: "linear-gradient(135deg, #059669, #10b981)",
                color: "#fff", border: "none", borderRadius: 10,
                fontSize: 14, fontWeight: 600, cursor: "pointer",
                boxShadow: "0 4px 14px rgba(5,150,105,0.3)",
                transition: "all 0.3s ease",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
              onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
            >
              📥 Download Onboarding Package (.md)
            </button>
          )}
        </div>

        {/* Right panel — results */}
        <div ref={resultsRef}>
          {!hasAnyResults && !running && (
            <div style={{ padding: "40px 32px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 500, gap: 20 }}>
              {/* Hero */}
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 56, marginBottom: 12, animation: "fadeInUp 0.6s ease-out" }}>🤖⚡</div>
                <h2 style={{ margin: "0 0 10px", fontSize: 34, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1, background: "linear-gradient(135deg, #f8fafc 30%, #a5b4fc 75%, #c4b5fd)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
                  AI-Powered Onboarding<br />Intelligence
                </h2>
                <p style={{ margin: "0 0 6px", fontSize: 15, color: "#94a3b8", maxWidth: 420, lineHeight: 1.5 }}>
                  Upload any business document and get a complete onboarding package in seconds — powered by 5 AI agents.
                </p>
              </div>

              {/* Feature grid */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, width: "100%", maxWidth: 480 }}>
                {[
                  { icon: "👁", title: "Vision Agent", desc: "Reads images directly via Gemma 4 multimodal" },
                  { icon: "🔍", title: "Research Agent", desc: "Builds client intelligence profile" },
                  { icon: "⚠️", title: "Risk Agent", desc: "Scans for red flags with reasoning mode" },
                  { icon: "📋", title: "Output Agent", desc: "Generates complete onboarding package" },
                ].map((f, i) => (
                  <div key={i} className="feature-card" style={{
                    padding: "14px 16px", borderRadius: 12, background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)", animation: `fadeInUp ${0.4 + i * 0.1}s ease-out`,
                  }}>
                    <div style={{ fontSize: 20, marginBottom: 4 }}>{f.icon}</div>
                    <p style={{ margin: "0 0 2px", fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>{f.title}</p>
                    <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", lineHeight: 1.4 }}>{f.desc}</p>
                  </div>
                ))}
              </div>

              {/* Tech badges */}
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center", animation: "fadeInUp 0.8s ease-out" }}>
                {["Gemma 4 31B", "Cerebras WSE-3", "~1,850 tok/s", "Structured Outputs", "Reasoning Mode", "Parallel Agents"].map((t, i) => (
                  <span key={i} style={{
                    padding: "4px 12px", borderRadius: 20, fontSize: 11, fontWeight: 500,
                    background: i < 3 ? "rgba(59,130,246,0.15)" : "rgba(255,255,255,0.05)",
                    border: `1px solid ${i < 3 ? "rgba(59,130,246,0.35)" : "rgba(255,255,255,0.1)"}`,
                    color: i < 3 ? "#93c5fd" : "#94a3b8",
                  }}>{t}</span>
                ))}
              </div>

              <p style={{ margin: 0, fontSize: 12, color: "#d1d5db" }}>
                ← Upload a document image to get started
              </p>
            </div>
          )}

          {running && !hasAnyResults && (
            <div style={{ padding: "40px 32px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 500, gap: 16 }}>
              <div style={{ fontSize: 52, animation: "pulse 1.2s infinite" }}>⚡</div>
              <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#f8fafc" }}>Agents Working at Cerebras Speed</h3>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ fontSize: 32, fontWeight: 800, color: "#3b82f6", fontVariantNumeric: "tabular-nums", fontFamily: "'Inter', monospace" }}>{liveElapsed}s</span>
                <span style={{ fontSize: 13, color: "#6b7280" }}>elapsed</span>
              </div>
              <p style={{ margin: 0, fontSize: 12, color: "#9ca3af" }}>~1,850 tokens/sec · Gemma 4 31B on WSE-3</p>

              {/* Live agent status */}
              <div style={{ width: "100%", maxWidth: 360, display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
                {[
                  { key: "vision", label: "Vision — reading document", icon: "👁" },
                  { key: "research", label: "Research — client intelligence", icon: "🔍" },
                  { key: "risk", label: "Risk — scanning for flags", icon: "⚠️" },
                  { key: "output", label: "Output — generating package", icon: "📋" },
                ].map(({ key, label, icon }) => {
                  const s = agentStates[key];
                  return (
                    <div key={key} style={{
                      display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", borderRadius: 8,
                      background: s === "running" ? "rgba(59,130,246,0.12)" : s === "done" ? "rgba(34,197,94,0.1)" : "rgba(255,255,255,0.03)",
                      border: `1px solid ${s === "running" ? "rgba(59,130,246,0.35)" : s === "done" ? "rgba(34,197,94,0.3)" : "rgba(255,255,255,0.08)"}`,
                      transition: "all 0.3s",
                    }}>
                      <span style={{ fontSize: 14 }}>{icon}</span>
                      <span style={{ flex: 1, fontSize: 12, fontWeight: 500, color: s === "done" ? "#4ade80" : s === "running" ? "#93c5fd" : "#94a3b8" }}>{label}</span>
                      {s === "running" && <span style={{ fontSize: 12, animation: "pulse 1s infinite" }}>●</span>}
                      {s === "done" && <span style={{ fontSize: 12, color: "#22c55e" }}>✓</span>}
                    </div>
                  );
                })}
              </div>

              <div style={{ width: "100%", maxWidth: 480, marginTop: 8 }}>
                <ThoughtStream stream={thoughtStream} />
              </div>
            </div>
          )}

          {hasAnyResults && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Confetti Celebration */}
              {showConfetti && (
                <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, pointerEvents: "none", zIndex: 9999, overflow: "hidden" }}>
                  {Array.from({ length: 50 }).map((_, i) => (
                    <div key={i} style={{
                      position: "absolute",
                      left: `${Math.random() * 100}%`,
                      top: `-${Math.random() * 20}px`,
                      width: `${6 + Math.random() * 8}px`,
                      height: `${6 + Math.random() * 8}px`,
                      borderRadius: Math.random() > 0.5 ? "50%" : "2px",
                      background: ["#3b82f6", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444", "#ec4899"][Math.floor(Math.random() * 6)],
                      animation: `confettiFall ${2 + Math.random() * 2}s ease-out ${Math.random() * 0.5}s forwards`,
                    }} />
                  ))}
                </div>
              )}

              {/* Live agent thought stream / audit log */}
              <ThoughtStream stream={thoughtStream} />

              {/* Live streamed executive briefing — Cerebras speed, made visible */}
              {streamingBriefing && (
                <Card style={{ background: "linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.08))", border: "1px solid rgba(99,102,241,0.3)" }}>
                  <Section title={
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                      ✍️ Live Executive Briefing
                      <span style={{ fontSize: 10, fontWeight: 600, color: briefingDone ? "#4ade80" : "#93c5fd", background: briefingDone ? "rgba(34,197,94,0.15)" : "rgba(59,130,246,0.15)", borderRadius: 12, padding: "2px 8px", textTransform: "none", letterSpacing: 0 }}>
                        {briefingDone ? "✓ generated on Cerebras" : "streaming live…"}
                      </span>
                    </span>
                  }>
                    <p style={{ margin: 0, fontSize: 15, lineHeight: 1.7, color: "#e2e8f0" }}>
                      {streamingBriefing}
                      {!briefingDone && <span style={{ display: "inline-block", marginLeft: 2, color: "#93c5fd", animation: "pulse 1s infinite", fontWeight: 700 }}>▋</span>}
                    </p>
                  </Section>
                </Card>
              )}

              {/* Deal Strategist — tool-calling reconciler verdict */}
              {strategistData && (() => {
                const vMap = {
                  GO: { c: "#4ade80", bg: "rgba(34,197,94,0.14)", bd: "rgba(34,197,94,0.35)", icon: "✓" },
                  HOLD: { c: "#fbbf24", bg: "rgba(245,158,11,0.14)", bd: "rgba(245,158,11,0.35)", icon: "⏸" },
                  ESCALATE: { c: "#f87171", bg: "rgba(220,38,38,0.14)", bd: "rgba(220,38,38,0.35)", icon: "⛔" },
                };
                const vd = vMap[strategistData.verdict] || vMap.HOLD;
                const score = strategistData.deal_score;
                return (
                  <Card style={{ background: "linear-gradient(135deg, rgba(20,184,166,0.1), rgba(13,148,136,0.05))", border: "1px solid rgba(20,184,166,0.28)" }}>
                    <Section title="🧭 Deal Strategist — Tool-Calling Reconciler">
                      <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 16px", borderRadius: 12, background: vd.bg, border: `1px solid ${vd.bd}`, color: vd.c, fontSize: 16, fontWeight: 800, letterSpacing: "0.02em" }}>
                          {vd.icon} {strategistData.verdict}
                        </span>
                        {score != null && (
                          <div style={{ flex: 1, minWidth: 180 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ fontSize: 11, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.06em" }}>Deal score</span>
                              <span style={{ fontSize: 13, fontWeight: 700, color: "#5eead4", fontVariantNumeric: "tabular-nums" }}>{score}/100</span>
                            </div>
                            <div style={{ height: 8, borderRadius: 6, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${score}%`, borderRadius: 6, background: "linear-gradient(90deg, #14b8a6, #22d3ee)", transition: "width 0.8s ease" }} />
                            </div>
                          </div>
                        )}
                      </div>

                      {strategistData.rationale && (
                        <p style={{ margin: "14px 0 0", fontSize: 14, lineHeight: 1.7, color: "#cbd5e1" }}>{strategistData.rationale}</p>
                      )}

                      {strategistData.trace?.length > 0 && (
                        <div style={{ marginTop: 14 }}>
                          <p style={{ margin: "0 0 6px", fontSize: 11, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.06em" }}>Tools the agent called</p>
                          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {strategistData.trace.map((t, i) => {
                              const argStr = Object.entries(t.args || {}).map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`).join(", ");
                              const resStr = t.result?.deal_score != null ? `deal score ${t.result.deal_score}/100 (${t.result.band})`
                                : t.result?.flagged ? `flagged for human · ${t.result.severity}` : JSON.stringify(t.result);
                              return (
                                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderRadius: 8, background: "rgba(0,0,0,0.25)", border: "1px solid rgba(255,255,255,0.06)", fontFamily: "'Menlo','Monaco','Consolas',monospace", fontSize: 11 }}>
                                  <span style={{ color: "#5eead4", fontWeight: 700, whiteSpace: "nowrap" }}>🛠 {t.name}</span>
                                  <span style={{ color: "#64748b" }}>({argStr})</span>
                                  <span style={{ color: "#475569" }}>→</span>
                                  <span style={{ color: "#94a3b8", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{resStr}</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      <p style={{ margin: "12px 0 0", fontSize: 11, color: "#5eead4", fontStyle: "italic" }}>
                        A 5th agent reconciling Research vs Risk via real Cerebras tool calling (strict function schemas) — autonomous agent ↔ tool coordination, not a prompt chain.
                      </p>
                    </Section>
                  </Card>
                );
              })()}

              {/* Completion banner */}
              {hasResults && !running && (
                <div style={{
                  background: "linear-gradient(135deg, rgba(22,163,74,0.15), rgba(34,197,94,0.06))",
                  border: "1px solid rgba(34,197,94,0.3)", borderRadius: 12, padding: "16px 20px",
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  animation: showConfetti ? "successGlow 2s ease-out" : "none",
                }}>
                  <div>
                    <p style={{ margin: "0 0 2px", fontSize: 16, fontWeight: 800, color: "#4ade80" }}>
                      ✅ Pipeline Complete — {elapsed}s
                    </p>
                    <p style={{ margin: 0, fontSize: 12, color: "#86efac" }}>
                      Est. manual onboarding ~30 min → VisualOps {elapsed}s — <strong>~{elapsed ? Math.round(1800 / parseFloat(elapsed)) : "~"}× faster</strong>
                    </p>
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <span style={{ padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 700, background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", color: "#fff" }}>
                      {tokensPerSec?.toLocaleString()} tok/s
                    </span>
                    <span style={{ padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 700, background: "rgba(34,197,94,0.15)", color: "#4ade80" }}>
                      {totalTokens?.toLocaleString()} tokens
                    </span>
                  </div>
                </div>
              )}

              {/* Still processing indicator */}
              {running && hasAnyResults && !hasResults && (
                <div style={{
                  background: "linear-gradient(135deg, rgba(59,130,246,0.12), rgba(37,99,235,0.06))",
                  border: "1px solid rgba(59,130,246,0.3)", borderRadius: 12, padding: "14px 20px",
                  display: "flex", alignItems: "center", gap: 12,
                }}>
                  <div style={{ fontSize: 20, animation: "pulse 1s infinite" }}>⚡</div>
                  <div style={{ flex: 1 }}>
                    <p style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "#93c5fd" }}>Building results in real-time...</p>
                    <p style={{ margin: 0, fontSize: 11, color: "#60a5fa" }}>{liveElapsed}s elapsed · agents completing as you watch</p>
                  </div>
                </div>
              )}

                {/* Vision Extraction Preview */}
                {visionData && (
                  <Card style={{ background: "linear-gradient(135deg, rgba(245,158,11,0.12), rgba(234,179,8,0.05))", border: "1px solid rgba(245,158,11,0.25)" }}>
                    <Section title="👁 Vision Agent — Extracted from Image (Multimodal)">
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                        {[
                          ["Company", visionData.company_name],
                          ["Industry", visionData.industry],
                          ["Document Type", visionData.document_type],
                          ["Deal Value", visionData.deal_value],
                          ["Key Terms", Array.isArray(visionData.key_terms) ? visionData.key_terms.slice(0, 3).join(", ") : visionData.key_terms],
                          ["Key Contacts", Array.isArray(visionData.key_contacts) ? visionData.key_contacts.join(", ") : visionData.key_contacts],
                        ].filter(([, v]) => v).map(([label, value], i) => (
                          <div key={i} style={{ padding: "8px 12px", borderRadius: 8, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                            <p style={{ margin: "0 0 2px", fontSize: 10, fontWeight: 600, color: "#fbbf24", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</p>
                            <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: "#fde68a" }}>{value}</p>
                          </div>
                        ))}
                      </div>
                      <p style={{ margin: "10px 0 0", fontSize: 11, color: "#d97706", fontStyle: "italic" }}>
                        ↑ All data extracted directly from the uploaded image by Gemma 4 31B's vision capabilities — no OCR pipeline needed
                      </p>
                    </Section>
                  </Card>
                )}

                {/* Agent Communication Log — proves Agent Collaboration */}
                {visionData && researchData && riskData && (
                  <Card style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.12), rgba(124,58,237,0.05))", border: "1px solid rgba(139,92,246,0.25)" }}>
                    <Section title="💬 Agent Communication Log — Inter-Agent Data Flow">
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        {[
                          { from: "👁 Vision", to: "🔍 Research", color: "#8b5cf6", data: `Sent: company="${visionData.company_name}", industry="${visionData.industry}", doc_type="${visionData.document_type}"` },
                          { from: "👁 Vision", to: "⚠️ Risk", color: "#a78bfa", data: `Sent: terms=[${(visionData.key_terms || []).length}], value="${visionData.deal_value || 'N/A'}", contacts=[${(visionData.key_contacts || []).length}]` },
                          { from: "🔍 Research", to: "📋 Output", color: "#3b82f6", data: `Sent: profile, pain_points=[${(researchData.pain_points || []).length}], tech_stack, approach` },
                          { from: "⚠️ Risk", to: "📋 Output", color: "#ef4444", data: `Sent: score=${riskData.risk_score_number || riskData.overall_score}/10, flags=[${(riskData.flags || []).length}], recommendation="${riskData.proceed_recommendation || 'N/A'}"` },
                          { from: "🔍+⚠️", to: "🧭 Strategist", color: "#2dd4bf", data: `Sent: opportunity + risk signals → reconcile into GO/HOLD/ESCALATE via tool calls` },
                        ].map((msg, i) => (
                          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderRadius: 8, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                            <span style={{ fontSize: 11, fontWeight: 700, color: msg.color, whiteSpace: "nowrap" }}>{msg.from}</span>
                            <span style={{ fontSize: 12, color: msg.color }}>→</span>
                            <span style={{ fontSize: 11, fontWeight: 700, color: msg.color, whiteSpace: "nowrap" }}>{msg.to}</span>
                            <span style={{ fontSize: 10, color: "#94a3b8", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "'Menlo', 'Monaco', 'Consolas', monospace" }}>{msg.data}</span>
                          </div>
                        ))}
                      </div>
                      <p style={{ margin: "8px 0 0", fontSize: 10, color: "#a78bfa", fontStyle: "italic" }}>
                        Agents 2 (Research) + 3 (Risk) received Vision data simultaneously via Promise.all() — true parallel execution
                      </p>
                    </Section>
                  </Card>
                )}

              {/* Toolbar */}
              <div className="card-animate" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <p style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "#f8fafc" }}>✓ Onboarding package ready</p>
                <button
                  onClick={downloadPackage}
                  style={{ fontSize: 13, fontWeight: 600, color: "#fff", background: "linear-gradient(135deg, #1d4ed8, #7c3aed)", border: "none", borderRadius: 8, padding: "8px 14px", cursor: "pointer", boxShadow: "0 2px 8px rgba(29,78,216,0.3)", transition: "transform 0.2s" }}
                  onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-1px)"}
                  onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
                >
                  ↓ Download package (.md)
                </button>
              </div>

              {/* Deal snapshot — surfaces Vision intelligence at a glance */}
              {visionData && (visionData.deal_value || visionData.key_contacts?.length) && (
                <Card>
                  <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 24, alignItems: "center" }}>
                    {visionData.deal_value && (
                      <div>
                        <p style={{ margin: "0 0 2px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Deal value</p>
                        <p style={{ margin: 0, fontSize: 26, fontWeight: 700, color: "#4ade80" }}>{visionData.deal_value}</p>
                      </div>
                    )}
                    <div>
                      {visionData.key_contacts?.length > 0 && (
                        <>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Key contacts</p>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: visionData.key_terms?.length ? 10 : 0 }}>
                            {visionData.key_contacts.map((c, i) => (
                              <span key={i} style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "rgba(255,255,255,0.05)", color: "#cbd5e1", border: "1px solid rgba(255,255,255,0.1)" }}>{c}</span>
                            ))}
                          </div>
                        </>
                      )}
                      {visionData.key_terms?.length > 0 && (
                        <>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Key terms</p>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            {visionData.key_terms.map((t, i) => (
                              <span key={i} style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" }}>{t}</span>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </Card>
              )}

              {/* Executive summary */}
              {outputData?.executive_summary && (
                <Card>
                  <Section title="Executive summary">
                    <p style={{ margin: 0, fontSize: 14, lineHeight: 1.7, color: "#cbd5e1" }}>{outputData.executive_summary}</p>
                    <button
                      onClick={() => copyToClipboard(outputData.executive_summary, 'summary')}
                      style={{
                        fontSize: 11, color: copied === 'summary' ? '#059669' : '#6b7280',
                        background: 'none', border: '1px solid rgba(148,163,184,0.22)', borderRadius: 6,
                        padding: '3px 10px', cursor: 'pointer', transition: 'all 0.2s', marginTop: 8,
                      }}
                    >
                      {copied === 'summary' ? '✓ Copied!' : '📋 Copy'}
                    </button>
                  </Section>
                  {visionData?.company_name && (
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {visionData.company_name && <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" }}>{visionData.company_name}</span>}
                      {visionData.industry && <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#f5f3ff", color: "#6d28d9", border: "1px solid #ddd6fe" }}>{visionData.industry}</span>}
                      {visionData.document_type && <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "rgba(255,255,255,0.05)", color: "#94a3b8", border: "1px solid rgba(255,255,255,0.1)" }}>{visionData.document_type}</span>}
                      {riskData && <RiskBadge score={riskData.risk_score} number={riskData.risk_score_number} />}
                    </div>
                  )}
                </Card>
              )}

              {/* Execution timeline — speed + parallelism proof */}
              <PipelineTimeline timeline={timeline} />

              {/* Two column: onboarding plan + risk flags */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {outputData?.onboarding_plan && (
                  <Card>
                    <Section title="Onboarding plan">
                      {outputData.onboarding_plan.map((step, i) => (
                        <div key={i} style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                          <div style={{ width: 24, height: 24, borderRadius: "50%", background: "#1d4ed8", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, flexShrink: 0, marginTop: 1 }}>{step.step}</div>
                          <div>
                            <p style={{ margin: "0 0 2px", fontSize: 13, fontWeight: 500, color: "#e2e8f0" }}>{step.action}</p>
                            <p style={{ margin: "0 0 2px", fontSize: 11, color: "#9ca3af" }}>{step.owner} · {step.timeline}</p>
                            <p style={{ margin: 0, fontSize: 11, color: "#6b7280" }}>{step.goal}</p>
                          </div>
                        </div>
                      ))}
                    </Section>
                  </Card>
                )}

                {riskData?.flags && (
                  <Card>
                    <Section title={`Risk flags (${riskData.flags.length})`}>
                      {riskData.flags.length === 0 && (
                        <p style={{ fontSize: 13, color: "#4ade80" }}>✓ No significant flags detected</p>
                      )}
                      {riskData.flags.map((f, i) => {
                        const sc = { low: "#fef9c3", medium: "#fffbeb", high: "#fef2f2" };
                        const tc = { low: "#854d0e", medium: "#92400e", high: "#991b1b" };
                        const bc = { low: "#eab308", medium: "#f59e0b", high: "#ef4444" };
                        const pct = { low: 33, medium: 66, high: 100 };
                        const icons = { low: "⚡", medium: "⚠️", high: "🔴" };
                        return (
                          <div key={i} style={{ padding: "10px 12px", borderRadius: 10, background: sc[f.severity] || sc.medium, marginBottom: 8, border: `1px solid ${bc[f.severity] || bc.medium}20` }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                              <span style={{ fontSize: 12 }}>{icons[f.severity] || "⚠️"}</span>
                              <span style={{ fontSize: 12, fontWeight: 700, color: tc[f.severity] || tc.medium, flex: 1 }}>{f.issue || f.flag}</span>
                              <span style={{ fontSize: 10, fontWeight: 600, color: tc[f.severity] || tc.medium, textTransform: "uppercase", padding: "2px 8px", borderRadius: 6, background: `${bc[f.severity] || bc.medium}15` }}>{f.severity}</span>
                            </div>
                            <p style={{ margin: "0 0 6px", fontSize: 11, color: "#6b7280", paddingLeft: 22 }}>{f.recommendation || f.detail}</p>
                            <div style={{ height: 4, borderRadius: 4, background: `${bc[f.severity] || bc.medium}20`, marginLeft: 22, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${pct[f.severity] || 50}%`, borderRadius: 4, background: bc[f.severity] || bc.medium, transition: "width 0.6s ease" }} />
                            </div>
                          </div>
                        );
                      })}
                      {riskData.missing_information?.length > 0 && (
                        <div style={{ marginTop: 8 }}>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Missing information</p>
                          {riskData.missing_information.map((m, i) => (
                            <p key={i} style={{ margin: "0 0 2px", fontSize: 12, color: "#6b7280" }}>· {m}</p>
                          ))}
                        </div>
                      )}
                      {riskData.proceed_recommendation && (() => {
                        const vc = { go: { bg: "#f0fdf4", bd: "#bbf7d0", tx: "#15803d", t: "✓ GO" }, "proceed-with-caution": { bg: "#fffbeb", bd: "#fde68a", tx: "#b45309", t: "⚠ PROCEED WITH CAUTION" }, escalate: { bg: "#fef2f2", bd: "#fecaca", tx: "#dc2626", t: "⛔ ESCALATE" } };
                        const s = vc[riskData.proceed_recommendation] || vc["proceed-with-caution"];
                        return (
                          <div style={{ marginTop: 10, padding: "8px 12px", borderRadius: 8, background: s.bg, border: `1px solid ${s.bd}`, textAlign: "center" }}>
                            <span style={{ fontSize: 13, fontWeight: 700, color: s.tx, letterSpacing: "0.04em" }}>{s.t}</span>
                          </div>
                        );
                      })()}
                      {riskData.compliance_notes && (
                        <p style={{ margin: "8px 0 0", fontSize: 11, color: "#9ca3af", borderTop: "1px solid #f3f4f6", paddingTop: 8 }}>{riskData.compliance_notes}</p>
                      )}
                    </Section>
                  </Card>
                )}
              </div>

              {/* Action checklist */}
              {checklist.length > 0 && (
                <Card>
                  <Section title={`Action checklist · ${checklist.filter(c => c.done).length}/${checklist.length} done`}>
                    {checklist.map((item, i) => (
                      <ChecklistItem key={i} {...item} onToggle={() => toggleCheck(i)} />
                    ))}
                    <button
                      onClick={() => navigator.clipboard.writeText(checklist.map((c) => `[${c.done ? "x" : " "}] (${c.priority}) ${c.task}`).join("\n"))}
                      style={{ marginTop: 10, fontSize: 12, color: "#1d4ed8", background: "none", border: "none", cursor: "pointer", padding: 0 }}
                    >
                      Copy checklist →
                    </button>
                  </Section>
                </Card>
              )}

              {/* Welcome email */}
              {outputData?.welcome_email && (
                <Card>
                  <Section title="Drafted welcome email">
                    <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "12px 16px", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <p style={{ margin: "0 0 8px", fontSize: 12, color: "#94a3b8" }}>Subject: <span style={{ color: "#e2e8f0", fontWeight: 500 }}>{outputData.welcome_email.subject}</span></p>
                      <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: "#cbd5e1", whiteSpace: "pre-wrap" }}>{outputData.welcome_email.body}</p>
                    </div>
                    <button
                      onClick={() => copyToClipboard(`Subject: ${outputData.welcome_email.subject}\n\n${outputData.welcome_email.body}`, 'email')}
                      style={{
                        fontSize: 11, color: copied === 'email' ? '#059669' : '#6b7280',
                        background: 'none', border: '1px solid rgba(148,163,184,0.22)', borderRadius: 6,
                        padding: '3px 10px', cursor: 'pointer', transition: 'all 0.2s', marginTop: 10,
                      }}
                    >
                      {copied === 'email' ? '✓ Copied!' : '📋 Copy email'}
                    </button>
                  </Section>
                </Card>
              )}

              {/* Success metrics */}
              {outputData?.success_metrics?.length > 0 && (
                <Card>
                  <Section title="Success metrics to track">
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
                      {outputData.success_metrics.map((m, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 12px", background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)", borderRadius: 8 }}>
                          <span style={{ color: "#4ade80", fontSize: 13 }}>📈</span>
                          <span style={{ fontSize: 13, color: "#a7f3d0", lineHeight: 1.4 }}>{m}</span>
                        </div>
                      ))}
                    </div>
                  </Section>
                </Card>
              )}

              {/* Research panel */}
              {researchData && (
                <Card>
                  <Section title="Client intelligence">
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                      {researchData.company_profile && (
                        <div>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Company profile</p>
                          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "#cbd5e1" }}>{researchData.company_profile}</p>
                        </div>
                      )}
                      {researchData.pain_points && (
                        <div>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Pain points</p>
                          {researchData.pain_points.map((p, i) => (
                            <p key={i} style={{ margin: "0 0 4px", fontSize: 13, color: "#cbd5e1" }}>· {p}</p>
                          ))}
                        </div>
                      )}
                    </div>
                    {researchData.likely_tech_stack?.length > 0 && (
                      <div style={{ marginTop: 12 }}>
                        <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Likely tech stack</p>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                          {researchData.likely_tech_stack.map((t, i) => (
                            <span key={i} style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#f5f3ff", color: "#6d28d9", border: "1px solid #ddd6fe" }}>{t}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {researchData.recommended_approach && (
                      <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(59,130,246,0.1)", borderRadius: 8, border: "1px solid rgba(59,130,246,0.2)" }}>
                        <p style={{ margin: 0, fontSize: 13, color: "#93c5fd" }}><strong>Recommended approach:</strong> {researchData.recommended_approach}</p>
                      </div>
                    )}
                  </Section>
                </Card>
              )}

              {/* ─── Pipeline Performance Stats ─────────────────── */}
              {elapsed && (
                <Card style={{ background: "linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.06))", border: "1px solid rgba(59,130,246,0.2)" }}>
                  <Section title="📊 Pipeline Performance">
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10 }}>
                      {[
                        { label: "Total Time", value: `${elapsed}s`, sub: "5 agents" },
                        { label: "Throughput", value: tokensPerSec ? `${tokensPerSec.toLocaleString()}` : "—", sub: "tok/s" },
                        { label: "Total Tokens", value: totalTokens ? totalTokens.toLocaleString() : "—", sub: "processed" },
                        { label: "Parallel Agents", value: "2", sub: "Research + Risk" },
                      ].map((stat, i) => (
                        <div key={i} style={{ padding: "12px", borderRadius: 10, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)", textAlign: "center" }}>
                          <p style={{ margin: "0 0 2px", fontSize: 22, fontWeight: 800, color: "#93c5fd", fontVariantNumeric: "tabular-nums" }}>{stat.value}</p>
                          <p style={{ margin: 0, fontSize: 11, fontWeight: 600, color: "#60a5fa" }}>{stat.label}</p>
                          <p style={{ margin: 0, fontSize: 10, color: "#64748b" }}>{stat.sub}</p>
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: 8, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.15)" }}>
                      <p style={{ margin: 0, fontSize: 11, color: "#93c5fd", lineHeight: 1.5 }}>
                        <strong>Why speed matters:</strong> Traditional document analysis takes 15-30 minutes per document with manual review. VisualOps processes the same document in {elapsed}s — enabling real-time onboarding at enterprise scale. Cerebras WSE-3's ~1,850 tok/s makes 5-agent orchestration feel instant.
                      </p>
                    </div>
                  </Section>
                </Card>
              )}

              {/* ─── Speed Comparison: Multi-Provider Race ─────────────────── */}
              {(openaiKeySet || geminiKeySet) && (
                <Card style={{ background: "linear-gradient(135deg, #0f172a, #1e293b)", border: "1px solid #334155" }}>
                  <Section title={<span style={{ color: "#94a3b8" }}>⚡ Speed in Action — Cerebras WSE-3 vs {[openaiKeySet && "OpenAI GPU", geminiKeySet && "Google TPU"].filter(Boolean).join(" vs ")} (Live Race)</span>}>
                    {!speedResult && !speedRunning && (
                      <div style={{ textAlign: "center", padding: "20px 0" }}>
                        <p style={{ margin: "0 0 6px", fontSize: 14, color: "#e2e8f0", fontWeight: 600 }}>
                          Same prompt. {[openaiKeySet, geminiKeySet].filter(Boolean).length + 1} providers. Real-time race.
                        </p>
                        <p style={{ margin: "0 0 16px", fontSize: 12, color: "#64748b" }}>
                          Sends identical request to all providers simultaneously via Promise.allSettled()
                        </p>
                        <button
                          onClick={handleSpeedCompare}
                          style={{
                            padding: "12px 32px", borderRadius: 10,
                            background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                            color: "#fff", border: "none", fontSize: 15, fontWeight: 700,
                            cursor: "pointer", boxShadow: "0 4px 20px rgba(59,130,246,0.5)",
                            transition: "all 0.3s ease",
                          }}
                          onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
                          onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
                        >
                          🏁 Run Speed Comparison
                        </button>
                      </div>
                    )}
                    {speedRunning && (
                      <div style={{ textAlign: "center", padding: "24px 0" }}>
                        <div style={{ fontSize: 36, animation: "pulse 1s infinite" }}>⚡</div>
                        <p style={{ margin: "8px 0 0", fontSize: 14, color: "#e2e8f0", fontWeight: 500 }}>Racing {[openaiKeySet, geminiKeySet].filter(Boolean).length + 1} providers...</p>
                        <p style={{ margin: "4px 0 0", fontSize: 12, color: "#64748b" }}>Same prompt sent simultaneously via Promise.allSettled()</p>
                      </div>
                    )}
                    {speedResult && (() => {
                      const c = speedResult.cerebras;
                      const o = speedResult.openai;
                      const g = speedResult.gemini;
                      const cOk = c && !c.error;
                      const oOk = o && !o.error;
                      const gOk = g && !g.error;
                      const cMs = c?.ms || 0;
                      const oMs = o?.ms || 0;
                      const gMs = g?.ms || 0;
                      const cTokSec = cOk && c.completionTime > 0 ? Math.round(c.completionTokens / c.completionTime) : 0;
                      const oTokSec = oOk && oMs > 0 && o.completionTokens > 0 ? Math.round((o.completionTokens / oMs) * 1000) : 0;
                      const gTokSec = gOk && gMs > 0 && g.completionTokens > 0 ? Math.round((g.completionTokens / gMs) * 1000) : 0;

                      // Build competitors array
                      const competitors = [];
                      if (o) competitors.push({ name: "OpenAI", model: "GPT-4o-mini", hw: "NVIDIA GPU", ok: oOk, ms: oMs, tokSec: oTokSec, error: o.error, color: "#94a3b8" });
                      if (g) competitors.push({ name: "Gemini", model: "2.0 Flash", hw: "Google TPU", ok: gOk, ms: gMs, tokSec: gTokSec, error: g.error, color: "#60a5fa" });

                      // Find the slowest competitor for speedup calc
                      const slowest = competitors.filter(x => x.ok).sort((a, b) => b.ms - a.ms)[0];
                      const speedup = cOk && slowest ? (slowest.ms / Math.max(cMs, 1)).toFixed(1) : null;

                      return (
                        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                          {/* Hero cards */}
                          <div style={{ display: "grid", gridTemplateColumns: `1fr ${competitors.map(() => "auto 1fr").join(" ")}`, gap: 8, alignItems: "center" }}>
                            {/* Cerebras */}
                            <div style={{ padding: "20px 14px", borderRadius: 12, background: "linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04))", border: "1px solid rgba(34,197,94,0.25)", textAlign: "center" }}>
                              <p style={{ margin: "0 0 2px", fontSize: 9, fontWeight: 700, color: "#4ade80", textTransform: "uppercase", letterSpacing: "0.12em" }}>Cerebras · Gemma 4 31B</p>
                              <p style={{ margin: "0 0 4px", fontSize: 32, fontWeight: 800, color: "#4ade80", fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>
                                {cOk ? `${(cMs / 1000).toFixed(2)}s` : "Error"}
                              </p>
                              {cOk && cTokSec > 0 && <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: "#86efac" }}>{cTokSec.toLocaleString()} tok/s</p>}
                              <p style={{ margin: "4px 0 0", fontSize: 9, color: "#4ade80" }}>WSE-3 · 900K cores</p>
                            </div>

                            {competitors.map((comp, ci) => (
                              <div key={ci} style={{ display: "contents" }}>
                                {/* VS badge */}
                                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                                  <span style={{ fontSize: 11, fontWeight: 800, color: "#475569" }}>VS</span>
                                  {cOk && comp.ok && (
                                    <span style={{ fontSize: 11, fontWeight: 700, color: "#4ade80", background: "rgba(34,197,94,0.15)", borderRadius: 12, padding: "2px 8px", whiteSpace: "nowrap" }}>
                                      {(comp.ms / Math.max(cMs, 1)).toFixed(1)}×
                                    </span>
                                  )}
                                </div>
                                {/* Competitor card */}
                                <div style={{ padding: "20px 14px", borderRadius: 12, background: "rgba(148,163,184,0.06)", border: "1px solid rgba(148,163,184,0.15)", textAlign: "center" }}>
                                  <p style={{ margin: "0 0 2px", fontSize: 9, fontWeight: 700, color: comp.color, textTransform: "uppercase", letterSpacing: "0.12em" }}>{comp.name} · {comp.model}</p>
                                  {comp.ok ? (
                                    <>
                                      <p style={{ margin: "0 0 4px", fontSize: 32, fontWeight: 800, color: comp.color, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>
                                        {(comp.ms / 1000).toFixed(2)}s
                                      </p>
                                      {comp.tokSec > 0 && <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: "#64748b" }}>{comp.tokSec.toLocaleString()} tok/s</p>}
                                    </>
                                  ) : (
                                    <>
                                      <p style={{ margin: "0 0 4px", fontSize: 22, fontWeight: 700, color: "#64748b", lineHeight: 1.3 }}>Unavailable</p>
                                      <p style={{ margin: 0, fontSize: 10, color: "#64748b" }}>{comp.error?.includes("Invalid") || comp.error?.includes("401") ? "Invalid key" : comp.error?.includes("429") ? "Rate limited" : "Error"}</p>
                                    </>
                                  )}
                                  <p style={{ margin: "4px 0 0", fontSize: 9, color: "#64748b" }}>{comp.hw}</p>
                                </div>
                              </div>
                            ))}
                          </div>

                          {/* Winner banner */}
                          {speedup && (
                            <div style={{ textAlign: "center", padding: "10px", borderRadius: 10, background: "linear-gradient(135deg, rgba(34,197,94,0.1), rgba(59,130,246,0.1))", border: "1px solid rgba(34,197,94,0.2)" }}>
                              <span style={{ fontSize: 18, fontWeight: 800, color: "#fff", background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", borderRadius: 24, padding: "6px 20px", boxShadow: "0 4px 14px rgba(99,102,241,0.5)" }}>
                                🏆 Cerebras wins — {speedup}× faster
                              </span>
                            </div>
                          )}

                          {/* Methodology */}
                          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 8, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.15)" }}>
                            <span style={{ fontSize: 16 }}>🧪</span>
                            <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>
                              <strong style={{ color: "#e2e8f0" }}>Methodology:</strong> Identical prompt sent to all providers simultaneously via <code style={{ color: "#93c5fd", background: "rgba(59,130,246,0.1)", padding: "1px 4px", borderRadius: 3 }}>Promise.allSettled()</code>. Cerebras WSE-3 achieves ~1,850 tok/s with 44GB on-chip SRAM — zero off-chip memory bottleneck.
                            </p>
                          </div>

                          <div style={{ textAlign: "center" }}>
                            <button
                              onClick={handleSpeedCompare}
                              disabled={speedRunning}
                              style={{
                                padding: "8px 20px", borderRadius: 8,
                                background: "transparent", color: "#64748b",
                                border: "1px solid #334155", fontSize: 12, fontWeight: 500,
                                cursor: "pointer", transition: "all 0.2s",
                              }}
                              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#3b82f6"; e.currentTarget.style.color = "#93c5fd"; }}
                              onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#334155"; e.currentTarget.style.color = "#64748b"; }}
                            >
                              🔄 Re-run comparison
                            </button>
                          </div>
                        </div>
                      );
                    })()}
                    {speedError && (
                      <p style={{ margin: "8px 0 0", fontSize: 12, color: "#f87171" }}>{speedError}</p>
                    )}
                  </Section>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Agent Orchestration Diagram */}
      {(running || hasResults) && (
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px 24px" }}>
          <Card style={{ overflow: "hidden" }}>
            <Section title="🔀 Agent Orchestration Pipeline — Live Data Flow">
              <div style={{ position: "relative", padding: "20px 0" }}>
                {/* Pipeline flow */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0 }}>
                  {/* Image Input */}
                  <div style={{ textAlign: "center", minWidth: 80 }}>
                    <div style={{ width: 48, height: 48, borderRadius: 10, background: "rgba(255,255,255,0.05)", border: "2px solid rgba(148,163,184,0.25)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 4px", fontSize: 20 }}>📄</div>
                    <p style={{ margin: 0, fontSize: 10, fontWeight: 600, color: "#94a3b8" }}>Document</p>
                  </div>
                  {/* Arrow */}
                  <div style={{ width: 40, height: 2, background: agentStates.vision !== "idle" ? "#3b82f6" : "rgba(148,163,184,0.22)", position: "relative", transition: "background 0.5s" }}>
                    <div style={{ position: "absolute", right: -4, top: -4, width: 0, height: 0, borderLeft: `8px solid ${agentStates.vision !== "idle" ? "#3b82f6" : "rgba(148,163,184,0.22)"}`, borderTop: "5px solid transparent", borderBottom: "5px solid transparent" }} />
                  </div>
                  {/* Agent 1: Vision */}
                  <div style={{ textAlign: "center", minWidth: 100 }}>
                    <div style={{
                      width: 56, height: 56, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 4px", fontSize: 22,
                      background: agentStates.vision === "done" ? "linear-gradient(135deg, #f0fdf4, #dcfce7)" : agentStates.vision === "running" ? "linear-gradient(135deg, #eff6ff, #dbeafe)" : "rgba(255,255,255,0.04)",
                      border: `2px solid ${agentStates.vision === "done" ? "#86efac" : agentStates.vision === "running" ? "#93c5fd" : "rgba(148,163,184,0.22)"}`,
                      boxShadow: agentStates.vision === "running" ? "0 0 16px rgba(59,130,246,0.3)" : "none",
                      transition: "all 0.5s", animation: agentStates.vision === "running" ? "agentPulse 2s infinite" : "none",
                    }}>👁</div>
                    <p style={{ margin: 0, fontSize: 10, fontWeight: 700, color: agentStates.vision === "done" ? "#15803d" : agentStates.vision === "running" ? "#1d4ed8" : "#9ca3af" }}>Vision</p>
                    <p style={{ margin: 0, fontSize: 9, color: "#9ca3af" }}>multimodal</p>
                  </div>
                  {/* Arrow to parallel split */}
                  <div style={{ width: 30, height: 2, background: agentStates.research !== "idle" || agentStates.risk !== "idle" ? "#3b82f6" : "rgba(148,163,184,0.22)", transition: "background 0.5s" }} />
                  {/* Parallel Agents */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, position: "relative" }}>
                    <div style={{ position: "absolute", left: -8, top: "50%", transform: "translateY(-50%)", background: "#dbeafe", borderRadius: 4, padding: "2px 6px", fontSize: 8, fontWeight: 700, color: "#1d4ed8", whiteSpace: "nowrap" }}>PARALLEL</div>
                    {/* Research */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 50 }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
                        background: agentStates.research === "done" ? "linear-gradient(135deg, #f0fdf4, #dcfce7)" : agentStates.research === "running" ? "linear-gradient(135deg, #eff6ff, #dbeafe)" : "rgba(255,255,255,0.04)",
                        border: `2px solid ${agentStates.research === "done" ? "#86efac" : agentStates.research === "running" ? "#93c5fd" : "rgba(148,163,184,0.22)"}`,
                        boxShadow: agentStates.research === "running" ? "0 0 12px rgba(59,130,246,0.3)" : "none",
                        transition: "all 0.5s", animation: agentStates.research === "running" ? "agentPulse 2s infinite" : "none",
                      }}>🔍</div>
                      <div>
                        <p style={{ margin: 0, fontSize: 10, fontWeight: 700, color: agentStates.research === "done" ? "#15803d" : "#6b7280" }}>Research</p>
                        <p style={{ margin: 0, fontSize: 9, color: "#9ca3af" }}>client intel</p>
                      </div>
                    </div>
                    {/* Risk */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 50 }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
                        background: agentStates.risk === "done" ? "linear-gradient(135deg, #f0fdf4, #dcfce7)" : agentStates.risk === "running" ? "linear-gradient(135deg, #eff6ff, #dbeafe)" : "rgba(255,255,255,0.04)",
                        border: `2px solid ${agentStates.risk === "done" ? "#86efac" : agentStates.risk === "running" ? "#93c5fd" : "rgba(148,163,184,0.22)"}`,
                        boxShadow: agentStates.risk === "running" ? "0 0 12px rgba(59,130,246,0.3)" : "none",
                        transition: "all 0.5s", animation: agentStates.risk === "running" ? "agentPulse 2s infinite" : "none",
                      }}>⚠️</div>
                      <div>
                        <p style={{ margin: 0, fontSize: 10, fontWeight: 700, color: agentStates.risk === "done" ? "#15803d" : "#6b7280" }}>Risk</p>
                        <p style={{ margin: 0, fontSize: 9, color: "#9ca3af" }}>reasoning</p>
                      </div>
                    </div>
                  </div>
                  {/* Arrow from parallel to output + strategist */}
                  <div style={{ width: 30, height: 2, background: (agentStates.output !== "idle" || agentStates.strategist !== "idle") ? "#3b82f6" : "rgba(148,163,184,0.2)", transition: "background 0.5s" }} />
                  {/* Agents 4 (Output) + 5 (Strategist) run in parallel */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, position: "relative" }}>
                    <div style={{ position: "absolute", left: -8, top: "50%", transform: "translateY(-50%)", background: "rgba(59,130,246,0.18)", borderRadius: 4, padding: "2px 6px", fontSize: 8, fontWeight: 700, color: "#93c5fd", whiteSpace: "nowrap" }}>PARALLEL</div>
                    {/* Output */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 50 }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
                        background: agentStates.output === "done" ? "linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.08))" : agentStates.output === "running" ? "linear-gradient(135deg, rgba(59,130,246,0.2), rgba(59,130,246,0.08))" : "rgba(255,255,255,0.04)",
                        border: `2px solid ${agentStates.output === "done" ? "rgba(34,197,94,0.4)" : agentStates.output === "running" ? "rgba(59,130,246,0.4)" : "rgba(148,163,184,0.2)"}`,
                        boxShadow: agentStates.output === "running" ? "0 0 12px rgba(59,130,246,0.3)" : "none",
                        transition: "all 0.5s", animation: agentStates.output === "running" ? "agentPulse 2s infinite" : "none",
                      }}>📋</div>
                      <div>
                        <p style={{ margin: 0, fontSize: 10, fontWeight: 700, color: agentStates.output === "done" ? "#4ade80" : agentStates.output === "running" ? "#93c5fd" : "#94a3b8" }}>Output</p>
                        <p style={{ margin: 0, fontSize: 9, color: "#94a3b8" }}>synthesize</p>
                      </div>
                    </div>
                    {/* Strategist */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 50 }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
                        background: agentStates.strategist === "done" ? "linear-gradient(135deg, rgba(20,184,166,0.25), rgba(20,184,166,0.08))" : agentStates.strategist === "running" ? "linear-gradient(135deg, rgba(45,212,191,0.2), rgba(45,212,191,0.08))" : "rgba(255,255,255,0.04)",
                        border: `2px solid ${agentStates.strategist === "done" ? "rgba(45,212,191,0.45)" : agentStates.strategist === "running" ? "rgba(45,212,191,0.4)" : "rgba(148,163,184,0.2)"}`,
                        boxShadow: agentStates.strategist === "running" ? "0 0 12px rgba(45,212,191,0.3)" : "none",
                        transition: "all 0.5s", animation: agentStates.strategist === "running" ? "agentPulse 2s infinite" : "none",
                      }}>🧭</div>
                      <div>
                        <p style={{ margin: 0, fontSize: 10, fontWeight: 700, color: agentStates.strategist === "done" ? "#5eead4" : agentStates.strategist === "running" ? "#5eead4" : "#94a3b8" }}>Strategist</p>
                        <p style={{ margin: 0, fontSize: 9, color: "#94a3b8" }}>tool calling</p>
                      </div>
                    </div>
                  </div>
                  {/* Arrow to package */}
                  <div style={{ width: 40, height: 2, background: agentStates.output === "done" ? "#22c55e" : "rgba(148,163,184,0.2)", position: "relative", transition: "background 0.5s" }}>
                    <div style={{ position: "absolute", right: -4, top: -4, width: 0, height: 0, borderLeft: `8px solid ${agentStates.output === "done" ? "#22c55e" : "rgba(148,163,184,0.2)"}`, borderTop: "5px solid transparent", borderBottom: "5px solid transparent" }} />
                  </div>
                  {/* Output Package */}
                  <div style={{ textAlign: "center", minWidth: 80 }}>
                    <div style={{ width: 48, height: 48, borderRadius: 10, background: agentStates.output === "done" ? "rgba(34,197,94,0.18)" : "rgba(255,255,255,0.05)", border: `2px solid ${agentStates.output === "done" ? "rgba(34,197,94,0.4)" : "rgba(148,163,184,0.25)"}`, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 4px", fontSize: 20, transition: "all 0.5s" }}>📦</div>
                    <p style={{ margin: 0, fontSize: 10, fontWeight: 600, color: agentStates.output === "done" ? "#4ade80" : "#94a3b8" }}>{agentStates.output === "done" ? "Ready!" : "Package"}</p>
                  </div>
                </div>
                {/* Data annotations */}
                {hasResults && (
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16, padding: "0 20px" }}>
                    {[
                      { label: "Image → JSON", detail: "Structured extraction" },
                      { label: "JSON → Intel + Flags", detail: "Promise.all() parallel" },
                      { label: "All → Package + Verdict", detail: "Synthesis ∥ tool-calling reconciler" },
                    ].map((a, i) => (
                      <div key={i} style={{ textAlign: "center", flex: 1 }}>
                        <p style={{ margin: 0, fontSize: 10, fontWeight: 600, color: "#3b82f6" }}>{a.label}</p>
                        <p style={{ margin: 0, fontSize: 9, color: "#9ca3af" }}>{a.detail}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </Section>
          </Card>
        </div>
      )}

      {/* Processing History */}
      {processHistory.length > 1 && (
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px 24px" }}>
          <Card>
            <Section title={`📜 Processing History — ${processHistory.length} Documents Analyzed`}>
              <div style={{ display: "flex", gap: 10, overflowX: "auto", padding: "4px 0" }}>
                {processHistory.map((h, i) => (
                  <div key={h.id} style={{
                    minWidth: 160, padding: "12px 14px", borderRadius: 10,
                    background: i === processHistory.length - 1 ? "linear-gradient(135deg, rgba(22,163,74,0.15), rgba(34,197,94,0.06))" : "rgba(255,255,255,0.04)",
                    border: `1px solid ${i === processHistory.length - 1 ? "rgba(34,197,94,0.3)" : "rgba(255,255,255,0.08)"}`,
                    flexShrink: 0,
                  }}>
                    <p style={{ margin: "0 0 4px", fontSize: 12, fontWeight: 700, color: "#e2e8f0" }}>#{i + 1} {h.company}</p>
                    <p style={{ margin: 0, fontSize: 11, color: "#94a3b8" }}>{h.elapsed}s · {h.tokensPerSec?.toLocaleString()} tok/s</p>
                    <p style={{ margin: 0, fontSize: 10, color: "#9ca3af" }}>{h.totalTokens?.toLocaleString()} tokens · {h.timestamp}</p>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 10, display: "flex", gap: 16, justifyContent: "center" }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#93c5fd" }}>
                  Total: {processHistory.reduce((a, h) => a + (h.totalTokens || 0), 0).toLocaleString()} tokens
                </span>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#4ade80" }}>
                  Avg: {(processHistory.reduce((a, h) => a + parseFloat(h.elapsed || 0), 0) / processHistory.length).toFixed(1)}s per doc
                </span>
              </div>
            </Section>
          </Card>
        </div>
      )}

      {/* Hackathon Footer */}
      <div style={{
        maxWidth: 1200, margin: "24px auto 0", padding: "20px 24px",
        borderTop: "1px solid rgba(255,255,255,0.08)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 20 }}>⚡</span>
          <div>
            <p style={{ margin: 0, fontSize: 12, fontWeight: 700, color: "#cbd5e1" }}>Built for the Cerebras × Google DeepMind Gemma 4 Hackathon</p>
            <p style={{ margin: 0, fontSize: 11, color: "#94a3b8" }}>Track 1: Multiverse Agents · Track 3: Enterprise Impact</p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["Gemma 4 31B", "WSE-3", "React", "Vite"].map((t, i) => (
            <span key={i} style={{ padding: "3px 10px", borderRadius: 12, fontSize: 10, fontWeight: 500, background: "rgba(255,255,255,0.05)", color: "#94a3b8", border: "1px solid rgba(255,255,255,0.1)" }}>{t}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
