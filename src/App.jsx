import { useState, useRef, useCallback } from "react";

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
        content: `You are a B2B risk assessment agent. Analyze this document data for risks, red flags, and compliance concerns.

Document data:
${JSON.stringify(visionData, null, 2)}

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

// ─── Pipeline orchestrator ────────────────────────────────────────────────────
async function runPipeline(apiKey, base64Image, mimeType, onUpdate) {
  const t0 = Date.now();
  const timeline = [];
  const mark = (stage, start, end) => timeline.push({ stage, start: start - t0, end: end - t0 });

  onUpdate({ stage: "vision", status: "running" });
  const vStart = Date.now();
  const vision = await agentVision(apiKey, base64Image, mimeType);
  mark("vision", vStart, Date.now());
  onUpdate({ stage: "vision", status: "done", data: vision.data, meta: { ms: vision.ms, tokens: vision.tokens } });

  // Agents 2 and 3 run in parallel
  onUpdate({ stage: "research", status: "running" });
  onUpdate({ stage: "risk", status: "running" });
  const pStart = Date.now();

  const [research, risk] = await Promise.all([
    agentResearch(apiKey, vision.data).then((r) => {
      mark("research", pStart, Date.now());
      onUpdate({ stage: "research", status: "done", data: r.data, meta: { ms: r.ms, tokens: r.tokens } });
      return r;
    }),
    agentRisk(apiKey, vision.data).then((r) => {
      mark("risk", pStart, Date.now());
      onUpdate({ stage: "risk", status: "done", data: r.data, meta: { ms: r.ms, tokens: r.tokens } });
      return r;
    }),
  ]);

  onUpdate({ stage: "output", status: "running" });
  const oStart = Date.now();
  const output = await agentOutput(apiKey, vision.data, research.data, risk.data);
  mark("output", oStart, Date.now());
  onUpdate({ stage: "output", status: "done", data: output.data, meta: { ms: output.ms, tokens: output.tokens } });

  const elapsedMs = Date.now() - t0;
  const elapsed = (elapsedMs / 1000).toFixed(1);
  const all = [vision, research, risk, output];
  const totalTokens = all.reduce((s, a) => s + (a.tokens || 0), 0);
  // Real output tokens/sec from the API's time_info (matches Cerebras' headline
  // metric); fall back to a wall-clock estimate if time_info isn't present.
  const genTokens = all.reduce((s, a) => s + (a.completionTokens || 0), 0);
  const genTime = all.reduce((s, a) => s + (a.completionTime || 0), 0);
  const tokensPerSec = genTime > 0
    ? Math.round(genTokens / genTime)
    : (elapsedMs > 0 ? Math.round((totalTokens / elapsedMs) * 1000) : 0);

  return {
    visionData: vision.data,
    researchData: research.data,
    riskData: risk.data,
    outputData: output.data,
    elapsed,
    totalTokens,
    tokensPerSec,
    timeline,
  };
}

// ─── UI Components ────────────────────────────────────────────────────────────

function AgentBadge({ label, icon, status, meta }) {
  const colors = {
    idle: { bg: "#f5f5f5", border: "#e0e0e0", text: "#999", dot: "#ccc" },
    running: { bg: "#eff6ff", border: "#bfdbfe", text: "#1d4ed8", dot: "#3b82f6" },
    done: { bg: "#f0fdf4", border: "#bbf7d0", text: "#15803d", dot: "#22c55e" },
  };
  const c = colors[status] || colors.idle;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "8px 14px", borderRadius: 8,
      background: c.bg, border: `1px solid ${c.border}`,
      transition: "all 0.3s ease",
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: "50%",
        background: c.dot,
        boxShadow: status === "running" ? `0 0 0 3px ${c.border}` : "none",
        animation: status === "running" ? "pulse 1.5s infinite" : "none",
      }} />
      <span style={{ fontSize: 13, fontWeight: 500, color: c.text }}>{icon} {label}</span>
      {status === "running" && (
        <span style={{ fontSize: 11, color: c.text, marginLeft: "auto" }}>running...</span>
      )}
      {status === "done" && (
        <span style={{ fontSize: 11, color: c.text, marginLeft: "auto", fontVariantNumeric: "tabular-nums" }}>
          {meta?.ms != null ? `${(meta.ms / 1000).toFixed(1)}s` : "✓"}
        </span>
      )}
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
                <span style={{ width: 92, fontSize: 12, color: "#374151", flexShrink: 0 }}>{r.label}</span>
                <div style={{ flex: 1, position: "relative", height: 22, background: "#f3f4f6", borderRadius: 6 }}>
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
      <p style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "#9ca3af", marginBottom: 10 }}>{title}</p>
      {children}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "16px 20px", ...style,
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
      padding: "8px 0", borderBottom: "1px solid #f3f4f6",
    }}>
      <button
        onClick={onToggle}
        style={{
          width: 18, height: 18, borderRadius: 4, flexShrink: 0, marginTop: 1,
          border: done ? "none" : "2px solid #d1d5db",
          background: done ? "#22c55e" : "transparent",
          cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        {done && <span style={{ color: "#fff", fontSize: 11 }}>✓</span>}
      </button>
      <span style={{ flex: 1, fontSize: 13, color: done ? "#9ca3af" : "#1f2937", textDecoration: done ? "line-through" : "none" }}>{task}</span>
      <span style={{ fontSize: 11, color: pc[priority], fontWeight: 500, textTransform: "uppercase" }}>{priority}</span>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function VisualOps() {
  const [apiKey, setApiKey] = useState("");
  const [apiKeySet, setApiKeySet] = useState(false);
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

  const [agentStates, setAgentStates] = useState({
    vision: "idle", research: "idle", risk: "idle", output: "idle",
  });
  const [timings, setTimings] = useState({
    vision: null, research: null, risk: null, output: null,
  });
  const [results, setResults] = useState({
    vision: null, research: null, risk: null, output: null,
  });

  const dropRef = useRef();

  const updateAgent = useCallback(({ stage, status, data, meta }) => {
    setAgentStates((prev) => ({ ...prev, [stage]: status }));
    if (data) setResults((prev) => ({ ...prev, [stage]: data }));
    if (meta) setTimings((prev) => ({ ...prev, [stage]: meta }));
    if (stage === "output" && status === "done" && data?.action_checklist) {
      setChecklist(data.action_checklist);
    }
  }, []);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults({ vision: null, research: null, risk: null, output: null });
    setAgentStates({ vision: "idle", research: "idle", risk: "idle", output: "idle" });
    setTimings({ vision: null, research: null, risk: null, output: null });
    setElapsed(null);
    setTokensPerSec(null);
    setTotalTokens(null);
    setTimeline(null);
    setError(null);
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
    try {
      const { base64, mimeType } = await prepareImage(file);
      const res = await runPipeline(apiKey, base64, mimeType, updateAgent);
      setElapsed(res.elapsed);
      setTokensPerSec(res.tokensPerSec);
      setTotalTokens(res.totalTokens);
      setTimeline(res.timeline);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const toggleCheck = (i) => {
    setChecklist((prev) => prev.map((item, idx) => idx === i ? { ...item, done: !item.done } : item));
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

  const downloadPackage = () => {
    const v = results.vision || {};
    const r = results.research || {};
    const rk = results.risk || {};
    const o = results.output || {};
    const list = (arr) => (Array.isArray(arr) ? arr : []);
    const md = [
      `# Onboarding Package — ${v.company_name || "Client"}`,
      ``,
      `_Generated by VisualOps · Gemma 4 31B on Cerebras · ${elapsed}s · ${tokensPerSec?.toLocaleString()} tok/s_`,
      ``,
      `## Snapshot`,
      `- **Company:** ${v.company_name || "—"}`,
      `- **Industry:** ${v.industry || "—"}`,
      `- **Document type:** ${v.document_type || "—"}`,
      `- **Deal value:** ${v.deal_value || "—"}`,
      `- **Risk:** ${(rk.risk_score || "—").toUpperCase()} (${rk.risk_score_number ?? "—"}/10) · ${rk.proceed_recommendation || "—"}`,
      `- **Key contacts:** ${list(v.key_contacts).join("; ") || "—"}`,
      ``,
      `## Executive summary`,
      o.executive_summary || "—",
      ``,
      `## Onboarding plan`,
      ...list(o.onboarding_plan).map((s) => `${s.step}. **${s.action}** — ${s.owner} · ${s.timeline} (${s.goal})`),
      ``,
      `## Risk flags`,
      ...(list(rk.flags).length ? list(rk.flags).map((f) => `- [${(f.severity || "").toUpperCase()}] ${f.issue} → ${f.recommendation}`) : ["- None detected"]),
      rk.compliance_notes ? `\n_Compliance: ${rk.compliance_notes}_` : "",
      ``,
      `## Action checklist`,
      ...list(checklist).map((c) => `- [${c.done ? "x" : " "}] (${c.priority}) ${c.task}`),
      ``,
      `## Success metrics`,
      ...list(o.success_metrics).map((m) => `- ${m}`),
      ``,
      `## Welcome email`,
      o.welcome_email ? `**Subject:** ${o.welcome_email.subject}\n\n${o.welcome_email.body}` : "—",
      ``,
    ].join("\n");
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `onboarding-${(v.company_name || "client").toLowerCase().replace(/[^a-z0-9]+/g, "-")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const hasResults = results.output !== null;
  const riskData = results.risk;
  const outputData = results.output;
  const visionData = results.vision;
  const researchData = results.research;

  return (
    <div style={{ minHeight: "100vh", background: "#f9fafb", fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        * { box-sizing: border-box; }
        textarea { resize: vertical; }
        input:focus, textarea:focus, button:focus { outline: 2px solid #3b82f6; outline-offset: 2px; }
      `}</style>

      {/* Header */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e5e7eb", padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: "#1d4ed8", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ color: "#fff", fontSize: 16 }}>⚡</span>
          </div>
          <div>
            <p style={{ margin: 0, fontWeight: 700, fontSize: 16, color: "#111827" }}>VisualOps</p>
            <p style={{ margin: 0, fontSize: 11, color: "#9ca3af" }}>4-agent onboarding intelligence · Gemma 4 on Cerebras</p>
          </div>
        </div>
        {elapsed && (
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: "6px 12px" }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: "#15803d" }}>⚡ {elapsed}s</span>
              <span style={{ fontSize: 11, color: "#15803d" }}>Cerebras</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, padding: "6px 12px" }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: "#9ca3af", textDecoration: "line-through" }}>~75s</span>
              <span style={{ fontSize: 11, color: "#9ca3af" }}>typical GPU</span>
            </div>
            <span style={{ fontSize: 13, fontWeight: 700, color: "#1d4ed8", background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 8, padding: "6px 12px" }}>
              {Math.round(75 / parseFloat(elapsed))}× faster
            </span>
            {tokensPerSec ? (
              <div style={{ display: "flex", alignItems: "center", gap: 6, background: "#1d4ed8", borderRadius: 8, padding: "6px 12px" }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: "#fff", fontVariantNumeric: "tabular-nums" }}>{tokensPerSec.toLocaleString()}</span>
                <span style={{ fontSize: 11, color: "#bfdbfe" }}>tok/s</span>
              </div>
            ) : null}
          </div>
        )}
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px", display: "grid", gridTemplateColumns: "340px 1fr", gap: 24 }}>

        {/* Left panel */}
        <div>
          {/* API Key */}
          {!apiKeySet ? (
            <Card style={{ marginBottom: 16 }}>
              <p style={{ margin: "0 0 8px", fontSize: 13, fontWeight: 500, color: "#374151" }}>Cerebras API Key</p>
              <input
                type="password"
                placeholder="csk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                style={{ width: "100%", padding: "8px 12px", borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, marginBottom: 8 }}
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
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px", background: "#f0fdf4", borderRadius: 8, border: "1px solid #bbf7d0" }}>
                <span style={{ fontSize: 12, color: "#15803d" }}>✓ API key set</span>
                <div style={{ display: "flex", gap: 12 }}>
                  <button onClick={checkModels} style={{ fontSize: 11, color: "#1d4ed8", background: "none", border: "none", cursor: "pointer" }}>check models</button>
                  <button onClick={() => setApiKeySet(false)} style={{ fontSize: 11, color: "#6b7280", background: "none", border: "none", cursor: "pointer" }}>change</button>
                </div>
              </div>
              {models && (
                <div style={{ marginTop: 8, padding: "8px 12px", background: "#fff", borderRadius: 8, border: "1px solid #e5e7eb" }}>
                  <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af" }}>Models your key can use:</p>
                  {models.map((m) => (
                    <p key={m} style={{ margin: "0 0 2px", fontSize: 12, color: m === "gemma-4-31b" ? "#15803d" : "#374151", fontWeight: m === "gemma-4-31b" ? 700 : 400 }}>
                      {m === "gemma-4-31b" ? "✓ " : "· "}{m}
                    </p>
                  ))}
                </div>
              )}
              {modelsErr && (
                <div style={{ marginTop: 8, padding: "8px 12px", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, fontSize: 12, color: "#dc2626" }}>{modelsErr}</div>
              )}
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
                border: "2px dashed #d1d5db", borderRadius: 10,
                padding: preview ? 0 : "32px 16px",
                textAlign: "center", cursor: "pointer",
                background: "#fafafa", overflow: "hidden",
                transition: "border-color 0.2s",
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = "#3b82f6"}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = "#d1d5db"}
            >
              {preview ? (
                <img src={preview} alt="uploaded document" style={{ width: "100%", borderRadius: 8, display: "block" }} />
              ) : (
                <>
                  <p style={{ fontSize: 28, margin: "0 0 8px" }}>📄</p>
                  <p style={{ margin: "0 0 4px", fontSize: 14, fontWeight: 500, color: "#374151" }}>Drop a business document</p>
                  <p style={{ margin: 0, fontSize: 12, color: "#9ca3af" }}>Contract, proposal, invoice, org chart, logo</p>
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
                <div style={{ width: 1, height: 12, background: "#e5e7eb", marginLeft: 20 }} />
                <span style={{ fontSize: 10, color: "#d1d5db" }}>parallel</span>
              </div>
              <AgentBadge label="Research — client intel" icon="🔍" status={agentStates.research} meta={timings.research} />
              <AgentBadge label="Risk — flag detector" icon="⚠️" status={agentStates.risk} meta={timings.risk} />
              <div style={{ width: 1, height: 12, background: "#e5e7eb", marginLeft: 20 }} />
              <AgentBadge label="Output — action generator" icon="📋" status={agentStates.output} meta={timings.output} />
            </div>
          </Card>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!file || !apiKeySet || running}
            style={{
              width: "100%", padding: "14px",
              background: (!file || !apiKeySet || running) ? "#e5e7eb" : "#1d4ed8",
              color: (!file || !apiKeySet || running) ? "#9ca3af" : "#fff",
              border: "none", borderRadius: 10,
              fontSize: 15, fontWeight: 600, cursor: (!file || !apiKeySet || running) ? "not-allowed" : "pointer",
              transition: "background 0.2s",
            }}
          >
            {running ? "⚡ Agents running..." : "Run 4-agent pipeline"}
          </button>

          {error && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, fontSize: 13, color: "#dc2626" }}>
              {error}
            </div>
          )}
        </div>

        {/* Right panel — results */}
        <div>
          {!hasResults && !running && (
            <div style={{ height: 400, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 8 }}>
              <p style={{ fontSize: 40, margin: 0 }}>🤖</p>
              <p style={{ margin: 0, fontSize: 15, color: "#9ca3af" }}>Upload a document and run the pipeline to see results</p>
              <p style={{ margin: 0, fontSize: 12, color: "#d1d5db" }}>Contract · Proposal · Invoice · Org chart · Logo</p>
            </div>
          )}

          {running && !hasResults && (
            <div style={{ height: 400, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12 }}>
              <div style={{ fontSize: 40, animation: "pulse 1.5s infinite" }}>⚡</div>
              <p style={{ margin: 0, fontSize: 15, color: "#374151", fontWeight: 500 }}>Agents working at Cerebras speed...</p>
              <p style={{ margin: 0, fontSize: 12, color: "#9ca3af" }}>1,500 tokens/sec · Gemma 4 31B</p>
            </div>
          )}

          {hasResults && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Toolbar */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <p style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "#111827" }}>✓ Onboarding package ready</p>
                <button
                  onClick={downloadPackage}
                  style={{ fontSize: 13, fontWeight: 600, color: "#fff", background: "#1d4ed8", border: "none", borderRadius: 8, padding: "8px 14px", cursor: "pointer" }}
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
                        <p style={{ margin: 0, fontSize: 26, fontWeight: 700, color: "#15803d" }}>{visionData.deal_value}</p>
                      </div>
                    )}
                    <div>
                      {visionData.key_contacts?.length > 0 && (
                        <>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Key contacts</p>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: visionData.key_terms?.length ? 10 : 0 }}>
                            {visionData.key_contacts.map((c, i) => (
                              <span key={i} style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#f9fafb", color: "#374151", border: "1px solid #e5e7eb" }}>{c}</span>
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
                    <p style={{ margin: 0, fontSize: 14, lineHeight: 1.7, color: "#374151" }}>{outputData.executive_summary}</p>
                  </Section>
                  {visionData?.company_name && (
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {visionData.company_name && <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" }}>{visionData.company_name}</span>}
                      {visionData.industry && <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#f5f3ff", color: "#6d28d9", border: "1px solid #ddd6fe" }}>{visionData.industry}</span>}
                      {visionData.document_type && <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 20, background: "#f9fafb", color: "#6b7280", border: "1px solid #e5e7eb" }}>{visionData.document_type}</span>}
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
                            <p style={{ margin: "0 0 2px", fontSize: 13, fontWeight: 500, color: "#111827" }}>{step.action}</p>
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
                        <p style={{ fontSize: 13, color: "#15803d" }}>✓ No significant flags detected</p>
                      )}
                      {riskData.flags.map((f, i) => {
                        const sc = { low: "#fef9c3", medium: "#fffbeb", high: "#fef2f2" };
                        const tc = { low: "#854d0e", medium: "#92400e", high: "#991b1b" };
                        return (
                          <div key={i} style={{ padding: "8px 10px", borderRadius: 8, background: sc[f.severity] || sc.medium, marginBottom: 6 }}>
                            <p style={{ margin: "0 0 2px", fontSize: 12, fontWeight: 600, color: tc[f.severity] || tc.medium }}>{f.severity?.toUpperCase()} · {f.issue}</p>
                            <p style={{ margin: 0, fontSize: 11, color: "#6b7280" }}>{f.recommendation}</p>
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
                    <div style={{ background: "#f9fafb", borderRadius: 8, padding: "12px 16px" }}>
                      <p style={{ margin: "0 0 8px", fontSize: 12, color: "#9ca3af" }}>Subject: <span style={{ color: "#374151", fontWeight: 500 }}>{outputData.welcome_email.subject}</span></p>
                      <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: "#374151", whiteSpace: "pre-wrap" }}>{outputData.welcome_email.body}</p>
                    </div>
                    <button
                      onClick={() => navigator.clipboard.writeText(`Subject: ${outputData.welcome_email.subject}\n\n${outputData.welcome_email.body}`)}
                      style={{ marginTop: 10, fontSize: 12, color: "#1d4ed8", background: "none", border: "none", cursor: "pointer", padding: 0 }}
                    >
                      Copy email →
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
                        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 12px", background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8 }}>
                          <span style={{ color: "#15803d", fontSize: 13 }}>📈</span>
                          <span style={{ fontSize: 13, color: "#166534", lineHeight: 1.4 }}>{m}</span>
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
                          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "#374151" }}>{researchData.company_profile}</p>
                        </div>
                      )}
                      {researchData.pain_points && (
                        <div>
                          <p style={{ margin: "0 0 4px", fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>Pain points</p>
                          {researchData.pain_points.map((p, i) => (
                            <p key={i} style={{ margin: "0 0 4px", fontSize: 13, color: "#374151" }}>· {p}</p>
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
                      <div style={{ marginTop: 12, padding: "10px 14px", background: "#eff6ff", borderRadius: 8, border: "1px solid #bfdbfe" }}>
                        <p style={{ margin: 0, fontSize: 13, color: "#1d4ed8" }}><strong>Recommended approach:</strong> {researchData.recommended_approach}</p>
                      </div>
                    )}
                  </Section>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
