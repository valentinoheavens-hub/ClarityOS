// ─────────────────────────────────────────────────────────────────────────────
// BGC Coach — provider. SERVER ONLY.
//
// One brain, one voice across every channel (web, WhatsApp, evidence scoring).
// Anthropic Claude is the PRIMARY coaching intelligence (the BGC frameworks are
// designed around it); Groq/Llama is a fast fallback so coaching never goes dark
// if Claude is unavailable. Models are overridable via env.
//
//   CLAUDE_COACH_MODEL  default: claude-opus-4-8
//   GROQ_COACH_MODEL    default: llama-3.3-70b-versatile  (legacy BGC_COACH_MODEL
//                       is still honoured for backwards compatibility)
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from "@anthropic-ai/sdk";
import Groq from "groq-sdk";

export const CLAUDE_MODEL = process.env.CLAUDE_COACH_MODEL ?? "claude-opus-4-8";
export const GROQ_MODEL =
  process.env.GROQ_COACH_MODEL ?? process.env.BGC_COACH_MODEL ?? "llama-3.3-70b-versatile";

/** The primary model, surfaced for logging/session metadata. */
export const COACH_MODEL = CLAUDE_MODEL;

let anthropicClient: Anthropic | null = null;
function anthropic(): Anthropic {
  if (!anthropicClient) anthropicClient = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  return anthropicClient;
}

let groqClient: Groq | null = null;
function groq(): Groq {
  if (!groqClient) groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });
  return groqClient;
}

export type CoachMessage = { role: "user" | "assistant"; content: string };

const hasAnthropicKey = () => Boolean(process.env.ANTHROPIC_API_KEY);

// ── Claude (primary) ──────────────────────────────────────────────────────────
// Thinking is intentionally left off: coaching is a real-time chat and the
// frameworks live in the system prompt, so low latency beats deliberation.

async function* streamClaude(
  system: string,
  messages: CoachMessage[],
  maxTokens: number
): AsyncGenerator<string> {
  const stream = anthropic().messages.stream({
    model: CLAUDE_MODEL,
    max_tokens: maxTokens,
    system,
    messages,
  });
  for await (const event of stream) {
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
      yield event.delta.text;
    }
  }
}

async function generateClaude(
  system: string,
  messages: CoachMessage[],
  maxTokens: number
): Promise<string> {
  const res = await anthropic().messages.create({
    model: CLAUDE_MODEL,
    max_tokens: maxTokens,
    system,
    messages,
  });
  return res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}

// ── Groq (fallback) ─────────────────────────────────────────────────────────—

async function* streamGroq(
  system: string,
  messages: CoachMessage[],
  maxTokens: number
): AsyncGenerator<string> {
  const stream = await groq().chat.completions.create({
    model: GROQ_MODEL,
    max_tokens: maxTokens,
    messages: [{ role: "system", content: system }, ...messages],
    stream: true,
  });
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content ?? "";
    if (text) yield text;
  }
}

async function generateGroq(
  system: string,
  messages: CoachMessage[],
  maxTokens: number
): Promise<string> {
  const res = await groq().chat.completions.create({
    model: GROQ_MODEL,
    max_tokens: maxTokens,
    messages: [{ role: "system", content: system }, ...messages],
  });
  return res.choices[0]?.message?.content ?? "";
}

// ── Public API — Claude first, Groq fallback ───────────────────────────────────

/** Stream the coach response as text deltas (used by the web chat endpoint). */
export async function* streamCoach(
  system: string,
  messages: CoachMessage[],
  maxTokens = 1024
): AsyncGenerator<string> {
  if (hasAnthropicKey()) {
    let yielded = false;
    try {
      for await (const text of streamClaude(system, messages, maxTokens)) {
        yielded = true;
        yield text;
      }
      return;
    } catch (err) {
      // If Claude already started streaming, don't restart on Groq (would
      // duplicate the reply) — surface the error to the caller instead.
      if (yielded) throw err;
      console.error("[coach] Claude stream failed before output — falling back to Groq:", err);
    }
  }
  yield* streamGroq(system, messages, maxTokens);
}

/** One-shot coach completion returning the full text (WhatsApp, scoring). */
export async function generateCoach(
  system: string,
  messages: CoachMessage[],
  maxTokens = 600
): Promise<string> {
  if (hasAnthropicKey()) {
    try {
      return await generateClaude(system, messages, maxTokens);
    } catch (err) {
      console.error("[coach] Claude completion failed — falling back to Groq:", err);
    }
  }
  return generateGroq(system, messages, maxTokens);
}
