/**
 * NEURON transport. Replaces the Gemini/Ollama/OpenAI-compatible provider layer entirely.
 *
 * NEURON has exactly one backend, so there is no provider to choose, no base URL to probe and
 * no model picker: the network's tier controller decides which model is served, bounded by its
 * weakest member. `modelId` is therefore something to DISPLAY, never something to select.
 *
 * The event shapes below are `ui/app.py`'s, read from the endpoint rather than assumed:
 *
 *   meta     { request_id, nodes, node_ids, cost_nrn, conversation_id, local }
 *   sources  { sources, used }                      -- RAG, when web search was used
 *   token    { text }                               -- a FRAGMENT, not the accumulated answer
 *   reroute  { at_token, nodes }                    -- recovery, NOT an error (see below)
 *   done     { tokens, latency_ms, tok_per_s, cost_nrn, reroutes, text, finish_reason }
 *   error    { detail, code }
 *
 * Three behaviours here are load-bearing, and each exists because of a real incident recorded
 * in this repo. They are implemented in this module, not in a component, so they can be tested
 * without a browser:
 *
 *  1. `reroute` IS NOT AN ERROR. A node dying mid-answer is recovered token-for-token --
 *     `test_node_death.py` SIGKILLs a node mid-generation and requires output identical to an
 *     uninterrupted run. Presenting it as a failure teaches users to distrust the one thing the
 *     network handles best. It is a neutral status: the stream paused, it is coming back.
 *  2. A PARTIAL ANSWER SURVIVES THE ERROR THAT ENDED IT. An `error` after tokens have streamed
 *     must not discard them -- the user has already been given, and charged for, that text.
 *  3. `done.reroutes` IS REPORTED. The driver records it deliberately ("a recovered answer is
 *     still a degraded one"), and it was once dropped on the way to the page, so a request that
 *     survived two node deaths looked identical to one that sailed through.
 */

export interface NeuronMeta {
  requestId: string;
  nodes: number;
  nodeIds: string[];
  costNrn: number;
  conversationId: string | null;
  /** True when THIS machine answered rather than the node network. Never present it as the
   *  network's work: the header used to claim credit for a local answer. */
  local: boolean;
}

export interface NeuronDone {
  tokens: number;
  latencyMs: number;
  tokPerS: number;
  costNrn: number | null;
  /** >0 means a node dropped and the answer was rebuilt. Visible, never silent. */
  reroutes: number;
  text: string;
  finishReason: string | null;
}

export interface NeuronError {
  detail: string;
  /** e.g. "insufficient_funds", "content_policy_violation". Drives the specific message. */
  code: string | null;
}

export interface StreamHandlers {
  onMeta?: (meta: NeuronMeta) => void;
  onSources?: (sources: unknown[], used: boolean) => void;
  /** Called with each FRAGMENT. Accumulation is the caller's job. */
  onToken?: (text: string) => void;
  onReroute?: (atToken: number, nodes: number) => void;
  onDone?: (done: NeuronDone) => void;
  onError?: (err: NeuronError) => void;
}

export interface StreamRequest {
  prompt: string;
  maxTokens?: number;
  useRag?: boolean;
  conversationId?: string | null;
  signal?: AbortSignal;
}

/** Parse one SSE frame ("event: x\ndata: {...}") into [event, data]. */
export function parseFrame(frame: string): [string, unknown] | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return [event, JSON.parse(dataLines.join('\n'))];
  } catch {
    // A malformed frame must not kill the stream: the answer so far is still the user's.
    return null;
  }
}

/**
 * Stream one answer. Resolves when the stream ends, whether by `done` or by `error`.
 *
 * Never throws for a server-reported failure -- those arrive through `onError`, so the caller
 * treats "the model refused" and "the chain could not be built" the same way it treats any
 * other end-of-stream, and keeps whatever text already arrived. It rejects only when the
 * request could not be made at all.
 */
/**
 * Reply length used when the caller does not specify one.
 *
 * This used to be 128 here while the UI's own default was 8192 — two numbers for one setting,
 * and the small one won for any thread whose `maxTokens` was undefined (a conversation restored
 * from localStorage that predates the field). The user saw "8192" in the composer and got answers
 * cut off at 128 tokens with "stopped at the limit, not because the answer was finished".
 *
 * A transport must not quietly substitute a different value from the one the product promises.
 * Kept here, exported, so there is one source of truth rather than a literal in each caller.
 */
export const DEFAULT_MAX_TOKENS = 8192;

export async function streamChat(req: StreamRequest, handlers: StreamHandlers): Promise<void> {
  const resp = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: req.prompt,
      max_tokens: req.maxTokens ?? DEFAULT_MAX_TOKENS,
      use_rag: req.useRag ?? false,
      conversation_id: req.conversationId ?? null,
    }),
    signal: req.signal,
  });

  if (!resp.ok || !resp.body) {
    handlers.onError?.({
      detail: `The chat service answered ${resp.status}.`,
      code: 'transport',
    });
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. Anything after the last separator is a partial
    // frame and stays in the buffer -- splitting on every newline would emit half a JSON object.
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';

    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (!parsed) continue;
      const [event, raw] = parsed;
      const d = raw as Record<string, unknown>;

      switch (event) {
        case 'meta':
          handlers.onMeta?.({
            requestId: String(d.request_id ?? ''),
            nodes: Number(d.nodes ?? 0),
            nodeIds: (d.node_ids as string[]) ?? [],
            costNrn: Number(d.cost_nrn ?? 0),
            conversationId: (d.conversation_id as string) ?? null,
            local: Boolean(d.local),
          });
          break;
        case 'sources':
          handlers.onSources?.((d.sources as unknown[]) ?? [], Boolean(d.used));
          break;
        case 'token':
          handlers.onToken?.(String(d.text ?? ''));
          break;
        case 'reroute':
          handlers.onReroute?.(Number(d.at_token ?? 0), Number(d.nodes ?? 0));
          break;
        case 'done':
          handlers.onDone?.({
            tokens: Number(d.tokens ?? 0),
            latencyMs: Number(d.latency_ms ?? 0),
            tokPerS: Number(d.tok_per_s ?? 0),
            costNrn: d.cost_nrn == null ? null : Number(d.cost_nrn),
            reroutes: Number(d.reroutes ?? 0),
            text: String(d.text ?? ''),
            finishReason: (d.finish_reason as string) ?? null,
          });
          break;
        case 'error':
          handlers.onError?.({
            detail: String(d.detail ?? 'The request failed.'),
            code: (d.code as string) ?? null,
          });
          break;
        default:
          break;
      }
    }
  }
}

/**
 * The message shown when a stream ends in an error.
 *
 * Deliberately never blames a node going offline. That event is RECOVERED, not fatal, and
 * saying otherwise is both untrue and corrosive -- it teaches people to distrust the one thing
 * the network handles best. What actually reaches this path is recovery being impossible.
 */
export function errorMessage(err: NeuronError, hadText: boolean): string {
  const tail = hadText
    ? ' The answer above is correct as far as it goes.'
    : '';
  switch (err.code) {
    // [P29]. "Contributing a machine earns more" was a FALSE PROMISE until 2026-08-17: node
    // earnings landed in a ledger row belonging to the machine, with no route into the wallet
    // the person signs in with, so the one action the message offered did not solve the problem
    // it pointed at. [P39] closed that — but only for a node whose owner has been RECORDED, and
    // recording it is a separate deliberate act. Naming both steps is the difference between an
    // instruction that works and one that sounds like it should.
    // The message a person gets at the end of the free grant, so it says WHY rather than just
    // no. NRN buys other volunteers' electricity and CPU time; that is the thing that ran out,
    // and there is no recurring faucet because refilling it would commit THEIR hardware to
    // unlimited free use, which is not the project's to give away. What is owed instead is a
    // straight answer about what is still available — and contributing needs LESS machine than
    // running the model locally does, which is the part nobody would guess.
    case 'insufficient_funds':
      return 'Your free NRN is spent. It paid other people to run answers on their machines, '
        + 'and there is no automatic top-up. Contributing this computer as a node earns more — '
        + 'it needs less memory than running the model here would — and claiming it links those '
        + 'earnings to this wallet.' + tail;
    case 'content_policy_violation':
      return 'That response was stopped by the acceptable-use policy.' + tail;
    case 'no_local_engine':
      return 'This machine has no local engine, and the network could not be reached.' + tail;
    default:
      return (err.detail || 'The request could not be completed.') + tail;
  }
}
