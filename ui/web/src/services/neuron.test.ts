/**
 * Tests for the NEURON transport.  Run: `npm test` in ui/web/
 *
 * WHY THIS FILE EXISTS, because it is not obvious and it matters.
 *
 * `ui/test_chat_ui.py` guards the old page with 67 assertions, and **59 of them grep
 * chat.html's SOURCE TEXT** — `'event==="reroute"' in SRC`. That worked because the page was
 * one hand-written HTML file. A React build compiles to minified bundles, so every one of those
 * 59 checks breaks the moment the new UI replaces the old one, and the guarantees they encode
 * would be silently lost.
 *
 * Those guarantees are not stylistic. Each was bought with a real incident recorded in
 * sessions.md and PROBLEMS.md. So the behaviours live in THIS module — plain functions over
 * plain data, no components, no browser — and are asserted here instead. That is the actual
 * answer to "the tests all break": not fewer tests, a better place to put them.
 *
 * The three that matter most:
 *   1. A reroute is NOT an error. A node dying mid-answer is recovered token-for-token
 *      (test_node_death.py SIGKILLs one and requires identical output). Reporting it as a
 *      failure teaches people to distrust the one thing the network handles best.
 *   2. A partial answer SURVIVES the error that ended it. The user has already been given, and
 *      billed for, that text; replacing it with an error message deletes what they paid for.
 *   3. `done.reroutes` is REPORTED. The driver records it deliberately — "a recovered answer is
 *      still a degraded one" — and it was once dropped on the way to the page.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { parseFrame, streamChat, errorMessage, StreamHandlers } from './neuron';

/** Feed a canned SSE body through streamChat, as the server would send it. */
function serve(body: string, ok = true) {
  const encoder = new TextEncoder();
  const chunks = body.split('|CHUNK|').map(c => encoder.encode(c));
  let i = 0;
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok,
    body: {
      getReader: () => ({
        read: async () =>
          i < chunks.length ? { done: false, value: chunks[i++] } : { done: true, value: undefined },
      }),
    },
    status: ok ? 200 : 500,
  })));
}

function collect() {
  const seen: Record<string, unknown[]> = {
    meta: [], token: [], reroute: [], done: [], error: [], sources: [],
  };
  const handlers: StreamHandlers = {
    onMeta: m => seen.meta.push(m),
    onToken: t => seen.token.push(t),
    onReroute: (at, n) => seen.reroute.push({ at, n }),
    onDone: d => seen.done.push(d),
    onError: e => seen.error.push(e),
    onSources: (s, used) => seen.sources.push({ s, used }),
  };
  return { seen, handlers };
}

const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

afterEach(() => vi.unstubAllGlobals());

describe('parseFrame', () => {
  it('reads the event name and payload', () => {
    expect(parseFrame('event: token\ndata: {"text":"hi"}')).toEqual(['token', { text: 'hi' }]);
  });

  it('returns null on malformed JSON rather than throwing', () => {
    // A bad frame must not kill the stream: the answer so far is still the user's.
    expect(parseFrame('event: token\ndata: {not json')).toBeNull();
  });

  it('returns null when there is no data line', () => {
    expect(parseFrame('event: ping')).toBeNull();
  });
});

describe('streamChat', () => {
  it('reports meta, tokens and done from a normal answer', async () => {
    serve(
      frame('meta', { request_id: 'r1', nodes: 2, node_ids: ['a', 'b'], cost_nrn: 0.012,
                      conversation_id: 'c1', local: false }) +
      frame('token', { text: 'Hel' }) +
      frame('token', { text: 'lo' }) +
      frame('done', { tokens: 2, latency_ms: 900, tok_per_s: 2.2, cost_nrn: 0.012,
                      reroutes: 0, text: 'Hello', finish_reason: 'stop' })
    );
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);

    expect(seen.meta).toHaveLength(1);
    expect((seen.meta[0] as any).nodes).toBe(2);
    expect((seen.meta[0] as any).conversationId).toBe('c1');
    expect(seen.token).toEqual(['Hel', 'lo']);
    expect((seen.done[0] as any).text).toBe('Hello');
    expect(seen.error).toHaveLength(0);
  });

  it('emits each token as a FRAGMENT, not the accumulated answer', async () => {
    serve(frame('token', { text: 'a' }) + frame('token', { text: 'b' }));
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);
    // 'ab' here would mean the caller double-counts every token when it concatenates.
    expect(seen.token).toEqual(['a', 'b']);
  });

  it('handles a frame split across two network chunks', async () => {
    // TCP does not respect message boundaries. Splitting on every newline instead of on the
    // blank-line separator would emit half a JSON object and lose the token.
    serve('event: token\ndata: {"te|CHUNK|xt":"split"}\n\n');
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);
    expect(seen.token).toEqual(['split']);
  });

  it('treats a reroute as its own event, never as an error', async () => {
    serve(
      frame('token', { text: 'part' }) +
      frame('reroute', { at_token: 4, nodes: 3 }) +
      frame('token', { text: ' more' }) +
      frame('done', { tokens: 2, latency_ms: 1, tok_per_s: 1, cost_nrn: 0,
                      reroutes: 1, text: 'part more', finish_reason: 'stop' })
    );
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);

    expect(seen.reroute).toHaveLength(1);
    expect(seen.error).toHaveLength(0);              // the load-bearing assertion
    expect(seen.token).toEqual(['part', ' more']);   // and the answer continues
  });

  it('reports done.reroutes so a recovered answer is not silent', async () => {
    serve(frame('done', { tokens: 1, latency_ms: 1, tok_per_s: 1, cost_nrn: 0,
                          reroutes: 2, text: 'x', finish_reason: 'stop' }));
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);
    expect((seen.done[0] as any).reroutes).toBe(2);
  });

  it('delivers an error AFTER tokens without discarding them', async () => {
    serve(
      frame('token', { text: 'half an answer' }) +
      frame('error', { detail: 'chain lost', code: 'no_chain' })
    );
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);
    // Both must arrive: the caller keeps the text and appends the failure.
    expect(seen.token).toEqual(['half an answer']);
    expect((seen.error[0] as any).code).toBe('no_chain');
  });

  it('surfaces a transport failure as an error event, not an exception', async () => {
    serve('', false);
    const { seen, handlers } = collect();
    await expect(streamChat({ prompt: 'hi' }, handlers)).resolves.toBeUndefined();
    expect((seen.error[0] as any).code).toBe('transport');
  });

  it('skips an unparseable frame and keeps the rest of the stream', async () => {
    serve('event: token\ndata: {broken\n\n' + frame('token', { text: 'survived' }));
    const { seen, handlers } = collect();
    await streamChat({ prompt: 'hi' }, handlers);
    expect(seen.token).toEqual(['survived']);
  });

  it('sends the fields /chat actually accepts', async () => {
    serve(frame('done', { tokens: 0, latency_ms: 0, tok_per_s: 0, cost_nrn: 0,
                          reroutes: 0, text: '', finish_reason: 'stop' }));
    const { handlers } = collect();
    await streamChat({ prompt: 'q', maxTokens: 64, useRag: true, conversationId: 'c9' }, handlers);
    const body = JSON.parse((globalThis.fetch as any).mock.calls[0][1].body);
    // ui/app.py's ChatBody: prompt, max_tokens, use_rag, conversation_id. Anything else is
    // ignored by the server, so sending it would be a control that does nothing.
    expect(body).toEqual({ prompt: 'q', max_tokens: 64, use_rag: true, conversation_id: 'c9' });
  });
});

describe('errorMessage', () => {
  it('never blames a node going offline', () => {
    // That event is RECOVERED. Naming it as the cause of a failure is untrue and corrosive.
    const all = ['insufficient_funds', 'content_policy_violation', 'no_local_engine', null]
      .map(code => errorMessage({ detail: 'something failed', code }, false).toLowerCase());
    for (const m of all) {
      expect(m).not.toContain('went offline');
      expect(m).not.toContain('dropped out');
    }
  });

  it('names the specific cause it knows about', () => {
    expect(errorMessage({ detail: '', code: 'insufficient_funds' }, false)).toContain('NRN');
    expect(errorMessage({ detail: '', code: 'content_policy_violation' }, false))
      .toContain('acceptable-use');
  });

  it('vouches for a partial answer when there is one', () => {
    const withText = errorMessage({ detail: 'x', code: null }, true);
    const without = errorMessage({ detail: 'x', code: null }, false);
    expect(withText).toContain('correct as far as it goes');
    expect(without).not.toContain('correct as far as it goes');
  });
});
