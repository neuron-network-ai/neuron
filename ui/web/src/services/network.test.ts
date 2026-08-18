/**
 * Do not send into a chain that cannot answer.  Run: `npm test` in ui/web/
 *
 * chat.html has refused this since Session 55; React did not. `canSend` was
 * `(text || attachments) && !isGenerating` with no network term at all, so on a degraded
 * network the user typed, sent, waited, and collected an error the page could have predicted
 * before they pressed a key. That was the one gap in the /next swap that could actually cost
 * somebody something ([P48] item 3).
 *
 * The refusals it must NOT make are the more interesting half, and each is a real incident
 * shape:
 *   * a failed status POLL is not a dead network. Every field falls back to a pessimistic
 *     default on a fetch error, so blocking on those defaults turns a monitoring blip into an
 *     outage nobody caused — a self-inflicted one, in chat.html's words.
 *   * a machine that can serve the model itself does not care that the network is short of
 *     nodes. It answers locally by design (engine/local_gguf.py), and blocking it would take
 *     chat away from the one person who never needed the chain.
 *   * the FIRST render, before anything has been polled, must not block. Otherwise every page
 *     load starts unable to send.
 */
import { describe, it, expect } from 'vitest';
import { blockReason, degradedNotice, NetworkState } from './wallet';

const base: NetworkState = {
  reachable: true, onlineNodes: 3, localCapable: false, healthy: true,
  servingModel: 'Qwen/Qwen2.5-1.5B-Instruct', statusKnown: true,
  layersCovered: 28, totalLayers: 28, uncoveredLayers: [],
};

describe('blockReason', () => {
  it('a healthy network sends', () => {
    expect(blockReason(base)).toBeNull();
  });

  it('an unroutable chain is refused BEFORE the user sends', () => {
    const r = blockReason({ ...base, healthy: false });
    expect(r).toBeTruthy();
    expect(r).toContain('short of nodes');
  });

  it('a failed status poll does NOT block — that is a self-inflicted outage', () => {
    // The shape of the real fetch-error fallback: everything pessimistic, statusKnown false.
    expect(blockReason({
      ...base, statusKnown: false, reachable: false, healthy: false,
      localCapable: false, onlineNodes: 0,
    })).toBeNull();
  });

  it('the initial state, before anything is polled, does not block the first message', () => {
    expect(blockReason({
      reachable: false, onlineNodes: 0, localCapable: false, healthy: false,
      servingModel: null, statusKnown: false, layersCovered: null,
      totalLayers: null, uncoveredLayers: [],
    })).toBeNull();
  });

  it('a locally-capable machine is never blocked by a short network', () => {
    expect(blockReason({ ...base, healthy: false, localCapable: true })).toBeNull();
  });
});

describe('degradedNotice', () => {
  it('names the missing layers rather than counting them', () => {
    const n = degradedNotice({
      ...base, healthy: false, layersCovered: 19, uncoveredLayers: [[10, 12], [27, 27]],
    });
    expect(n).toContain('19/28');
    expect(n).toContain('missing 10–12, 27');
  });

  it('takes the total from the coordinator, not a hardcoded 28', () => {
    const n = degradedNotice({
      ...base, healthy: false, layersCovered: 12, totalLayers: 36, uncoveredLayers: [],
    });
    expect(n).toContain('12/36');
  });

  it('a single missing layer reads as one number, not a range', () => {
    const n = degradedNotice({ ...base, healthy: false, uncoveredLayers: [[7, 7]] });
    expect(n).toContain('missing 7');
    expect(n).not.toContain('7–7');
  });

  it('says nothing when the network is fine, unpolled, or served locally', () => {
    expect(degradedNotice(base)).toBeNull();
    expect(degradedNotice({ ...base, healthy: false, statusKnown: false })).toBeNull();
    expect(degradedNotice({ ...base, healthy: false, localCapable: true })).toBeNull();
  });
});
