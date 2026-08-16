/**
 * Recording who owns this machine's node earnings.  [P39] phase 2, ported to the React app.
 *
 * A node's ledger row is created by `register_node` with no email, no login and no owner — its
 * entire credential is the `node_token` in one config.json on one disk. Lose the disk, lose the
 * money, and `blockchain/migrate_ledger.py` marks such an account `unmapped` and skips it. The
 * Chat UI is the one screen where both halves are present: the agent knows its node_id, and the
 * session knows who is signed in. Recording the owner here is the difference between "your
 * computer earns NRN" being true and being technically true.
 *
 * WHY A SERVICE MODULE AND NOT JUST A COMPONENT. The same reason `neuron.ts` exists: the old
 * page's guarantees were asserted by grepping chat.html's source text, which a compiled bundle
 * makes impossible. The rules below were each bought with a real decision recorded in [P39], so
 * they live in plain functions over plain data and are asserted in `nodeOwner.test.ts`.
 *
 * THE RULES, in the order they would hurt if broken:
 *
 *   1. **Never send a wallet id from the page.** `ui/app.py` injects `owner_wallet_id` from the
 *      SESSION and `NodeBindBody` has no such field, precisely so a page cannot nominate
 *      somebody else as the owner of this machine's earnings. This module must not reintroduce
 *      what the server took away.
 *   2. **A panel, never a gate.** Shown only when this machine serves a node AND has no owner
 *      recorded. A driver-only machine sees nothing, nobody is asked twice, and a volunteer
 *      without a browser wallet must still be able to run a node.
 *   3. **A failure here must not touch the chat.** `/node/owner` failing means render nothing —
 *      a coordinator outage is not a reason the chat stops working.
 *   4. **A declined signature is not an error.** `err.code === 4001` is a person saying no.
 *      It has to say nothing changed and leave the button usable, not die.
 */

/** The minimum of EIP-1193 this needs. Declared here rather than pulling in a wallet SDK. */
export interface Eip1193Provider {
  request(args: { method: string; params?: unknown[] }): Promise<unknown>;
}

declare global {
  interface Window {
    ethereum?: Eip1193Provider;
  }
}

export interface NodeOwnerState {
  /** Does this machine serve a node at all? A driver-only install does not. */
  isNode: boolean;
  /** True only when there is a node AND no owner recorded yet. Gates the whole panel. */
  needsOwner: boolean;
  nodeId: string | null;
}

export const NO_NODE: NodeOwnerState = { isNode: false, needsOwner: false, nodeId: null };

/**
 * Ask whether this machine has unclaimed node earnings.
 *
 * Any failure answers NO_NODE, which renders nothing. That is rule 3: the page must not grow a
 * broken panel because a coordinator was slow, and the alternative — an error box in the
 * sidebar of a working chat — is worse than silence about a prompt nobody was waiting for.
 */
export async function fetchNodeOwner(signal?: AbortSignal): Promise<NodeOwnerState> {
  try {
    const r = await fetch('/node/owner', { signal });
    if (!r.ok) return NO_NODE;
    const d = await r.json();
    return {
      isNode: Boolean(d.is_node),
      needsOwner: Boolean(d.is_node) && Boolean(d.needs_owner),
      nodeId: (d.node_id as string) ?? null,
    };
  } catch {
    return NO_NODE;
  }
}

export type ClaimPhase =
  | 'idle'
  | 'connecting'
  | 'challenging'
  | 'signing'
  | 'binding'
  | 'done'
  | 'declined'
  | 'error';

export interface ClaimResult {
  phase: ClaimPhase;
  /** What to show the person. Always set. */
  message: string;
  /** The bound address, only on 'done'. */
  address?: string;
  /** True while the button should stay usable — every outcome except success. */
  retryable: boolean;
}

/** Progress callback so the caller can narrate without owning the sequence. */
export type ClaimProgress = (phase: ClaimPhase, message: string) => void;

const NO_WALLET =
  'No wallet extension in this browser. Open NEURON in the browser that holds your wallet, ' +
  'or use tools/sign_payout.html.';

/**
 * Is this the user declining, rather than something breaking?
 *
 * 4001 is EIP-1193's `userRejectedRequest`. The message test is a fallback for wallets that
 * throw a plain Error — getting this wrong in the strict direction is the expensive one: it
 * tells somebody who simply changed their mind that the software failed.
 */
export function isUserDeclined(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const e = err as { code?: unknown; message?: unknown };
  if (e.code === 4001) return true;
  return typeof e.message === 'string' && /reject|denied|declined/i.test(e.message);
}

function describe(err: unknown): string {
  if (err && typeof err === 'object' && typeof (err as Error).message === 'string') {
    return (err as Error).message;
  }
  return String(err);
}

/**
 * Connect, sign a challenge, and record the owner. One call, because the three steps are only
 * meaningful together and a half-finished claim has no state worth keeping.
 *
 * `provider` is injected rather than read off `window` so the whole path is testable — the old
 * page's tests stubbed `window.ethereum` and therefore never exercised any of this.
 */
export async function claimNodeEarnings(
  provider: Eip1193Provider | undefined,
  onProgress: ClaimProgress = () => {},
): Promise<ClaimResult> {
  if (!provider) {
    onProgress('error', NO_WALLET);
    return { phase: 'error', message: NO_WALLET, retryable: true };
  }
  try {
    onProgress('connecting', 'Connecting…');
    const accounts = (await provider.request({ method: 'eth_requestAccounts' })) as string[];
    const from = accounts?.[0];
    if (!from) {
      const m = 'Your wallet returned no account. Unlock it and try again.';
      onProgress('error', m);
      return { phase: 'error', message: m, retryable: true };
    }

    onProgress('challenging', `Preparing a challenge for ${from} …`);
    const chRes = await fetch(`/node/payout/challenge?address=${encodeURIComponent(from)}`);
    const ch = await chRes.json();
    if (ch.error) {
      onProgress('error', String(ch.error));
      return { phase: 'error', message: String(ch.error), retryable: true };
    }

    onProgress('signing', 'Approve the signature in your wallet — it moves no funds.');
    const signature = (await provider.request({
      method: 'personal_sign',
      params: [ch.message, from],
    })) as string;

    onProgress('binding', 'Recording…');
    // NO WALLET ID IN THIS BODY, and that is rule 1. `ui/app.py` takes the owner from the
    // session; sending one here would hand the page the ability to name somebody else.
    const res = await (
      await fetch('/node/payout/bind', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address: ch.address, nonce: ch.nonce, signature }),
      })
    ).json();
    if (res.error) {
      onProgress('error', String(res.error));
      return { phase: 'error', message: String(res.error), retryable: true };
    }

    const done =
      'Claimed. These earnings are now recorded as yours, and survive losing this machine.';
    onProgress('done', done);
    return { phase: 'done', message: done, address: res.payout_address, retryable: false };
  } catch (err) {
    if (isUserDeclined(err)) {
      const m = 'You declined the signature. Nothing changed — you can do this any time.';
      onProgress('declined', m);
      return { phase: 'declined', message: m, retryable: true };
    }
    const m = `Could not claim: ${describe(err)}`;
    onProgress('error', m);
    return { phase: 'error', message: m, retryable: true };
  }
}

/** The unclaimed-state line, kept here so the wording is asserted rather than typed twice. */
export function unclaimedLabel(nodeId: string | null): string {
  return `not claimed — this node's NRN is held under ${nodeId ?? 'this machine'}`;
}

export const CLAIM_EXPLANATION =
  'Sign once to record that these earnings are yours. Without it they are tied to a file on ' +
  'this disk, and are lost with it.';
