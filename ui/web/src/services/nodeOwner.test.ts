/**
 * Tests for the node-earnings claim.  Run: `npm test` in ui/web/
 *
 * WHY THESE EXIST SEPARATELY FROM ui/test_node_owner_ui.py. That file has 16 tests and they
 * cover the SERVER half — that `ui/app.py` injects `owner_wallet_id` from the session, that
 * `node_token` never reaches the browser, that `NodeBindBody` has no wallet field. All of that
 * still holds and none of it is retested here.
 *
 * What was never covered anywhere is the BROWSER half: the old page's tests stubbed
 * `window.ethereum` entirely, so connect → challenge → sign → bind was asserted by grepping
 * chat.html for substrings and executed by nobody. `claimNodeEarnings` takes its provider as an
 * argument for exactly that reason — the sequence can be driven here, including the paths a
 * real wallet takes and a stub never does.
 *
 * The four that matter, from [P39]:
 *   1. no wallet id is ever sent — the server takes it from the session on purpose;
 *   2. the panel is gated on is_node AND needs_owner, so it is a prompt and not a nag;
 *   3. a failed /node/owner renders nothing rather than an error in a working chat;
 *   4. a DECLINED signature says nothing changed and stays retryable. A person changing their
 *      mind is not a failure, and telling them it is teaches them the software is broken.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  fetchNodeOwner,
  claimNodeEarnings,
  claimWithAccount,
  isUserDeclined,
  unclaimedLabel,
  NO_NODE,
  Eip1193Provider,
  ClaimPhase,
} from './nodeOwner';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Canned JSON responses keyed by URL prefix, plus a record of what was POSTed. */
function serve(routes: Record<string, unknown>, opts: { status?: number } = {}) {
  const posted: { url: string; body: any }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'POST') posted.push({ url, body: JSON.parse(String(init.body)) });
      const key = Object.keys(routes).find(k => String(url).startsWith(k));
      if (key === undefined) throw new Error(`unrouted fetch: ${url}`);
      return { ok: (opts.status ?? 200) < 400, status: opts.status ?? 200,
               json: async () => routes[key] };
    }),
  );
  return posted;
}

/** A wallet that answers, or throws whatever it is given. */
function wallet(accounts: string[], sig: string | Error = '0xsig'): Eip1193Provider {
  return {
    request: vi.fn(async ({ method }) => {
      if (method === 'eth_requestAccounts') return accounts;
      if (method === 'personal_sign') {
        if (sig instanceof Error) throw sig;
        return sig;
      }
      throw new Error(`unexpected method ${method}`);
    }),
  };
}

const OK_ROUTES = {
  '/node/payout/challenge': { message: 'neuron node:abc', nonce: 'n1', address: '0xAbC' },
  '/node/payout/bind': { ok: true, payout_address: '0xAbC' },
};

describe('fetchNodeOwner — the gate on whether the panel exists at all', () => {
  it('reports a node that needs an owner', async () => {
    serve({ '/node/owner': { is_node: true, needs_owner: true, unclaimed: true,
                             logged_in: true, node_id: 'node-c-pavilion' } });
    expect(await fetchNodeOwner()).toEqual({
      isNode: true, needsOwner: true, unclaimed: true, loggedIn: true,
      nodeId: 'node-c-pavilion',
    });
  });

  it('a LOGGED-OUT operator still sees that the earnings are unclaimed', async () => {
    // The bug this field exists for. `needs_owner` is `logged_in AND not owned`, so it is false
    // exactly when nobody is signed in — and the panel keyed on it, so the person who most
    // needed the prompt got a blank sidebar. Zero of three live nodes had ever been claimed.
    serve({ '/node/owner': { is_node: true, needs_owner: false, unclaimed: true,
                             logged_in: false, node_id: 'agent-optinovate-6ff49d' } });
    const s = await fetchNodeOwner();
    expect(s.unclaimed).toBe(true);
    expect(s.needsOwner).toBe(false);
    expect(s.loggedIn).toBe(false);
  });

  it('an OWNED node is not unclaimed, so nobody is nagged twice', async () => {
    serve({ '/node/owner': { is_node: true, needs_owner: false, unclaimed: false,
                             logged_in: true, node_id: 'n1' } });
    expect((await fetchNodeOwner()).unclaimed).toBe(false);
  });

  it('a driver-only machine is never unclaimed, whatever the server says', async () => {
    serve({ '/node/owner': { is_node: false, unclaimed: true, node_id: null } });
    expect((await fetchNodeOwner()).unclaimed).toBe(false);
  });

  it('a node that ALREADY has an owner needs nothing — nobody is asked twice', async () => {
    serve({ '/node/owner': { is_node: true, needs_owner: false, node_id: 'n1' } });
    expect((await fetchNodeOwner()).needsOwner).toBe(false);
  });

  it('a driver-only machine is not a node, so there is nothing to claim', async () => {
    serve({ '/node/owner': { is_node: false, needs_owner: true, node_id: null } });
    // needs_owner true with is_node false is nonsense the server should not send; if it does,
    // it must not produce a panel offering to claim earnings that do not exist.
    expect((await fetchNodeOwner()).needsOwner).toBe(false);
  });

  it('a coordinator outage renders nothing, it does not break the chat', async () => {
    serve({ '/node/owner': {} }, { status: 503 });
    expect(await fetchNodeOwner()).toEqual(NO_NODE);
  });

  it('...and neither does a thrown fetch', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('offline'); }));
    expect(await fetchNodeOwner()).toEqual(NO_NODE);
  });
});

describe('claimNodeEarnings — the path no test has ever executed', () => {
  it('connects, signs the challenge, binds, and reports the address', async () => {
    const posted = serve(OK_ROUTES);
    const phases: ClaimPhase[] = [];
    const res = await claimNodeEarnings(wallet(['0xAbC']), p => phases.push(p));

    expect(res.phase).toBe('done');
    expect(res.address).toBe('0xAbC');
    expect(res.retryable).toBe(false);
    expect(phases).toEqual(['connecting', 'challenging', 'signing', 'binding', 'done']);
    expect(posted).toHaveLength(1);
    expect(posted[0].url).toBe('/node/payout/bind');
  });

  it('NEVER sends a wallet id — the server takes it from the session on purpose', async () => {
    const posted = serve(OK_ROUTES);
    await claimNodeEarnings(wallet(['0xAbC']));
    const body = posted[0].body;
    expect(Object.keys(body).sort()).toEqual(['address', 'nonce', 'signature']);
    for (const k of ['wallet_id', 'walletId', 'owner_wallet_id', 'email']) {
      expect(body).not.toHaveProperty(k);
    }
  });

  it('signs the exact message the server challenged with, for the address it returned',
    async () => {
      serve(OK_ROUTES);
      const w = wallet(['0xAbC']);
      await claimNodeEarnings(w);
      expect(w.request).toHaveBeenCalledWith({
        method: 'personal_sign', params: ['neuron node:abc', '0xAbC'],
      });
    });

  it('DECLINING is not an error: nothing changed, and the button stays usable', async () => {
    serve(OK_ROUTES);
    const denied = Object.assign(new Error('User rejected the request'), { code: 4001 });
    const res = await claimNodeEarnings(wallet(['0xAbC'], denied));
    expect(res.phase).toBe('declined');
    expect(res.retryable).toBe(true);
    expect(res.message).toMatch(/nothing changed/i);
    expect(res.message).not.toMatch(/error|failed/i);
  });

  it('a wallet that throws a bare rejection message is still a decline, not a fault',
    async () => {
      serve(OK_ROUTES);
      const res = await claimNodeEarnings(wallet(['0xAbC'], new Error('user denied signature')));
      expect(res.phase).toBe('declined');
    });

  it('a real failure says so and stays retryable', async () => {
    serve(OK_ROUTES);
    const res = await claimNodeEarnings(wallet(['0xAbC'], new Error('wallet exploded')));
    expect(res.phase).toBe('error');
    expect(res.retryable).toBe(true);
    expect(res.message).toContain('wallet exploded');
  });

  it('no wallet extension explains where to go instead of failing silently', async () => {
    const res = await claimNodeEarnings(undefined);
    expect(res.phase).toBe('error');
    expect(res.retryable).toBe(true);
    expect(res.message).toMatch(/sign_payout\.html/);
  });

  it('a locked wallet returning no account is explained, not thrown', async () => {
    serve(OK_ROUTES);
    const res = await claimNodeEarnings(wallet([]));
    expect(res.phase).toBe('error');
    expect(res.message).toMatch(/unlock/i);
  });

  it('a challenge the server refuses stops before any signature is requested', async () => {
    serve({ '/node/payout/challenge': { error: 'that address is already bound' } });
    const w = wallet(['0xAbC']);
    const res = await claimNodeEarnings(w);
    expect(res.phase).toBe('error');
    expect(res.message).toBe('that address is already bound');
    expect(w.request).toHaveBeenCalledTimes(1);   // eth_requestAccounts only
  });

  it('a bind the server refuses is reported and left retryable', async () => {
    serve({
      '/node/payout/challenge': OK_ROUTES['/node/payout/challenge'],
      '/node/payout/bind': { error: 'signature did not match' },
    });
    const res = await claimNodeEarnings(wallet(['0xAbC']));
    expect(res.phase).toBe('error');
    expect(res.message).toBe('signature did not match');
    expect(res.retryable).toBe(true);
  });
});

describe('isUserDeclined', () => {
  it('4001 is the EIP-1193 code for "the person said no"', () => {
    expect(isUserDeclined({ code: 4001 })).toBe(true);
  });
  it('matches the wording wallets use when they throw plain errors', () => {
    for (const m of ['User rejected', 'user denied', 'Request declined']) {
      expect(isUserDeclined(new Error(m))).toBe(true);
    }
  });
  it('does not swallow real faults as declines', () => {
    expect(isUserDeclined(new Error('network error'))).toBe(false);
    expect(isUserDeclined({ code: -32603 })).toBe(false);
    expect(isUserDeclined(null)).toBe(false);
  });
});

describe('wording', () => {
  it('names the node holding the money, because that is what makes it concrete', () => {
    expect(unclaimedLabel('node-c-pavilion')).toContain('node-c-pavilion');
  });
  it('still reads as a sentence when the node id is missing', () => {
    expect(unclaimedLabel(null)).toBe("not claimed — this node's NRN is held under this machine");
  });
});

describe('claimWithAccount — the default path, and why it exists ([P53])', () => {
  it('claims with one POST to /node/claim and no wallet involved', async () => {
    const posted = serve({ '/node/claim': { node_id: 'n1', owner_wallet_id: 'w_x',
                                            payout_address: '0xAbC', rebound: false } });
    const res = await claimWithAccount();
    expect(res.phase).toBe('done');
    expect(res.address).toBe('0xAbC');
    expect(posted).toHaveLength(1);
    expect(posted[0].url).toBe('/node/claim');
  });

  it('sends NO wallet id — the server takes the owner from the session, same as rule 1', async () => {
    const posted = serve({ '/node/claim': { node_id: 'n1', payout_address: '0xAbC' } });
    await claimWithAccount();
    expect(JSON.stringify(posted[0].body ?? {})).not.toMatch(/w_|wallet/i);
  });

  it('never touches window.ethereum, so a machine with no wallet extension can still claim', async () => {
    serve({ '/node/claim': { node_id: 'n1', payout_address: '0xAbC' } });
    vi.stubGlobal('ethereum', undefined);
    const res = await claimWithAccount();
    expect(res.phase).toBe('done');
  });

  it('surfaces the 409 verbatim and stays retryable — a foreign bound address is not a bug', async () => {
    serve({ '/node/claim': { error: 'this node pays out to 0xOther, which is not an address '
                                    + 'this machine holds a key for.' } }, { status: 409 });
    const res = await claimWithAccount();
    expect(res.phase).toBe('error');
    expect(res.retryable).toBe(true);
    expect(res.message).toContain('0xOther');
  });

  it('reports progress so the button can narrate', async () => {
    serve({ '/node/claim': { node_id: 'n1', payout_address: '0xAbC' } });
    const seen: ClaimPhase[] = [];
    await claimWithAccount(p => seen.push(p));
    expect(seen).toContain('binding');
    expect(seen).toContain('done');
  });
});
