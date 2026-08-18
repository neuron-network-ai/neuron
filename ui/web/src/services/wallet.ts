/**
 * Wallet and network state, from the two endpoints `ui/app.py` already proxies.
 *
 * Both go through the agent's own server, never to the coordinator directly — the browser has
 * no coordinator credentials and should not learn its address.
 *
 *   GET /wallet/balance -> { logged_in: false }
 *                        | { logged_in: true, wallet_id, email, balance, total_earned }
 *                        | { logged_in: true, wallet_id, error }        <- ask failed
 *   GET /network        -> { reachable, online_nodes, local_capable, healthy, ... }
 *
 * The three-state balance is the point, and it is a mistake this project has already made
 * once: "you earned nothing" and "we could not ask" render identically as 0.00 unless they are
 * kept apart. A wallet whose read FAILED must never display a confident zero — on a network
 * that pays people, that reads as "this does not pay".
 */

export interface Wallet {
  loggedIn: boolean;
  walletId: string | null;
  email: string | null;
  /** null when not logged in, or when the balance could not be read. Never coerce to 0. */
  balance: number | null;
  /** Lifetime NRN earned by contributing this machine. null under the same rules. */
  totalEarned: number | null;
  /** Set when logged in but the balance read failed. The UI must say so, not show zero. */
  error: string | null;
}

export interface NetworkState {
  reachable: boolean;
  onlineNodes: number;
  /** True when THIS machine can serve the model itself. Decides whether the header may
   *  credit the network for an answer — it answered locally by design, not by failure. */
  localCapable: boolean;
  healthy: boolean;
  servingModel: string | null;
  /**
   * Did the poll itself succeed? NOT the same as `healthy`, and conflating them is a real
   * outage: on a failed fetch every other field falls back to its pessimistic default, so a
   * network that is merely unpolled looks exactly like one that is down. Anything that REFUSES
   * to act — see `blockReason` — must key on this first, or a flaky status endpoint takes chat
   * away from a user whose chain is perfectly fine.
   */
  statusKnown: boolean;
  /** Layers of the serving model currently covered by online nodes, and the total. */
  layersCovered: number | null;
  totalLayers: number | null;
  /** Inclusive [lo, hi] ranges with no node behind them. Named rather than counted, because
   *  "3 layers missing" is not something an operator can act on and "missing 10–12" is. */
  uncoveredLayers: [number, number][];
}

export const EMPTY_WALLET: Wallet = {
  loggedIn: false, walletId: null, email: null,
  balance: null, totalEarned: null, error: null,
};

export async function fetchWallet(signal?: AbortSignal): Promise<Wallet> {
  try {
    const r = await fetch('/wallet/balance', { signal });
    if (!r.ok) return { ...EMPTY_WALLET, error: `wallet service answered ${r.status}` };
    const d = await r.json();
    if (!d.logged_in) return EMPTY_WALLET;
    if (d.error) {
      // Logged in, but the coordinator could not be asked. Keep loggedIn true so the UI shows
      // the account and says the figure is unavailable, rather than logging the user out.
      return { ...EMPTY_WALLET, loggedIn: true, walletId: d.wallet_id ?? null,
               email: d.email ?? null, error: String(d.error) };
    }
    return {
      loggedIn: true,
      walletId: d.wallet_id ?? null,
      email: d.email ?? null,
      balance: typeof d.balance === 'number' ? d.balance : null,
      totalEarned: typeof d.total_earned === 'number' ? d.total_earned : null,
      error: null,
    };
  } catch {
    // A failed poll is not a logout, and not a zero balance.
    return { ...EMPTY_WALLET, error: 'could not reach the wallet service' };
  }
}

export async function fetchNetwork(signal?: AbortSignal): Promise<NetworkState> {
  try {
    const r = await fetch('/network', { signal });
    if (!r.ok) throw new Error(String(r.status));
    const d = await r.json();
    return {
      reachable: Boolean(d.reachable),
      onlineNodes: Number(d.online_nodes ?? 0),
      localCapable: Boolean(d.local_capable),
      healthy: Boolean(d.healthy ?? d.network_healthy),
      servingModel: (d.model_id as string) ?? null,
      statusKnown: true,
      layersCovered: typeof d.layers_covered === 'number' ? d.layers_covered : null,
      totalLayers: typeof d.total_layers === 'number' ? d.total_layers : null,
      uncoveredLayers: Array.isArray(d.uncovered_layers) ? d.uncovered_layers : [],
    };
  } catch {
    // statusKnown:false is the whole point of this branch. Every other field is a pessimistic
    // default and must not be read as evidence about the network.
    return { reachable: false, onlineNodes: 0, localCapable: false, healthy: false,
             servingModel: null, statusKnown: false, layersCovered: null,
             totalLayers: null, uncoveredLayers: [] };
  }
}

/**
 * Why sending is refused right now, or null when it is not.
 *
 * A prompt sent into an incomplete chain cannot be answered, so it costs the user a wait and
 * then an error — when the page already knew. chat.html has refused this since Session 55;
 * React did not, and `canSend` was `(text || attachments) && !isGenerating` with no network
 * term at all.
 *
 * Two conditions it deliberately does NOT block on:
 *   * a failed status poll (`statusKnown` false). That says nothing about /chat, and refusing
 *     on it turns a monitoring blip into an outage.
 *   * a machine that can serve the model itself. An incomplete chain means the NETWORK is
 *     short of nodes; local inference does not care.
 */
export function blockReason(n: NetworkState): string | null {
  if (!n.statusKnown) return null;
  if (n.localCapable) return null;
  if (!n.healthy) {
    return 'The network is short of nodes right now, so there is nothing to send this to.';
  }
  return null;
}

/** The degraded banner's text, or null. Names the missing layers rather than counting them,
 *  and takes the total from the coordinator instead of a hardcoded 28 that goes stale on the
 *  next model tier. */
export function degradedNotice(n: NetworkState): string | null {
  if (!n.statusKnown || n.localCapable || n.healthy) return null;
  const total = n.totalLayers ?? 28;
  const where = n.uncoveredLayers
    .map(r => (r[0] === r[1] ? String(r[0]) : `${r[0]}–${r[1]}`))
    .join(', ');
  return `Network degraded — ${n.layersCovered ?? 0}/${total} model layers are online`
    + (where ? ` (missing ${where})` : '')
    + ', and this machine cannot run the model on its own yet. '
    + 'Sending is paused until the chain is complete.';
}

/** NRN as shown to a person. Four decimals: a single answer can cost ~0.0120. */
export function formatNrn(v: number | null): string {
  return v == null ? '—' : v.toFixed(4).replace(/\.?0+$/, '') || '0';
}

/**
 * Roughly what a network answer costs: `PRICE_PER_1K_WEIGHTED` (1.0 NRN/1k weighted tokens)
 * against a typical turn. The coordinator holds ~0.158 NRN before it will dispatch at all, so
 * this is also the floor below which the next message is refused outright.
 */
export const NRN_PER_MESSAGE = 0.158;

/**
 * About how many more network answers this balance buys. null when unknown — never 0, because
 * "we could not read your balance" and "you have none" must not render the same ([P43]'s
 * three-state rule, applied to a number people plan around).
 *
 * WHY THIS IS SHOWN AT ALL, and it is the whole of the [P29] decision. The grant is 25 NRN,
 * about 158 messages, and then it is gone for good — there is no recurring faucet and there
 * should not be one (see below). Given that, a person is entitled to know where they are in it
 * BEFORE they arrive at the end. Discovering a hard limit by hitting it is the part that is
 * actually unfair; the limit itself is not.
 */
export function messagesLeft(balance: number | null): number | null {
  if (balance == null || !isFinite(balance)) return null;
  return Math.max(0, Math.floor(balance / NRN_PER_MESSAGE));
}

/** Worth warning about: roughly a fifth of the initial grant, or unable to send at all. */
export function isLowBalance(balance: number | null): boolean {
  const n = messagesLeft(balance);
  return n !== null && n <= 30;
}

/**
 * How the header describes who served an answer.
 *
 * A machine that can hold the serving model answers LOCALLY BY DESIGN, so the network must not
 * be credited for that work — the old header read "Powered by N nodes worldwide" directly above
 * a reply whose own line said "this machine · Cost: 0.0000 NRN".
 */
export function networkLabel(net: NetworkState): string {
  if (!net.reachable) return 'Coordinator unreachable';
  const nodes = `${net.onlineNodes} node${net.onlineNodes === 1 ? '' : 's'} online`;
  return net.localCapable ? `Answers run here · ${nodes} for bigger models` : `Powered by ${nodes}`;
}
