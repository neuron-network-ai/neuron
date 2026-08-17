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
    };
  } catch {
    return { reachable: false, onlineNodes: 0, localCapable: false,
             healthy: false, servingModel: null };
  }
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
