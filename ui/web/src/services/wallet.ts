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
