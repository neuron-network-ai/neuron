/**
 * Is there a newer build, and does this person need to do anything about it?
 *
 * 0.20.2 exists because nothing reported the running build, so a rollout was invisible and a
 * rollback could not be CONFIRMED. That was fixed for the operator's dashboard; the app — the
 * one place the person who must act actually looks — still said nothing. Auto-update covers
 * most machines and is not enough on its own: it is a setting a person can switch off, its
 * download can fail, and a source checkout is deliberately never touched by it.
 *
 * The comparison is done SERVER-SIDE (`ui/app.py:/app/update`) using the same numeric rule as
 * the coordinator ([P48]): `"0.20.10" > "0.20.9"` is false as strings, so a lexical compare
 * breaks at the tenth patch release, and a node AHEAD of the network must never be told to
 * downgrade. This module only renders what it is told.
 */

export interface UpdateState {
  running: string | null;
  latest: string | null;
  /** True only when the server determined a newer build exists. Never inferred here. */
  available: boolean;
  /** The coordinator is asking nodes to move to an OLDER build (NEURON_AGENT_ROLLBACK). */
  rollback: boolean;
  url: string | null;
  /** Set when the check could not be made. Must never render as "you are up to date". */
  error: string | null;
}

export const NO_UPDATE: UpdateState = {
  running: null, latest: null, available: false, rollback: false, url: null, error: null,
};

export const RELEASES_URL = 'https://github.com/neuron-network-ai/neuron/releases/latest';

export async function fetchUpdate(signal?: AbortSignal): Promise<UpdateState> {
  try {
    const r = await fetch('/app/update', { signal });
    if (!r.ok) return { ...NO_UPDATE, error: `update check answered ${r.status}` };
    const d = await r.json();
    return {
      running: d.running ?? null,
      latest: d.latest ?? null,
      available: Boolean(d.available),
      rollback: Boolean(d.rollback),
      url: d.url ?? null,
      error: d.error ?? null,
    };
  } catch {
    return { ...NO_UPDATE, error: 'could not reach the update check' };
  }
}

/**
 * The sentence to show, or null for nothing.
 *
 * A rollback is worded as the network ASKING rather than as an upgrade: telling somebody to
 * "update" to an older version reads as a bug and gets ignored, which is the one case where
 * being ignored is most expensive.
 */
export function updateNotice(u: UpdateState): string | null {
  if (!u.available || !u.latest) return null;
  return u.rollback
    ? `NEURON ${u.latest} is the build this network is asking nodes to run`
      + (u.running ? ` — you have ${u.running}.` : '.')
    : `NEURON ${u.latest} is available`
      + (u.running ? ` — you are running ${u.running}.` : '.');
}

/** Where "Get it" points. The coordinator's own url when it gives one, so a ROLLBACK sends
 *  people to the build the network actually wants rather than to whatever is newest. */
export function updateHref(u: UpdateState): string {
  return u.url || RELEASES_URL;
}
