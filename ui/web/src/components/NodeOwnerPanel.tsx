import React, { useCallback, useEffect, useState } from 'react';
import { Wallet2 } from 'lucide-react';
import {
  NodeOwnerState,
  NO_NODE,
  ClaimPhase,
  fetchNodeOwner,
  claimNodeEarnings,
  unclaimedLabel,
  CLAIM_EXPLANATION,
} from '../services/nodeOwner';

/**
 * "This computer's earnings" — the claim prompt, ported from chat.html's
 * buildNodeOwnerSection(). See services/nodeOwner.ts for why the behaviour lives outside the
 * component and what [P39] requires of it.
 *
 * A PANEL, NOT A GATE. It renders null unless this machine serves a node with no owner
 * recorded, so a driver-only install shows nothing, an already-claimed node shows nothing, and
 * a volunteer with no browser wallet is never blocked from running a node — they simply do not
 * press the button. Nothing here can prevent chatting, signing in, or serving.
 */
export const NodeOwnerPanel: React.FC = () => {
  const [state, setState] = useState<NodeOwnerState>(NO_NODE);
  const [phase, setPhase] = useState<ClaimPhase>('idle');
  const [status, setStatus] = useState('');
  const [address, setAddress] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    // Failure answers NO_NODE, which renders nothing — a coordinator outage must not put an
    // error box in the sidebar of a chat that is working perfectly well.
    fetchNodeOwner(controller.signal).then(setState).catch(() => {});
    return () => controller.abort();
  }, []);

  const onClaim = useCallback(async () => {
    setPhase('connecting');
    const res = await claimNodeEarnings(window.ethereum, (p, m) => {
      setPhase(p);
      setStatus(m);
    });
    setPhase(res.phase);
    setStatus(res.message);
    if (res.phase === 'done' && res.address) setAddress(res.address);
  }, []);

  if (!state.needsOwner && !address) return null;

  const busy = phase === 'connecting' || phase === 'challenging'
    || phase === 'signing' || phase === 'binding';
  const claimed = phase === 'done';

  return (
    <div className="px-3 py-2.5 border-t border-line space-y-1.5">
      <div className="label-mono text-ink-faint px-0.5">This computer&rsquo;s earnings</div>

      <div
        className={`text-[11px] break-all ${claimed ? 'text-ink' : 'text-ink-muted'}`}
        title={claimed ? address ?? '' : undefined}
      >
        {claimed ? address : unclaimedLabel(state.nodeId)}
      </div>

      {!claimed && (
        <button
          onClick={onClaim}
          disabled={busy}
          className="w-full h-7 rounded-lg text-[12px] font-medium text-accent hover:bg-surface-2
                     transition disabled:opacity-50 disabled:cursor-default
                     flex items-center justify-center gap-1.5"
        >
          <Wallet2 className="w-3.5 h-3.5" />
          <span>{busy ? 'Claiming…' : 'Claim with browser wallet'}</span>
        </button>
      )}

      {/* The explanation shows until something happens, then gets out of the way. A declined
          signature reads as ordinary text, not as a failure, because it is not one. */}
      <div className={`text-[11px] leading-snug ${
        phase === 'error' ? 'text-ink' : 'text-ink-faint'}`}>
        {status || CLAIM_EXPLANATION}
      </div>
    </div>
  );
};
