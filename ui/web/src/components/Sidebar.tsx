import React, { useState } from 'react';
import { ChatThread, Folder } from '../types';
import { Wallet, NetworkState, formatNrn, networkLabel, messagesLeft, isLowBalance }
  from '../services/wallet';
import { NodeOwnerPanel } from './NodeOwnerPanel';
import {
  Plus,
  Pin,
  Trash2,
  Search,
  Settings,
  UserCheck,
  Download,
  X,
} from 'lucide-react';

interface SidebarProps {
  threads: ChatThread[];
  activeThreadId: string | null;
  folders: Folder[];
  wallet: Wallet;
  network: NetworkState;
  onSignIn: () => void;
  onSelectThread: (id: string) => void;
  onNewThread: () => void;
  onDeleteThread: (id: string) => void;
  onPinThread: (id: string) => void;
  onOpenPersonas: () => void;
  onExportData: () => void;
  isMobileOpen: boolean;
  setIsMobileOpen: (open: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  threads,
  activeThreadId,
  folders,
  wallet,
  network,
  onSignIn,
  onSelectThread,
  onNewThread,
  onDeleteThread,
  onPinThread,
  onOpenPersonas,
  onExportData,
  isMobileOpen,
  setIsMobileOpen,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);

  const filteredThreads = threads.filter(t => {
    const matchesSearch =
      t.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.messages.some(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesFolder = selectedFolderId === null || t.folderId === selectedFolderId;
    return matchesSearch && matchesFolder;
  });

  const pinnedThreads = filteredThreads.filter(t => t.pinned);
  const unpinnedThreads = filteredThreads.filter(t => !t.pinned);

  return (
    <>
      {/* Mobile backdrop */}
      {isMobileOpen && (
        <div
          onClick={() => setIsMobileOpen(false)}
          className="fixed inset-0 bg-black/40 backdrop-blur-[2px] z-40 lg:hidden"
        />
      )}

      <aside
        className={`fixed lg:static inset-y-0 left-0 z-50 w-[268px] flex flex-col bg-surface border-r border-line transition-transform duration-200 ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Brand */}
        <div className="h-14 px-4 flex items-center justify-between border-b border-line flex-shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            {/* The NEURON mark from docs/logo.svg, inlined rather than fetched: a driver node
                feeds a hub which fans out to two peers -- each circle a machine holding a slice
                of the model, each line activations crossing between them, the hub largest
                because it carries the most. Inlined so it needs no network request and
                recolours with the theme; currentColor drives the strokes. */}
            <svg viewBox="0 0 60 60" className="w-6 h-6 flex-shrink-0 text-accent"
                 role="img" aria-label="NEURON">
              <g stroke="currentColor" strokeLinecap="round" fill="none">
                <line x1="16" y1="30" x2="22" y2="30" strokeWidth="2.5" />
                <line x1="38" y1="27" x2="44" y2="17" strokeWidth="2" />
                <line x1="38" y1="33" x2="44" y2="43" strokeWidth="2" />
              </g>
              <circle cx="12" cy="30" r="6" fill="currentColor" />
              <circle cx="30" cy="30" r="8" fill="currentColor" opacity="0.75" />
              <circle cx="47" cy="15" r="4.5" fill="var(--accent-soft)"
                      stroke="currentColor" strokeWidth="2" />
              <circle cx="47" cy="45" r="4.5" fill="var(--accent-soft)"
                      stroke="currentColor" strokeWidth="2" />
            </svg>
            <div className="min-w-0">
              <div className="text-[13px] font-semibold tracking-tight text-ink leading-tight">NEURON</div>
              <div className="label-mono text-ink-faint leading-tight">Distributed inference</div>
            </div>
          </div>

          <button
            onClick={() => setIsMobileOpen(false)}
            className="lg:hidden grid place-items-center w-7 h-7 rounded-md text-ink-muted hover:text-ink hover:bg-surface-2 transition"
            title="Close sidebar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* New conversation */}
        <div className="p-3 pb-2">
          <button
            onClick={onNewThread}
            className="w-full h-9 flex items-center justify-center gap-2 rounded-lg bg-accent hover:bg-accent-hover text-accent-ink text-[13px] font-semibold transition"
          >
            <Plus className="w-4 h-4" />
            <span>New conversation</span>
          </button>
        </div>

        {/* Search */}
        <div className="px-3 pb-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-faint pointer-events-none" />
            <input
              type="text"
              placeholder="Search conversations"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full h-8 pl-8 pr-7 rounded-lg bg-surface-2 border border-transparent text-[12.5px] text-ink placeholder-ink-faint focus:outline-none focus:bg-surface focus:border-accent-line transition"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 grid place-items-center w-4 h-4 rounded text-ink-faint hover:text-ink transition"
                title="Clear search"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
          {searchQuery.trim() && (
            <div className="label-mono text-ink-faint mt-1.5 px-0.5">
              {filteredThreads.length} {filteredThreads.length === 1 ? 'match' : 'matches'}
            </div>
          )}
        </div>

        {/* Folder filter, shown only once folders exist */}
        {folders.length > 0 && (
          <div className="px-3 pb-2 flex flex-wrap gap-1">
            <button
              onClick={() => setSelectedFolderId(null)}
              className={`label-mono px-2 py-1 rounded-md border transition ${
                selectedFolderId === null
                  ? 'bg-accent-soft border-accent-line text-accent'
                  : 'bg-surface-2 border-transparent text-ink-muted hover:text-ink'
              }`}
            >
              All
            </button>
            {folders.map(folder => (
              <button
                key={folder.id}
                onClick={() => setSelectedFolderId(folder.id)}
                className={`label-mono px-2 py-1 rounded-md border transition ${
                  selectedFolderId === folder.id
                    ? 'bg-accent-soft border-accent-line text-accent'
                    : 'bg-surface-2 border-transparent text-ink-muted hover:text-ink'
                }`}
              >
                {folder.name}
              </button>
            ))}
          </div>
        )}

        {/* Threads */}
        <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-3 min-h-0">
          {pinnedThreads.length > 0 && (
            <section className="space-y-0.5">
              <div className="label-mono text-ink-faint px-2 py-1 flex items-center gap-1.5">
                <Pin className="w-3 h-3" />
                <span>Pinned</span>
              </div>
              {pinnedThreads.map(thread => renderThreadItem(thread))}
            </section>
          )}

          <section className="space-y-0.5">
            <div className="label-mono text-ink-faint px-2 py-1">Recent</div>
            {filteredThreads.length === 0 ? (
              <div className="px-2 py-6 text-center space-y-2">
                <p className="text-[12px] text-ink-muted">
                  {searchQuery ? `Nothing matches "${searchQuery}"` : 'No conversations yet.'}
                </p>
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="text-[12px] font-medium text-accent hover:underline"
                  >
                    Clear search
                  </button>
                )}
              </div>
            ) : (
              unpinnedThreads.map(thread => renderThreadItem(thread))
            )}
          </section>
        </div>

        {/* Wallet + network. Replaces the engine list: there are no engines to choose
            between, and what a contributor actually wants to see is what they have earned. */}
        <div className="px-3 py-2.5 border-t border-line space-y-1.5">
          <div className="label-mono text-ink-faint px-0.5">Wallet</div>
          {wallet.loggedIn ? (
            <div className="flex flex-col gap-1">
              {renderWalletRow('Balance', wallet.balance, wallet.error)}
              {renderWalletRow('Earned', wallet.totalEarned, wallet.error)}
              {/* Where you are in the free grant, BEFORE you reach the end of it. The grant
                  is one-time by design — NRN buys other volunteers' compute, and refilling it
                  automatically would commit their hardware to unlimited free use. Given that,
                  discovering the limit by hitting it is the genuinely unfair part, and this is
                  the fix for it. Hidden when the balance could not be read: "we could not ask"
                  must never render as a confident number. */}
              {messagesLeft(wallet.balance) !== null && (
                <div className={`text-[11px] pt-0.5 ${
                  isLowBalance(wallet.balance) ? 'text-ink' : 'text-ink-faint'}`}>
                  {`≈ ${messagesLeft(wallet.balance)} network answers left`}
                  {isLowBalance(wallet.balance) && ' · contribute this machine to earn more'}
                </div>
              )}
              {wallet.email && (
                <div className="text-[11px] text-ink-faint truncate pt-0.5" title={wallet.email}>
                  {wallet.email}
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={onSignIn}
              className="w-full h-7 rounded-lg text-[12px] font-medium text-accent hover:bg-surface-2 transition"
            >
              Sign in to chat and earn
            </button>
          )}
          <div className="flex items-center gap-1.5 pt-1" title={networkLabel(network)}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              network.reachable && network.healthy ? 'bg-ok' : 'bg-line-strong'}`} />
            <span className="text-[11px] text-ink-muted truncate">{networkLabel(network)}</span>
          </div>
        </div>

        {/* [P39]: this machine's node earnings have no owner until someone signs for them.
            Renders null unless there IS a node and it has no owner yet, so a driver-only
            install and an already-claimed one both show nothing. A panel, not a gate. */}
        <NodeOwnerPanel />

        {/* Footer actions */}
        <div className="p-2 border-t border-line flex items-center gap-1">
          <button
            onClick={onOpenPersonas}
            className="flex-1 h-8 flex items-center justify-center gap-1.5 rounded-lg text-[12px] font-medium text-ink-muted hover:text-ink hover:bg-surface-2 transition"
          >
            <UserCheck className="w-3.5 h-3.5" />
            <span>Personas</span>
          </button>
          <button
            onClick={onExportData}
            className="grid place-items-center w-8 h-8 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-2 transition"
            title="Export chat history"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </aside>
    </>
  );

  /** One wallet figure. A FAILED read must never render as a confident 0.00 -- on a network
   *  that pays people, "we could not ask" shown as zero reads as "this does not pay". */
  function renderWalletRow(label: string, value: number | null, error: string | null) {
    return (
      <div className="flex items-center justify-between gap-2"
           title={error ? `Could not read your wallet: ${error}` : undefined}>
        <span className="text-[11.5px] text-ink-muted truncate">{label}</span>
        <span className="label-mono text-ink-faint flex-shrink-0">
          {error ? 'unavailable' : `${formatNrn(value)} NRN`}
        </span>
      </div>
    );
  }

  function highlightText(text: string, query: string) {
    if (!query.trim()) return text;
    const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escapedQuery})`, 'gi');
    const parts = text.split(regex);

    return (
      <>
        {parts.map((part, i) =>
          part.toLowerCase() === query.toLowerCase() ? (
            <mark key={i} className="bg-accent-soft text-accent font-semibold rounded-sm px-0.5">
              {part}
            </mark>
          ) : (
            part
          )
        )}
      </>
    );
  }

  function renderThreadItem(thread: ChatThread) {
    const isActive = thread.id === activeThreadId;

    // Show the matching line from the conversation when searching.
    let snippet = '';
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchMsg = thread.messages.find(m => m.content.toLowerCase().includes(q));
      if (matchMsg) {
        const idx = matchMsg.content.toLowerCase().indexOf(q);
        const start = Math.max(0, idx - 15);
        const end = Math.min(matchMsg.content.length, idx + q.length + 25);
        snippet =
          (start > 0 ? '…' : '') +
          matchMsg.content.substring(start, end).replace(/\n/g, ' ') +
          (end < matchMsg.content.length ? '…' : '');
      }
    }

    return (
      <div
        key={thread.id}
        onClick={() => {
          onSelectThread(thread.id);
          setIsMobileOpen(false);
        }}
        className={`group relative flex items-start gap-2 pl-3 pr-1.5 py-2 rounded-lg cursor-pointer transition ${
          isActive ? 'bg-surface-2' : 'hover:bg-surface-2/70'
        }`}
      >
        {/* Active marker */}
        <span
          className={`absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full transition ${
            isActive ? 'bg-accent' : 'bg-transparent'
          }`}
        />

        <div className="min-w-0 flex-1">
          <div
            className={`text-[12.5px] truncate leading-snug ${
              isActive ? 'text-ink font-medium' : 'text-ink-muted group-hover:text-ink'
            }`}
          >
            {highlightText(thread.title || 'Untitled', searchQuery)}
          </div>
          {snippet ? (
            <div className="text-[11px] text-ink-faint truncate mt-0.5">
              {highlightText(snippet, searchQuery)}
            </div>
          ) : (
            <div className="label-mono text-ink-faint mt-0.5">
              {thread.messages.length} {thread.messages.length === 1 ? 'msg' : 'msgs'}
            </div>
          )}
        </div>

        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity flex-shrink-0">
          <button
            onClick={e => {
              e.stopPropagation();
              onPinThread(thread.id);
            }}
            className={`grid place-items-center w-6 h-6 rounded-md transition ${
              thread.pinned ? 'text-accent' : 'text-ink-faint hover:text-ink hover:bg-surface-3'
            }`}
            title={thread.pinned ? 'Unpin' : 'Pin'}
          >
            <Pin className={`w-3 h-3 ${thread.pinned ? 'fill-current' : ''}`} />
          </button>
          <button
            onClick={e => {
              e.stopPropagation();
              onDeleteThread(thread.id);
            }}
            className="grid place-items-center w-6 h-6 rounded-md text-ink-faint hover:text-danger hover:bg-danger-soft transition"
            title="Delete"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      </div>
    );
  }
};
