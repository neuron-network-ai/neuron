import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ChatThread,
  ChatMessage,
  Settings,
  Persona,
  Folder,
  Attachment,
  AnswerMeta,
} from './types';
import {
  loadSettings,
  saveSettings,
  loadThreads,
  saveThreads,
  loadPersonas,
  saveCustomPersonas,
  loadFolders,
  saveFolders,
} from './utils/storage';
import { streamChat, errorMessage } from './services/neuron';
import { Wallet, NetworkState, EMPTY_WALLET, fetchWallet, fetchNetwork, blockReason, degradedNotice } from './services/wallet';
import { Sidebar } from './components/Sidebar';
import { ChatMessageList } from './components/ChatMessageList';
import { ChatInput } from './components/ChatInput';
import { PersonasModal } from './components/PersonasModal';
import { ThemeProvider, useTheme } from './contexts/ThemeContext';
import { Menu, UserCheck, Trash2, Sun, Moon, Monitor } from 'lucide-react';

function AppContent() {
  const { theme, setTheme } = useTheme();
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [wallet, setWallet] = useState<Wallet>(EMPTY_WALLET);
  // statusKnown starts FALSE: nothing has been polled yet, and the initial state must not
  // read as "the network is down" — that would block the very first message on every load.
  const [network, setNetwork] = useState<NetworkState>(
    { reachable: false, onlineNodes: 0, localCapable: false, healthy: false, servingModel: null,
      statusKnown: false, layersCovered: null, totalLayers: null, uncoveredLayers: [] });
  const [threads, setThreads] = useState<ChatThread[]>(loadThreads);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [personas, setPersonas] = useState<Persona[]>(loadPersonas);
  const [folders, setFolders] = useState<Folder[]>(loadFolders);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Lets the Stop button cancel the in-flight request rather than just hiding it.
  const abortRef = useRef<AbortController | null>(null);

  // Message rows are memoised so a streaming token only re-renders the message
  // it belongs to. That only holds if their callbacks are referentially stable,
  // so callbacks read state through this ref instead of closing over it.
  const latestRef = useRef({ threads, activeThreadId, settings });
  latestRef.current = { threads, activeThreadId, settings };

  const getActiveThread = useCallback(() => {
    const { threads: ts, activeThreadId: id } = latestRef.current;
    return ts.find(t => t.id === id) || ts[0];
  }, []);

  // Modals visibility
  const [showPersonasModal, setShowPersonasModal] = useState(false);

  // Sync theme with settings. The provider is seeded from stored settings in
  // App(), so this only records later changes made from the header toggle.
  useEffect(() => {
    setSettings(prev => (prev.theme === theme ? prev : { ...prev, theme }));
  }, [theme]);

  // Wallet and network state, polled.
  //
  // This replaces the local-server probe that used to run here, walking a list of candidate
  // Ollama / LM Studio ports. NEURON has one backend and it is the server that served this
  // page, so there is nothing to discover -- and what a contributor actually wants on screen
  // is what their machine has earned.
  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;

    const poll = async () => {
      const [w, n] = await Promise.all([
        fetchWallet(controller.signal),
        fetchNetwork(controller.signal),
      ]);
      if (controller.signal.aborted) return;
      setWallet(w);
      setNetwork(n);
      timer = window.setTimeout(poll, 15000);
    };

    poll();
    return () => {
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  // Create initial thread if none exists
  useEffect(() => {
    if (threads.length === 0) {
      const initialThread: ChatThread = makeThread();
      setThreads([initialThread]);
      setActiveThreadId(initialThread.id);
      saveThreads([initialThread]);
    } else if (!activeThreadId) {
      setActiveThreadId(threads[0].id);
    }
  }, []);

  // Persist threads, but coalesce writes. Streaming mutates this state many
  // times a second and serialising the whole history on every token costs
  // real time once a conversation gets long.
  useEffect(() => {
    const handle = window.setTimeout(() => saveThreads(threads), 400);
    return () => window.clearTimeout(handle);
  }, [threads]);

  // A debounce can always lose the last write, so flush when the page goes away.
  useEffect(() => {
    const flush = () => saveThreads(latestRef.current.threads);
    window.addEventListener('pagehide', flush);
    document.addEventListener('visibilitychange', flush);
    return () => {
      window.removeEventListener('pagehide', flush);
      document.removeEventListener('visibilitychange', flush);
      flush();
    };
  }, []);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  useEffect(() => {
    saveCustomPersonas(personas);
  }, [personas]);

  useEffect(() => {
    saveFolders(folders);
  }, [folders]);

  // Cancel any in-flight generation if the app unmounts.
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const activeThread = threads.find(t => t.id === activeThreadId) || threads[0];

  const updateThread = useCallback((threadId: string, updater: (thread: ChatThread) => ChatThread) => {
    setThreads(prev => prev.map(t => (t.id === threadId ? updater(t) : t)));
  }, []);

  /** A new thread. The model is recorded from what the network is serving, never chosen:
   *  the tier controller decides it, bounded by the weakest machine in the chain. */
  const makeThread = useCallback((from?: ChatThread | null): ChatThread => ({
    id: 'thread_' + Date.now(),
    title: 'New conversation',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    modelId: from?.modelId || network.servingModel || '',
    systemPrompt: from?.systemPrompt || settings.defaultSystemPrompt || 'You are a helpful AI assistant.',
    maxTokens: from?.maxTokens ?? settings.defaultMaxTokens ?? 8192,
    conversationId: null,
    useRag: from?.useRag ?? settings.useRag,
  }), [network.servingModel, settings]);

  const handleNewThread = () => {
    const newThread: ChatThread = makeThread(activeThread);

    setThreads(prev => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
  };

  const handleDeleteThread = (id: string) => {
    const filtered = threads.filter(t => t.id !== id);

    if (filtered.length === 0) {
      const newThread: ChatThread = makeThread();
      setThreads([newThread]);
      setActiveThreadId(newThread.id);
      return;
    }

    setThreads(filtered);
    if (activeThreadId === id) {
      setActiveThreadId(filtered[0].id);
    }
  };

  const handlePinThread = (id: string) => {
    setThreads(prev =>
      prev.map(t => (t.id === id ? { ...t, pinned: !t.pinned } : t))
    );
  };

  const updateActiveThread = (updater: (thread: ChatThread) => ChatThread) => {
    if (!activeThreadId) return;
    updateThread(activeThreadId, updater);
  };

  const estimateTokens = (text: string) => Math.max(1, Math.ceil((text || '').length / 4));

  /**
   * Single entry point for producing an assistant reply. Send, regenerate and
   * edit-and-resend all funnel through here with an explicit message list, so
   * none of them can read a stale copy of the thread.
   */
  const runGeneration = useCallback(async (thread: ChatThread, contextMessages: ChatMessage[]) => {
    const threadId = thread.id;
    const assistantMsgId = 'msg_asst_' + Date.now();
    const assistantPlaceholder: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      model: thread.modelId,
    };

    updateThread(threadId, t => ({
      ...t,
      updatedAt: Date.now(),
      messages: [...contextMessages, assistantPlaceholder],
    }));

    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    setIsGenerating(true);

    const genStartTime = Date.now();
    const promptToks = contextMessages.reduce((sum, m) => sum + estimateTokens(m.content), 0);

    const applyToAssistant = (patch: (msg: ChatMessage) => ChatMessage) => {
      updateThread(threadId, t => ({
        ...t,
        messages: t.messages.map(m => (m.id === assistantMsgId ? patch(m) : m)),
      }));
    };

    // A fast local model can emit tokens quicker than the browser can paint.
    // Deltas are buffered and applied once per animation frame, so render cost
    // is bounded by the display rate rather than by the token rate.
    let pendingDelta = '';
    let frame: number | null = null;

    const flushDelta = () => {
      frame = null;
      if (!pendingDelta) return;
      const delta = pendingDelta;
      pendingDelta = '';
      const elapsed = Date.now() - genStartTime;
      applyToAssistant(m => {
        const content = (m.content || '') + delta;
        const compToks = estimateTokens(content);
        return {
          ...m,
          content,
          latencyMs: elapsed,
          promptTokens: promptToks,
          completionTokens: compToks,
          tokensEstimate: promptToks + compToks,
        };
      });
    };

    const cancelPendingFlush = () => {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      pendingDelta = '';
    };

    // Carried out of the stream so the finally-block and the error path can both see them.
    let meta: AnswerMeta | null = null;
    let reroutes = 0;

    try {
      await streamChat(
        {
          prompt: contextMessages[contextMessages.length - 1]?.content ?? '',
          maxTokens: thread.maxTokens,
          useRag: thread.useRag,
          conversationId: thread.conversationId ?? null,
          signal: controller.signal,
        },
        {
          onMeta: (m) => {
            meta = { nodes: m.nodes, local: m.local, costNrn: m.costNrn };
            // The server creates the conversation on the first turn; keeping its id is what
            // makes history survive a reload.
            if (m.conversationId) {
              updateThread(threadId, t => ({ ...t, conversationId: m.conversationId }));
            }
            applyToAssistant(msg => ({ ...msg, neuron: meta ?? undefined }));
          },
          onToken: (text) => {
            pendingDelta += text;
            if (frame === null) frame = requestAnimationFrame(flushDelta);
          },
          // A node dying mid-answer is RECOVERED, token for token -- test_node_death.py
          // SIGKILLs one and requires output identical to an uninterrupted run. So this is a
          // neutral status, never an error: presenting it as a failure teaches people to
          // distrust the one thing the network handles best.
          onReroute: () => {
            reroutes += 1;
            applyToAssistant(msg => ({
              ...msg,
              neuron: { ...(msg.neuron ?? { nodes: 0, local: false, costNrn: null }), reroutes },
            }));
          },
          onDone: (d) => {
            cancelPendingFlush();
            const compToks = d.tokens || estimateTokens(d.text);
            applyToAssistant(msg => ({
              ...msg,
              // d.text is authoritative -- it is what the driver actually produced and what
              // was billed, so it replaces whatever the token stream assembled.
              content: d.text || msg.content,
              latencyMs: d.latencyMs || (Date.now() - genStartTime),
              promptTokens: promptToks,
              completionTokens: compToks,
              tokensEstimate: promptToks + compToks,
              neuron: {
                nodes: meta?.nodes ?? 0,
                local: meta?.local ?? false,
                costNrn: d.costNrn,
                tokPerS: d.tokPerS,
                latencyMs: d.latencyMs,
                reroutes: d.reroutes || reroutes,
              },
            }));
            // An answer costs NRN, so the sidebar figure is stale the moment one finishes.
            fetchWallet().then(setWallet).catch(() => {});
          },
          // THE PARTIAL ANSWER SURVIVES. Text already streamed has been given to the user and
          // billed to them; replacing it with an error message deletes something they paid
          // for. The failure is appended, never substituted.
          onError: (err) => {
            if (frame !== null) cancelAnimationFrame(frame);
            frame = null;
            flushDelta();
            applyToAssistant(msg => ({
              ...msg,
              content: msg.content
                ? `${msg.content}

⚠️ ${errorMessage(err, true)}`
                : `⚠️ ${errorMessage(err, false)}`,
              error: true,
              errorDetail: err.detail,
            }));
          },
        }
      );
    } catch (err: any) {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      flushDelta();

      if (controller.signal.aborted) {
        // Stopped on purpose: keep whatever streamed in so far.
        applyToAssistant(msg => ({
          ...msg,
          content: msg.content ? `${msg.content}

⏹ Stopped` : '⏹ Stopped before any output.',
        }));
      } else {
        applyToAssistant(msg => ({
          ...msg,
          content: msg.content
            ? `${msg.content}

⚠️ The connection to this machine dropped.`
            : '⚠️ Could not reach the chat service on this machine.',
          error: true,
          errorDetail: String(err?.message ?? err),
        }));
      }
    } finally {
      if (frame !== null) cancelAnimationFrame(frame);
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsGenerating(false);
      }
    }
  }, [updateThread]);

  const handleSendMessage = async (text: string, attachments: Attachment[]) => {
    if (!activeThread) return;

    const userMessage: ChatMessage = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
      attachments,
    };

    const updatedMessages = [...activeThread.messages, userMessage];

    // Generate smart thread title if it's the first message
    if (activeThread.messages.length === 0) {
      const newTitle = text.length > 30 ? text.slice(0, 30) + '...' : text || 'New AI Conversation';
      updateThread(activeThread.id, t => ({ ...t, title: newTitle }));
    }

    await runGeneration(activeThread, updatedMessages);
  };

  // The handlers below are referentially stable and read live state from
  // latestRef, so memoised message rows never see a stale thread.
  const handleRegenerate = useCallback(async () => {
    const thread = getActiveThread();
    if (!thread || thread.messages.length === 0) return;

    // Drop everything after the last user message and answer it again.
    const lastUserIdx = thread.messages.reduce((acc, m, idx) => (m.role === 'user' ? idx : acc), -1);
    if (lastUserIdx === -1) return;

    await runGeneration(thread, thread.messages.slice(0, lastUserIdx + 1));
  }, [getActiveThread, runGeneration]);

  const handleEditMessage = useCallback((msgId: string, newContent: string) => {
    const thread = getActiveThread();
    if (!thread) return;
    const msgIdx = thread.messages.findIndex(m => m.id === msgId);
    if (msgIdx === -1) return;

    const updated = [...thread.messages];
    updated[msgIdx] = { ...updated[msgIdx], content: newContent };

    // Answer the edited message directly; passing the list avoids regenerating
    // from the pre-edit thread state.
    runGeneration(thread, updated.slice(0, msgIdx + 1));
  }, [getActiveThread, runGeneration]);

  const handleStopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleBranchMessage = useCallback((messageId: string) => {
    const activeThread = getActiveThread();
    if (!activeThread) return;
    const msgIdx = activeThread.messages.findIndex(m => m.id === messageId);
    if (msgIdx === -1) return;

    const branchedMessages = activeThread.messages.slice(0, msgIdx + 1);
    const branchTitle = activeThread.title
      ? `Branch: ${activeThread.title.replace(/^Branch:\s*/, '')}`
      : 'Branched Conversation';

    const newThread: ChatThread = {
      id: 'thread_' + Date.now(),
      title: branchTitle,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: branchedMessages,
      modelId: activeThread.modelId,
      systemPrompt: activeThread.systemPrompt,
      maxTokens: activeThread.maxTokens,
      conversationId: null,
      useRag: activeThread.useRag,
      folderId: activeThread.folderId,
      personaId: activeThread.personaId,
    };

    setThreads(prev => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
  }, [getActiveThread]);

  const handleFeedbackMessage = useCallback((messageId: string, feedback: 'up' | 'down' | null) => {
    const id = latestRef.current.activeThreadId;
    if (!id) return;
    updateThread(id, t => ({
      ...t,
      messages: t.messages.map(m => (m.id === messageId ? { ...m, feedback } : m)),
      updatedAt: Date.now(),
    }));
  }, [updateThread]);

  const handleClearConversation = () => {
    if (!activeThread || activeThread.messages.length === 0) return;
    if (window.confirm('Are you sure you want to clear all messages from this conversation?')) {
      updateActiveThread(t => ({
        ...t,
        messages: [],
        updatedAt: Date.now(),
      }));
    }
  };

  const handleSelectPersona = (persona: Persona) => {
    updateActiveThread(t => ({
      ...t,
      personaId: persona.id,
      systemPrompt: persona.systemPrompt,
    }));
  };

  const handleExportBackup = () => {
    const data = {
      threads,
      settings,
      personas: personas.filter(p => p.isCustom),
      folders,
      exportDate: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `localai_chat_backup_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportBackup = (jsonStr: string) => {
    try {
      const data = JSON.parse(jsonStr);
      if (data.threads && Array.isArray(data.threads)) {
        setThreads(data.threads);
        setActiveThreadId(data.threads[0]?.id ?? null);
      }
      if (data.settings) {
        setSettings(prev => ({ ...prev, ...data.settings }));
      }
      // Personas and folders are part of the export, so restore them too.
      if (Array.isArray(data.personas)) {
        setPersonas(prev => {
          const known = new Set(prev.map(p => p.id));
          return [...prev, ...data.personas.filter((p: Persona) => p.isCustom && !known.has(p.id))];
        });
      }
      if (Array.isArray(data.folders)) {
        setFolders(data.folders);
      }
      alert('Backup imported successfully!');
    } catch (e) {
      alert('Invalid backup JSON file format.');
    }
  };

  // What the NETWORK is serving. Displayed, never selected -- the tier controller decides it
  // and it changes as machines join or leave.
  const servingModelName = (network.servingModel || activeThread?.modelId || '')
    .split('/').pop() || 'the network model';
  const activePersona = personas.find(p => p.id === activeThread?.personaId);

  const cycleTheme = () => setTheme(theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light');
  const ThemeIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor;

  return (
    <div className="flex h-screen font-sans overflow-hidden bg-canvas text-ink">
      {/* Sidebar */}
      <Sidebar
        threads={threads}
        activeThreadId={activeThreadId}
        folders={folders}
        wallet={wallet}
        network={network}
        onSignIn={() => { window.location.href = '/login'; }}
        onSelectThread={setActiveThreadId}
        onNewThread={handleNewThread}
        onDeleteThread={handleDeleteThread}
        onPinThread={handlePinThread}
        onOpenPersonas={() => setShowPersonasModal(true)}
        onExportData={handleExportBackup}
        isMobileOpen={isMobileSidebarOpen}
        setIsMobileOpen={setIsMobileSidebarOpen}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between gap-3 h-14 px-3 sm:px-5 border-b border-line bg-surface/80 backdrop-blur-sm flex-shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <button
              onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
              className="lg:hidden p-2 -ml-1 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-2 transition"
              title="Toggle sidebar"
            >
              <Menu className="w-[18px] h-[18px]" />
            </button>
            <div className="min-w-0">
              <h2 className="text-[13px] font-semibold text-ink truncate leading-tight">
                {activeThread?.title || 'New Conversation'}
              </h2>
              {activePersona && (
                <div className="label-mono text-ink-faint truncate leading-tight mt-0.5">
                  {activePersona.avatar} {activePersona.name}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1.5 flex-shrink-0">
            {/* What the network is serving. A STATEMENT, not a control: the tier controller
                picks the model from how many machines are online, bounded by the weakest one,
                so there is nothing here for a user to choose. It upgrades on its own as the
                network grows, which is worth saying out loud rather than hiding. */}
            <div
              className="flex items-center gap-2 h-8 pl-2.5 pr-2.5 rounded-lg border border-line bg-surface-2"
              title={network.reachable
                ? `Serving ${servingModelName}. The network chooses this automatically and upgrades it as more machines join.`
                : 'The coordinator is unreachable, so the serving model is unknown.'}
            >
              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                network.reachable && network.healthy ? 'bg-ok' : 'bg-line-strong'}`} />
              <span className="text-[12px] font-medium text-ink max-w-[9rem] sm:max-w-[14rem] truncate">
                {servingModelName}
              </span>
            </div>

            <button
              onClick={() => setShowPersonasModal(true)}
              className="hidden sm:grid place-items-center w-8 h-8 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-2 transition"
              title="Select persona"
            >
              <UserCheck className="w-[17px] h-[17px]" />
            </button>

            {activeThread && activeThread.messages.length > 0 && (
              <button
                onClick={handleClearConversation}
                className="grid place-items-center w-8 h-8 rounded-lg text-ink-muted hover:text-danger hover:bg-danger-soft transition"
                title="Clear all messages in this conversation"
              >
                <Trash2 className="w-[17px] h-[17px]" />
              </button>
            )}

            <button
              onClick={cycleTheme}
              className="grid place-items-center w-8 h-8 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-2 transition"
              title={`Theme: ${theme} (click to change)`}
            >
              <ThemeIcon className="w-[17px] h-[17px]" />
            </button>
          </div>
        </header>

        {/* Message Thread List */}
        <ChatMessageList
          messages={activeThread?.messages || []}
          isGenerating={isGenerating}
          onRegenerate={handleRegenerate}
          onEditMessage={handleEditMessage}
          onBranchMessage={handleBranchMessage}
          onFeedbackMessage={handleFeedbackMessage}
          speechVoice={settings.speechVoice}
          speechRate={settings.speechRate}
          speechPitch={settings.speechPitch}
        />

        {/* Chat Input Component */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isGenerating={isGenerating}
          onStopGeneration={handleStopGeneration}
          maxTokens={activeThread?.maxTokens ?? settings.defaultMaxTokens ?? 8192}
          setMaxTokens={(val) => updateActiveThread(t => ({ ...t, maxTokens: val }))}
          useRag={activeThread?.useRag ?? false}
          setUseRag={(val) => updateActiveThread(t => ({ ...t, useRag: val }))}
          systemPrompt={activeThread?.systemPrompt ?? ''}
          setSystemPrompt={(val) => updateActiveThread(t => ({ ...t, systemPrompt: val }))}
          blockedReason={blockReason(network)}
          degradedNotice={degradedNotice(network)}
        />
      </main>

      {/* Modals */}

      {showPersonasModal && (
        <PersonasModal
          personas={personas}
          activePersonaId={activeThread?.personaId}
          onSelectPersona={handleSelectPersona}
          onSaveCustomPersona={(newP) => setPersonas(prev => [...prev, newP])}
          onClose={() => setShowPersonasModal(false)}
        />
      )}

    </div>
  );
}

export default function App() {
  // Seeded from storage so the saved theme applies on the first paint instead
  // of flashing the system theme and being written back over the stored value.
  const [initialTheme] = useState(() => loadSettings().theme || 'system');

  return (
    <ThemeProvider defaultTheme={initialTheme}>
      <AppContent />
    </ThemeProvider>
  );
}
