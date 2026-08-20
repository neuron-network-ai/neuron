import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage } from '../types';
import { MarkdownRenderer } from './MarkdownRenderer';
import { TextToSpeech } from '../utils/speech';
import { splitThinking } from '../utils/thinking';
import {
  Volume2,
  VolumeX,
  Copy,
  Check,
  RotateCcw,
  Edit2,
  ChevronDown,
  ChevronUp,
  Brain,
  FileText,
  GitFork,
  ArrowDown,
  ThumbsUp,
  ThumbsDown,
  Terminal,
} from 'lucide-react';

interface ChatMessageListProps {
  messages: ChatMessage[];
  isGenerating: boolean;
  onRegenerate: () => void;
  onEditMessage: (messageId: string, newContent: string) => void;
  onBranchMessage: (messageId: string) => void;
  onFeedbackMessage?: (messageId: string, feedback: 'up' | 'down' | null) => void;
  speechVoice?: string;
  speechRate?: number;
  speechPitch?: number;
}

const formatLatency = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`);

export const ChatMessageList: React.FC<ChatMessageListProps> = ({
  messages,
  isGenerating,
  onRegenerate,
  onEditMessage,
  onBranchMessage,
  onFeedbackMessage,
  speechVoice,
  speechRate = 1.0,
  speechPitch = 1.0,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isUserScrolledUp = useRef<boolean>(false);

  // Read through refs inside stable callbacks so memoised rows keep working.
  const speakingIdRef = useRef<string | null>(null);
  speakingIdRef.current = speakingId;
  const voiceRef = useRef({ speechVoice, speechRate, speechPitch });
  voiceRef.current = { speechVoice, speechRate, speechPitch };

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior, block: 'end' });
    }
  }, []);

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const scrolledUp = scrollHeight - scrollTop - clientHeight > 100;
    isUserScrolledUp.current = scrolledUp;
    setShowScrollBottom(prev => (prev === scrolledUp ? prev : scrolledUp));
  }, []);

  useEffect(() => {
    if (isGenerating) {
      isUserScrolledUp.current = false;
      scrollToBottom('smooth');
    }
  }, [isGenerating, scrollToBottom]);

  const lastMessageContent = messages[messages.length - 1]?.content;
  useEffect(() => {
    if (!isUserScrolledUp.current) {
      scrollToBottom(isGenerating ? 'auto' : 'smooth');
    }
  }, [messages.length, lastMessageContent, isGenerating, scrollToBottom]);

  // Don't leave speech running after the list unmounts.
  useEffect(() => {
    return () => TextToSpeech.stop();
  }, []);

  const handleCopy = useCallback((id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }, []);

  const handleSpeak = useCallback((id: string, text: string) => {
    if (speakingIdRef.current === id) {
      TextToSpeech.stop();
      setSpeakingId(null);
      return;
    }
    const { speechVoice: v, speechRate: r, speechPitch: p } = voiceRef.current;
    setSpeakingId(id);
    TextToSpeech.speak(text, v || undefined, r, p, () => setSpeakingId(null));
  }, []);

  const startEdit = useCallback((id: string, content: string) => {
    setEditingId(id);
    setEditContent(content);
  }, []);

  const cancelEdit = useCallback(() => setEditingId(null), []);

  const saveEdit = useCallback(
    (msgId: string, value: string) => {
      if (value.trim()) onEditMessage(msgId, value);
      setEditingId(null);
    },
    [onEditMessage]
  );

  const toggleThinking = useCallback((id: string) => {
    setExpandedThinking(prev => ({ ...prev, [id]: !prev[id] }));
  }, []);

  const handleFeedback = useCallback(
    (id: string, feedback: 'up' | 'down' | null) => onFeedbackMessage?.(id, feedback),
    [onFeedbackMessage]
  );

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto grid place-items-center p-8">
        <div className="max-w-md text-center rise-in">
          <div className="w-11 h-11 rounded-xl bg-accent-soft border border-accent-line grid place-items-center mx-auto mb-4">
            <Terminal className="w-5 h-5 text-accent" />
          </div>
          <h2 className="text-[17px] font-semibold tracking-tight text-ink mb-2">
            Ask anything
          </h2>
          <p className="text-[13px] leading-relaxed text-ink-muted">
            Answers run on this machine when it can hold the model, and across the network of
            volunteer machines when it cannot. Nothing to configure and nothing to install —
            pick a persona to set the tone, then start typing.
          </p>
          <div className="flex items-center justify-center gap-2 mt-5 label-mono text-ink-faint">
            <span className="px-2 py-1 rounded-md bg-surface-2 border border-line">Enter to send</span>
            <span className="px-2 py-1 rounded-md bg-surface-2 border border-line">Shift+Enter for newline</span>
          </div>
        </div>
      </div>
    );
  }

  const lastIndex = messages.length - 1;

  return (
    <div ref={containerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto relative">
      <div className="max-w-3xl w-full mx-auto px-4 sm:px-8 py-8 space-y-7">
        {messages.map((msg, index) => {
          const isLastAssistant = msg.role !== 'user' && index === lastIndex;
          const isEditing = editingId === msg.id;
          return (
            <MessageRow
              key={msg.id}
              msg={msg}
              isLastAssistant={isLastAssistant}
              isStreaming={isLastAssistant && isGenerating}
              showRetry={isLastAssistant && !isGenerating}
              isCopied={copiedId === msg.id}
              isSpeaking={speakingId === msg.id}
              isEditing={isEditing}
              editValue={isEditing ? editContent : ''}
              thinkingExpanded={!!expandedThinking[msg.id]}
              onEditValueChange={setEditContent}
              onCopy={handleCopy}
              onSpeak={handleSpeak}
              onStartEdit={startEdit}
              onCancelEdit={cancelEdit}
              onSaveEdit={saveEdit}
              onToggleThinking={toggleThinking}
              onBranch={onBranchMessage}
              onFeedback={handleFeedback}
              onRegenerate={onRegenerate}
            />
          );
        })}

        {/* Waiting indicator, shown only before the first token lands */}
        {isGenerating && !messages[lastIndex]?.content && (
          <div className="flex items-center gap-2 label-mono text-ink-faint">
            <span className="flex gap-1">
              <Dot delay="0ms" />
              <Dot delay="150ms" />
              <Dot delay="300ms" />
            </span>
            <span>Thinking</span>
          </div>
        )}

        <div ref={messagesEndRef} className="h-1" />
      </div>

      {showScrollBottom && (
        <button
          onClick={() => {
            isUserScrolledUp.current = false;
            scrollToBottom('smooth');
          }}
          className="sticky bottom-5 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1.5 h-8 px-3 rounded-full bg-surface border border-line shadow-[var(--shadow-lg)] text-[12px] font-medium text-ink hover:border-accent-line transition"
          title="Jump to latest"
        >
          <ArrowDown className="w-3.5 h-3.5 text-accent" />
          <span>Latest</span>
        </button>
      )}
    </div>
  );
};

interface MessageRowProps {
  msg: ChatMessage;
  isLastAssistant: boolean;
  isStreaming: boolean;
  showRetry: boolean;
  isCopied: boolean;
  isSpeaking: boolean;
  isEditing: boolean;
  editValue: string;
  thinkingExpanded: boolean;
  onEditValueChange: (value: string) => void;
  onCopy: (id: string, text: string) => void;
  onSpeak: (id: string, text: string) => void;
  onStartEdit: (id: string, content: string) => void;
  onCancelEdit: () => void;
  onSaveEdit: (id: string, value: string) => void;
  onToggleThinking: (id: string) => void;
  onBranch: (id: string) => void;
  onFeedback: (id: string, feedback: 'up' | 'down' | null) => void;
  onRegenerate: () => void;
}

/**
 * One message. Wrapped in React.memo with every callback stable upstream, so a
 * token arriving in the newest reply re-renders only that reply — not the whole
 * conversation. This is what keeps long chats fast.
 */
const MessageRow = React.memo<MessageRowProps>(function MessageRow({
  msg,
  isStreaming,
  showRetry,
  isCopied,
  isSpeaking,
  isEditing,
  editValue,
  thinkingExpanded,
  onEditValueChange,
  onCopy,
  onSpeak,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onToggleThinking,
  onBranch,
  onFeedback,
  onRegenerate,
}) {
  const isUser = msg.role === 'user';

  // Reasoning models emit <think>…</think> inline. Split it out for display
  // only — msg.content keeps the raw text that gets replayed as history.
  const { thinking, answer, thinkingOpen } = isUser
    ? { thinking: '', answer: msg.content, thinkingOpen: false }
    : splitThinking(msg.content);
  const reasoning = thinking || msg.thinkingProcess || '';

  return (
    <article className="group rise-in">
      {/* Byline */}
      <div className={`flex items-center gap-2 mb-1.5 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {isUser ? (
          <span className="label-mono text-ink-faint">You</span>
        ) : (
          <>
            <span className="label-mono text-ink-muted font-semibold">{msg.model || 'Assistant'}</span>
            {msg.neuron && (
              <span className="label-mono text-ink-faint">
                · {msg.neuron.local ? 'this machine' : `${msg.neuron.nodes} node${msg.neuron.nodes === 1 ? '' : 's'}`}
                {msg.neuron.reroutes ? ` · recovered from ${msg.neuron.reroutes} node drop${msg.neuron.reroutes === 1 ? '' : 's'}` : ''}
              </span>
            )}
          </>
        )}
      </div>

      {/* Reasoning trace */}
      {reasoning && (
        <div className="mb-2 rounded-lg border border-line bg-surface-2 overflow-hidden">
          <button
            onClick={() => onToggleThinking(msg.id)}
            className="w-full flex items-center justify-between px-3 py-2 text-[12px] font-medium text-ink-muted hover:text-ink transition"
          >
            <span className="flex items-center gap-1.5">
              <Brain className={`w-3.5 h-3.5 text-accent ${thinkingOpen ? 'animate-pulse' : ''}`} />
              {thinkingOpen ? 'Thinking…' : 'Reasoning'}
              <span className="label-mono text-ink-faint">
                {reasoning.length > 900 ? `${Math.round(reasoning.length / 4)} tok` : ''}
              </span>
            </span>
            {thinkingExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
          {thinkingExpanded && (
            <div className="px-3 py-2.5 border-t border-line font-mono text-[11.5px] leading-relaxed text-ink-muted whitespace-pre-wrap max-h-80 overflow-y-auto">
              {reasoning}
            </div>
          )}
        </div>
      )}

      {/* Attachments */}
      {msg.attachments && msg.attachments.length > 0 && (
        <div className={`flex flex-wrap gap-1.5 mb-2 ${isUser ? 'justify-end' : ''}`}>
          {msg.attachments.map(att => (
            <div
              key={att.id}
              className="flex items-center gap-1.5 pl-1.5 pr-2.5 py-1 rounded-lg bg-surface-2 border border-line text-[11.5px] text-ink-muted"
            >
              {att.type === 'image' ? (
                <img src={att.data} alt={att.name} className="w-5 h-5 rounded object-cover flex-shrink-0" />
              ) : (
                <FileText className="w-3.5 h-3.5 text-accent flex-shrink-0" />
              )}
              <span className="truncate max-w-[140px]">{att.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* Body */}
      {isEditing ? (
        <div className="space-y-2">
          <textarea
            value={editValue}
            onChange={e => onEditValueChange(e.target.value)}
            autoFocus
            className="w-full p-3 rounded-xl bg-surface border border-accent-line text-[14px] text-ink resize-none min-h-[90px] focus:outline-none focus:border-accent transition"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={onCancelEdit}
              className="h-7 px-3 rounded-lg text-[12px] font-medium text-ink-muted hover:text-ink hover:bg-surface-2 transition"
            >
              Cancel
            </button>
            <button
              onClick={() => onSaveEdit(msg.id, editValue)}
              className="h-7 px-3 rounded-lg bg-accent hover:bg-accent-hover text-accent-ink text-[12px] font-semibold transition"
            >
              Save &amp; resend
            </button>
          </div>
        </div>
      ) : isUser ? (
        <div className="flex justify-end">
          <div className="max-w-[85%] px-3.5 py-2.5 rounded-2xl rounded-tr-md bg-surface-2 border border-line text-[14px] leading-relaxed text-ink whitespace-pre-wrap">
            {msg.content}
          </div>
        </div>
      ) : msg.error ? (
        <div className="px-3.5 py-3 rounded-xl bg-danger-soft border border-danger/25 text-[13.5px] leading-relaxed text-danger whitespace-pre-wrap">
          {msg.content}
        </div>
      ) : (
        <div className="text-[14.5px] leading-[1.7]">
          <MarkdownRenderer content={answer} />
          {isStreaming && (
            <span className="inline-block w-[7px] h-[15px] -mb-[2px] ml-0.5 bg-accent animate-pulse" />
          )}
        </div>
      )}

      <Metrics msg={msg} />

      {/* Actions */}
      {!isEditing && (
        <div
          className={`flex items-center gap-0.5 mt-1.5 transition-opacity ${
            msg.feedback ? 'opacity-100' : 'opacity-0 group-hover:opacity-100 focus-within:opacity-100'
          } ${isUser ? 'justify-end' : 'justify-start'}`}
        >
          <IconAction title="Copy" onClick={() => onCopy(msg.id, msg.content)} active={isCopied}>
            {isCopied ? <Check className="w-3.5 h-3.5 text-ok" /> : <Copy className="w-3.5 h-3.5" />}
          </IconAction>

          <IconAction
            title={isSpeaking ? 'Stop reading' : 'Read aloud'}
            onClick={() => onSpeak(msg.id, msg.content)}
            active={isSpeaking}
          >
            {isSpeaking ? <VolumeX className="w-3.5 h-3.5 text-accent" /> : <Volume2 className="w-3.5 h-3.5" />}
          </IconAction>

          {isUser && (
            <IconAction title="Edit and resend" onClick={() => onStartEdit(msg.id, msg.content)}>
              <Edit2 className="w-3.5 h-3.5" />
            </IconAction>
          )}

          {!isUser && (
            <>
              <IconAction title="Branch a new chat from here" onClick={() => onBranch(msg.id)}>
                <GitFork className="w-3.5 h-3.5" />
              </IconAction>

              <span className="w-px h-4 bg-line mx-1" />

              <IconAction
                title={msg.feedback === 'up' ? 'Remove rating' : 'Good response'}
                onClick={() => onFeedback(msg.id, msg.feedback === 'up' ? null : 'up')}
                active={msg.feedback === 'up'}
              >
                <ThumbsUp className={`w-3.5 h-3.5 ${msg.feedback === 'up' ? 'text-ok fill-current' : ''}`} />
              </IconAction>
              <IconAction
                title={msg.feedback === 'down' ? 'Remove rating' : 'Poor response'}
                onClick={() => onFeedback(msg.id, msg.feedback === 'down' ? null : 'down')}
                active={msg.feedback === 'down'}
              >
                <ThumbsDown className={`w-3.5 h-3.5 ${msg.feedback === 'down' ? 'text-danger fill-current' : ''}`} />
              </IconAction>
            </>
          )}

          {showRetry && (
            <button
              onClick={onRegenerate}
              className="ml-1 h-6 px-2 flex items-center gap-1 rounded-md text-[11px] font-medium text-ink-muted hover:text-ink hover:bg-surface-2 transition"
              title="Regenerate"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          )}
        </div>
      )}
    </article>
  );
});

const Metrics: React.FC<{ msg: ChatMessage }> = ({ msg }) => {
  if (msg.role === 'user' || msg.error || !msg.content) return null;

  const compTokens =
    msg.completionTokens || msg.tokensEstimate || Math.max(1, Math.ceil((msg.content || '').length / 4));
  const promptTokens = msg.promptTokens;
  const latency = msg.latencyMs;
  // Reported by the server on `meta`, never guessed. A machine that can hold the serving
  // model answers locally BY DESIGN, and that is not a lesser answer -- it is a free one.
  const isLocal = msg.neuron?.local ?? false;

  // Prefer the DECODE rate the server measured. Deriving it here -- which is what this line
  // used to do -- divides output tokens by the WHOLE request, prefill included, so a short
  // answer to a long prompt reads as a slow network when nothing is wrong: 16 tokens showed
  // 1.16 tok/s where 119 showed 2.18 over the same chain, with the hop measuring 49.7 ms
  // against 49.5 either side. The fallback stays for an engine or a build that sends neither.
  // See stream_timing.py, and note that ui/static/chat.html carries the same rule -- these two
  // pages have diverged before ([P53]).
  const decodeRate = msg.neuron?.decodeTokPerS;
  const ttftMs = msg.neuron?.ttftMs;
  const tokensPerSec =
    decodeRate != null
      ? decodeRate.toFixed(1)
      : latency && latency > 0 && compTokens > 0
        ? (compTokens / (latency / 1000)).toFixed(1)
        : null;
  const rateTitle =
    decodeRate != null
      ? ttftMs != null && ttftMs > 0
        ? `Steady speed once the reply started. Reading the prompt took ${(ttftMs / 1000).toFixed(1)}s before the first word.`
        : 'Steady speed once the reply started.'
      : 'Output tokens over the whole request, including reading the prompt.';

  // What the answer ACTUALLY cost, as the coordinator settled it -- never a client-side
  // estimate. This block used to price the reply against Gemini's per-token rates and render
  // the result as "$0.00012", which was wrong twice over: the arithmetic described a service
  // NEURON does not use, and the dollar sign asserted a cash value NRN does not have. The
  // installer's own disclosure, the landing page and the footer of this app all state that
  // plainly; a currency symbol in the one place a user looks after every answer contradicts
  // all three.
  //
  // 0 NRN is not "unknown": a locally-served answer is genuinely free, because nobody else's
  // hardware ran it and there is nothing to settle.
  const cost = msg.neuron?.costNrn;
  const costStr = isLocal
    ? 'this machine · free'
    : cost == null
      ? 'cost pending'
      : cost === 0
        ? 'free'
        : `${cost.toFixed(4).replace(/\.?0+$/, '')} NRN`;

  const stats = [
    `${compTokens} tok${promptTokens ? ` · ${promptTokens + compTokens} total` : ''}`,
    latency !== undefined && latency > 0 ? formatLatency(latency) : null,
    tokensPerSec ? `${tokensPerSec} tok/s` : null,
    costStr,
  ].filter(Boolean) as string[];

  return (
    <div
      title={rateTitle}
      className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-2.5 font-mono text-[10.5px] text-ink-faint"
    >
      {stats.map((s, i) => (
        <React.Fragment key={i}>
          {i > 0 && <span className="text-line-strong select-none">/</span>}
          <span>{s}</span>
        </React.Fragment>
      ))}
    </div>
  );
};

const IconAction: React.FC<{
  title: string;
  onClick: () => void;
  active?: boolean;
  children: React.ReactNode;
}> = ({ title, onClick, active, children }) => (
  <button
    onClick={onClick}
    title={title}
    className={`grid place-items-center w-6.5 h-6.5 p-1 rounded-md transition ${
      active ? 'bg-surface-2' : 'text-ink-faint hover:text-ink hover:bg-surface-2'
    }`}
  >
    {children}
  </button>
);

const Dot: React.FC<{ delay: string }> = ({ delay }) => (
  <span
    className="w-1 h-1 rounded-full bg-accent animate-bounce"
    style={{ animationDelay: delay, animationDuration: '900ms' }}
  />
);
