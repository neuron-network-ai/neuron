import React, { useState, useRef, useEffect } from 'react';
import { Attachment } from '../types';
import { SpeechRecognitionService } from '../utils/speech';
import {
  ArrowUp,
  Square,
  Paperclip,
  Mic,
  Sliders,
  X,
  FileText,
  Image as ImageIcon,
  Globe,
} from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (text: string, attachments: Attachment[]) => void;
  isGenerating: boolean;
  onStopGeneration: () => void;
  maxTokens: number;
  setMaxTokens: (val: number) => void;
  /** Retrieve current web context before answering (`use_rag` on /chat). */
  useRag: boolean;
  setUseRag: (val: boolean) => void;
  systemPrompt: string;
  setSystemPrompt: (val: string) => void;
  /**
   * Why sending is refused, or null. A prompt sent into an incomplete chain cannot be
   * answered, so it costs the user a wait and then an error when the page already knew.
   * Computed by `blockReason` — which deliberately does NOT block on a failed status poll.
   */
  blockedReason?: string | null;
  /** The degraded-network explanation, naming the layers nobody is serving. */
  degradedNotice?: string | null;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isGenerating,
  onStopGeneration,
  maxTokens,
  setMaxTokens,
  useRag,
  setUseRag,
  systemPrompt,
  setSystemPrompt,
  blockedReason = null,
  degradedNotice = null,
}) => {
  const [inputText, setInputText] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [showParams, setShowParams] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sttRef = useRef<SpeechRecognitionService | null>(null);

  // Stop the microphone if the input unmounts mid-recording.
  useEffect(() => {
    return () => sttRef.current?.stop();
  }, []);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputText]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if ((!inputText.trim() && attachments.length === 0) || isGenerating) return;

    onSendMessage(inputText.trim(), attachments);
    setInputText('');
    setAttachments([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file: File) => {
      const reader = new FileReader();
      const isImage = file.type.startsWith('image/');

      const makeAttachment = (type: 'image' | 'text'): Attachment => ({
        id: 'att_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6),
        name: file.name,
        type,
        mimeType: file.type || (type === 'text' ? 'text/plain' : 'image/png'),
        data: reader.result as string,
        size: file.size,
      });

      if (isImage) {
        reader.readAsDataURL(file);
        reader.onload = () => setAttachments(prev => [...prev, makeAttachment('image')]);
      } else {
        reader.readAsText(file);
        reader.onload = () => setAttachments(prev => [...prev, makeAttachment('text')]);
      }
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  const toggleMic = () => {
    if (isRecording) {
      if (sttRef.current) sttRef.current.stop();
      setIsRecording(false);
    } else {
      if (!SpeechRecognitionService.isSupported()) {
        alert('Speech recognition is not supported in this browser.');
        return;
      }
      setIsRecording(true);
      sttRef.current = new SpeechRecognitionService(
        transcript => {
          setInputText(prev => prev + (prev ? ' ' : '') + transcript);
        },
        err => {
          console.error('Speech recognition error:', err);
          setIsRecording(false);
        },
        () => {
          setIsRecording(false);
        }
      );
      sttRef.current.start();
    }
  };

  // `!blockedReason` is the term this was missing. Note it is only consulted here, on the SEND
  // button: while `isGenerating` the control is Stop, and taking that away would strand a
  // running generation with no way to cancel it — the same reasoning as chat.html's
  // `disabled = !!blockedReason && !busy`.
  const canSend = (inputText.trim() || attachments.length > 0) && !isGenerating && !blockedReason;

  return (
    <footer className="flex-shrink-0 px-4 sm:px-8 pb-5 pt-2 bg-canvas">
      <div className="max-w-3xl mx-auto">
        {/* Say WHY before the user types, not after they send. role=status rather than alert:
            it re-renders on every network poll, and an assertive region would interrupt a
            screen reader each time. */}
        {(degradedNotice || blockedReason) && (
          <div
            role="status"
            className="mb-2 px-3 py-2 rounded-lg bg-warn-soft text-[12.5px] text-warn text-center"
          >
            {degradedNotice || blockedReason}
          </div>
        )}
        {/* Parameters drawer */}
        {showParams && (
          <div className="mb-2 p-3.5 rounded-xl border border-line bg-surface rise-in">
            <div className="flex items-center justify-between mb-3">
              <span className="label-mono text-ink-faint">Answer settings</span>
              <button
                onClick={() => setShowParams(false)}
                className="grid place-items-center w-5 h-5 rounded text-ink-faint hover:text-ink transition"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-3">
              {/* Web search. `use_rag` on /chat -- rag/retriever.py fetches current context and
                  prepends it, so this is a real capability, not decoration. It is the only
                  per-answer switch NEURON actually honours: temperature, top-p and the rest
                  were removed because /chat takes prompt, max_tokens, use_rag and
                  conversation_id, and a control that moves a number nothing reads is worse
                  than no control at all. */}
              <label className="block sm:col-span-2 cursor-pointer">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[12px] font-medium text-ink flex items-center gap-1.5">
                    <Globe className="w-3.5 h-3.5 text-ink-faint" />
                    Web search
                  </span>
                  <input
                    type="checkbox"
                    checked={useRag}
                    onChange={e => setUseRag(e.target.checked)}
                    className="w-4 h-4 accent-[var(--accent)] cursor-pointer"
                  />
                </div>
                <p className="text-[11px] text-ink-muted mt-1">
                  Look up current information before answering. Slower, and the sources are
                  listed with the reply.
                </p>
              </label>

              <label className="block sm:col-span-2">
                <div className="flex items-baseline justify-between mb-1.5">
                  <span className="text-[12px] font-medium text-ink">Max output tokens</span>
                  <span className="font-mono text-[11px] text-ink-muted">
                    {maxTokens.toLocaleString()}
                    {maxTokens <= 2048 ? ' · may truncate long answers' : ''}
                  </span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="32768"
                  step="512"
                  value={maxTokens}
                  onChange={e => setMaxTokens(parseInt(e.target.value, 10))}
                  className="w-full cursor-pointer"
                />
              </label>
            </div>

            <label className="block">
              <span className="text-[12px] font-medium text-ink">System instruction</span>
              <textarea
                value={systemPrompt}
                onChange={e => setSystemPrompt(e.target.value)}
                placeholder="You are a helpful software engineering assistant…"
                className="mt-1.5 w-full h-16 p-2.5 rounded-lg bg-surface-2 border border-line text-[12.5px] text-ink placeholder-ink-faint resize-none focus:outline-none focus:border-accent-line transition"
              />
            </label>
          </div>
        )}

        {/* Attachments */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2">
            {attachments.map(att => (
              <div
                key={att.id}
                className="group/att flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-lg bg-surface border border-line text-[11.5px] text-ink-muted"
              >
                {att.type === 'image' ? (
                  <ImageIcon className="w-3.5 h-3.5 text-accent" />
                ) : (
                  <FileText className="w-3.5 h-3.5 text-accent" />
                )}
                <span className="truncate max-w-[150px]">{att.name}</span>
                <button
                  onClick={() => removeAttachment(att.id)}
                  className="grid place-items-center w-4 h-4 rounded text-ink-faint hover:text-danger transition"
                  title="Remove"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Composer */}
        <form
          onSubmit={handleSubmit}
          className="relative rounded-2xl border border-line bg-surface shadow-[var(--shadow-sm)] focus-within:border-accent-line transition"
        >
          <textarea
            ref={textareaRef}
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message…"
            rows={1}
            className="w-full px-4 pt-3.5 pb-1 bg-transparent text-[14px] leading-relaxed text-ink placeholder-ink-faint resize-none focus:outline-none max-h-[200px] min-h-[44px]"
          />

          <div className="flex items-center justify-between px-2 pb-2">
            <div className="flex items-center gap-0.5">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="grid place-items-center w-8 h-8 rounded-lg text-ink-faint hover:text-ink hover:bg-surface-2 transition"
                title="Attach file or image"
              >
                <Paperclip className="w-4 h-4" />
              </button>

              <button
                type="button"
                onClick={toggleMic}
                className={`grid place-items-center w-8 h-8 rounded-lg transition ${
                  isRecording
                    ? 'text-danger bg-danger-soft'
                    : 'text-ink-faint hover:text-ink hover:bg-surface-2'
                }`}
                title={isRecording ? 'Stop recording' : 'Voice input'}
              >
                <Mic className={`w-4 h-4 ${isRecording ? 'animate-pulse' : ''}`} />
              </button>

              <button
                type="button"
                onClick={() => setShowParams(!showParams)}
                className={`grid place-items-center w-8 h-8 rounded-lg transition ${
                  showParams
                    ? 'text-accent bg-accent-soft'
                    : 'text-ink-faint hover:text-ink hover:bg-surface-2'
                }`}
                title="Answer settings"
              >
                <Sliders className="w-4 h-4" />
              </button>

              <span className="hidden sm:inline label-mono text-ink-faint ml-1.5">
                {isRecording ? 'listening…' : 'enter to send'}
              </span>
            </div>

            {isGenerating ? (
              <button
                type="button"
                onClick={onStopGeneration}
                className="flex items-center gap-1.5 h-8 px-3 rounded-lg border border-line bg-surface-2 text-[12px] font-semibold text-ink hover:border-danger hover:text-danger transition"
              >
                <Square className="w-3 h-3 fill-current" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!canSend}
                className="grid place-items-center w-8 h-8 rounded-lg bg-accent text-accent-ink transition hover:bg-accent-hover disabled:bg-surface-3 disabled:text-ink-faint disabled:cursor-not-allowed"
                title="Send"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>
      </div>
    </footer>
  );
};
