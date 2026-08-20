export type Theme = 'light' | 'dark' | 'system';

/**
 * There is no ProviderType any more.
 *
 * The app that became this one could talk to Gemini, Ollama, LM Studio and KoboldCPP, so a
 * provider had to be carried on every thread, message and setting, discovered by probing base
 * URLs, and chosen by the user. NEURON has exactly one backend, and the MODEL is chosen by the
 * network's tier controller — bounded by its weakest member, not by whoever is typing. So
 * `modelId` is something to DISPLAY, never to select, and the whole abstraction collapses.
 */

export interface Attachment {
  id: string;
  name: string;
  type: 'image' | 'text' | 'file';
  mimeType: string;
  data: string; // Base64 or plain text content
  size: number;
}

/** What a finished answer cost and who produced it. Straight off the `done`/`meta` events. */
export interface AnswerMeta {
  /** Machines in the chain. 0 with `local` true means this machine answered alone. */
  nodes: number;
  /** This machine served it. Answering locally is BY DESIGN when it can hold the model. */
  local: boolean;
  costNrn: number | null;
  tokPerS?: number;
  latencyMs?: number;
  /** Time to the first token -- setup, handshake and reading the prompt. */
  ttftMs?: number;
  /** Steady rate after the first token: the figure that actually describes the network. */
  decodeTokPerS?: number;
  /** >0 means a node dropped and the answer was rebuilt. Recorded because a recovered answer
   *  is still a degraded one — and it must never be silent. */
  reroutes?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  model?: string;
  attachments?: Attachment[];
  thinkingProcess?: string;
  latencyMs?: number;
  tokensEstimate?: number;
  promptTokens?: number;
  completionTokens?: number;
  /** Set when the answer ended in an error. The TEXT IS STILL KEPT: a partial answer the user
   *  has already been given, and charged for, must survive whatever ended it. */
  error?: boolean;
  errorDetail?: string;
  neuron?: AnswerMeta;
  feedback?: 'up' | 'down' | null;
}

export interface ChatThread {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  /** The model the NETWORK was serving for this thread. Recorded, not chosen. */
  modelId: string;
  systemPrompt: string;
  maxTokens: number;
  /** Server-side conversation id, so history survives a reload (`/conversations`). */
  conversationId?: string | null;
  /** Answer with retrieved web context (`use_rag` on /chat). */
  useRag?: boolean;
  pinned?: boolean;
  folderId?: string | null;
  personaId?: string;
}

export interface Persona {
  id: string;
  name: string;
  description: string;
  avatar: string;
  systemPrompt: string;
  category: 'coding' | 'writing' | 'reasoning' | 'assistant' | 'productivity' | 'custom';
  isCustom?: boolean;
}

export interface Folder {
  id: string;
  name: string;
  color?: string;
}

/**
 * What is left to configure once there is no provider, no base URL and no model picker.
 *
 * Sampling knobs are gone too: /chat takes `prompt`, `max_tokens`, `use_rag` and
 * `conversation_id`, so temperature and top-p had nowhere to go and would have been controls
 * that visibly did nothing.
 */
export interface Settings {
  defaultMaxTokens: number;
  defaultSystemPrompt: string;
  useRag: boolean;
  autoSpeechOutput: boolean;
  speechVoice: string;
  speechRate: number;
  speechPitch: number;
  theme: Theme;
}
