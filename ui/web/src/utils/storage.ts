import { ChatThread, Settings, Persona, Folder } from '../types';
import { DEFAULT_PERSONAS } from '../data/personas';

const SETTINGS_KEY = 'neuron_chat_settings_v1';
const THREADS_KEY = 'neuron_chat_threads_v1';
const PERSONAS_KEY = 'neuron_chat_personas_v1';
const FOLDERS_KEY = 'neuron_chat_folders_v1';

export const DEFAULT_SETTINGS: Settings = {
  // /chat caps at this; the server clamps too. 8192 because reasoning models spend a large
  // share of their budget before writing anything, and 2048 truncated answers mid-sentence.
  defaultMaxTokens: 8192,
  defaultSystemPrompt: 'You are a helpful AI assistant.',
  useRag: false,
  autoSpeechOutput: false,
  speechVoice: '',
  speechRate: 1.0,
  speechPitch: 1.0,
  theme: 'system',
};

export const loadSettings = (): Settings => {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch (err) {
    console.error('Failed to load settings from storage', err);
    return DEFAULT_SETTINGS;
  }
};

export const saveSettings = (settings: Settings): void => {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (err) {
    console.error('Failed to save settings to storage', err);
  }
};

export const loadThreads = (): ChatThread[] => {
  try {
    const raw = localStorage.getItem(THREADS_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch (err) {
    console.error('Failed to load threads from storage', err);
    return [];
  }
};

export const saveThreads = (threads: ChatThread[]): void => {
  try {
    localStorage.setItem(THREADS_KEY, JSON.stringify(threads));
  } catch (err) {
    console.error('Failed to save threads to storage', err);
  }
};

export const loadPersonas = (): Persona[] => {
  try {
    const raw = localStorage.getItem(PERSONAS_KEY);
    if (!raw) return DEFAULT_PERSONAS;
    const custom: Persona[] = JSON.parse(raw);
    return [...DEFAULT_PERSONAS, ...custom.filter(p => p.isCustom)];
  } catch (err) {
    return DEFAULT_PERSONAS;
  }
};

export const saveCustomPersonas = (personas: Persona[]): void => {
  try {
    const customOnly = personas.filter(p => p.isCustom);
    localStorage.setItem(PERSONAS_KEY, JSON.stringify(customOnly));
  } catch (err) {
    console.error('Failed to save custom personas', err);
  }
};

export const loadFolders = (): Folder[] => {
  try {
    const raw = localStorage.getItem(FOLDERS_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch (err) {
    return [];
  }
};

export const saveFolders = (folders: Folder[]): void => {
  try {
    localStorage.setItem(FOLDERS_KEY, JSON.stringify(folders));
  } catch (err) {
    console.error('Failed to save folders', err);
  }
};
