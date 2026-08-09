export class TextToSpeech {
  private static synth: SpeechSynthesis | null = typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null;

  static getVoices(): SpeechSynthesisVoice[] {
    if (!this.synth) return [];
    return this.synth.getVoices();
  }

  /**
   * Voices load asynchronously in most browsers, so a plain getVoices() call at
   * render time usually returns an empty list. Fires immediately with whatever
   * is available, then again once the list is populated. Returns an unsubscribe.
   */
  static onVoicesChanged(callback: (voices: SpeechSynthesisVoice[]) => void): () => void {
    const synth = this.synth;
    if (!synth) {
      callback([]);
      return () => {};
    }

    const handler = () => callback(synth.getVoices());
    handler();
    synth.addEventListener('voiceschanged', handler);
    return () => synth.removeEventListener('voiceschanged', handler);
  }

  static speak(text: string, voiceName?: string, rate: number = 1.0, pitch: number = 1.0, onEnd?: () => void) {
    if (!this.synth) return;
    this.stop();

    // Clean text of markdown syntax before reading out
    const cleanText = text
      .replace(/```[\s\S]*?```/g, 'Code block omitted.')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/[*_~#]/g, '');

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = rate;
    utterance.pitch = pitch;

    if (voiceName) {
      const voices = this.getVoices();
      const match = voices.find(v => v.name === voiceName);
      if (match) utterance.voice = match;
    }

    if (onEnd) utterance.onend = onEnd;

    this.synth.speak(utterance);
  }

  static stop() {
    if (this.synth) {
      this.synth.cancel();
    }
  }

  static isSupported(): boolean {
    return typeof window !== 'undefined' && 'speechSynthesis' in window;
  }
}

export class SpeechRecognitionService {
  private recognition: any = null;

  constructor(onResult: (text: string) => void, onError?: (err: any) => void, onEnd?: () => void) {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      // Interim results fire repeatedly for the same phrase; emitting them would
      // append the same words over and over to the input box.
      this.recognition.interimResults = false;

      this.recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) {
            transcript += result[0].transcript;
          }
        }
        if (transcript.trim()) {
          onResult(transcript.trim());
        }
      };

      if (onError) this.recognition.onerror = onError;
      if (onEnd) this.recognition.onend = onEnd;
    }
  }

  start() {
    if (this.recognition) {
      try {
        this.recognition.start();
      } catch (e) {
        // Already started
      }
    }
  }

  stop() {
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch (e) {}
    }
  }

  static isSupported(): boolean {
    return typeof window !== 'undefined' && (!!(window as any).SpeechRecognition || !!(window as any).webkitSpeechRecognition);
  }
}
