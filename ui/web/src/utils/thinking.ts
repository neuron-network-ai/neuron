/**
 * Reasoning models (Qwen3.x, DeepSeek-R1, …) wrap their chain of thought in
 * <think>…</think> before the real answer.
 *
 * Important: this splits for DISPLAY ONLY. The stored message keeps the raw
 * model output, and that raw text is what gets sent back as history. Recurrent
 * models like Qwen3.6-35B-A3B reuse a cached prefix only when the re-rendered
 * history matches byte-for-byte — stripping the think block on the way back out
 * would invalidate the cache and force a full re-read of the conversation.
 */
export interface SplitThinking {
  /** Chain-of-thought text, if any. */
  thinking: string;
  /** The answer with the think block removed. */
  answer: string;
  /** True while a think block is open and not yet closed (mid-stream). */
  thinkingOpen: boolean;
}

const OPEN = '<think>';
const CLOSE = '</think>';

export function splitThinking(content: string): SplitThinking {
  if (!content || content.indexOf(OPEN) === -1) {
    return { thinking: '', answer: content || '', thinkingOpen: false };
  }

  const thinkingParts: string[] = [];
  let answer = '';
  let rest = content;
  let open = false;

  while (rest.length > 0) {
    const start = rest.indexOf(OPEN);
    if (start === -1) {
      answer += rest;
      break;
    }

    answer += rest.slice(0, start);
    const afterOpen = rest.slice(start + OPEN.length);
    const end = afterOpen.indexOf(CLOSE);

    if (end === -1) {
      // Still streaming inside the block: everything left is reasoning.
      thinkingParts.push(afterOpen);
      open = true;
      break;
    }

    thinkingParts.push(afterOpen.slice(0, end));
    rest = afterOpen.slice(end + CLOSE.length);
  }

  return {
    thinking: thinkingParts.join('\n').trim(),
    // Models usually emit a blank line after </think>; drop the leading gap.
    answer: answer.replace(/^\s*\n/, '').trimStart(),
    thinkingOpen: open,
  };
}
