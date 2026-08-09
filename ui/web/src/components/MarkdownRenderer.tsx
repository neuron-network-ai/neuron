import React, { useMemo } from 'react';
import { CodeBlock } from './CodeBlock';

interface MarkdownRendererProps {
  content: string;
}

const MarkdownRendererImpl: React.FC<MarkdownRendererProps> = ({ content }) => {
  // Split content into code blocks and normal text. Memoised because a
  // streaming reply re-renders continuously and this walks the whole string.
  const parts = useMemo(() => {
    const acc: Array<{ type: 'code' | 'markdown'; content: string; language?: string }> = [];
    if (!content) return acc;

    const codeBlockRegex = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        acc.push({ type: 'markdown', content: content.slice(lastIndex, match.index) });
      }
      acc.push({ type: 'code', language: match[1] || 'text', content: match[2] });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      acc.push({ type: 'markdown', content: content.slice(lastIndex) });
    }
    return acc;
  }, [content]);

  if (!content) return null;

  return (
    <div className="space-y-1">
      {parts.map((part, index) => {
        if (part.type === 'code') {
          return <CodeBlock key={index} language={part.language || 'text'} code={part.content} />;
        }

        const lines = part.content.split('\n');
        return (
          <div key={index}>
            {lines.map((line, lIdx) => {
              const trimmed = line.trim();

              if (trimmed.startsWith('### ')) {
                return (
                  <h3 key={lIdx} className="text-[15px] font-semibold text-ink mt-4 mb-1.5">
                    {renderInline(trimmed.slice(4))}
                  </h3>
                );
              }
              if (trimmed.startsWith('## ')) {
                return (
                  <h2 key={lIdx} className="text-[16.5px] font-semibold tracking-tight text-ink mt-5 mb-2">
                    {renderInline(trimmed.slice(3))}
                  </h2>
                );
              }
              if (trimmed.startsWith('# ')) {
                return (
                  <h1
                    key={lIdx}
                    className="text-[18px] font-bold tracking-tight text-ink mt-5 mb-2 pb-1.5 border-b border-line"
                  >
                    {renderInline(trimmed.slice(2))}
                  </h1>
                );
              }

              // Horizontal rule
              if (/^(---|\*\*\*|___)$/.test(trimmed)) {
                return <hr key={lIdx} className="my-4 border-line" />;
              }

              if (trimmed.startsWith('> ')) {
                return (
                  <blockquote
                    key={lIdx}
                    className="my-2 pl-3 border-l-2 border-accent-line text-ink-muted italic"
                  >
                    {renderInline(trimmed.slice(2))}
                  </blockquote>
                );
              }

              if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                return (
                  <div key={lIdx} className="flex gap-2.5 my-1 pl-1">
                    <span className="text-accent select-none mt-[2px] leading-none">•</span>
                    <span className="flex-1">{renderInline(trimmed.slice(2))}</span>
                  </div>
                );
              }

              const numListMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
              if (numListMatch) {
                return (
                  <div key={lIdx} className="flex gap-2.5 my-1 pl-1">
                    <span className="font-mono text-[12px] text-accent select-none mt-[3px]">
                      {numListMatch[1]}.
                    </span>
                    <span className="flex-1">{renderInline(numListMatch[2])}</span>
                  </div>
                );
              }

              if (trimmed === '') {
                return <div key={lIdx} className="h-2.5" />;
              }

              return (
                <p key={lIdx} className="my-1">
                  {renderInline(line)}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};

/**
 * Only re-renders when the text itself changes, so a token arriving in the
 * newest reply doesn't re-parse every earlier message in the conversation.
 */
export const MarkdownRenderer = React.memo(
  MarkdownRendererImpl,
  (prev, next) => prev.content === next.content
);

/** Inline code, then bold/italic/links inside the remaining text. */
function renderInline(text: string): React.ReactNode {
  if (!text) return null;

  const codeParts = text.split(/`([^`]+)`/g);
  return codeParts.map((part, i) => {
    if (i % 2 === 1) {
      return (
        <code
          key={i}
          className="px-1.5 py-0.5 mx-px rounded-md bg-surface-2 border border-line font-mono text-[0.85em] text-accent"
        >
          {part}
        </code>
      );
    }
    return <React.Fragment key={i}>{renderFormatting(part, i)}</React.Fragment>;
  });
}

/** Bold, italic and links. Applied in that order so nesting resolves sensibly. */
function renderFormatting(text: string, keyPrefix: number): React.ReactNode {
  if (!text) return null;

  const boldParts = text.split(/\*\*([^*]+)\*\*/g);
  return boldParts.map((part, i) => {
    if (i % 2 === 1) {
      return (
        <strong key={`${keyPrefix}-b-${i}`} className="font-semibold text-ink">
          {part}
        </strong>
      );
    }
    return <React.Fragment key={`${keyPrefix}-i-${i}`}>{renderItalic(part, `${keyPrefix}-${i}`)}</React.Fragment>;
  });
}

function renderItalic(text: string, keyPrefix: string): React.ReactNode {
  if (!text) return null;

  const italicParts = text.split(/(?:\*([^*\n]+)\*|_([^_\n]+)_)/g);
  return italicParts.map((part, i) => {
    if (part === undefined) return null;
    // The split produces two capture slots per match; either one means italic.
    if (i % 3 !== 0) {
      return (
        <em key={`${keyPrefix}-em-${i}`} className="italic">
          {part}
        </em>
      );
    }
    return <React.Fragment key={`${keyPrefix}-t-${i}`}>{renderLinks(part, `${keyPrefix}-${i}`)}</React.Fragment>;
  });
}

function renderLinks(text: string, keyPrefix: string): React.ReactNode {
  if (!text) return null;

  const parts = text.split(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g);
  const out: React.ReactNode[] = [];

  for (let i = 0; i < parts.length; i += 3) {
    if (parts[i]) out.push(<React.Fragment key={`${keyPrefix}-x-${i}`}>{parts[i]}</React.Fragment>);
    const label = parts[i + 1];
    const href = parts[i + 2];
    if (label && href) {
      out.push(
        <a
          key={`${keyPrefix}-a-${i}`}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent underline underline-offset-2 decoration-accent-line hover:decoration-accent"
        >
          {label}
        </a>
      );
    }
  }

  return out;
}
