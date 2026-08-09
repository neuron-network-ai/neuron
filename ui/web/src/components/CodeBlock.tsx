import React, { useState, useMemo } from 'react';
import { Check, Copy, WrapText } from 'lucide-react';
// The "common" build covers ~40 mainstream languages; the full package pulls in
// nearly 200 and roughly triples the client bundle.
import hljs from 'highlight.js/lib/common';
import 'highlight.js/styles/github-dark.css';

interface CodeBlockProps {
  language: string;
  code: string;
}

const CodeBlockImpl: React.FC<CodeBlockProps> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);
  const [wrap, setWrap] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const cleanCode = code.replace(/\n+$/, '');
  const lines = cleanCode.split('\n');

  const { highlightedHtml, displayLang, isAutoDetected } = useMemo(() => {
    const rawLang = (language || '').toLowerCase().trim();
    const hasValidLang =
      rawLang && rawLang !== 'text' && rawLang !== 'code' && rawLang !== 'auto' && hljs.getLanguage(rawLang);

    if (hasValidLang) {
      try {
        return {
          highlightedHtml: hljs.highlight(cleanCode, { language: rawLang, ignoreIllegals: true }).value,
          displayLang: rawLang,
          isAutoDetected: false,
        };
      } catch {
        // fall through to auto-detection
      }
    }

    try {
      const autoRes = hljs.highlightAuto(cleanCode);
      return {
        highlightedHtml: autoRes.value,
        displayLang: autoRes.language || 'text',
        isAutoDetected: true,
      };
    } catch {
      return {
        highlightedHtml: cleanCode.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'),
        displayLang: rawLang || 'text',
        isAutoDetected: false,
      };
    }
  }, [cleanCode, language]);

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-line bg-[#0d1117]">
      {/* Header */}
      <div className="flex items-center justify-between h-9 pl-3.5 pr-1.5 bg-[#161b22] border-b border-[#21262d]">
        <div className="flex items-center gap-2 min-w-0">
          <span className="label-mono text-[#7d8590]">{displayLang}</span>
          {isAutoDetected && (
            <span className="label-mono text-[#57606a]" title="Language auto-detected">
              auto
            </span>
          )}
          <span className="label-mono text-[#57606a]">
            · {lines.length} {lines.length === 1 ? 'line' : 'lines'}
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          <button
            onClick={() => setWrap(!wrap)}
            className={`grid place-items-center w-7 h-7 rounded-md transition ${
              wrap ? 'text-[#e6edf3] bg-[#21262d]' : 'text-[#7d8590] hover:text-[#e6edf3] hover:bg-[#21262d]'
            }`}
            title={wrap ? 'Disable wrapping' : 'Wrap long lines'}
          >
            <WrapText className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 h-7 px-2 rounded-md text-[#7d8590] hover:text-[#e6edf3] hover:bg-[#21262d] transition"
            title="Copy code"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-[#3fb950]" />
                <span className="label-mono text-[#3fb950]">copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span className="label-mono">copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex overflow-x-auto">
        <div
          aria-hidden="true"
          className="flex-shrink-0 py-3 px-3 text-right select-none font-mono text-[11.5px] leading-[20px] text-[#484f58] border-r border-[#21262d]"
        >
          {lines.map((_, idx) => (
            <div key={idx}>{idx + 1}</div>
          ))}
        </div>

        <pre
          className={`flex-1 py-3 px-4 m-0 font-mono text-[12.5px] leading-[20px] ${
            wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'
          }`}
        >
          <code className={`hljs language-${displayLang}`} dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
        </pre>
      </div>
    </div>
  );
};

/**
 * Syntax highlighting is the most expensive thing rendered in a message —
 * highlightAuto alone probes dozens of grammars — so skip it entirely unless
 * this block's own code changed.
 */
export const CodeBlock = React.memo(
  CodeBlockImpl,
  (prev, next) => prev.code === next.code && prev.language === next.language
);
