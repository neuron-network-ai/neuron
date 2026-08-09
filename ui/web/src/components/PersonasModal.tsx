import React, { useState } from 'react';
import { Persona } from '../types';
import { useModalDismiss } from '../hooks/useModalDismiss';
import { X, Plus, Check, ArrowLeft } from 'lucide-react';

interface PersonasModalProps {
  personas: Persona[];
  activePersonaId?: string;
  onSelectPersona: (persona: Persona) => void;
  onSaveCustomPersona: (persona: Persona) => void;
  onClose: () => void;
}

export const PersonasModal: React.FC<PersonasModalProps> = ({
  personas,
  activePersonaId,
  onSelectPersona,
  onSaveCustomPersona,
  onClose,
}) => {
  const [isCreatingCustom, setIsCreatingCustom] = useState(false);
  const { onBackdropClick } = useModalDismiss(onClose);

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [avatar, setAvatar] = useState('🤖');
  const [systemPrompt, setSystemPrompt] = useState('');

  const handleCreatePersona = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !systemPrompt.trim()) return;

    const newP: Persona = {
      id: 'custom_' + Date.now(),
      name: name.trim(),
      description: description.trim() || 'Custom created AI persona',
      avatar: avatar.trim() || '🤖',
      systemPrompt: systemPrompt.trim(),
      category: 'custom',
      isCustom: true,
    };

    onSaveCustomPersona(newP);
    onSelectPersona(newP);
    setIsCreatingCustom(false);
  };

  const inputClass =
    'w-full h-9 px-3 rounded-lg bg-surface-2 border border-line text-[12.5px] text-ink placeholder-ink-faint focus:outline-none focus:border-accent-line focus:bg-surface transition';

  return (
    <div
      onClick={onBackdropClick}
      className="fixed inset-0 z-50 flex items-start sm:items-center justify-center p-4 sm:p-6 bg-black/40 backdrop-blur-[3px]"
    >
      <div className="w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl bg-surface border border-line shadow-[var(--shadow-lg)] overflow-hidden rise-in">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-line flex-shrink-0">
          <div className="flex items-center gap-2">
            {isCreatingCustom && (
              <button
                onClick={() => setIsCreatingCustom(false)}
                className="grid place-items-center w-7 h-7 rounded-md text-ink-faint hover:text-ink hover:bg-surface-2 transition"
                title="Back"
              >
                <ArrowLeft className="w-4 h-4" />
              </button>
            )}
            <h2 className="text-[13.5px] font-semibold text-ink">
              {isCreatingCustom ? 'New persona' : 'Personas'}
            </h2>
          </div>

          <div className="flex items-center gap-1">
            {!isCreatingCustom && (
              <button
                onClick={() => setIsCreatingCustom(true)}
                className="flex items-center gap-1.5 h-7 px-2.5 rounded-lg bg-accent hover:bg-accent-hover text-accent-ink text-[12px] font-semibold transition"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Create</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="grid place-items-center w-7 h-7 rounded-md text-ink-faint hover:text-ink hover:bg-surface-2 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 min-h-0">
          {isCreatingCustom ? (
            <form onSubmit={handleCreatePersona} className="space-y-3.5">
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
                <label className="sm:col-span-3 block">
                  <span className="label-mono text-ink-faint">Name</span>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={e => setName(e.target.value)}
                    placeholder="Code Reviewer"
                    className={`${inputClass} mt-1.5`}
                  />
                </label>

                <label className="block">
                  <span className="label-mono text-ink-faint">Emoji</span>
                  <input
                    type="text"
                    value={avatar}
                    onChange={e => setAvatar(e.target.value)}
                    className={`${inputClass} mt-1.5 text-center text-lg`}
                  />
                </label>
              </div>

              <label className="block">
                <span className="label-mono text-ink-faint">Description</span>
                <input
                  type="text"
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Reviews changes for security and readability"
                  className={`${inputClass} mt-1.5`}
                />
              </label>

              <label className="block">
                <span className="label-mono text-ink-faint">System instruction</span>
                <textarea
                  required
                  value={systemPrompt}
                  onChange={e => setSystemPrompt(e.target.value)}
                  placeholder="You are an expert code reviewer. Point out complexity, edge cases and security issues…"
                  className="mt-1.5 w-full h-28 p-3 rounded-lg bg-surface-2 border border-line text-[12.5px] leading-relaxed text-ink placeholder-ink-faint resize-none focus:outline-none focus:border-accent-line focus:bg-surface transition"
                />
              </label>

              <div className="flex justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setIsCreatingCustom(false)}
                  className="h-8 px-3 rounded-lg text-[12px] font-medium text-ink-muted hover:text-ink hover:bg-surface-2 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="h-8 px-3.5 rounded-lg bg-accent hover:bg-accent-hover text-accent-ink text-[12px] font-semibold transition"
                >
                  Save &amp; apply
                </button>
              </div>
            </form>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {personas.map(persona => {
                const isSelected = persona.id === activePersonaId;

                return (
                  <button
                    key={persona.id}
                    onClick={() => {
                      onSelectPersona(persona);
                      onClose();
                    }}
                    className={`text-left p-3 rounded-xl border transition ${
                      isSelected
                        ? 'bg-accent-soft border-accent-line'
                        : 'bg-surface border-line hover:bg-surface-2'
                    }`}
                  >
                    <div className="flex items-start gap-2.5">
                      <span className="text-xl leading-none mt-0.5">{persona.avatar}</span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[13px] font-semibold text-ink truncate">{persona.name}</span>
                          {isSelected && (
                            <span className="grid place-items-center w-4 h-4 rounded-full bg-accent flex-shrink-0">
                              <Check className="w-2.5 h-2.5 text-accent-ink" />
                            </span>
                          )}
                        </div>
                        <span className="label-mono text-ink-faint">{persona.category}</span>
                        <p className="text-[12px] text-ink-muted leading-relaxed mt-1.5 line-clamp-2">
                          {persona.description}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
