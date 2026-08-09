import React, { useEffect } from 'react';

/**
 * Standard modal dismissal: Escape closes, and clicking the backdrop (but not
 * the panel inside it) closes. Returns the handler to spread on the backdrop.
 */
export function useModalDismiss(onClose: () => void) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const onBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  return { onBackdropClick };
}
