/**
 * Tests for the balance a person plans around.  Run: `npm test` in ui/web/
 *
 * The free grant is 25 NRN, about 158 network answers, and it does not come back — see the
 * [P29] decision in PROBLEMS.md. That makes it a number people need to SEE while they still
 * have some, because the unfair part of a hard limit is discovering it by hitting it, not the
 * limit itself.
 *
 * The three-state rule from wallet.ts applies here too and is the sharp edge: an unreadable
 * balance must not render as a confident zero. On a network that pays people, "0 answers left"
 * when the truth is "we could not ask" reads as "this is broken and you are out".
 */
import { describe, it, expect } from 'vitest';
import { messagesLeft, isLowBalance, NRN_PER_MESSAGE } from './wallet';

describe('messagesLeft', () => {
  it('turns a balance into answers a person can plan around', () => {
    expect(messagesLeft(25)).toBe(Math.floor(25 / NRN_PER_MESSAGE));
    expect(messagesLeft(1)).toBe(6);
  });

  it('a balance too small to send anything is 0, not a fraction', () => {
    expect(messagesLeft(0.1)).toBe(0);
    expect(messagesLeft(0)).toBe(0);
  });

  it('UNKNOWN is null, never 0 — the distinction the whole wallet module exists for', () => {
    expect(messagesLeft(null)).toBeNull();
    expect(messagesLeft(NaN)).toBeNull();
    expect(messagesLeft(Infinity)).toBeNull();
  });
});

describe('isLowBalance', () => {
  it('warns while there is still room to act, not at zero', () => {
    expect(isLowBalance(25)).toBe(false);
    expect(isLowBalance(3)).toBe(true);
    expect(isLowBalance(0)).toBe(true);
  });

  it('an unknown balance is not a warning — it is not evidence of anything', () => {
    expect(isLowBalance(null)).toBe(false);
  });
});
