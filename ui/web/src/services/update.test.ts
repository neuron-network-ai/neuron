/**
 * The update notice.  Run: `npm test` in ui/web/
 *
 * A release the operator has to install was visible on the coordinator's dashboard and nowhere
 * the operator looks. The rules that matter here are all about NOT saying the wrong thing:
 * never render "up to date" when the check failed, never tell a node ahead of the network to
 * downgrade, and never word a rollback as an upgrade.
 */
import { describe, it, expect } from 'vitest';
import { updateNotice, updateHref, NO_UPDATE, RELEASES_URL, UpdateState } from './update';

const at = (o: Partial<UpdateState>): UpdateState => ({ ...NO_UPDATE, ...o });

describe('updateNotice', () => {
  it('says nothing when there is nothing to install', () => {
    expect(updateNotice(NO_UPDATE)).toBeNull();
    expect(updateNotice(at({ running: '0.20.3', latest: '0.20.3' }))).toBeNull();
  });

  it('names both versions when a newer build exists', () => {
    const n = updateNotice(at({ running: '0.20.3', latest: '0.20.4', available: true }));
    expect(n).toContain('0.20.4');
    expect(n).toContain('0.20.3');
  });

  it('a failed check is NOT "you are up to date"', () => {
    // The server returns available:false with a reason when it could not ask. Silence is the
    // right render — what must never happen is a positive claim about being current.
    const n = updateNotice(at({ running: '0.20.3', error: 'coordinator unreachable' }));
    expect(n).toBeNull();
  });

  it('never invents an update from the version strings alone', () => {
    // `available` is the server's numeric verdict ([P48]). A node running 0.20.4 against a
    // published 0.20.3 is AHEAD and must not be told to downgrade — the live case on the
    // founder's own machine.
    expect(updateNotice(at({ running: '0.20.4', latest: '0.20.3', available: false }))).toBeNull();
  });

  it('a rollback is worded as the network asking, not as an upgrade', () => {
    const n = updateNotice(at({
      running: '0.20.4', latest: '0.19.0', available: true, rollback: true,
    }));
    expect(n).toContain('asking nodes to run');
    expect(n).not.toContain('is available');
  });
});

describe('updateHref', () => {
  it("follows the coordinator's own url when it gives one", () => {
    const u = at({ url: 'https://example.test/NEURON-Setup-0.19.0.exe' });
    expect(updateHref(u)).toBe('https://example.test/NEURON-Setup-0.19.0.exe');
  });

  it('falls back to the releases page, unpinned so it follows the next release', () => {
    expect(updateHref(NO_UPDATE)).toBe(RELEASES_URL);
    expect(RELEASES_URL).toContain('releases/latest');
  });
});
