# NEURON v0.20.5

The release that makes three sessions of work reachable. Everything below existed in the
repository and none of it reached a user, because the app serves its own packaged copy — which
is itself the shape of two of these bugs.

## Your earnings can now be claimed, because you can now find the claim

A node's entire credential is a `node_token` in one `config.json` on one disk. Lose the file,
lose the balance. The panel that binds those earnings to the account you sign into has existed
since Session 61 and **had never once been used**: zero of three live nodes had an owner
recorded.

Not because it was broken. Because it rendered only inside the wallet panel, which needs you
signed in *and* to click "Wallet ID" — and the server's `needs_owner` was `logged_in AND not
owned`, so a signed-OUT operator got `false` and saw nothing at all. **The message explaining
why to sign in was gated on being signed in.**

There is now a strip above the composer, on both the classic page and the React one, that says
what is at stake and offers the sign-in links. It disappears once an owner exists, and a machine
that serves no node never sees it.

## A broken install says so instead of rendering a blank page

Vite content-hashes every bundle, so `index.html` names a different file each build, and the
installer's `ignoreversion` overwrites same-named files while never pruning ones that vanished.
Any partial copy therefore *merges* two builds and the page silently renders nothing.

- the installer now **clears the hashed bundles before copying**, making it a replace rather than
  a merge, and **closes the running app first** so no file is skipped for being locked;
- the app **verifies its own install** at startup and per request, and serves a page naming the
  missing files and the remedy rather than a blank screen.

## The app tells you when there is a newer build

A release you have to install was visible on the coordinator's dashboard and nowhere you look.
It now appears in the app, comparing versions numerically — `0.20.10` is newer than `0.20.9`,
which string comparison gets wrong — so a node *ahead* of the network is never told to
downgrade. A failed check says so rather than claiming you are up to date.

## The node can no longer fail quietly

Two fixes for the same class of fault, both found live:

- the agent loop ran in a thread whose exceptions went to a `stderr` that does not exist in a
  windowed app. It could stop dead while the tray, the Chat UI and the network listener carried
  on — alive, holding its port, answering probes, unregistered, and silent for 81 minutes. Its
  death is now recorded, through a path that does not depend on the logging it may have broken,
  and the tray goes red with the remedy;
- a **heartbeat watchdog** covers what no exception handler can: a loop still running and no
  longer reaching the coordinator.

`neuron_doctor` gained a matching check, and stopped failing a healthy network — it was reading
a field `/models` has never returned.

## Proof of compute, and being paid for it

- a node holding the whole model answered the *right answer to a different question* and was
  scored as wrong. It reports what it holds; the verifier now reads it. Serving was never
  affected — only the proof;
- emission pays for **proven** work rather than **observed** work. The verifier was not running
  for 53% of its own history, and honest nodes earned nothing for every hour it slept;
- deleting a node refuses while it still holds NRN, and leaves a record rather than a hole.

Every download link and version reference is checked against a real release, so the app cannot
point the network at a build that was never published.
