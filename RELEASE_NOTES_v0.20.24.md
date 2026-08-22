The network answers everything now, and one feature was removed rather than fixed.

## Your prompt no longer goes to the coordinator

Before your machine can use the network it asks the coordinator which machines to use. That
request used to carry **your whole prompt**. The coordinator used it exactly twice, both times
as a length — a cost estimate and a character count — and never read the text, never logged it,
never stored it. But PRIVACY.md told people, in a table, that the coordinator **cannot** read
their prompt. It could.

"Nothing reads it" is a weaker promise than "it is not sent", and only the second one was made.
Your machine now sends the number of characters instead of the characters.

## The network is the path, and a dead chain says so

NEURON used to run the model on your own computer whenever it fitted and use the network only
for models one machine could not hold. That is no longer the default: requests go to the node
chain, and **if the chain cannot serve one, it fails and says so** rather than quietly
answering locally.

The reason is not speed — local is faster for a model that fits, and that has not changed. It
is that a network nobody's requests reach is a network nobody can tell is broken. This project
has already had its encrypted network path fail on *every* request for a day while looking
perfectly healthy, because every token anyone had seen came from their own PC. A silent local
fallback would rebuild exactly that blindness.

Local execution is still there for anyone who wants it: set `NEURON_LOCAL_FIRST=1`.

## A re-placement now reaches a node that is already running

The coordinator can move which layers a node serves. Nothing told the node — it learned its new
range only the next time it happened to re-register, which in practice meant the next restart.
On 2026-08-21 that gap served word salad while every indicator stayed green, at the fastest
tokens-per-second this network has ever produced. The assignment now rides on the heartbeat, and
a node on the wrong range downloads and reloads without being restarted.

## The assistant was removed, not repaired

The task list came from a different product and never worked: every task anybody typed was
rejected by a field-name mismatch, silently, since the day it shipped.

It was removed rather than fixed because of what it did when it *was* on. The open list was
attached to your message on every turn — so with the network now answering everything, a private
to-do list was being tokenised, computed across volunteers' machines, and charged for on every
message. Gone entirely: the panel, the composer toggle, the tools the model could call, the
prompt injection, and the endpoints behind them.

## Also in this build

- **Your account, from the header.** Wallet, payout address and machines, with the wallet id
  **masked behind an eye toggle** — it is a spending key, and it was previously printed in full
  every time the panel opened. Copy still works while it is hidden.
- **Signed out is no longer a dead end** — the account panel offers Google and GitHub sign-in.
- **Send feedback** goes to the project directly, with your version and OS attached only if you
  tick the box. Never your conversations, prompts or wallet.
- **Bring a machine** — share the install page, or join the community.
- The empty state shows whether the network can actually answer **before** you type.
- The sidebar folds away on a desktop, and remembers.

## Verifying this release

The SHA-256 is published on the release page and at `/agent/version`, and it is the same hash
every installed agent checks before it will run an update. The installer is not code-signed, so
Windows will warn about an unrecognised publisher — the source is public and the hash is the
thing to check.
