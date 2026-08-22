This release exists because someone asked a question the code could not answer, and the honest
answer turned out to be worse than the question.

## Your prompt no longer goes to the coordinator

The landing page rated NEURON "partial" on privacy by architecture. Asked to prove why, the
reason found was not the one written down.

Before your machine can use the network it has to ask the coordinator which machines to use.
That request carried **your whole prompt** — `{"prompt": "..."}` — on every network-served
message. The coordinator used it exactly twice, both times as a LENGTH: a worst-case cost
estimate, and the character count kept against the request. It never read the text, never logged
it, and never wrote it to the database.

But PRIVACY.md told people, in a table, that the coordinator **cannot** read their prompt. It
could. **"Nothing reads it" is a weaker promise than "it is not sent",** and only the second one
was made. A field that arrives is in that process's memory, in any core dump taken of it, and in
anything put in front of it that logs request bodies — none of which is visible to the person
who read the promise and decided to type something private.

The storage half of this had been fixed months ago — the coordinator stopped writing prompt text
and kept a character count — and that is exactly what hid the rest. The database was clean, so
the question looked closed while the prompt still crossed the wire every time.

**Your machine now sends the number of characters instead of the characters.** The coordinator
still accepts the old shape, because agents already installed still send it, so nothing breaks
while the network updates.

Recorded as [P65] in PROBLEMS.md, which is where this project keeps its mistakes rather than
its achievements.

## A re-placement now reaches a node that is already running

The coordinator can move which layers a node serves. Nothing told the node: it learned its new
range only the next time it happened to re-register, which in practice meant the next restart.

On 2026-08-21 that gap served **word salad while every indicator was green**, at the fastest
tokens-per-second this network has ever produced — because a node running nine layers instead of
eighteen is genuinely quicker. Speed went up as correctness went to zero.

0.20.22 made that safe: a node asked for a role it cannot serve now refuses and the request is
rerouted, instead of computing over layers it never downloaded. It did not make the gap shorter.
This does. The assignment rides on the heartbeat — the one message every live node sends
continuously, and the only channel that reaches a machine behind a home router — and a node that
finds itself on the wrong range downloads and reloads without being restarted. A slice that
already covers the new range is reloaded with no download at all.

## Why "partial" is now a tick

Two things made that rating honest. One was the defect above, and it is gone. The other remains
and is stated rather than hidden: a volunteer machine computing part of your answer holds those
numbers in plaintext inside its own process. That is the work it is doing. What limits it is
structural — no single machine holds the whole model, so it cannot turn those numbers back into
your text, and no single machine holds your whole conversation.

That limit is shared by every distributed inference network. What is not shared: NEURON runs the
model on your own machine when it fits, so most answers never leave it at all; every network hop
is separately encrypted with a one-time key, so neither the relay nor the coordinator can read
the wire; and from this release the coordinator is sent a length rather than your words.

## Verifying this release

SHA-256 is published at `/agent/version` and on the release page, and it is the same hash every
installed agent checks before it will run an update. The installer is not code-signed, so
Windows will warn about an unrecognised publisher — the source is public and the hash is the
thing to check.
