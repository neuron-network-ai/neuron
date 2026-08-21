Everything in this release was found by using the product rather than by reading it.

## The workspace could be blanked by a web-search citation

`rag/retriever.py` has always sent citations as `{title, href}`. The client declared them
`string[]` and rendered each one directly, so an object reached React as a child, React threw,
and **the render unwound the entire tree** — no sidebar, no threads, no composer, on a server
that was answering correctly. It fired for anyone who ticked "Web search", and it persisted,
because the object form is written into the browser's stored conversations.

Fixed at the render site so threads already on disk work, and **a per-message error boundary**
now contains a broken message instead of letting it take the app with it.

## A deleted chat came back on refresh

Deleting removed the local thread — which was the very thing marking it "already imported" — so
the next load pulled it back from the server. The client never called `DELETE
/conversations/{id}`. It does now.

## Signing out hid the way back in

`fetchSession` collapsed "could not ask" and "nobody is signed in" into one value, and the
account bar hid on both — taking the wallet, the balance **and the sign-in link** with it.

## Node earnings reach their owner without a sweep

Ownership has been recorded since [P39], and hourly emission already went to the owner's wallet.
Per-request settlement did not: it paid the node and stopped, so request earnings piled up on a
machine account nobody can sign into, drained only by an operator running a script by hand.
Settlement now forwards each share to the owner. The machine still shows what it earned.

## A headless node can be claimed at all

Claiming was a browser action, and a node running without a chat UI has no page to click — so it
earned into an account nobody owned and nothing said so. `neuron-agent --claim <wallet-id>` does
it now, and an unclaimed node says so on every registration.

## Also

* The signing tool is served at `/static/sign_payout.html` instead of being named at a source
  path that shipped nowhere and 404'd in three different UIs.
* The workspace page is never served from cache, and its bundle is purged on upgrade, so a
  stale page cannot silently load an app from two builds ago.
* A node asked to serve layers it does not hold now refuses instead of computing over
  uninitialised weights and returning fluent nonsense.
* Auto-repair no longer stands down indefinitely for a migration that is not converging, and
  `/status` reports why a repair is not happening.

Full detail is in `PROBLEMS.md` under [P56] and [P58]–[P63].
