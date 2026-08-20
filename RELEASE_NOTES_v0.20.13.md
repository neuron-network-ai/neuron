# NEURON v0.20.13

A second chat interface, at `/workspace` — personas, skills, a task list, attachments, voice —
served by NEURON itself and answering through NEURON. The existing page at `/` is untouched.

## What it is

The same app, wearing NEURON's own mark and palette (both lifted from `ui/static/chat.html`
and `coordinator/theme.py` rather than approximated), offering NEURON and nothing else. No
Gemini, Ollama or KoboldCPP tabs, no image mode, no model picker — the coordinator decides
which model the network serves, so a menu whose every option is the current one is a decision
nobody has.

Everything the existing page does, it does:

- signed-in account, NRN balance, Wallet ID, log out
- `Answers run here · N nodes online`
- **Web search** and **Use the network**, per request
- what a reply cost, and **where it ran** — "this machine" or "2 nodes"
- unclaimed-earnings, low-balance and update notices
- **Claim with my account**

## History is shared, both ways

`/v1/chat/completions` now takes and returns a `conversation_id` and writes into the same
store the existing page reads. A chat started in one appears in the other, and the workspace
imports what is already there on first load. Without the id every turn would have opened its
own conversation and filled the old page with one-message threads.

## Memory, and why it needed a second model

Long-term memory across conversations, **entirely on your machine**: the notes and their
vectors live in a file in NEURON's state directory, the embedding runs in NEURON's own
process, and nothing reaches the coordinator or the node network. It costs no NRN.

It needed a new endpoint — `/v1/embeddings` — and then it needed the *right* model. Embedding
with the chat model looked like it worked and was quietly wrong:

```
"how much memory does my laptop have?"
  0.908  [Cooking]  My favourite pasta is cacio e pepe…     <- wrong, ranked first
  0.839  [Hardware] The Pavilion laptop has 12 GB of RAM…
```

A chat model's vectors encode sentence shape more than meaning, so everything lands in a
0.85–0.91 band and nothing separates. That is not weak recall, it is confident wrong recall.
With a real embedding model (~80 MB, fetched once, cached like every other GGUF):

```
  0.715  [Hardware] The Pavilion laptop has 12 GB of RAM…
  0.389  [Cooking]  My favourite pasta is cacio e pepe…
```

**It costs 174 MB while in use** (measured, not estimated) and is **released after ten minutes
idle**, returning 147 MB. If you never turn memory on you never pay for it at all.

## Honest gaps

- **The tool loop is off.** It runs shell and file actions through a route that does not exist
  here. Shipping that inside an app handed to strangers is a security decision, not a port.
- **Memory is per-machine.** Conversations follow your account; memory does not.
- **The workspace bundle is ~915 kB** against the existing page's few kB. That is page load,
  not answer speed.

## Speed

Unchanged. One regression was found and fixed before release: resolving your API key called a
coordinator-backed endpoint before *every* message, putting ~600 ms in front of every first
token. It is cached now.

## Upgrading

`/` is exactly as it was. `/workspace` is additive — try it, and keep using the old page if you
prefer it.
