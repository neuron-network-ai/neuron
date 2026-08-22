NEURON had no front door. This release is that door.

## Clicking NEURON opens NEURON

Until now, installing NEURON started a background process whose only surface was a tray icon —
and Windows hides new tray icons behind a chevron by default. You ticked "Start NEURON now",
and as far as the screen was concerned, nothing happened. The chat page existed at
`localhost:8080` and was unreachable unless you already knew it was there.

**The app now opens its own chat page**, once the server is genuinely answering rather than
immediately — opening early would greet a new install with a connection error, which is worse
than the silence it replaces.

**And it opens it every time you click**, not once. An earlier version of this fix opened the
page a single time per install; that only moves the problem to day two, when you are back to
hunting a hidden icon.

## The icon no longer starts a second copy of NEURON

Every shortcut — Start Menu, desktop, sign-in — points at the same program, so double-clicking
one while NEURON was already running started a **second** agent. It then lost a fight over the
ports the first one held and retried forever: two processes, one of them useless, and no window
from either.

That same double-click now simply shows you the app.

## Starting at sign-in stays quiet

The difference is intent, not a counter. The shortcut that runs NEURON when you sign in says so,
and gets no browser tab. Anything you click yourself opens the page.

## Verifying this release

The SHA-256 is on the release page and at `/agent/version`, and it is the hash every installed
agent checks before it will run an update. The installer is not code-signed, so Windows warns
about an unrecognised publisher — the source is public and the hash is the thing to check.
