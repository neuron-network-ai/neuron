Two failures this release exists to fix were found by running the product against machines it
had never run against.

## The account claim only ever worked on Windows

`POST /node/claim` looked for this machine's payout key in `LOCALAPPDATA/NEURON`, while the
agent stores it beside the config file — `state_dir=os.path.dirname(config_path)`. Those
coincide on a Windows installer install and nowhere else, so every Linux node, every source
run and every self-hoster was refused with *"this machine holds no key at all"* about a key
sitting one directory away, and told to go find a browser wallet — the exact thing the
account-claim was built to make unnecessary.

Fixed by making the two rules one. Verified on a node that was genuinely unclaimed.

## Proof-of-compute certified a path users are not served by

The verifier built its own challenge message. The node reads its ROLE out of that message, so
the verifier could put a node in a different role than a real request does — and did, for a
day, while reporting 5662 consecutive passes on a node returning garbage to every user.

* one constructor for the `config` message, shared by the driver and the verifier, so the two
  cannot drift apart again;
* the MIDDLE role is challenged as a relay for the first time — the node is given a next hop it
  can really reach and graded on what it forwards, rather than on an isolated probe no user
  request produces;
* every attestation records which path it exercised, and a probe used as a fallback says so.

It earned its keep within hours: a placement change left a node serving nine layers where the
coordinator thought it served eighteen, and one challenge named the fault, the node, and the
fact that it was placement rather than bad arithmetic.

## Also in this release

* **The workspace UI is the front door.** `/` now redirects to `/workspace/`; the previous page
  keeps a URL at `/classic`.
* **A middle node prefers a same-LAN next hop.** Two machines on one switch no longer route
  every token through the relay. The relay stays the address of record for every failure, an
  address outside the subnets we asked about is refused, and an unreachable offer is remembered
  rather than retried on every request.
* **A config from an older build no longer crash-loops the agent.** Missing keys fall back to
  the build's defaults, which a comment had claimed for months while `self.cfg["slice_dir"]`
  raised `KeyError` out of startup.
* **`agent/requirements.txt` now lists `cryptography`.** A venv built from it produced an agent
  that could not import its own modules.
* **Auto-repair no longer stands down indefinitely** for a migration that is preparing and not
  converging, and `/status` reports why a repair is not happening instead of leaving the reason
  in the coordinator's stdout.

Full detail for every item is in `PROBLEMS.md` under [P56] and [P58]–[P60].
