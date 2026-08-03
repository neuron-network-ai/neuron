# NLnet — NGI Zero application

## 1. Project name

**NEURON** — Network of Existing Utilised Resources, Open Nodes.

Repository: https://github.com/neuron-network-ai/neuron · Apache 2.0

## 2. What problem does it solve?

Running a large language model requires more memory than any ordinary computer has, so the
capability sits with a handful of companies who can read what is sent to them, price it as they
choose, and withdraw it. Meanwhile the world already contains billions of general-purpose
processors idling at a few percent of capacity. NEURON lets ordinary machines pool that spare
capacity to run models none of them could run alone, so the capability does not depend on any
single operator's permission. A researcher in a country where these services are restricted, or
a developer who cannot afford per-token pricing, has no practical alternative today.

## 3. What is your solution?

A transformer is a stack of layers. NEURON splits that stack across volunteer machines: each one
holds a contiguous slice of layers, runs them on the activations arriving from the previous
machine, and passes the result on. No machine holds the whole model.

A property falls out of that design rather than from a policy: only the machine the user is
sitting at handles readable text. It holds the embedding and the output projection. Every other
machine in the chain receives opaque numeric activations and never touches a tokenizer, so it
cannot read what it is computing.

Working today, measured on three ordinary computers (a 16-core desktop, a 6-core Dell OptiPlex,
a 4-core HP Pavilion):

- **Correctness:** the split is **bit-exact** against running the whole model on one machine
  (`selftest_shard.py`, max delta 0.000e+00).
- **Throughput scales with machines:** ~3.2 tokens/s on one, **4.61 on two, 6.16 on three**, at
  3.82× pipeline overlap. Distribution buys capacity and concurrent users, not single-answer
  speed — we are explicit about that.
- **Bandwidth:** activations cross the network **4.25× smaller** (12,508 → 2,946 bytes per
  message) with output identical to full precision, in a format that carries nothing executable.
- **Joining is automatic:** a new machine registers itself, is assigned a layer range, downloads
  **only the bytes for its own layers** (1.40 GB rather than 3.09 GB), and is verified by other
  nodes in **254 ms** — no operator, no approval queue, no shared secret. It survives a power cut
  and is serving again **13 seconds** after boot.
- **NAT traversal is built in.** A node makes only outbound connections; no VPN, no port
  forwarding, no router configuration.

**Honest current state.** This is an early alpha. The live network is **2 machines covering 21 of
28 layers** — an incomplete chain — having served **38 requests**. No person outside the project
has yet run a node; that is the next milestone, not a claim. There is one coordinator on one
virtual machine, which is the single point of failure this application is largely about removing.

NEURON also carries a compute accounting layer: a record of how much work each machine
contributed, held in an ordinary database. It has no cash value, there is no exchange and no sale.

## 4. What are you requesting funding for?

| Work | Amount |
|---|---|
| **DHT peer discovery (libp2p)** — machines find each other and form chains with no central service. This is what removes the coordinator from the critical path. | €15,000 |
| **Coordinator redundancy + PostgreSQL** — replace the single SQLite instance with a replicated database and several stateless coordinators, so the network survives losing one machine. | €10,000 |
| **Android agent (ARM NEON kernel)** — phones charging overnight are the most abundant idle hardware there is. Requires porting the compute kernel from AVX2 to NEON. | €15,000 |
| **Code signing certificate and infrastructure review** — the installer is unsigned, so Windows warns every newcomer that it is unrecognised. A signed binary removes the single largest practical barrier to a non-technical person joining, and a security review before wider distribution is the responsible step alongside it. | €5,000 |
| **Total** | **€45,000** |

## 5. Why does this advance the open internet?

The ability to run capable language models is becoming infrastructure, and it is concentrating
into very few hands. NEURON is an attempt to make that capability something people can operate
themselves, on hardware they already own, without asking anyone's permission and without a
central operator able to read, meter, or withdraw it.

The funded work is specifically the decentralisation: today one machine decides routing and
placement, and item 1 exists to remove it. Everything is Apache 2.0, and the engineering log —
the measurements above and the failures behind them — is public.

It also reuses hardware that already exists rather than requiring new datacenters, though we are
careful not to overclaim: inference draws real power, roughly 15–45 W above idle on a CPU. What
is avoided is the manufacturing, not the electricity.

## 6. Relation to existing work

**Petals** (BigScience) is the proven reference and the closest relative — it runs 100B-class
models across volunteer machines using DHT discovery with relay fallback, and it demonstrates
that the networking works at scale. We do not intend to reinvent those primitives; item 1 builds
on `libp2p`/`hivemind` deliberately.

NEURON's contribution is the part Petals does not address: a one-click install for a
non-technical person, automatic layer assignment and slice download, proof-of-compute
verification of machines nobody vouches for, and NAT traversal that needs no router
configuration. Bittensor, Gensyn and io.net all require GPU hardware and substantial operator
setup; NEURON requires neither.

## 7. Who are you?

**Raman Kumar Sharma**, Rotterdam, Netherlands. An optometrist by profession and a self-taught
developer. NEURON is a solo project, built in the open under Apache 2.0, with the full
engineering history — every measurement, and every approach that was tried and rejected — in the
public repository.
