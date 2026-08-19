# NEURON v0.20.9

**Network chat gives correct answers again.** One fix, and it is the whole release.

## The network returned nonsense and charged for it ([P55])

Ticking **Use the network** produced `  1  2   3` and then whitespace until the 128-token cap,
every time, and billed about **0.13 NRN** for it. Local chat — the default — was never affected.

The last node in the chain was running the right layers with **the final normalisation left
off**, and handing that back to the driver, which put the output head on a hidden state several
times larger than anything the head has ever seen.

Nothing looked broken from any single angle, and that is the point of this entry:

- the sharding maths is bit-exact against the unsplit model (`max|delta| = 0`);
- the wire, the weights, the layer ranges, the model id and the KV cache all checked out;
- the node computed **layers 10-27 correctly** when asked to in isolation;
- **proof-of-compute passed 5662 challenges** on that node while it was doing this.

The cause is a seam, not a component. A node picks its role from the `config` message it is
handed, because a machine holding the model's final layer is "the last stage" for every question
anyone can ask it — including a verifier challenging some *other* range. Since 0.20.4 that
choice was made by asking whether the message carried an `s1` field: the verifier sends one, so
`s1` present meant "I am being probed, run the layers and skip the norm".

**The driver sends `s1` too.** In a three-machine chain the driver's message goes to the middle
node, which forwards a different one, so the last node was reached correctly — which is why the
network answered coherently earlier the same day. In a **two-machine chain** the driver *is* the
stage before the last one, its message goes straight there, and every real request was served as
though it were a verifier's challenge.

Two machines is the ordinary state of a network built out of other people's spare computers, and
it is what this network has been running.

The fix has two halves and **neither needs the other**, which matters because a driver and the
machines it talks to are updated by different people on different days:

- the driver stops sending `s1` to a hop that has never read it — it goes only to a middle
  relay now. **This alone makes answers correct against every node already installed**, without
  waiting for a single one of them to update;
- the node reads what `s1` *means* when an older driver does send it: equal `s1`/`s2` is the
  junction between two stages of a real chain, `s1 < s2` is a range somebody is challenging.

The driver and the verifier also now say which they are outright (`stage`, `probe`), so the next
reader does not have to re-derive it.

Verified end to end on real weights over a real socket: the node's answer is now bit-identical
to `layers[19:]` **with** the final norm, where before it was bit-identical to the same layers
**without** it.

## What this release does not fix

**Proof-of-compute only ever exercises the probe path.** It certified this node as healthy
throughout, correctly — it was answering the challenge's question perfectly while answering
users' questions with the challenge's computation. A verifier that never drives the serve path
cannot vouch for the serve path. Filed as **[P56]**, and it is the more important finding of
the two.

## Upgrading

Nodes update themselves within 24 hours, and **there is no window where the network is worse
off**: an upgraded node serves an old driver correctly, and an upgraded driver is served
correctly by an old node. If you only update one machine, update the one running the app you
chat from — that is enough.
