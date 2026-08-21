**[P60] — a node answers real pipeline traffic only for a role it can actually serve.**

`node_server` chose its role in three branches, and the last was a catch-all for "everything
else". A node whose range does not include the model's final layer was assumed to be talking to
a verifier — but it can equally be a node the coordinator believes is last while the node knows
it is not, because placement moves in the coordinator's database and the node finds out only
when it next registers.

So real pipeline traffic fell into the probe branch, ran the node's own layers, **skipped the
final norm**, and handed the driver something to run `lm_head` on. A question about the sky came
back as `Sovereberg Sovere ABCDEFGHITestCategory` — while `/status` read routable, stage1_ok,
healthy and 28/28 covered, and decode hit the fastest figure this network has ever produced,
because a node running half its layers is genuinely quicker. Speed rose as correctness went to
zero.

Such a config is now refused with `range_mismatch`, naming the range the node actually holds.
Refusing costs nothing: the driver already turns a refused config into a reroute, and
proof-of-compute already reads `range_mismatch` as stale placement rather than as a failed
challenge, so an honest node is never flagged for the coordinator's bookkeeping.

Also carries everything in v0.20.16.
