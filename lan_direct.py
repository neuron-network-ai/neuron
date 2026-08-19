"""lan_direct.py — let two machines in the same house skip the relay, and leak nothing doing it.

**The cost this removes.** Every node registers `behind_nat: true` by default
(`agent/agent.py:651`), so the coordinator publishes the RELAY's address for it
(`coordinator/main.py:574`) and every activation of every token is proxied through a 1 GB VM.
Measured in [P57] with `tools/bench_hop.py`: dialling **this machine's own node** through the
relay costs **88 ms** median, where the same machine's neighbour on the LAN is **5.3 ms**. Two
thirds of a NEURON token is compute and a third is that detour, and a volunteer running two
machines at home is the ordinary case, not an exotic one.

**Why the obvious fix was refused, and rightly.** Turning the relay off (`--no-relay`) publishes
a machine's own address to the whole network. That is a Tailscale dependency for anyone who
wants it to work across houses, and — more seriously — the relay is a PRIVACY feature, not just
a NAT workaround. Peers see `150.230.22.250` and never where a volunteer lives. `/node/list`
already hides node addresses from public callers for exactly this reason, citing "the
private-earnings and private-address decisions". None of that may regress for a latency win.

**So the disclosure is inverted, and that is the whole idea.** The node does not advertise where
it is. The DRIVER states which private subnets IT is already on, and the node answers only if it
holds an address inside one of them:

    driver -> node   config { ..., "lan_hint": ["192.168.1"] }
    node   -> driver ack    { ..., "direct": {"ip": "192.168.1.11", "port": 50999} }

The driver therefore learns nothing it could not have learned by scanning its own LAN, and a
node on a different network answers with no `direct` at all — the field is simply absent and the
relay carries the request exactly as it does today. Nothing is stored, nothing is published, and
the coordinator never sees a home address.

**Rules that keep it honest, each load-bearing:**

  * **RFC1918 only.** 10/8, 172.16/12, 192.168/16. A public address is never offered and never
    accepted, so this can never disclose where a volunteer lives even if the rest is wrong.
  * **100.64/10 is deliberately EXCLUDED** — that is Tailscale's range (and carrier-grade NAT).
    Using it would make a private mesh a dependency of being fast, which is the thing that was
    refused. A tailnet address is not a LAN address.
  * **The prefix must MATCH.** A hint is a /24 the driver is genuinely on. Without that rule a
    node could name any private address and steer a peer's activations at a machine of its
    choosing on that peer's own network.
  * **It is an optimisation, never a requirement.** Every failure — no match, no answer, a
    refused connection, a wrong build — falls back to the address the coordinator gave.
"""
import ipaddress
import socket

# RFC1918 only. See the module docstring for why 100.64.0.0/10 is not here.
PRIVATE_NETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def is_private(ip):
    """True for an RFC1918 IPv4 literal. False for anything else, including Tailscale's
    100.64/10, loopback, link-local and every public address."""
    if not ip or not isinstance(ip, str):
        return False
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return False
    return any(addr in net for net in PRIVATE_NETS)


def local_addresses():
    """This machine's own RFC1918 IPv4 addresses.

    `psutil` when it is available (the agent already depends on it and it sees every
    interface); otherwise the stdlib, which sees fewer but is never wrong about the ones it
    does see. A machine with no private address simply returns [] and this whole feature turns
    itself off.
    """
    out = []
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and is_private(a.address):
                    out.append(a.address)
    except Exception:                                          # noqa: BLE001
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if is_private(ip):
                    out.append(ip)
        except OSError:
            pass
    return sorted(set(out))


def prefix_of(ip):
    """The /24 an address sits in, as a string: '192.168.1.11' -> '192.168.1'."""
    return ip.rsplit(".", 1)[0] if is_private(ip) else None


def local_prefixes():
    """The /24 prefixes this machine is on — what a driver puts in `lan_hint`.

    Prefixes rather than addresses on purpose: it tells the peer which network to answer about
    without naming this machine's own position on it.
    """
    return sorted({p for p in (prefix_of(ip) for ip in local_addresses()) if p})


def address_for(hint, addresses=None):
    """This machine's address on one of the caller's subnets, or None.

    `hint` is the caller's `lan_hint` list. Returns None for anything unusable — a missing or
    malformed hint, no shared subnet, no private address of our own — because every one of
    those means "use the relay", which is the safe answer and the current behaviour.
    """
    if not isinstance(hint, (list, tuple)):
        return None
    wanted = {h for h in hint if isinstance(h, str)}
    if not wanted:
        return None
    for ip in (addresses if addresses is not None else local_addresses()):
        if prefix_of(ip) in wanted:
            return ip
    return None


def usable(direct, hint=None):
    """Is a peer's advertised `direct` block safe for THIS machine to dial?

    Checked on the driver side as well as offered on the node side, deliberately: the node
    decides what to reveal, and the caller still decides what to trust. A node naming an
    address outside the subnets we actually asked about is refused here even though it answered
    our question — that is the rule that stops a peer steering our activations at some other
    machine on our own network.
    """
    if not isinstance(direct, dict):
        return False
    ip, port = direct.get("ip"), direct.get("port")
    if not is_private(ip) or not isinstance(port, int) or not 0 < port < 65536:
        return False
    allowed = set(hint if hint is not None else local_prefixes())
    return prefix_of(ip) in allowed
