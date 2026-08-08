"""ui/test_chat_ui.py — run: python -m ui.test_chat_ui

`ui/static/chat.html` is a single hand-written file with no build step and no framework, so
nothing catches a regression in it. These are static checks on the source: crude, but they
hold the properties that were actually broken, and they run in milliseconds with no browser.

What they defend, all of it found by reading the page against what the network really does:

  * **The wait before the first token is the experience.** At ~1 tok/s over volunteer machines
    ([P1]), the gap between Send and the first character is tens of seconds. It was a blinking
    cursor — indistinguishable from a hung page — while the `meta` event carrying the chain
    had *already arrived* and was being held until the answer finished.
  * **A partial answer must survive the failure that ended it.** A node dropping mid-answer is
    expected on a volunteer network (RESILIENCE.md [R2]); `showError` overwrote the message
    body, deleting text the user had already been given and already paid for.
  * **A capped answer is not a finished answer.** `max_tokens` was hardcoded and silent.
  * **Do not send into a chain that cannot answer** — but never disable the Stop button.
  * **The palette is the product's** (see coordinator/theme.py, docs/index.html), and contrast
    holds in both themes.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "static", "chat.html"), encoding="utf-8").read()

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def contrast(hex_a, hex_b):
    """WCAG contrast ratio between two #rrggbb colours."""
    def lum(h):
        parts = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        conv = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in parts]
        return 0.2126 * conv[0] + 0.7152 * conv[1] + 0.0722 * conv[2]
    a, b = sorted((lum(hex_a), lum(hex_b)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def var_block(selector):
    """The custom properties declared in one selector block."""
    i = SRC.index(selector)
    body = SRC[i:SRC.index("}", i)]
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6}|[^;]+);", body))


def main():
    print("\n-- the wait before the first token is shown, not hidden")
    check("a pipeline indicator exists", "function startPipeline" in SRC)
    check("it starts before the response is read",
          SRC.index("startPipeline(body)") < SRC.index('await fetch("/chat"'),
          "the indicator must be up before the request, not after the first byte")
    check("the chain from `meta` is shown while waiting", "pipeline.meta(d)" in SRC,
          "meta carries the node count and used to be held until the answer finished")
    check("an elapsed timer runs", "pipe-t" in SRC and "setInterval" in SRC)
    check("it is removed on the first token",
          "if(!gotToken){ pipeline.stop();" in SRC)
    check("it is removed on done and on error",
          SRC.count("pipeline.stop()") >= 4, "every exit path must clear the timer")
    check("the wait is explained rather than apologised for",
          "first token takes longest" in SRC)
    check("the indicator is announced to screen readers",
          'el.setAttribute("role", "status")' in SRC)

    print("\n-- a partial answer survives the failure that ended it")
    check("showError takes the partial text", "function showError(body, detail, code, partial)"
          in SRC)
    check("...and re-renders it instead of overwriting",
          "if(partial){" in SRC and "body.innerHTML = renderMarkdown(partial)" in SRC)
    check("...saying what is above is all that arrived", "is what arrived" in SRC)
    check("...and offering to ask again", "ask again" in SRC)
    check("a dead connection reads differently from an exhausted network",
          "connection to this machine dropped" in SRC and "retried on other machines" in SRC)
    check("a stream with no tokens at all explains itself",
          "returned no tokens" in SRC, "this used to render as a bare ellipsis")
    # An `error` event ALSO ends the stream with no token and no .meta, so the no-tokens
    # fallback ran straight after showError() and replaced a precise message with a guess
    # about an unrelated failure. Live on 2026-08-07: the coordinator refused with
    # "insufficient NRN balance; this request needs 0.158 NRN held" -- the page instead said a
    # node had dropped out while loading its slice, and an hour went into restarting a machine
    # that was serving correctly. The specific branch at `code === "insufficient_funds"` was
    # already written and simply never reached.
    check("...but never OVER a real error the server already sent",
          "shownError = true" in SRC and "!shownError" in SRC,
          "the generic fallback must not overwrite showError()'s specific message")
    check("the guard is declared with the other per-stream flags",
          "shownError = false" in SRC)
    check("an insufficient balance says so, rather than blaming a node",
          'code === "insufficient_funds"' in SRC
          and "Not enough NRN balance for this request" in SRC)

    print("\n-- an empty wallet is stated BEFORE the message is sent, not after it fails")
    # /infer takes its hold before it builds a chain, so a wallet under the hold can never
    # reach a node -- the failure is an account limit that looks exactly like a broken
    # network. The only prior signal was a small grey number in the header.
    check("there is a strip for it, above the composer", 'id="lowbal"' in SRC)
    check("...announced politely, not as an interrupting alert",
          '<div id="lowbal" role="status">' in SRC,
          "assertive would re-interrupt a screen reader on every refreshAuth")
    check("...themed, not a hardcoded amber",
          "#lowbal{" in SRC and "var(--warn-bg)" in SRC and "var(--warn)" in SRC)
    check("a zero balance is named as an ACCOUNT limit, not a network fault",
          "not a fault in the network" in SRC,
          "the whole point: this failure is indistinguishable from a dead chain")
    # Running out is the moment a shared network most easily reads as a bait-and-switch, so
    # the copy has to invite rather than charge -- and say plainly there is no paywall.
    check("...and says there is no paywall", "has no paywall" in SRC)
    check("...invites the user to contribute instead of to pay",
          "Share your machine's idle time" in SRC)
    check("...with a real destination, unpinned so it follows the next release",
          "releases/latest" in SRC,
          "a version-pinned installer link would keep sending strangers to 0.18")
    check("...and promises only what the agent actually does",
          "only runs while you are not using it" in SRC,
          "agent.py pauses on user activity and the donation ceiling")
    # Scoped to the two message strings, NOT the whole file: a source-wide keyword ban also
    # matches the comment above them explaining why the claim is not made, so it failed on
    # a file that was already correct.
    _copy = SRC[SRC.find("el.innerHTML = bal <= 0"):SRC.find("async function refreshAuth")]
    check("no unmeasured environmental claim in the copy itself",
          bool(_copy) and not any(w in _copy.lower()
                                  for w in ("carbon", "co2", "green energy", "emissions")),
          "nothing here measures energy; an unmeasured green claim is the real dishonesty")
    check("a low-but-nonzero balance warns before it bites", "Low balance:" in SRC)
    check("the threshold is a named constant, not a literal in the branch",
          "MIN_SENDABLE_NRN" in SRC and "const MIN_SENDABLE_NRN" in SRC)
    check("an unreadable balance stays silent rather than accusing the account",
          "bal === null" in SRC and "noteBalance(null)" in SRC)
    check("the header itself flags the zero", "empty-bal" in SRC)
    check("stopping keeps what was generated",
          'note.textContent = "Stopped. You were charged only for what had been generated."'
          in SRC)

    print("\n-- a node dying is NOT presented as a failure")
    # The load-bearing correction in this file. A node dying mid-answer is RECOVERED, token
    # for token: neuron_driver._reroute takes a fresh chain and replays the junction cache into
    # it, and test_node_death.py SIGKILLs a node mid-generation and requires the output to
    # match an uninterrupted run exactly. An earlier version of this page told the user "a
    # machine in the chain went offline mid-answer" as the reason an answer FAILED — naming the
    # ordinary, handled event as the cause, and teaching people to distrust the one thing the
    # network handles best. What actually reaches the error path is recovery being impossible.
    check("a reroute is handled as its own event", 'event==="reroute"' in SRC)
    check("...and is explicitly not an error", "NOT an error" in SRC)
    check("...it says the answer continues", "picking up where it left off" in SRC)
    check("...and is not styled as an error",
          '.className = "pipe reroute-live"' in SRC,
          "it must reuse the neutral pipeline style, not .err or .partial-note")
    check("a failed answer does NOT blame a node going offline",
          "went offline mid-answer" not in SRC,
          "a single node going offline is recovered; saying otherwise is simply untrue")
    check("...it says the RETRIES were exhausted", "retried on other machines" in SRC)
    check("...and vouches for the partial text", "correct as far as it goes" in SRC)
    check("a recovered answer records the recovery", "recovered from " in SRC,
          "RESILIENCE.md: a fallback must be visible in the response metadata, never silent")

    print("\n-- a capped answer says it was capped")
    check("the cap is a named constant", "const MAX_TOKENS" in SRC)
    check("the request uses it", "max_tokens:MAX_TOKENS" in SRC,
          "a hardcoded 128 in the body drifts from the constant the UI reports")
    check("hitting it is stated", "d.tokens >= MAX_TOKENS" in SRC
          and "because the answer was finished" in SRC)
    check("...with a continue affordance", "→ continue" in SRC)

    print("\n-- the header is a status, not a claim about who answered")
    # Observed on a real screen: the header read "Powered by 2 nodes worldwide" directly above
    # a reply whose own meta line said "this machine · Cost: 0.0000 NRN". Both true; together
    # they read as a contradiction, and the louder one took credit the network had not earned.
    # A locally-capable machine answers locally BY DESIGN (engine/local_gguf.py), so the header
    # has to describe the network's state, not attribute the answer.
    # No `or` fallbacks in these: this file has already shipped one vacuous check
    # (`or ">" in priv`, always true), so each assertion below has exactly one condition.
    check("the node count is phrased as a status, not an attribution",
          '" online"' in SRC and "worldwide" not in SRC,
          "'N nodes online' is a fact about the network; 'powered by N nodes worldwide' "
          "reads as 'N nodes answered you', which is false whenever this machine served it")
    check("a locally-capable machine does not credit the network for its own answer",
          '"Answers run here ' in SRC,
          "when this machine serves the model, the header must say so")
    check("...and the network count is still shown, as capacity for bigger models",
          "for bigger models\"" in SRC)
    check("'Powered by' remains only on the branch where the network really does answer",
          SRC.count('"Powered by ') == 1,
          "exactly one use, in the else branch of localCapable")

    print("\n-- never send into a chain that cannot answer")
    check("a block state exists", "function setBlocked" in SRC)
    check("it is driven by real coverage, not a guess",
          "!n.healthy && !localCapable" in SRC)
    check("a locally-capable machine is NOT blocked", "localCapable" in SRC,
          "this machine serving the model itself makes a short network irrelevant")
    check("Stop stays clickable while busy",
          'btn.disabled = !!blockedReason && !v' in SRC,
          "disabling the button mid-request would strand a running generation")
    check("a failed status poll does not block sending",
          "self-inflicted outage" in SRC)
    check("the degraded banner names the missing layers",
          "uncovered_layers" in SRC and "missing " in SRC)
    check("the layer total comes from the coordinator",
          "n.total_layers" in SRC, "a hardcoded 28 goes stale on the next model tier")

    print("\n-- accessibility and mobile")
    check("the thread is a live region", 'aria-live="polite"' in SRC)
    check("icon-only buttons are labelled", SRC.count("aria-label=") >= 3)
    check("the composer is labelled", 'aria-label="Message"' in SRC)
    check("the app uses dvh so the composer clears the address bar",
          "height:100dvh" in SRC)
    check("...with a vh fallback for browsers without dvh",
          "height:100vh;height:100dvh" in SRC)
    check("motion can be turned off", "prefers-reduced-motion" in SRC)

    print("\n-- the palette is NEURON's, in both themes")
    light, dark = var_block(":root{"), var_block(':root[data-theme="dark"]')
    check("light brand is the site green", light["brand"] == "#15803d", light.get("brand"))
    check("no indigo anywhere", "#4f46e5" not in SRC.replace("brand was indigo #4f46e5", ""))
    check("dark mode does not reuse the light green",
          dark["brand"] != light["brand"],
          "#15803d is a text colour on white and is unreadable on #0e1116")
    r_light = contrast(light["brand"], light["on-brand"])
    r_dark = contrast(dark["brand"], dark["on-brand"])
    check(f"Send button passes AA in light ({r_light:.2f}:1)", r_light >= 4.5)
    check(f"Send button passes AA in dark ({r_dark:.2f}:1)", r_dark >= 4.5,
          "white on the dark theme's green was 2.9:1 before --on-brand existed")
    check("the warning strip is themed, not a hardcoded near-white",
          "--warn-bg" in SRC and "background:var(--warn-bg)" in SRC)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
