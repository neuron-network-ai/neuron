"""coordinator/theme.py — the look the coordinator's pages share with the landing page.

`docs/index.html` (the public site) and `/dashboard` (the live network) are the two things a
person actually looks at, and they looked like different products: the site is a green,
rounded, hairline-ruled page, while the dashboard was unstyled system-grey with Google's
palette (#1a73e8, #f9ab00, #c5221f) left over from its first hour of existence. Anyone
following "Live network dashboard →" from the site crossed a visible seam.

So the palette, type scale and component shapes live here, once, and both dashboards render
through them. Values are lifted directly from `docs/index.html` — if that page changes, change
these to match; they are a copy on purpose (the coordinator VM does not serve the site and must
not fetch anything from it).

**No web fonts.** The landing page loads Inter and Space Grotesk from Google, and its own
comment says what that costs: every visitor's IP reaches Google for a typeface. That trade is
worse here — the dashboard hard-refreshes every 5 seconds, so it would be a repeated
third-party request from a page whose whole subject is a privacy-preserving network. The
families are still *named* first in the stack, so a visitor who has them installed sees them,
and nobody's browser goes and asks for them.
"""

# The palette, named rather than repeated. Straight out of docs/index.html.
BG = "#f6fdf7"           # page
SURFACE = "#fff"         # cards, nav, table headers
LINE = "#d1fae5"         # hairline borders
LINE_SOFT = "#f0fdf4"    # inner table rules
INK = "#1a2e1c"          # body text
MUTED = "#6b7280"        # secondary text
GREEN = "#15803d"        # brand green for TEXT — #16a34a is 3.30:1 on white, under AA
GREEN_DARK = "#166534"   # headings
GREEN_BRIGHT = "#16a34a" # large shapes and the logo mark only
MINT = "#f0fdf4"         # tinted fills
MINT_LINE = "#bbf7d0"

SANS = "'Inter',system-ui,-apple-system,sans-serif"
DISPLAY = "'Space Grotesk',system-ui,-apple-system,sans-serif"

# Soft pills, one per state, in the landing page's `.flow-pill` shape. The old badges were
# solid fills in four saturated colours, which gave a table of ordinary healthy nodes the
# visual weight of an alarm.
PILLS = {
    "online":        (MINT, GREEN_DARK, MINT_LINE),
    "offline":       ("#fef2f2", "#b91c1c", "#fecaca"),
    "trusted":       (MINT, GREEN_DARK, MINT_LINE),
    "verified":      ("#eff6ff", "#1d4ed8", "#bfdbfe"),
    "probationary":  ("#fffbeb", "#92400e", "#fde68a"),
    "flagged":       ("#fef2f2", "#b91c1c", "#fecaca"),
    "serving":       (MINT, GREEN_DARK, MINT_LINE),
    "ready":         ("#eff6ff", "#1d4ed8", "#bfdbfe"),
    "locked":        ("#f9fafb", MUTED, "#e5e7eb"),
}

# The mark from docs/index.html, inline so the coordinator serves no image files and the tab
# icon matches the site.
LOGO = ("<svg width='26' height='26' viewBox='0 0 28 28' aria-hidden='true'>"
        "<circle cx='5' cy='14' r='4' fill='#16a34a'/>"
        "<circle cx='23' cy='5' r='3' fill='#bbf7d0' stroke='#16a34a' stroke-width='1.5'/>"
        "<circle cx='23' cy='23' r='3' fill='#bbf7d0' stroke='#16a34a' stroke-width='1.5'/>"
        "<circle cx='14' cy='14' r='4.5' fill='#22c55e'/>"
        "<line x1='9' y1='14' x2='9.5' y2='14' stroke='#16a34a' stroke-width='2'/>"
        "<line x1='18.5' y1='14' x2='20.5' y2='7' stroke='#16a34a' stroke-width='1.5'/>"
        "<line x1='18.5' y1='14' x2='20.5' y2='21' stroke='#16a34a' stroke-width='1.5'/>"
        "</svg>")

FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 28'"
           "%3E%3Ccircle cx='5' cy='14' r='4' fill='%2316a34a'/%3E%3Ccircle cx='23' cy='5'"
           " r='3' fill='%23bbf7d0'/%3E%3Ccircle cx='23' cy='23' r='3' fill='%23bbf7d0'/%3E"
           "%3Ccircle cx='14' cy='14' r='4.5' fill='%2322c55e'/%3E%3C/svg%3E")

SITE = "https://github.com/neuron-network-ai/neuron"

# Plain string, not an f-string, so the CSS needs no brace-doubling and stays readable.
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:__SANS__;background:__BG__;color:__INK__;line-height:1.6;font-size:14px}
a{color:__GREEN__;text-decoration:none}
a:hover{text-decoration:underline}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:__MINT__;
  border:1px solid __MINT_LINE__;border-radius:4px;padding:1px 5px}

nav{background:__SURFACE__;border-bottom:1px solid __LINE__;padding:0 1.5rem;display:flex;
  align-items:center;justify-content:space-between;height:56px;position:sticky;top:0;z-index:10}
.nav-logo{display:flex;align-items:center;gap:10px;font-family:__DISPLAY__;font-size:17px;
  font-weight:700;color:__GREEN_DARK__;letter-spacing:-.2px}
.nav-links{display:flex;gap:1.25rem;font-size:13px}
.nav-links a{color:__MUTED__}

main{max-width:1080px;margin:0 auto;padding:2rem 1.5rem 3rem}
h1{font-family:__DISPLAY__;font-size:26px;font-weight:700;color:__GREEN_DARK__;
  letter-spacing:-.5px;margin-bottom:.2rem}
.sub{color:__MUTED__;font-size:13px;margin-bottom:1.5rem}
h2{font-size:11px;font-weight:700;color:__GREEN__;text-transform:uppercase;letter-spacing:1.5px;
  margin:2rem 0 .9rem}

.banner{display:inline-flex;align-items:center;gap:9px;color:#fff;padding:.55rem 1.1rem;
  border-radius:8px;font-size:13px;font-weight:600;margin-bottom:1.5rem}
.banner .dot{width:8px;height:8px;border-radius:50%;background:#fff;opacity:.9}

/* Hairline grid: a 1px gap over a __LINE__ background, exactly as the landing page's stats
   strip does it, instead of five separately-bordered boxes. */
/* Flex, not grid, and the reason is the hairline trick itself: the 1px "borders" are the
   container's background showing through the gaps, so any cell a grid leaves empty on the last
   row shows up as a solid green block. Five stats in a four-across viewport did exactly that.
   Flex wrapping lets the last row's items grow to fill it, so there is no hole to tint. */
.stats{display:flex;flex-wrap:wrap;gap:1px;background:__LINE__;border:1px solid __LINE__;
  border-radius:10px;overflow:hidden}
.stat{flex:1 1 150px;background:__SURFACE__;padding:1.15rem 1rem;text-align:center}
.stat .n{font-family:__DISPLAY__;font-size:24px;font-weight:700;color:__GREEN_DARK__;
  letter-spacing:-.5px;line-height:1.25}
.stat .l{font-size:12px;color:__MUTED__;margin-top:3px}

.panel{background:__SURFACE__;border:1px solid __LINE__;border-radius:10px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px 14px;color:__MUTED__;font-weight:600;background:__SURFACE__;
  border-bottom:2px solid __LINE__}
td{padding:10px 14px;border-bottom:1px solid __LINE_SOFT__;color:#374151}
tr:last-child td{border-bottom:none}
td.key{color:__MUTED__;width:190px}
td.nw,th{white-space:nowrap}
td.id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:__INK__;
  white-space:nowrap}
/* Always, not only on phones: the node table has ten columns, so it outgrows a laptop window
   long before it outgrows a phone. Without this it clipped instead of scrolling. */
.table-wrap{overflow-x:auto}

.pill{display:inline-block;border-radius:20px;padding:2px 11px;font-size:11px;font-weight:600;
  border:1px solid transparent;white-space:nowrap}
.tick{color:__GREEN__;font-weight:700}
.dash{color:#d1d5db}

/* Layer coverage. "21/28 covered" says a chain is broken without saying WHERE, and the gap is
   the one thing an operator needs in order to place a node that fixes it. */
.cov{display:flex;flex-wrap:wrap;gap:3px}
.cov i{flex:1 0 22px;max-width:44px;height:28px;border-radius:4px;font-style:normal;
  font-size:10px;font-weight:600;display:flex;align-items:center;justify-content:center;
  background:#dcfce7;color:__GREEN_DARK__;border:1px solid __MINT_LINE__}
.cov i.gap{background:#fef2f2;color:#b91c1c;border-color:#fecaca}
.cov-key{font-size:12px;color:__MUTED__;margin-top:.7rem}

.note{background:__MINT__;border:1px solid __MINT_LINE__;border-radius:8px;padding:.9rem 1.1rem;
  font-size:12.5px;color:#374151;line-height:1.6;margin-top:1.25rem}
.note strong{color:__GREEN_DARK__}
.callout{font-size:13px;color:#374151;margin-top:.9rem}
.callout b{color:__GREEN_DARK__}

footer{border-top:1px solid __LINE__;background:__SURFACE__;padding:1.4rem 1.5rem;
  text-align:center;font-size:12px;color:__MUTED__}

@media(max-width:640px){
  .nav-links a:not(:last-child){display:none}
  main{padding:1.25rem 1rem 2rem}
}
"""
for _name, _value in (("__SANS__", SANS), ("__DISPLAY__", DISPLAY), ("__BG__", BG),
                      ("__SURFACE__", SURFACE), ("__LINE_SOFT__", LINE_SOFT),
                      ("__LINE__", LINE), ("__INK__", INK), ("__MUTED__", MUTED),
                      ("__GREEN_DARK__", GREEN_DARK), ("__GREEN__", GREEN),
                      ("__MINT_LINE__", MINT_LINE), ("__MINT__", MINT)):
    CSS = CSS.replace(_name, _value)


def pill(state, label=None):
    """A soft status pill. Unknown states get the neutral grey rather than no styling."""
    bg, fg, line = PILLS.get(str(state), PILLS["locked"])
    return (f"<span class='pill' style='background:{bg};color:{fg};border-color:{line}'>"
            f"{label or state}</span>")


def banner(healthy, text):
    """The one place a strong fill is still right. A degraded network means no request can
    complete, so it keeps the full-strength red instead of the soft pill treatment."""
    colour = GREEN if healthy else "#b91c1c"
    return f"<div class='banner' style='background:{colour}'><span class='dot'></span>{text}</div>"


def page(title, body, refresh=5, nav_links=True, no_referrer=False):
    """Wrap page content in the shared shell: nav, main, footer.

    `refresh` keeps the existing meta-refresh behaviour (both dashboards advertise
    "auto-refresh 5s" in their subtitle); pass None to switch it off.

    `no_referrer` is for pages whose URL is itself a credential. The private node dashboard is
    reached as `?token=<node token>`, so following ANY link from it hands that token to the
    destination in the Referer header — a nav link to github.com would post a node's token to
    GitHub's logs. Two defences, because the page should not depend on remembering the first:
    such pages carry no outbound nav, AND they tell the browser to send no referrer at all.
    """
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    if no_referrer:
        meta += '<meta name="referrer" content="no-referrer">'
    links = (f"<a href='/dashboard'>Network</a><a href='/docs'>API</a>"
             f"<a href='{SITE}'>GitHub</a>") if nav_links else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{meta}
<link rel="icon" type="image/svg+xml" href="{FAVICON}">
<title>{title}</title>
<style>{CSS}</style></head><body>
<nav>
  <div class="nav-logo">{LOGO}NEURON</div>
  <div class="nav-links">{links}</div>
</nav>
<main>
{body}
</main>
<footer>NEURON — distributed inference on ordinary machines ·
  {f'<a href="{SITE}">source</a> · ' if nav_links else ''}Apache 2.0</footer>
</body></html>"""
