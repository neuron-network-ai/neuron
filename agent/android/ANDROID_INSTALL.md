# Install NEURON on Android

Your Android phone contributes spare compute to the NEURON network. **You decide how much**, and
the app is built so that no setting you can reach will damage your phone.

Out of the box it is deliberately timid: WiFi only, charging only, only once the phone is
already charged past 80%, and only while you are not using it. Everything below tells you what
each setting costs you before you change it.

## What you need

- Android 8.0 or newer
- WiFi connection (or mobile data, if you turn that on — see [How much you give](#how-much-you-give))
- 2 GB free storage
- No root required

## What you get

- Your phone earns compute credits while it contributes
- A private AI chat at the local address shown in the app
- Nothing collected about you
- Full control over when it runs, and a single switch to stop it

## Honest expectations

- First start downloads about 800 MB (your slice of the model)
- Earnings are small while the network is small — they grow as more devices join
- Compute credits have no cash value today
- On a normal night your phone contributes for **4–6 hours**, not the whole night. It waits for
  the battery to finish charging, works in short bursts, and stops when it warms up. That is the
  design, not a fault.
- The app is unsigned — Android will warn you before installing

## Install

**Step 1 — Allow installs from this source**

On your phone:
Settings → Apps → Special app access → Install unknown apps → Your browser → Allow

**Step 2 — Download the APK**

Open your phone browser and go to:
https://github.com/neuron-network-ai/neuron/releases

Tap the latest `NEURON-android-vX.X.X.apk` file. Tap Download. When it finishes, tap Open.

SHA-256 is shown on the releases page — verify it matches before installing if you want to be
sure the file is genuine.

**Step 3 — Install**

Tap Install when Android asks. Tap Done when it finishes.

**Step 4 — Open NEURON**

Find NEURON in your app drawer and open it. Sign in with Google or GitHub. Read the safety
screen — it is five lines and you only see it once.

**Step 5 — Let it run**

Plug your phone in and connect to WiFi. Once the battery passes 80%, NEURON starts contributing.
A notification shows what it is doing at all times. Your phone is verified by the network within
about a minute — then it earns.

---

## Keeping your phone safe

Read this once. It is the part that actually needs you.

- **Don't cover the phone while it is contributing.** Not under a pillow, not in bedding, not
  down the side of a sofa. Charging makes heat and computing makes heat; a covered phone cannot
  get rid of either.
- **Keep it out of direct sun and out of parked cars.**
- **Use the charger that came with the phone, or a reputable replacement.** A failing charger is
  the most common reason a phone runs hot on charge.
- **A thick case traps heat.** If the app keeps saying it is easing off overnight, take the case
  off and see if that fixes it.
- **Stop, unplug, and get the phone looked at if** the back is bulging, the screen is lifting at
  the edges, it is too hot to hold comfortably, or there is any smell or hissing. Do not carry
  on charging it. A swollen battery is a fire risk whatever caused it, and no amount of compute
  credit is worth one.

## What the app will tell you

It never just stops. Every pause names its reason on the notification and the main screen.

| You'll see | It means |
|---|---|
| **contributing** | Working normally. |
| **waiting — battery at 63%** | Charging still. It starts at 80%, so the charge finishes first (that is when your phone is hottest). |
| **easing off — phone warm (37 °C)** | Halved its own workload to cool down. Normal on a warm night. |
| **paused — 38 °C** | Too warm. It resumes on its own at 34 °C. |
| **stopped for tonight — reached 41 °C** | Something is wrong: covered, in the sun, a bad charger, or a tired battery. Unplug and replug to reset. |
| **disabled for 24 hours** | Third overheat in a day. Check the causes above before re-enabling. |
| **disabled — have your battery checked** | Your phone reported a battery fault, or hit 45 °C. This one cannot be turned back on from the app, on purpose. |

Before you turn any setting up, the app shows you a one-time dialog naming exactly what it costs
you. Those dialogs are not skippable for the on-battery and wireless-charging settings.

---

## How much you give

### Contribution level

One choice that sets everything at once.

| Level | Runs when | While you use the phone | On battery | Earns |
|---|---|---|---|---|
| **Idle** *(default)* | Plugged in, past 80%, screen off | pauses | never | least |
| **Balanced** | Plugged in, past 80% | keeps going | never | more |
| **Generous** | Plugged in, or on battery above your floor | keeps going | yes, to your floor | most |

**Idle** is the default because it is the only level with no cost to you at all: the phone is
charged, asleep, on a charger, and would otherwise be doing nothing.

**Balanced** works while you use the phone. Expect it to feel slightly less responsive in heavy
apps — games and camera especially — and to charge a little more slowly.

**Generous** runs on battery. Be clear about the price: it can add **roughly one extra charge
cycle a day**. Phone batteries are rated for a few hundred cycles before they lose noticeable
capacity, so a year of this uses up a real share of your battery's life. It is the right setting
for a spare phone in a drawer, and the wrong one for the phone you depend on.

**There is no "maximum" level on Android.** The desktop agent has one, for servers with fans and
no battery. On a phone its only function would be to wear the hardware out, so it isn't offered.

### Switches

| Setting | Default | What changing it costs you |
|---|---|---|
| **WiFi only** | on | Off = contributes over mobile data. **Uses your data allowance**, including the ~800 MB first download. |
| **Monthly mobile data cap** | 2 GB | Only applies when *WiFi only* is off. Stops for the month when reached; resets on your billing day. |
| **Battery floor** | 50% | Only applies at Generous. Contribution stops below this. **Cannot be set below 30%** — deep discharges age a battery fastest. |
| **Allow on wireless charging** | off | Wireless charging runs the battery about 3–5 °C hotter than a cable for the same energy, and heat is what kills batteries. Wired is recommended. |
| **Quiet hours** | off | A window (e.g. 08:00–18:00) where the phone never contributes, whatever the level says. |

### Limits you cannot change

These protect the hardware, not the network, so they apply at every level:

- **Waits for 80% charge** before starting — bulk charging is when your phone is hottest, and
  stacking compute on top of it is the single worst thing this app could do.
- **Eases off at 36 °C, pauses at 38 °C, stops for the night at 41 °C.**
- **Pauses below 5 °C** — charging a cold battery causes permanent damage.
- **Stops permanently if your phone reports a battery fault**, or hits 45 °C.
- **Never uses more than half your phone's cores**, and works in bursts rather than
  continuously, so it doesn't sit at its thermal limit.
- **Android's own battery saver and Doze always win.** NEURON does not work around them.
- **Any sensor it can't read counts as a reason to stop.** If it doesn't know the temperature,
  it doesn't run.

The exact thresholds, and the reasoning behind each one, are in
[SAFETY_LIMITS.md](SAFETY_LIMITS.md).

---

## Check your earnings

Open the NEURON app at any time. Your compute credit balance is shown on the main screen.

## Stop contributing

Toggle the switch in the app to pause. At the default level, unplugging the phone also stops it.

## Remove everything

Settings → Apps → NEURON → Uninstall

This removes the app, your config, and your model slice. Your compute credit balance stays on
the network ledger — it is tied to your account, not your device.

## Privacy

- Your prompts never leave your device (your phone is not a driver node — it only processes
  opaque numbers)
- No location data collected
- No contacts, camera or microphone access
- Mobile data is off by default and only ever used if you turn it on
- Source code: github.com/neuron-network-ai/neuron

## Troubleshooting

**It says "waiting" even though it's plugged in:**
It starts at 80% battery, not at 0%. This is deliberate — see the safety limits above.

**It keeps saying "easing off" or "paused" all night:**
The phone is warm. Take the case off, make sure it isn't under bedding, and try a different
charger. If it still happens, the battery may be near end of life.

**It never runs while I'm using the phone:**
That is the Idle default. Settings → Contribution → Balanced.

**It stopped and says the data cap is reached:**
You have *WiFi only* off and hit your monthly mobile cap. Raise the cap, or connect to WiFi.

**It won't let me turn contribution back on:**
Your phone reported a battery fault or reached 45 °C. That lockout is intentional and cannot be
cleared from the app. Have the battery checked.

**Download stuck or slow:**
The first download is 800 MB. Leave it connected and plugged in.

**Node not appearing in the network:**
Wait 2 minutes after first start. The network verifies new nodes automatically.
Check the dashboard: neuronnet.duckdns.org/dashboard

**Notification disappeared:**
Some Android manufacturers (Samsung, Xiaomi, Oppo) aggressively kill background apps.
Go to Settings → Battery → NEURON → set to "Unrestricted" or "No restrictions".

---

Apache 2.0 — github.com/neuron-network-ai/neuron
© 2026 NEURON Labs, Rotterdam
