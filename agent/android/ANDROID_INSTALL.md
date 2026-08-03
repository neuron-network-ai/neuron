# Install NEURON on Android

Your Android phone contributes spare compute to the NEURON network. **You decide how much.**

Out of the box it is deliberately timid — WiFi only, charging only, and only while you are not
using the phone. Every one of those is a setting you can change in the app, and this page tells
you what each one costs you.

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

Find NEURON in your app drawer and open it. Sign in with Google or GitHub.

**Step 5 — Let it run**

Plug your phone in and connect to WiFi. NEURON starts contributing automatically. A notification
shows your node status. Your phone is verified by the network within about a minute — then it
earns.

To give more than the default, open **Settings → Contribution** and read on.

---

## How much you give

### Contribution level

One choice that sets everything at once. Same four levels as the desktop agent.

| Level | Runs when | Screen on | Battery | Earns |
|---|---|---|---|---|
| **Idle** *(default)* | Charging, screen off | pauses | never | least |
| **Balanced** | Charging | keeps going | never | more |
| **Generous** | Charging, or on battery above your floor | keeps going | yes, down to your floor | more still |
| **Max** | Always | keeps going | yes, down to 15% | most |

**Idle** is the default because it is the only level with no cost to you: the phone is asleep on
a charger and would otherwise be doing nothing.

**Balanced** contributes while you are using the phone. Expect it to feel slightly less
responsive in heavy apps — games and camera especially — and to charge more slowly.

**Generous** and **Max** run on battery. Your phone will lose charge measurably faster and run
warmer. Pick these on a spare phone, not your only one.

### Switches

Independent of the level. Change any of them at any time.

| Setting | Default | What turning it off/up costs you |
|---|---|---|
| **WiFi only** | on | Off = contributes over mobile data. **This uses your data allowance**, including the ~800 MB first download. Set a monthly cap below. |
| **Monthly mobile data cap** | 2 GB | Only applies when *WiFi only* is off. Contributing stops for the month when reached, and resets on your billing day. |
| **Pause while I'm using the phone** | on at Idle | Off = keeps working while the screen is on. Costs responsiveness, not battery (you are charging). |
| **Battery floor** | 40% | Only applies at Generous/Max. Contributing stops below this. Lower it for more earnings and less usable phone. Cannot go below 15%. |
| **Pause during my hours** | off | Set a window (e.g. 08:00–18:00) where the phone never contributes, whatever the level says. |

### Rails you cannot switch off

These are not contribution choices, they are protection for the hardware, and they apply at
every level including Max:

- **Battery temperature above 40 °C** — pauses until it cools. Sustained heat permanently
  degrades a phone battery, so this is not yours to override.
- **Battery below 15%** — pauses, regardless of your floor.
- **Android's own battery saver or Doze** — the OS wins; NEURON does not fight it.
- **Less than ~500 MB free RAM** — pauses, same rail the desktop agent has.

The app's main screen always shows which of these is why it is paused, in plain words — never a
silent stop.

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

**App says "not contributing" even when charging:**
Check the main screen — it names the reason. Most often: WiFi not connected, or Android's
battery saver is on, which kills background apps.

**It never runs while I'm using the phone:**
That is the Idle default. Settings → Contribution → Balanced.

**It stopped and says the data cap is reached:**
You have *WiFi only* off and hit your monthly mobile cap. Raise the cap, or connect to WiFi.

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
