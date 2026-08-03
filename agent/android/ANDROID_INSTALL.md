# Install NEURON on Android

Your Android phone contributes spare compute 
to the NEURON network while it charges. 
Nothing runs on mobile data. Nothing runs 
while you are using the phone.

## What you need

- Android 8.0 or newer
- WiFi connection
- Phone charging (plugged in)
- 2 GB free storage
- No root required

## What you get

- Your phone earns compute credits while 
  charging overnight
- A private AI chat at the local address 
  shown in the app
- Nothing collected about you
- Stops automatically when you unplug

## Honest expectations

- First start downloads about 800 MB 
  (your slice of the model)
- Earnings are small while the network 
  is small — they grow as more devices join
- Compute credits have no cash value today
- The app is unsigned — Android will warn 
  you before installing

## Install

**Step 1 — Allow installs from this source**

On your phone:
Settings → Apps → Special app access → 
Install unknown apps → Your browser → Allow

**Step 2 — Download the APK**

Open your phone browser and go to:
https://github.com/neuron-network-ai/neuron/releases

Tap the latest NEURON-android-vX.X.X.apk file.
Tap Download. When it finishes, tap Open.

SHA-256 is shown on the releases page — 
verify it matches before installing if you want 
to be sure the file is genuine.

**Step 3 — Install**

Tap Install when Android asks.
Tap Done when it finishes.

**Step 4 — Open NEURON**

Find NEURON in your app drawer and open it.
Sign in with Google or GitHub.

**Step 5 — Let it run**

Plug your phone in and connect to WiFi.
NEURON starts contributing automatically.
A notification shows your node status.
Your phone is verified by the network 
within about a minute — then it earns.

## How it behaves

```
Charging + WiFi connected → contributing
Battery low or unplugged → paused automatically
Mobile data only → paused automatically
Phone too warm (>38°C) → paused automatically
Screen on, you are using it → paused automatically
```

## Check your earnings

Open the NEURON app at any time.
Your compute credit balance is shown 
on the main screen.

## Stop contributing

Toggle the switch in the app to pause.
Or simply unplug your phone — it stops on its own.

## Remove everything

Settings → Apps → NEURON → Uninstall

This removes the app, your config, and your 
model slice. Your compute credit balance 
stays on the network ledger — it is tied to 
your account, not your device.

## Privacy

- Your prompts never leave your device 
  (your phone is not a driver node — 
  it only processes opaque numbers)
- No location data collected
- No contacts, camera or microphone access
- No mobile data used — WiFi only
- Source code: github.com/neuron-network-ai/neuron

## Troubleshooting

**App says "not contributing" even when charging:**
Check WiFi is connected. Check battery saver 
mode is off — Android battery saver kills 
background apps.

**Download stuck or slow:**
The first download is 800 MB. Leave it 
connected and plugged in.

**Node not appearing in the network:**
Wait 2 minutes after first start. The network 
verifies new nodes automatically. 
Check the dashboard: neuronnet.duckdns.org/dashboard

**Notification disappeared:**
Some Android manufacturers (Samsung, Xiaomi, 
Oppo) aggressively kill background apps. 
Go to Settings → Battery → NEURON → 
set to "Unrestricted" or "No restrictions".

---

Apache 2.0 — github.com/neuron-network-ai/neuron  
© 2026 NEURON Labs, Rotterdam
