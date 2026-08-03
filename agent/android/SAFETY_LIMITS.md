# Android safety limits — the specification

A phone is not a small laptop. It has no fan, a battery physically pressed against the SoC, and
it spends its nights on a charger under a pillow. Sustained compute on a phone can do three
things a desktop cannot: permanently degrade the battery, permanently degrade the phone, and —
in the worst combination of heat and insulation — hurt the person holding it.

So the Android client's limits are not tuned for throughput. They are tuned so that a phone
that contributes for two years is indistinguishable from one that never did.

This file is the specification the client implements. [ANDROID_INSTALL.md](ANDROID_INSTALL.md)
is the same rules written for the phone's owner.

---

## 1. The governing decisions

**Contribute only in the trickle phase, never during bulk charge.** A phone charging from 20%
to 80% is already dissipating the most heat it will produce all night — fast-charge losses land
directly on the battery. Adding compute on top stacks two heat sources at the worst moment.
The client therefore waits for **battery ≥ 80% and plugged in**: the charge current has dropped
to a trickle, the phone is thermally quiet, and the compute has the whole thermal budget to
itself. This costs some contributing hours per night and is not negotiable.

**Wireless charging is opt-out by default.** Inductive coupling loses energy as heat *at the
back of the phone*, exactly where the battery is. Qi charging routinely runs 3–5 °C hotter than
cable for the same delivered energy. Contributing on a wireless pad is allowed only if the owner
turns it on, with a warning.

**There is no `max` level on Android.** The desktop guard has one (`agent/resource_guard.py`)
because a server has no owner, no battery and usually a fan. A phone has all three problems.
Porting `max` to Android would mean shipping a setting whose only function is to damage the
device, so the Android client stops at `generous`.

**Half the cores, never all of them.** Saturating every core pins the SoC at its thermal
ceiling and makes the phone hot to hold. The worker takes `min(4, ceil(cores / 2))` threads and
requests the efficiency cluster where the scheduler allows it.

**Every rail fails closed.** If a sensor cannot be read, if the thermal API is missing, if the
battery temperature returns nonsense — the answer is *pause*, not *proceed*. This is the
opposite of the desktop GPU rule (`gpu.py`: "cannot tell" must never mean "pause"), and
deliberately so. On desktop, failing closed would silently empty the network; on a phone,
failing open risks the hardware. Different stakes, different default.

---

## 2. Thermal rails

Battery temperature comes from `ACTION_BATTERY_CHANGED` → `EXTRA_TEMPERATURE` (tenths of °C).
System thermal pressure comes from `PowerManager.getCurrentThermalStatus()` and
`addThermalStatusListener()` (API 29+). **Both** are consulted; whichever says pause, wins.

| Constant | Value | Behaviour |
|---|---|---|
| `TEMP_THROTTLE_C` | 36.0 | Drop to 50% duty cycle. Notification shows "easing off — phone warm". |
| `TEMP_PAUSE_C` | 38.0 | Pause. Nothing runs. |
| `TEMP_RESUME_C` | 34.0 | Resume permitted (4 °C hysteresis — never resume at the pause threshold, it oscillates and each cycle is another heat spike). |
| `TEMP_SESSION_STOP_C` | 41.0 | Stop for the rest of this charge session. No resume until unplugged and replugged. |
| `TEMP_FAULT_C` | 45.0 | Permanent disable. See §5. |
| `TEMP_COLD_PAUSE_C` | 5.0 | Pause. Charging a lithium cell near or below 0 °C causes lithium plating — permanent capacity loss and a genuine safety defect. 5 °C is the margin. |

Thermal status mapping (API 29+):

| `getCurrentThermalStatus()` | Action |
|---|---|
| `NONE` (0), `LIGHT` (1) | run |
| `MODERATE` (2) | pause — the system is already throttling for its own reasons |
| `SEVERE` (3) and above | stop for the session, and log it |

**Android 8.0–9.0 have no thermal status API.** On those versions the client runs on battery
temperature alone, with every threshold lowered by 1 °C, and says so on the settings screen. It
does not pretend to have a signal it cannot read.

---

## 3. Charge and battery rails

| Constant | Value | Behaviour |
|---|---|---|
| `MIN_LEVEL_PLUGGED_PCT` | 80 | Below this while charging: wait. Bulk charge finishes first (§1). |
| `BATTERY_FLOOR_DEFAULT_PCT` | 50 | On-battery contribution stops here. Owner-adjustable. |
| `BATTERY_FLOOR_MIN_PCT` | 30 | The floor cannot be set lower. Deep discharge cycles age the cell fastest. |
| `WIRELESS_ALLOWED_DEFAULT` | false | Opt-in, with a heat warning. |
| `MIN_FREE_RAM_MB` | 500 | Same rail as the desktop guard. |

`EXTRA_HEALTH` is read on every battery broadcast. `BATTERY_HEALTH_OVERHEAT`,
`BATTERY_HEALTH_DEAD`, `BATTERY_HEALTH_OVER_VOLTAGE` and `BATTERY_HEALTH_UNSPECIFIED_FAILURE`
all trigger permanent disable (§5) — a phone reporting a battery fault must not be given extra
work, whatever the temperature reads.

**Cycle cost is disclosed, because it is real.** On-battery contribution at `generous` can add
roughly one full charge cycle per day. Phone cells are typically rated for several hundred
cycles to 80% capacity, so a year of that is a measurable share of the battery's life. The app
states this before the owner enables it; it does not bury it here.

---

## 4. Load shaping

| Constant | Value |
|---|---|
| `MAX_THREADS` | `min(4, ceil(cores / 2))` |
| `DUTY_WORK_MS` / `DUTY_REST_MS` | 90 000 / 30 000 (75% duty) |
| `DUTY_WORK_MS` above `TEMP_THROTTLE_C` | 90 000 / 90 000 (50% duty) |
| `TARGET_SUSTAINED_C` | 36.0 — the temperature the duty cycle is trying to hold below |

The rest interval is not idle padding; it is what lets the case shed heat between batches. A
phone at 75% duty and 36 °C contributes more over a night than one that runs flat out, hits
38 °C in twenty minutes and then sits paused until morning.

Scheduling uses `WorkManager` constraints so the **OS** enforces the important ones rather than
our code alone:

```kotlin
Constraints.Builder()
    .setRequiresCharging(true)
    .setRequiresBatteryNotLow(true)
    .setRequiredNetworkType(if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
    .setRequiresDeviceIdle(level == Level.IDLE)
    .build()
```

Doze and the OEM battery managers are never fought or worked around. If the system stops the
worker, it stays stopped.

---

## 5. The warning system

Five tiers. Every one of them names the reason in plain words — the app must never simply stop.

**W0 — state, always visible.** The persistent foreground-service notification shows one of:
`contributing`, `waiting` (+ what for), `easing off` (+ why), `paused` (+ why). Battery
temperature is shown as a number whenever the state is anything other than `contributing`.

**W1 — consent gates, before the fact.** A one-time dialog when the owner first enables each of
these, naming the specific cost. Confirmation is required; there is no "don't show again" on the
battery and wireless ones:

| Setting | The dialog must state |
|---|---|
| Balanced / on-screen use | The phone will feel slower in games and camera, and charge more slowly. |
| Generous (on battery) | About one extra charge cycle per day; measurable battery-life cost over a year; the phone will run warmer and discharge visibly faster. |
| Wireless charging | Charges the battery 3–5 °C hotter than cable for the same energy; wired is recommended. |
| Mobile data | Uses the owner's data allowance, including the ~800 MB first download. Cap required. |

**W2 — live caution.** Above `TEMP_THROTTLE_C` the notification changes to `easing off — phone
warm (37 °C)`. If the phone stays above the throttle point for 30 minutes, a dismissible heads-up
notification suggests removing the case or uncovering the phone.

**W3 — automatic stop.** Any rail in §2 or §3 pauses the work and states which one. At
`TEMP_SESSION_STOP_C`, the notification is not dismissible and reads: *stopped for tonight —
your phone reached 41 °C. Check it isn't covered or in direct sun.*

**W4 — lockout.** Three session stops within 24 hours disable contribution for 24 hours, with a
notification explaining that the phone is repeatedly overheating and that this is usually
covering, a thick case, a failing charger, or a battery reaching end of life.

**W5 — permanent disable.** On `TEMP_FAULT_C` or any faulty `EXTRA_HEALTH` value, contribution
is disabled permanently and cannot be re-enabled from within the app. The notification advises
having the battery checked. This is the one state the owner cannot override, because the two
readings that trigger it are the two that precede a battery failure.

**First-run acknowledgement.** Before the first contribution ever starts, the owner sees the
physical-safety screen (§6) and must acknowledge it. It is not a EULA scroll-wrap; it is five
lines.

---

## 6. Physical safety copy

Shown at first run, and available at any time from the app. Wording matters more than
completeness here — this is the part a person actually has to act on.

- **Don't cover the phone while it is contributing.** Not under a pillow, in bedding, or down
  the side of a sofa. Charging and computing both make heat, and a covered phone cannot lose it.
- **Keep it out of direct sun and out of parked cars.**
- **Use the charger that came with the phone, or a reputable replacement.** A failing charger is
  the most common cause of a phone that runs hot on charge.
- **A thick case traps heat.** If the app keeps easing off overnight, take the case off.
- **Stop, unplug, and have the phone looked at if** the back is bulging or the screen is lifting
  at the edges, if it is too hot to hold comfortably, or if there is any smell or hissing. Do not
  keep charging it. Contribution is not worth a damaged cell, and a swollen battery is a fire
  risk regardless of what caused it.

---

## 7. What this costs the network

These limits mean a phone contributes for perhaps 4–6 hours of a 10-hour charge session, at 75%
duty, on half its cores. That is a small fraction of what the hardware could deliver, and it is
the correct trade: NEURON's proposition is that idle devices contribute at no cost to their
owners. A phone with a degraded battery a year in would falsify that proposition, and no amount
of throughput buys it back.
