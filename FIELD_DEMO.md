# Field demo — RC driving + live GPS + Telegram map links

What the audience sees:

> The bot is driven around the field on the FlySky. Its position moves live on
> a map on the laptop. You hold a diseased leaf in front of it; it detects the
> disease on its own, photographs it, and a message lands in the Telegram group
> with the photo and a tappable Google Maps link to exactly where it happened.

No autonomous driving, no ESP32 in the loop. Everything that runs is the part
that is actually finished.

---

## PART 1 — Wire the GPS (30 min, do this at home)

Your module is the round M8N-with-compass puck, 6-wire pigtail:
**red, black, white, yellow, green, purple.**

Six wires because it carries two devices: a GPS on UART (4 wires with power)
and a magnetometer on I2C (2 more). **You only need the GPS.** Leave the two
I2C wires unconnected.

### The colour mapping is NOT standard across sellers

Do not trust a diagram you found online for "an M8N". Find out safely instead.

**The safe method: the Pi's RX pin is an INPUT, so nothing can be damaged by
guessing wrong on it.**

1. Connect only these two first:

   | GPS | Pi |
   |---|---|
   | red | pin 4 (5 V) |
   | black | pin 6 (GND) |

   Most of these pucks have an onboard regulator and want 5 V. If yours is
   marked 3.3 V, use pin 1 instead. Power it and the LED should blink.

2. **Leave the Pi's TX (pin 8) disconnected entirely.** The GPS needs no
   commands from you. This removes any chance of shorting two outputs together.

3. Take ONE of the four remaining wires and connect it to **pin 10 (GPIO 15,
   the Pi's RX)**. Then:

       cd ~/acesss/aces
       python3 tools/gps_test.py --raw --port /dev/serial0

   - `$GNGGA,...` lines scrolling → that wire is the GPS TX. Done.
   - Nothing → power off, try the next wire, repeat.

   Four wires, four tries, maybe five minutes. Far faster than guessing.

4. Once found, note the colour. Write it on tape on the module.

### The antenna must face the sky

Flat ceramic side up, nothing above it, and as far from the Pi as the cable
allows. The Pi (especially anything in a USB 3 port) emits broadband noise
sitting right on the GPS band.

### Free the serial port first

    sudo raspi-config
      Interface Options -> Serial Port
        login shell over serial?         NO
        serial hardware enabled?         YES
    sudo reboot

Then set it permanently in `config/settings.py`:

    GPS_PORT = "/dev/serial0"

### First fix takes 15 minutes, ONCE

A new M8N downloads the satellite almanac at 50 bits per second. One blocked
satellite restarts it. **Do this at home, outdoors, the day before** — leave
`python3 tools/gps_test.py` running for 15 minutes with clear sky. Every fix
after that takes 30–60 s.

Skip this and you will burn your first quarter hour in the field staring at
`sats 3`.

---

## PART 2 — Telegram (20 min, at home)

1. Message **@BotFather** on Telegram, send `/newbot`, follow the prompts,
   copy the token.
2. Create a group, add the bot, **make it an admin** — otherwise it cannot
   post reliably in groups.
3. Send any message in the group, then open in a browser:

       https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates

   Find `"chat":{"id":-100...}`. **Group IDs are negative** — include the minus.
4. On the Pi:

       cd ~/acesss/aces
       cp config/secrets.example.py config/secrets.py
       nano config/secrets.py        # paste token and chat id

5. Test it before the field:

       python3 -c "
       import sys; sys.path.insert(0,'.')
       from telemetry.telegram_bot import Telegram
       t=Telegram(); t.message('ACES test'); t.drain()"

   A message must appear in the group. If not, fix it now, not in a field.

Each detection sends **two** messages: the photo with a caption containing a
`maps.google.com` link, and a native Telegram location pin that opens in the
phone's map app. Both are tappable.

---

## PART 3 — Networking in the field

No wifi out there. **Phone hotspot** — and connect the Pi to it AT HOME so it
remembers the network. In the field, turn the hotspot on, power the Pi, wait
~40 s, and it rejoins by itself.

Keep mobile data ON for this demo. You need it for:
- Telegram messages to actually send
- map tiles on the laptop's map view

(The map also has a SURVEY view that needs no internet at all — see Part 5.)

Test the whole hotspot flow at home. `hostname -I` on the Pi while on the
hotspot, and write that IP on tape.

---

## PART 4 — Run it

SSH in from the laptop, then **always inside tmux**, or a dropped connection
kills the run:

    tmux new -s demo
    cd ~/acesss/aces
    python3 run_demo.py --no-drive

`--no-drive` means: no ESP32, no serial port opened, motors never commanded.
You drive entirely on the FlySky. Detection, GPS, storage, map and Telegram
all run.

Detach with **ctrl-b then d**. Reattach with `tmux attach -t demo`.

Two browser tabs on the laptop:

    http://<pi-ip>:8080     live detection view
    http://<pi-ip>:8081     live map

**Wait for the GPS.** The readout line shows `gps N sats hdop X fix True`.
Do not start demoing until `fix True`. Detections without a fix still work but
have no location, which is the whole point of the demo.

---

## PART 5 — The map view

The map has two views, toggled top right:

- **SATELLITE** — real map tiles. Needs internet, so keep hotspot data on.
- **SURVEY** — everything plotted in local metres from your first fix, drawn
  as a plain SVG. Needs no network at all. In an open field this is often the
  clearer view, and it makes the track shape obvious.

Each detection appears in the right-hand log with severity, a severity bar,
and an **OPEN IN GOOGLE MAPS** link with the coordinates.

---

## PART 6 — Running the demo, in order

1. Arrive, power everything, hotspot on.
2. SSH in, `tmux new -s demo`, start `run_demo.py --no-drive`.
3. **Wait for `fix True`.** Talk through the hardware while it acquires.
4. Open both browser tabs. Show the live position moving as you drive.
5. Drive to a spot. Hold the diseased leaf in front of the camera.
6. Detection fires → photo saved → pin on the map → Telegram message.
7. Open the Telegram message on a phone, tap the Google Maps link. **This is
   the moment of the demo.** Let it land.
8. Drive somewhere else, repeat with a second leaf.

The same leaf will not pin twice within 3 m — that is deliberate, or one plant
becomes fifteen pins and the map stops meaning anything. To trigger again,
move a few metres.

---

## Suggestions, including things worth deciding in advance

**Light.** You tuned under a flashlight. Direct sun is a completely different
condition and your thresholds may not hold. Two options, pick one:

- Demo in **open shade** (a building's shadow, under a tree) which is closer
  to your tuning light, or
- Re-tune outdoors beforehand with `tools/tune_live.py`, and save a second
  profile. Doing this once in daylight the day before is 20 minutes well spent.

**Hold the leaf against the sky, not the ground.** Brown lesion against brown
soil is nearly the same colour, and the background model cannot separate them.
Sky or your shirt as a backdrop makes it trivially easy.

**Say the accuracy out loud before anyone asks.** "The NEO-M8N gives us about
2 to 5 metres, so a pin identifies the plant group, not the individual plant.
RTK would fix that and costs about ten times more." Naming a limitation before
it is pointed out reads as engineering judgement. Being caught by it does not.

**Run the whole thing once, start to finish, the day before.** A demo that has
never been run twice will fail.

**Bring:** charged power bank for the Pi, spare LiPo, the flashlight you tuned
under, 3–4 diseased leaves (they wilt), tape with the IP on it, and a phone
with the Telegram group already open.

---

## If something fails mid-demo

| Symptom | Say this, do this |
|---|---|
| No GPS fix | "Cold start, still acquiring." Demo detection + Telegram; the photo and severity still land. |
| Telegram silent | It queues and retries. Show the map instead — the pin is already there. |
| Detection not firing | Move to shade, change the backdrop. Show `:8080`, talk through the mask. |
| SSH drops | `tmux attach -t demo`. It never stopped running. |

---

## Quick reference

    tmux new -s demo
    cd ~/acesss/aces
    python3 run_demo.py --no-drive

    # ctrl-b then d   detach
    tmux attach -t demo

    http://<pi-ip>:8080     detection
    http://<pi-ip>:8081     map   (toggle SURVEY if no internet)
