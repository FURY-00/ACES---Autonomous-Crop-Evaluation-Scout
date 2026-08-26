# GPS field test — step by step

Goal: sit in a field with a laptop, drive nothing, and watch the bot's real
position move on a live map.

---

## PART A — At home, before you leave (30 min)

Do all of this indoors. The GPS will not fix inside, but everything else can
be proven, and debugging wiring in a field with a dying laptop is miserable.

### A1. Wire the NEO-M8N

| NEO-M8N | Raspberry Pi |
|---|---|
| VCC | pin 1 (3.3 V) — **not 5 V** |
| GND | pin 6 |
| TX  | pin 10 (GPIO 15, Pi RX) |
| RX  | pin 8  (GPIO 14, Pi TX) |

TX goes to RX. Crossed. Getting this backwards is the single most common
reason a GPS "does not work".

**The antenna must face the sky with nothing above it.** Not sideways, not
under the chassis, not under a lid. Keep it as far from the Pi as the cable
allows: the Pi (and especially anything plugged into a USB 3 port) emits
broadband noise sitting right on the GPS band.

### A2. Free up the serial port

The Pi uses the UART for a login console by default, which fights the GPS
for the same pins.

    sudo raspi-config

    Interface Options -> Serial Port
      "Would you like a login shell over serial?"      -> NO
      "Would you like the serial port hardware enabled?" -> YES

    sudo reboot

After rebooting:

    ls -l /dev/serial*

You want `serial0` to exist. On a Pi 4 it usually points at `ttyS0`.

### A3. Prove bytes are arriving

    cd ~/Desktop/acesss/aces
    python3 tools/gps_test.py --raw --port /dev/serial0

| What you see | What it means |
|---|---|
| `$GNGGA,...` lines | Working. Move on. |
| Nothing at all | Wiring, or the console is still enabled |
| Garbage characters | Wrong baud — try `--baud 9600` then `38400` |
| GGA lines with empty lat/lon | Module is fine, just no fix. Correct indoors. |

Do not leave the house until you see NMEA sentences.

### A4. Set the port permanently

Edit `config/settings.py`:

    GPS_PORT = "/dev/serial0"

### A5. First fix, near a window or on a balcony (15 min)

    python3 tools/gps_test.py

**A brand-new NEO-M8N can take 15 minutes for its first ever fix** — it has to
download the satellite almanac at 50 bits per second, and one blocked
satellite restarts the download. Do this once, at home, with the best sky view
you have. Every later fix takes 30–60 s.

Leave it running until you see `FIRST FIX`. If you never get one indoors,
that is normal — but do the 15 minutes somewhere outdoors before the field
trip, or you will spend your first 15 minutes there waiting.

---

## PART B — Networking: how your laptop reaches the Pi in a field

There is no wifi in a field. Pick one of these **and test it at home**.

### Option 1 — Phone hotspot (easiest)

1. At home, turn on your phone's hotspot.
2. Connect the Pi to it (desktop wifi menu, or `sudo raspi-config` ->
   System Options -> Wireless LAN). The Pi remembers it.
3. Connect your laptop to the same hotspot.
4. On the Pi: `hostname -I` — note the address.

In the field: turn the hotspot on, power the Pi, wait ~40 s, and it rejoins
automatically. **Mobile data does not need to be on.** The hotspot is just a
local network. Battery cost is small.

### Option 2 — Ethernet cable, laptop to Pi

No phone needed. Plug a cable between them, then on the Pi:

    sudo nano /etc/dhcpcd.conf
    # add at the end:
    interface eth0
    static ip_address=192.168.50.1/24

Set your laptop's ethernet to a static `192.168.50.2/24`. SSH to
`192.168.50.1`. Completely reliable, but you are tethered by a cable.

### Option 3 — Pi as its own hotspot

Most convenient in the field, most setup time. Only do this if you have a
spare evening.

### Finding the Pi if you forget the IP

    ssh pi@aces.local            # works if avahi/mDNS is running
    ping aces.local

Better: **write the IP on masking tape and stick it to the Pi.**

---

## PART C — SSH from your laptop

### C1. Enable SSH on the Pi (once)

    sudo raspi-config     # Interface Options -> SSH -> Yes

### C2. Connect

Windows PowerShell, macOS, or Linux — all the same:

    ssh pi@192.168.x.x

Use your actual username if it is not `pi`.

### C3. Use tmux. This is not optional in a field.

If your SSH drops — laptop sleeps, you walk out of range, the hotspot
hiccups — **every program you started dies with it**. In a field that happens
constantly.

    sudo apt install tmux      # once, at home

Then, every time:

    ssh pi@192.168.x.x
    tmux new -s gps            # start a named session
    cd ~/Desktop/acesss/aces
    python3 tools/gps_test.py

If SSH drops, the program keeps running. Reconnect and:

    tmux attach -t gps

Detach deliberately with **ctrl-b then d**. The program keeps running.

---

## PART D — In the field

### D1. Setup

1. Power the Pi (power bank is fine — GPS testing draws very little).
2. Phone hotspot on. Wait ~40 s for the Pi to join.
3. `ssh pi@<ip>` from the laptop, then `tmux new -s gps`.
4. **Antenna up, clear sky, away from your body.** Standing over it with a
   laptop is enough to degrade the fix.

### D2. Run the static accuracy test first

Put the bot down. Do not touch it.

    cd ~/Desktop/acesss/aces
    python3 tools/gps_test.py --static 300

Five minutes, completely still. It then reports how far the reported position
wandered while nothing moved. **That wander is your real accuracy** — no
software fixes it, and it decides whether you can honestly say "this pin is
that plant" or only "this pin is that corner of the field".

| Median error | Verdict |
|---|---|
| under 2.5 m | good for a NEO-M8N |
| 2.5 – 6 m | typical; good for zones, not individual plants |
| over 6 m | something is wrong — see D5 |

It also prints "phantom distance": how far GPS *thinks* the parked bot
travelled. That number is exactly why the robot uses wheel odometry for
distance and GPS only for pinning.

### D3. Walk the track

    python3 tools/gps_test.py

Open `http://<pi-ip>:8081` on the laptop.

**Switch to the SURVEY view** using the toggle at the top right. Map tiles
need internet; SURVEY draws everything in local metres from your first fix as
a plain SVG and needs no network at all. In a field it is also the more useful
view.

Now carry the bot in a rectangle, roughly 20 x 10 m, walking slowly and
pausing at each corner. Watch the green track draw itself.

Compare the corners on screen with where you actually turned. That comparison
is the honest measure of the system, and it is a good thing to show a teacher.

### D4. Everything is logged

Each run writes `data/gps_test_<timestamp>.csv` with lat, lon, satellites,
HDOP, and per-sample step distance. Pull it to your laptop afterwards:

    scp pi@192.168.x.x:~/Desktop/acesss/aces/data/gps_test_*.csv .

### D5. If accuracy is poor

In order of how often each is the culprit:

1. **Antenna orientation.** Ceramic patch antennas are directional. Flat
   side up, facing the sky, nothing above it.
2. **Proximity to the Pi.** Move the module 15–20 cm away. USB 3 ports are
   the worst offenders.
3. **Satellite count.** Under 6 sats or HDOP above 2, wait longer. Both are
   on screen every second.
4. **Buildings and trees.** Signals bounce off walls and arrive late, which
   the receiver reads as extra distance. Open ground, away from edges.
5. **Cold almanac.** If this module has never had a long outdoor session,
   give it 15 minutes untouched with clear sky, once.

---

## Quick reference

    # raw NMEA, for when nothing works
    python3 tools/gps_test.py --raw

    # stationary accuracy, 5 minutes
    python3 tools/gps_test.py --static 300

    # live track + map
    python3 tools/gps_test.py
    # then http://<pi-ip>:8081  -> SURVEY view

    # tmux
    tmux new -s gps        # start
    ctrl-b then d          # detach, program keeps running
    tmux attach -t gps     # come back
