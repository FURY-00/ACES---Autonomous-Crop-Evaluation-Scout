"""
GPS-only field test. No camera, no ESP32, no motors.

Reads the NEO-M8N, serves a live map at :8081, prints a terminal readout,
and logs the track to CSV. Everything works over SSH with no display.

Usage
-----
    python3 tools/gps_test.py                     # normal test
    python3 tools/gps_test.py --port /dev/serial0
    python3 tools/gps_test.py --raw               # dump raw NMEA (debugging)
    python3 tools/gps_test.py --static 300        # 5-min stationary drift test

The static test is the one that actually tells you something. Leave the bot
completely still and it measures how far the reported position wanders. That
wander is your real-world accuracy: no amount of software fixes it, and it
sets the floor on how precisely you can pin a diseased plant.
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S           # noqa: E402


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def raw_dump(port, baud):
    """Straight NMEA passthrough. Use this when nothing else works."""
    import serial
    print(f"raw NMEA from {port} @ {baud}   (ctrl-c to stop)\n")
    print("You should see lines starting with $GNGGA / $GPGGA within a second.")
    print("Nothing at all      -> wiring, or the serial console is still on.")
    print("Garbage characters  -> wrong baud rate. Try 9600 and 38400.")
    print("GGA with empty      -> module alive but no fix yet. Go outside.")
    print("lat/lon fields         Give it 60-90 s with a clear view of sky.\n")
    ser = serial.Serial(port, baud, timeout=1)
    try:
        while True:
            line = ser.readline().decode("ascii", "ignore").strip()
            if line:
                print(line)
    except KeyboardInterrupt:
        ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=S.GPS_PORT)
    ap.add_argument("--baud", type=int, default=S.GPS_BAUD)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--static", type=int, default=0,
                    help="run a stationary drift test for N seconds")
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args()

    if args.raw:
        raw_dump(args.port, args.baud)
        return

    # Point the shared reader at whichever port the user gave us.
    S.GPS_PORT, S.GPS_BAUD = args.port, args.baud
    from telemetry.gps_reader import GPSReader

    gps = GPSReader(args.port, args.baud)
    if not gps.available:
        print("\nCould not open the serial port. Check, in order:")
        print("  1. ls -l /dev/serial* /dev/ttyAMA*   -- does the port exist?")
        print("  2. sudo raspi-config -> Interface Options -> Serial Port")
        print("        login shell over serial? NO")
        print("        serial hardware enabled?  YES")
        print("  3. groups | grep dialout    -- add yourself if missing:")
        print("        sudo usermod -aG dialout $USER   (then log out and in)")
        print("  4. python3 tools/gps_test.py --raw    -- see if bytes arrive")
        return

    if not args.no_map:
        from telemetry import map_server
        map_server.start()
        print(f"\n  live map: http://<pi-ip>:{S.MAP_PORT}")
        print("  open it on your laptop or phone, and switch to SURVEY view")
        print("  if you have no internet for map tiles.\n")

    os.makedirs(S.SD_ROOT, exist_ok=True)
    csv_path = os.path.join(S.SD_ROOT,
                            f"gps_test_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["t", "lat", "lon", "alt_m", "sats", "hdop", "quality",
                 "fix_ok", "step_m", "dist_from_start_m"])
    print(f"  logging to {csv_path}\n")

    first = None
    prev = None
    total = 0.0
    samples = []
    t0 = time.time()
    n = 0

    print(f"{'time':>6s} {'sats':>4s} {'hdop':>5s} {'fix':>4s} "
          f"{'latitude':>12s} {'longitude':>12s} {'step':>7s} {'total':>8s}")
    print("-" * 70)

    try:
        while True:
            f = gps.read()
            el = time.time() - t0

            if f.stamp == 0:
                print(f"{el:6.0f} {f.sats:4d} {f.hdop:5.1f} {'--':>4s} "
                      f"{'waiting for first NMEA sentence':>36s}", end="\r")
                time.sleep(1.0)
                continue

            step = 0.0
            if f.fix_ok:
                if first is None:
                    first = (f.lat, f.lon)
                    print(f"\n  FIRST FIX after {el:.0f} s: "
                          f"{f.lat:.6f}, {f.lon:.6f}\n")
                if prev:
                    step = haversine_m(prev[0], prev[1], f.lat, f.lon)
                    total += step
                prev = (f.lat, f.lon)
                samples.append((f.lat, f.lon))
                if not args.no_map:
                    from telemetry import map_server
                    map_server.update_bot(lat=f.lat, lon=f.lon, fix_ok=True,
                                          sats=f.sats, state="GPS TEST")

            dist0 = (haversine_m(first[0], first[1], f.lat, f.lon)
                     if (first and f.fix_ok) else 0.0)
            n += 1
            wr.writerow([round(time.time(), 1), f.lat, f.lon, f.alt_m, f.sats,
                         f.hdop, f.quality, f.fix_ok, round(step, 2),
                         round(dist0, 2)])
            if n % 5 == 0:
                fh.flush()

            print(f"{el:6.0f} {f.sats:4d} {f.hdop:5.1f} "
                  f"{'YES' if f.fix_ok else 'no':>4s} "
                  f"{f.lat:12.6f} {f.lon:12.6f} {step:6.2f}m {total:7.1f}m",
                  end="\r")

            # ---- stationary drift test ----------------------------------
            if args.static and el > args.static and len(samples) > 10:
                lat0 = sum(s[0] for s in samples) / len(samples)
                lon0 = sum(s[1] for s in samples) / len(samples)
                errs = sorted(haversine_m(lat0, lon0, a, b) for a, b in samples)
                cep = errs[len(errs) // 2]
                p95 = errs[int(len(errs) * 0.95)]
                print("\n\n" + "=" * 62)
                print(f"STATIONARY DRIFT over {args.static}s, {len(samples)} fixes")
                print("=" * 62)
                print(f"  median error (CEP50)  {cep:6.2f} m")
                print(f"  95th percentile       {p95:6.2f} m")
                print(f"  worst                 {errs[-1]:6.2f} m")
                print(f"  phantom distance      {total:6.1f} m "
                      f"(the bot never moved)")
                print()
                if cep < 2.5:
                    print("  Good for a NEO-M8N. Pins will land on the right plant")
                    print("  to within a couple of metres.")
                elif cep < 6:
                    print("  Typical. Fine for 'which part of the field', not for")
                    print("  'which plant'. Say so when you present it.")
                else:
                    print("  Poor. Check: antenna facing the SKY with nothing above")
                    print("  it, away from the Pi and any USB 3 port (they emit")
                    print("  broadband RF right on the GPS band), and give it a")
                    print("  full 15 min outdoors once so it can download the")
                    print("  almanac.")
                print(f"\n  'phantom distance' is what your odometry would report")
                print(f"  from GPS alone while parked. That is why the robot uses")
                print(f"  wheel odometry for distance and GPS only for pinning.")
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        gps.close()
        print(f"\n\n  {n} samples -> {csv_path}")
        if first and prev:
            print(f"  start {first[0]:.6f},{first[1]:.6f}")
            print(f"  end   {prev[0]:.6f},{prev[1]:.6f}")
            print(f"  straight-line start to end: "
                  f"{haversine_m(first[0], first[1], prev[0], prev[1]):.1f} m")
            print(f"  path length as logged:      {total:.1f} m")


if __name__ == "__main__":
    main()
