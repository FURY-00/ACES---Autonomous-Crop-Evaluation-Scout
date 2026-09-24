"""
Find the steering trim for a chassis that pulls to one side.

Your bot is slightly out of square, so it curves even when told to go
straight. The vision controller then spends its authority fighting that bias
instead of following the row. Trim cancels it with a fixed offset.

Usage
-----
    python3 tools/trim_finder.py --port /dev/ttyACM0

It drives straight for 3 seconds at a time with a trim value you choose, so
you can see which way it curves and by how much. No cameras involved -- this
is purely about the chassis.

    ENTER       run again with the same trim
    a number    set a new trim and run (e.g.  8   or  -12)
    q           quit

Method
------
    1. clear floor, at least 3 m, bot pointing at a mark on the far wall
    2. run with trim 0, watch which way it curves
    3. curves LEFT  -> try a POSITIVE trim (steers right)
       curves RIGHT -> try a NEGATIVE trim
    4. increase in steps of 5 until it tracks straight
    5. put that number into your run command:  --trim 8
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()

    import serial
    ser = serial.Serial()
    ser.port = a.port
    ser.baudrate = 115200
    ser.timeout = 0.2
    ser.dtr = False          # do not reset the ESP32 on open
    ser.rts = False
    ser.open()
    time.sleep(1.0)
    ser.reset_input_buffer()
    print(f"connected on {a.port}\n")

    trim = 0
    print("Put the bot on a clear floor pointing at a mark 3 m away.")
    print("Watch which way it curves.\n")

    try:
        while True:
            cmd = input(f"trim {trim:+d}  [ENTER=run, number=set, q=quit] > ").strip()
            if cmd.lower() == "q":
                break
            if cmd:
                try:
                    trim = int(cmd)
                except ValueError:
                    print("  give a whole number, like 8 or -12")
                    continue

            print(f"  running {a.seconds:.0f}s at trim {trim:+d} ...")
            t0 = time.time()
            while time.time() - t0 < a.seconds:
                ser.write(f"$S,{trim}\n".encode())
                ser.write(b"$GO\n")
                time.sleep(0.1)
            for _ in range(5):
                ser.write(b"$STOP\n")
                time.sleep(0.05)
            print("  stopped.\n")
            print("  curved LEFT  -> try a MORE POSITIVE trim")
            print("  curved RIGHT -> try a MORE NEGATIVE trim")
            print("  straight     -> that is your number\n")
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            ser.write(b"$STOP\n")
            time.sleep(0.05)
        ser.close()
        print(f"\nuse it like this:")
        print(f"  python3 run_mission.py --webcam 1 --no-plant-check "
              f"--port {a.port} --trim {trim}")


if __name__ == "__main__":
    main()
