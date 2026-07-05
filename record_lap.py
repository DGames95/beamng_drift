"""Record a human-driven lap of the handling course to extract its centerline.

We control the BeamNG launch, so the beamngpy server is listening. This loads
the automation_test_track level, spawns a drivable car at the handling-circuit
start (from the mission's race file), leaves the sim running in REAL TIME (not
the deterministic lockstep sil_beamng uses), and polls the car's position while
YOU drive. Drive the loop staying roughly mid-track; it auto-stops after
`--laps` laps (or `--seconds`). The raw trajectory is saved for build_lap.py to
turn into tracks/automation_handling.npz.

Usage:
    python record_lap.py                 # launch, spawn at start, record 1 lap
    python record_lap.py --laps 2        # record two laps (build_lap picks the best)
    python record_lap.py --connect       # attach to an already-running BeamNG
    python record_lap.py --vehicle sunburst2
"""

import argparse
import math
import time

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle

import sil_beamng as S

LEVEL = "automation_test_track"
# handling-circuit forward start (from timeTrial/003-Handling race.race.json)
START_POS = (-296.3886108, 10.43276501, 118.3325653)
START_ROT_QUAT = (-0.01037953543, -0.01057732679, -0.7136703042, 0.7003249833)
RAW_OUT = "tracks/handling_lap_raw.npz"
RATE_HZ = 30.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vehicle", default="etk800", help="drivable car model id")
    p.add_argument("--laps", type=int, default=1, help="auto-stop after this many laps")
    p.add_argument("--seconds", type=float, default=240.0, help="hard time limit [s]")
    p.add_argument("--connect", action="store_true", help="attach to a running BeamNG")
    p.add_argument("--out", default=RAW_OUT)
    args = p.parse_args()

    bng = BeamNGpy(S.HOST, S.PORT, home=S.BNG_HOME, user=S.BNG_USER)
    bng.open(launch=not args.connect)

    scenario = Scenario(LEVEL, "record_lap")
    veh = Vehicle("ego", model=args.vehicle, license="DRIVE")
    scenario.add_vehicle(veh, pos=START_POS, rot_quat=START_ROT_QUAT)
    scenario.make(bng)
    print(f"[rec] loading {LEVEL} ...", flush=True)
    bng.scenario.load(scenario)
    bng.scenario.start()
    # REAL TIME: do not set_deterministic / pause — the user drives with keyboard
    try:
        veh.recover()
        time.sleep(1.0)
        veh.set_shift_mode("arcade")           # automatic; easy to drive
        veh.control(throttle=0.0, brake=0.0, parkingbrake=0.0)  # release handbrake
    except Exception as e:
        print(f"[rec] setup warning: {e}")

    print("\n" + "=" * 64)
    print(f"  DRIVE THE HANDLING LOOP NOW — staying roughly mid-track.")
    print(f"  Recording up to {args.laps} lap(s) / {args.seconds:.0f}s at {RATE_HZ:.0f} Hz.")
    print(f"  It auto-stops when you complete the lap(s). Ctrl-C to stop early.")
    print("=" * 64 + "\n", flush=True)

    rec = []           # (t, x, y, z)
    start_xy = np.array(START_POS[:2])
    t0 = time.perf_counter()
    laps = 0
    armed = False      # left the start zone at least once this lap
    dt = 1.0 / RATE_HZ
    try:
        while True:
            t = time.perf_counter() - t0
            if t > args.seconds:
                print("\n[rec] time limit reached.")
                break
            try:
                veh.sensors.poll()
                pos = veh.state["pos"]
                spd = float(np.hypot(veh.state["vel"][0], veh.state["vel"][1]))
            except Exception:
                time.sleep(dt); continue
            rec.append((t, pos[0], pos[1], pos[2]))
            d_start = float(np.hypot(pos[0] - start_xy[0], pos[1] - start_xy[1]))
            if d_start > 40.0:
                armed = True
            if armed and d_start < 8.0:
                laps += 1
                armed = False
                print(f"\n[rec] lap {laps} complete at t={t:.1f}s")
                if laps >= args.laps:
                    break
            print(f"  t={t:6.1f}s  pos=({pos[0]:7.1f},{pos[1]:7.1f})  "
                  f"v={spd*3.6:5.1f} km/h  d_start={d_start:6.1f}m  lap={laps}", end="\r")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[rec] stopped by user.")
    finally:
        try:
            bng.close()
        except Exception:
            pass

    if len(rec) < 50:
        print(f"[rec] only {len(rec)} samples — did the car move? Not saving.")
        return
    arr = np.array(rec, dtype=float)
    np.savez(args.out, t=arr[:, 0], xyz=arr[:, 1:4])
    dist = float(np.sum(np.hypot(*np.diff(arr[:, 1:3], axis=0).T)))
    print(f"[rec] saved {len(arr)} samples ({dist:.0f} m driven, {laps} lap(s)) -> {args.out}")
    print("[rec] now build the centerline:  python build_lap.py")


if __name__ == "__main__":
    main()
