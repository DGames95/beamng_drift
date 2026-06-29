"""Run a domain-randomized (sim-to-real) drift policy in BeamNG via the SIL.

The policy is trained in driftRL with DomainRandomizedDriftEnv (domain_rand.py)
so it is robust to the reality gap. This script drives a no-ESC RWD car in
BeamNG in its dedicated drift config (the Ibishu Miramar 'drift' preset by
default — override with --vehicle / --config) through
sil_beamng.BeamNGSIL with that policy, shows a live top-down overlay
(debug_view.DebugView), logs telemetry, and prints drift metrics so you can
judge whether the transfer worked.

The car gets a rolling start: beamngpy's set_velocity ramps it to ~11 m/s
(the speed the policy was trained to start at) and the track is anchored to the
car's measured heading, so the policy engages aligned and up to speed rather
than from a perpendicular standstill. Use --start-speed 0 for a standstill.

Usage:
    python test_sim2real_beamng.py                 # see defaults below
    python test_sim2real_beamng.py --check         # offline pipeline check, NO BeamNG
    python test_sim2real_beamng.py --track random --seconds 60
    python test_sim2real_beamng.py --connect       # attach to a running BeamNG
    python test_sim2real_beamng.py --no-overlay    # headless (no top-down view)

All CLI args default to the DEFAULT_* constants below -- edit those instead
of retyping flags every run; pass the flag to override for one run.

The policy consumes state["track_obs"] (the 8-vector [vx, vy, r, e_y, e_psi,
kappa@0/10/25], scaled by track.py's obs() — which MUST match driftRL's
OBS_SCALE) and outputs [delta in +-0.5 rad, T in +-1]; T splits into
throttle (max(0,T)) / brake (max(0,-T)). Same contract as controller.py "rl".
"""

import argparse
import csv
import math

import numpy as np

import sil_beamng as S
from track import Track, DEFAULT_LOOKAHEAD

# --- CLI defaults, edit these directly rather than retyping flags ---
DEFAULT_MODEL   = "models/drift_dr_sim2real/best_model"  # PPO .zip (no extension)
DEFAULT_TRACK   = "random"   # "circle" | "random"
DEFAULT_RADIUS  = 30.0       # circle radius [m]            (--track circle)
DEFAULT_LENGTH  = 600.0      # open-track length [m]        (--track random)
DEFAULT_SEED    = 55         # random-track layout seed     (--track random)
DEFAULT_START_SPEED = 11.0   # rolling-start speed [m/s] before engaging policy
                             # (== driftRL reset v0; 0 disables -> standstill start)
DEFAULT_SECONDS = 40.0       # run duration [s]
DEFAULT_CONNECT = False      # attach to a running BeamNG instead of launching one
DEFAULT_VEHICLE = "miramar"  # RWD, no ESC. Alts: "barstow" (V8 muscle), "bx",
                             # "sunburst2", "etkc", "pessima" — all have drift configs
DEFAULT_CONFIG  = "drift"    # vehicle config/variant (.pc preset) to spawn, e.g.
                             # "drift" (miramar/barstow/pessima/sbr/fullsize),
                             # "drift_pro" (sunburst2), "pro_drift_M" (bx),
                             # "kc8_drift_M" (etkc). "" / "stock" = default config.
DEFAULT_OVERLAY = True       # live top-down debug overlay (debug_view.DebugView)
DEFAULT_CSV     = None       # telemetry CSV output path (None = don't write)
DEFAULT_CHECK   = False      # offline obs/policy check; does NOT launch BeamNG

OVERLAY_HZ      = 15.0        # top-down overlay refresh rate [Hz]
BETA_DRIFT_MIN  = 0.4        # |beta| [rad] above which the car counts as drifting (== driftRL)


def _make_base_track(track_mode, radius, length, seed):
    """Canonical track in its own frame (circle start faces +Y, random +X)."""
    if track_mode == "circle":
        return Track.circle(radius=radius)
    rng = np.random.default_rng(seed)
    return Track.random_track(rng, length=length)


def anchor_track(track, x0, y0, heading):
    """Rigidly move a track so its start sits at (x0,y0) with tangent `heading`.

    The SIL does NOT spawn the car facing +X: SPAWN_ROT_QUAT=(0,0,1,0) is a
    180 deg Z-rotation, so the measured heading is ~-Y. Rather than assume an
    orientation (which left the car perpendicular to a +X-built track), we read
    the car's actual heading and rotate the track to match it, so the policy
    always starts aligned (e_psi ~ 0) — exactly its training distribution.
    """
    dpsi = heading - float(track.psi[0])
    c, s = math.cos(dpsi), math.sin(dpsi)
    R = np.array([[c, -s], [s, c]])
    new_xy = (track.xy - track.xy[0]) @ R.T + np.array([x0, y0])
    new_psi = track.psi + dpsi
    return Track(new_xy, new_psi, track.kappa, track.closed)


def rolling_start(sil, target):
    """Bring the car up to `target` m/s before engaging the policy.

    set_velocity alone under-delivered in paused lockstep (it ramps over
    wall-clock dt while we step faster than real time, so only part of the ramp
    lands), so we re-issue it periodically and add a light straight throttle,
    looping until the speed is reached or it times out. Returns reached speed.
    """
    sil.vehicle.set_velocity(target, dt=1.0)
    speed = 0.0
    n = int(6.0 * S.CONTROL_HZ)  # up to ~6 s to reach speed
    for i in range(n):
        sil.apply_control(throttle=0.4, brake=0.0, delta=0.0)  # straight, builds speed
        sil.step()
        speed = sil.get_state()["speed"]
        if speed >= target - 0.5:
            break
        if i % 25 == 24:  # top up the velocity command as it bleeds off
            sil.vehicle.set_velocity(target, dt=1.0)
    sil._prime_state()
    return speed


def calibrate_steering_sign(sil, probe=0.25, ticks=16, throttle=0.3, min_speed=4.0):
    """Sanity-check BeamNG's steering sign vs our convention (delta>0 -> left).

    The sign is a fixed BeamNG convention (sil.steer_sign defaults to -1), so
    this only VERIFIES it: command a known left steer and measure the yaw
    response. A positive command that yields negative yaw confirms sign = -1.

    Crucially, a reading is only trusted when the car is actually turning at
    speed; a low-speed or weak-yaw measurement is noise and returns None so the
    caller keeps the reliable default instead of flipping the steering on noise
    (which is what produced the intermittent inverted behaviour). Returns
    (sign | None, mean_r).
    """
    delta_cmd = probe * S.MAX_STEER_ANGLE  # raw +probe left command (pre steer_sign)
    rs = []
    for _ in range(ticks):
        # bypass steer_sign here so we measure BeamNG's raw response, not the
        # already-corrected command
        raw = float(np.clip(delta_cmd / S.MAX_STEER_ANGLE, -1.0, 1.0))
        sil.vehicle.control(throttle=throttle, brake=0.0, steering=raw, parkingbrake=0.0)
        sil.step()
        rs.append(sil.get_state()["r"])
    mean_r = float(np.mean(rs[ticks // 2:]))  # settled half
    speed = sil.get_state()["speed"]
    if speed < min_speed or abs(mean_r) < 0.05:
        return None, mean_r  # inconclusive -> keep the default sign
    sign = 1.0 if mean_r >= 0.0 else -1.0
    return sign, mean_r


def settle_straight(sil, target, max_ticks=80):
    """Drive straight (zero steer), holding speed, until the yaw rate and
    lateral velocity settle — so the policy engages on a clean trajectory.

    The steering-sign calibration leaves the car mid-turn (it commands a steer,
    then stops). Engaging from that is fine on a CCW circle (the car needs to
    turn left anyway) but wrong on a random track that starts straight: the car
    launches yawing and sliding off the line. Settling first nulls that out.
    Returns the settled state.
    """
    st = sil.get_state()
    for _ in range(max_ticks):
        thr = 0.3 if st["speed"] < target - 0.3 else 0.0
        sil.apply_control(throttle=thr, brake=0.0, delta=0.0)
        sil.step()
        st = sil.get_state()
        if abs(st["r"]) < 0.05 and abs(st["vy"]) < 0.4:
            break
    sil._prime_state()
    return st


def load_policy(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def policy_action(policy, obs):
    """One deterministic forward pass -> (throttle, brake, delta)."""
    action, _ = policy.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
    delta = float(np.clip(action[0], -0.5, 0.5))
    T = float(np.clip(action[1], -1.0, 1.0))
    return max(0.0, T), max(0.0, -T), delta


def check(model_path):
    """Offline sanity check: obs pipeline shape/scale + policy forward pass.

    Runs with no BeamNG — catches the usual transfer-breakers (wrong obs
    length, stale OBS_SCALE, model that won't load) before you spend a minute
    launching the sim.
    """
    print("[check] loading policy ...")
    policy = load_policy(model_path)
    print(f"[check] policy loaded: obs{policy.observation_space.shape} "
          f"act{policy.action_space.shape}")
    assert policy.observation_space.shape == (8,), "policy expects an 8-dim obs"

    # anchor a canonical circle at (0,0) heading 0 and probe the start pose at
    # the rolling-start speed (vx=11), exactly the regime the policy engages in
    track = anchor_track(_make_base_track("circle", DEFAULT_RADIUS, 0.0, 0), 0.0, 0.0, 0.0)
    obs, e_y, e_psi, kappa_p, idx = track.obs(
        0.0, 0.0, 0.0, DEFAULT_START_SPEED, 0.0, 0.0, 0, DEFAULT_LOOKAHEAD)
    print(f"[check] track.obs -> shape {obs.shape}, values {np.round(obs, 3).tolist()}")
    assert obs.shape == (8,), "track.obs must return an 8-vector"
    # vx with scale 4.0; a mismatched scale (e.g. old 20.0) would show ~1/5 of this
    expected_vx = DEFAULT_START_SPEED / 4.0
    assert abs(obs[0] - expected_vx) < 1e-3, (
        f"vx obs {obs[0]:.3f} != {expected_vx:.3f} — track.py obs() scale != driftRL OBS_SCALE")
    # anchored at heading 0 -> heading error ~0; confirms track orientation
    assert abs(e_psi) < 1e-2, f"e_psi {e_psi:.3f} != 0 — track not anchored to heading"

    thr, brk, delta = policy_action(policy, obs)
    print(f"[check] policy action on this obs: throttle={thr:.2f} brake={brk:.2f} "
          f"delta={math.degrees(delta):+.1f} deg")
    print("[check] OK — obs pipeline and policy are wired correctly.")


def run(args):
    policy = load_policy(args.model_path)
    print(f"[test] policy: {args.model_path}")

    # use a no-ESC RWD car, and (if given) a dedicated drift config/variant so
    # the chassis/tyres/diff are already set up to slide
    S.VEHICLE_MODEL = args.vehicle
    if args.config and args.config.lower() not in ("stock", "default", "none"):
        S.VEHICLE_CONFIG = f"vehicles/{args.vehicle}/{args.config}.pc"
        print(f"[test] vehicle: {args.vehicle}  config: {S.VEHICLE_CONFIG}")
    else:
        S.VEHICLE_CONFIG = None
        print(f"[test] vehicle: {args.vehicle}  config: <default>")

    sil = S.BeamNGSIL().open(launch=not args.connect)
    dt = S.DT

    # Get the car ready to actually move from a standstill:
    #  * recover() — programmatic "press R": clears the stuck/settling spawn
    #    state that otherwise leaves the car frozen until a manual reset.
    #  * release the parking brake — BeamNG spawns with the handbrake ON, which
    #    locks the rear wheels (symptom: gear 1, full throttle, no motion).
    #  * auto-shifting gearbox ("arcade") — drift configs usually have a MANUAL
    #    transmission and spawn in neutral, so plain throttle did nothing.
    try:
        sil.vehicle.recover()
        for _ in range(int(0.5 * S.CONTROL_HZ)):  # let it re-settle after reset
            sil.step()
        sil.vehicle.set_shift_mode("arcade")
        sil.vehicle.control(throttle=0.0, brake=0.0, parkingbrake=0.0)
        sil.step()
        sil._prime_state()
    except Exception as e:
        print(f"[test] startup recover/gearbox setup failed ({e}); continuing")

    # --- rolling start: bring the car up to speed BEFORE engaging the policy ---
    # The policy is trained always starting at ~11 m/s; BeamNG spawns at rest,
    # far out of distribution. Use beamngpy's set_velocity (forward-directed),
    # verified/topped-up until the target speed is actually reached.
    if args.start_speed > 0:
        print(f"[test] rolling start -> {args.start_speed:.1f} m/s ...", flush=True)
        reached = rolling_start(sil, args.start_speed)
        print(f"[test] rolling start reached {reached:.1f} m/s")

    # --- steering-sign sanity check: verify (don't blindly trust) BeamNG's
    # steering sign against the known default (sil.steer_sign = -1). Only an
    # at-speed, clear-yaw reading is allowed to override it; a noisy low-speed
    # reading is ignored, so a failed rolling start can't flip the steering and
    # cause inverted behaviour.
    sign, mean_r = calibrate_steering_sign(sil)
    if sign is None:
        print(f"[test] steering check inconclusive (r={mean_r:+.3f}) — keeping "
              f"default steer_sign={sil.steer_sign:+.0f} (BeamNG convention)")
    elif sign != sil.steer_sign:
        print(f"[test] WARNING: measured steer sign {sign:+.0f} != default "
              f"{sil.steer_sign:+.0f} (r={mean_r:+.3f}); using measured")
        sil.steer_sign = sign
    else:
        print(f"[test] steering sign confirmed {sign:+.0f} (r={mean_r:+.3f})")

    # Re-straighten after the calibration turn so the car engages on a clean,
    # straight trajectory (no residual yaw / lateral velocity). Critical for the
    # random track, whose start is straight; a circle tolerated engaging mid-turn.
    target = args.start_speed if args.start_speed > 0 else sil.get_state()["speed"]
    s0 = settle_straight(sil, target)

    # Anchor the track to the car's ACTUAL settled pose+heading, so the policy
    # engages aligned with the track (e_psi ~ 0) regardless of the SIL's spawn
    # orientation. Then re-zero the nearest-sample search hint.
    sil.track = anchor_track(
        _make_base_track(args.track, args.radius, args.length, args.seed),
        s0["X"], s0["Y"], s0["psi"])
    sil._track_hint = 0
    print(f"[test] engaged: X={s0['X']:.1f} Y={s0['Y']:.1f} "
          f"psi={math.degrees(s0['psi']):+.0f}deg  "
          f"vx={s0['vx']:.2f} vy={s0['vy']:.2f} speed={s0['speed']:.2f} m/s")
    if args.start_speed > 0:
        if abs(s0["speed"] - args.start_speed) > 3.0:
            print(f"[test] WARNING: rolling-start speed {s0['speed']:.1f} far from "
                  f"target {args.start_speed:.1f} — check set_velocity dt")
        if abs(s0["vy"]) > 3.0:
            print(f"[test] WARNING: large lateral vy={s0['vy']:.1f} at engage — "
                  f"set_velocity may not be forward-aligned (world-frame?)")

    max_steps = int(args.seconds / dt)
    print(f"[test] running {args.seconds:.0f} s  ({max_steps} ticks @ "
          f"{1/dt:.0f} Hz control)  track={args.track}")

    # live top-down overlay — daemon thread, never blocks the control loop
    view = None
    if args.overlay:
        from debug_view import DebugView
        view = DebugView(sil.track, update_hz=OVERLAY_HZ)

    log = []
    off_track_events = 0
    try:
        for k in range(max_steps):
            t = k * dt
            state = sil.get_state()
            if view is not None:
                view.update(state, t)

            if state.get("off_track"):
                off_track_events += 1
                e_y = state["track_frame"][0]
                print(f"\n[test] OFF TRACK e_y={e_y:+.2f} m at t={t:.1f} s")
                break
            if state.get("track_end"):
                print(f"\n[test] reached end of track at t={t:.1f} s")
                break

            obs = state.get("track_obs")
            if obs is None:
                print("[test] no track_obs (TRACK_MODE None?) — aborting")
                break

            throttle, brake, delta = policy_action(policy, obs)
            sil.apply_control(throttle, brake, delta)
            sil.step()

            vx, vy = state["vx"], state["vy"]
            beta = math.atan2(vy, vx) if state["speed"] > 0.5 else 0.0
            tf = state["track_frame"]
            log.append({
                "t": t, "X": state["X"], "Y": state["Y"],
                "vx": vx, "vy": vy, "speed": state["speed"], "r": state["r"],
                "beta": beta, "e_y": tf[0], "e_psi": tf[1],
                "delta": delta, "throttle": throttle, "brake": brake,
            })
            print(
                f"t={t:6.2f}  v={state['speed']:5.2f}  vx={vx:5.2f} vy={vy:5.2f}"
                f"  beta={math.degrees(beta):+6.1f}deg  e_y={tf[0]:+5.2f}"
                f"  delta={math.degrees(delta):+5.1f}deg  T={throttle-brake:+.2f}",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\n[test] interrupted.")
    except Exception as e:  # e.g. BNGDisconnectedError if the sim is closed/crashes
        print(f"\n[test] sim ended early ({type(e).__name__}: {e}) — "
              f"reporting partial telemetry.")
    finally:
        try:
            if view is not None:
                view.close()
            sil.close()
        except Exception:
            pass

    _report(log, off_track_events, args)


def _report(log, off_track_events, args):
    if not log:
        print("\n[test] no telemetry collected.")
        return
    arr = {k: np.array([d[k] for d in log]) for k in log[0]}
    n = len(arr["t"])
    settled = slice(n // 4, None)  # ignore the spin-up transient
    drift_frac = float(np.mean(np.abs(arr["beta"]) > BETA_DRIFT_MIN))
    dist = float(np.sum(np.hypot(np.diff(arr["X"]), np.diff(arr["Y"]))))

    print("\n\n--- sim-to-real BeamNG run summary ---")
    print(f"duration:        {arr['t'][-1]:.1f} s ({n} ticks)")
    print(f"distance:        {dist:.1f} m")
    print(f"speed:           mean {arr['speed'][settled].mean():.2f}  "
          f"max {arr['speed'].max():.2f} m/s")
    print(f"|beta|:          mean {math.degrees(np.abs(arr['beta'][settled]).mean()):.1f}  "
          f"max {math.degrees(np.abs(arr['beta']).max()):.1f} deg")
    print(f"drifting:        {100*drift_frac:.0f}% of ticks (|beta| > "
          f"{math.degrees(BETA_DRIFT_MIN):.0f} deg)")
    print(f"|e_y|:           mean {np.abs(arr['e_y'][settled]).mean():.2f}  "
          f"max {np.abs(arr['e_y']).max():.2f} m")
    print(f"off-track:       {off_track_events} event(s)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        print(f"telemetry CSV:   {args.csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL,
                   help="PPO .zip (without extension) to run")
    p.add_argument("--track", choices=["circle", "random"], default=DEFAULT_TRACK)
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                   help="circle radius [m] (--track circle)")
    p.add_argument("--length", type=float, default=DEFAULT_LENGTH,
                   help="open-track length [m] (--track random)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for the random track layout")
    p.add_argument("--start-speed", type=float, default=DEFAULT_START_SPEED,
                   help="rolling-start speed [m/s] before engaging policy (0 = standstill)")
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="run duration [s]")
    p.add_argument("--connect", action="store_true", default=DEFAULT_CONNECT,
                   help="attach to an already-running BeamNG instead of launching")
    p.add_argument("--vehicle", default=DEFAULT_VEHICLE,
                   help="BeamNG vehicle model id (default: miramar, a no-ESC RWD car)")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="vehicle config/variant .pc preset name (e.g. drift, drift_pro, "
                        "pro_drift_M); 'stock' for the model default")
    p.add_argument("--no-overlay", dest="overlay", action="store_false", default=DEFAULT_OVERLAY,
                   help="disable the live top-down debug overlay")
    p.add_argument("--csv", default=DEFAULT_CSV, help="optional telemetry CSV output path")
    p.add_argument("--check", action="store_true", default=DEFAULT_CHECK,
                   help="offline obs/policy sanity check; does NOT launch BeamNG")
    args = p.parse_args()

    if args.check:
        check(args.model_path)
    else:
        run(args)
