"""
Software-in-the-loop (SIL) bridge between a Python vehicle controller and
BeamNG.drive.

What it does:
  * launches / connects to BeamNG.drive
  * loads an empty flat world (smallgrid) and spawns one car
  * runs the simulation in deterministic lockstep
  * every tick: reads the car state, asks your controller for commands,
    and applies throttle / brake / steering

State is exposed in the form a kinematic/dynamic bicycle-model controller
expects:

    X, Y      world position            [m]
    psi       heading (yaw)             [rad]
    vx, vy    BODY-frame velocity       [m/s]   (vx = longitudinal, vy = lateral)
    r         yaw rate (dpsi/dt)        [rad/s]
    ax, ay    BODY-frame acceleration   [m/s^2] (finite-differenced)

Commands your controller returns:

    throttle  [0, 1]
    brake     [0, 1]
    delta     front road-wheel steer angle [rad]   (NOT normalized)

IMPORTANT - steering:
    BeamNG's vehicle.control(steering=...) takes a NORMALIZED value in
    [-1, 1] where +/-1 is full steering-wheel lock, not a physical angle.
    Your bicycle model produces a physical road-wheel angle `delta`, so we
    map  steering_input = clip(delta / MAX_STEER_ANGLE, -1, 1).
    MAX_STEER_ANGLE must be calibrated once for the chosen vehicle: hold
    steering=1.0, let the wheels reach full lock, and measure the resulting
    road-wheel angle (e.g. from the steady-state radius of a slow circle,
    delta ~= wheelbase / turn_radius). Set MAX_STEER_ANGLE to that value.
"""

from __future__ import annotations

import importlib
import math
import os
import time
import traceback

import numpy as np

import controller as controller_module

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

from track import Track, DEFAULT_LOOKAHEAD

# --------------------------------------------------------------------------- #
# Configuration  (edit these)                                                  #
# --------------------------------------------------------------------------- #
BNG_HOME = r"E:\SteamLibrary\steamapps\common\BeamNG.drive"
BNG_USER = r"C:\Users\damia\AppData\Local\BeamNG.drive"
HOST = "localhost"
PORT = 25252

MAP = "smallgrid"          # empty flat infinite plane; try "gridmap_v2" for a textured pad
VEHICLE_MODEL = "etk800"   # rwd sedan, decent for drifting; e.g. "sunburst2" (rwd) also good

SPAWN_POS = (0.0, 0.0, 0.5)
SPAWN_ROT_QUAT = (0.0, 0.0, 1.0, 0.0)   # identity -> facing +x

# Deterministic SIL timing.
PHYSICS_HZ = 100           # simulation steps per second BeamNG advances
CONTROL_HZ = 50            # how often your controller runs
# steps advanced per control tick (must be integer):
STEPS_PER_TICK = PHYSICS_HZ // CONTROL_HZ
DT = STEPS_PER_TICK / PHYSICS_HZ        # control timestep [s]

# Steering calibration: physical road-wheel angle [rad] that corresponds to
# full lock (normalized steering = 1.0). ~30 deg is a reasonable start for the
# etk800; calibrate with calibrate_max_steer() for accuracy.
MAX_STEER_ANGLE = math.radians(30.0)

# --------------------------------------------------------------------------- #
# Track configuration                                                          #
# --------------------------------------------------------------------------- #
# Set TRACK_MODE to "circle" or "random".  The track is centred near SPAWN_POS
# so the car starts on the line.  Set to None to disable path-following.
TRACK_MODE   = "circle"        # "circle" | "random" | None
TRACK_RADIUS = 30.0            # [m] used only for TRACK_MODE="circle" (matches driftRL training)
TRACK_SEED   = 42              # RNG seed for "random"
TRACK_LENGTH = 600.0           # [m] open-track arc length for "random"

# Lookahead distances fed to track.frame() and track.obs() [m].
# Matches driftRL defaults (0, 10, 25 m) so trained policies transfer.
LOOKAHEAD = DEFAULT_LOOKAHEAD  # override e.g. (0.0, 5.0, 15.0, 30.0)

OFF_TRACK_RESET = True         # True = teleport+continue; False = raise KeyboardInterrupt


# --------------------------------------------------------------------------- #
# Track factory                                                                #
# --------------------------------------------------------------------------- #
def _make_track() -> Track | None:
    if TRACK_MODE is None:
        return None
    # circle centred so spawn (0, 0) is on the track, car faces +X tangent
    origin = (0.0, -TRACK_RADIUS)  # centre below spawn so the track passes through (0,0)
    if TRACK_MODE == "circle":
        return Track.circle(radius=TRACK_RADIUS, origin=origin)
    elif TRACK_MODE == "random":
        rng = np.random.default_rng(TRACK_SEED)
        return Track.random_track(rng, length=TRACK_LENGTH, origin=(0.0, 0.0))
    else:
        raise ValueError(f"Unknown TRACK_MODE {TRACK_MODE!r}")


# --------------------------------------------------------------------------- #
# Interface                                                                    #
# --------------------------------------------------------------------------- #
class BeamNGSIL:
    """Thin wrapper: connect, load empty world, read state, apply control."""

    def __init__(self):
        self.bng: BeamNGpy | None = None
        self.vehicle: Vehicle | None = None
        self._prev = None  # (t, psi, vx_body, vy_body) for finite differencing

        # track (None when TRACK_MODE is None)
        self.track: Track | None = _make_track()
        self._track_hint: int = 0  # nearest-sample search hint, updated each tick

    # ---- lifecycle -------------------------------------------------------- #
    def open(self, launch: bool = True):
        self.bng = BeamNGpy(HOST, PORT, home=BNG_HOME, user=BNG_USER)
        self.bng.open(launch=launch)

        scenario = Scenario(MAP, "sil_drift")
        self.vehicle = Vehicle("ego", model=VEHICLE_MODEL, license="SIL")
        self.vehicle.attach_sensor("electrics", Electrics())
        scenario.add_vehicle(
            self.vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT_QUAT
        )
        scenario.make(self.bng)

        # deterministic lockstep: physics only advances when we step it
        print("[sil] setting deterministic...", flush=True)
        self.bng.settings.set_deterministic(PHYSICS_HZ)
        print("[sil] loading scenario...", flush=True)
        self.bng.scenario.load(scenario)
        print("[sil] starting scenario...", flush=True)
        self.bng.scenario.start()
        print("[sil] pausing...", flush=True)
        self.bng.control.pause()

        # let the car settle on the ground — step in small batches so BeamNG
        # doesn't time out on a single large RPC call
        print("[sil] settling (2 s)...", flush=True)
        for _ in range(4):
            self.bng.control.step(int(PHYSICS_HZ * 0.5))
        print("[sil] priming state...", flush=True)
        self._prime_state()
        print("[sil] ready.", flush=True)
        return self

    def close(self):
        if self.bng is not None:
            self.bng.close()

    # ---- simulation ------------------------------------------------------- #
    def step(self):
        """Advance physics by one control tick."""
        self.bng.control.step(STEPS_PER_TICK)

    def _prime_state(self):
        s = self._read_raw()
        self._prev = (s["t"], s["psi"], s["vx"], s["vy"])

    def _read_raw(self):
        self.vehicle.sensors.poll()
        st = self.vehicle.state
        pos = st["pos"]
        d = st["dir"]            # forward unit vector in world frame
        vel = st["vel"]          # world-frame velocity vector

        psi = math.atan2(d[1], d[0])
        cps, sps = math.cos(psi), math.sin(psi)
        # rotate world velocity into body frame
        vx = vel[0] * cps + vel[1] * sps      # longitudinal
        vy = -vel[0] * sps + vel[1] * cps     # lateral
        return {
            "t": time.perf_counter(),
            "X": pos[0],
            "Y": pos[1],
            "psi": psi,
            "vx": vx,
            "vy": vy,
            "vel_world": (vel[0], vel[1]),
        }

    def get_state(self) -> dict:
        """Return the controller-facing state dict.

        Yaw rate and body accelerations are finite-differenced over DT.
        """
        s = self._read_raw()
        pt, ppsi, pvx, pvy = self._prev

        # use fixed DT from deterministic stepping for clean derivatives
        dpsi = math.atan2(math.sin(s["psi"] - ppsi), math.cos(s["psi"] - ppsi))
        r = dpsi / DT
        ax = (s["vx"] - pvx) / DT
        ay = (s["vy"] - pvy) / DT

        self._prev = (s["t"], s["psi"], s["vx"], s["vy"])

        state = {
            "X": s["X"],
            "Y": s["Y"],
            "psi": s["psi"],
            "vx": s["vx"],
            "vy": s["vy"],
            "r": r,
            "ax": ax,
            "ay": ay,
            "speed": math.hypot(s["vx"], s["vy"]),
        }

        # --- track-frame augmentation ---
        if self.track is not None:
            e_y, e_psi, kappa_p, idx = self.track.frame(
                s["X"], s["Y"], s["psi"], self._track_hint, LOOKAHEAD
            )
            self._track_hint = idx
            state["track_frame"] = (e_y, e_psi, kappa_p, idx)

            # full RL obs vector (scaled) — ready for policy forward-pass
            obs, *_ = self.track.obs(
                s["X"], s["Y"], s["psi"],
                s["vx"], s["vy"], r,
                idx, LOOKAHEAD,
            )
            state["track_obs"] = obs

            state["off_track"] = self.track.off_track(e_y)
            state["track_end"] = self.track.at_end(idx)

        return state

    # ---- actuation -------------------------------------------------------- #
    def apply_control(self, throttle: float, brake: float, delta: float):
        steering = float(np.clip(delta / MAX_STEER_ANGLE, -1.0, 1.0))
        self.vehicle.control(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
            steering=steering,
        )

    def reset(self):
        """Teleport the car back to spawn (useful between controller trials)."""
        self.vehicle.teleport(SPAWN_POS, rot_quat=SPAWN_ROT_QUAT, reset=True)
        self.bng.control.step(int(PHYSICS_HZ * 0.5))
        self._prime_state()


# --------------------------------------------------------------------------- #
# Hot-reloadable controller wrapper                                            #
# --------------------------------------------------------------------------- #
GAINS_PATH = "gains.json"
CONTROLLER_FILE = controller_module.__file__


class HotController:
    """Wraps controller.DriftController with live reloading.

    * gains.json changes  -> re-read, push to the live controller instance.
    * controller.py changes -> importlib.reload, rebuild the instance while
      carrying over internal state. Errors are caught so the sim never dies
      from a bad edit; the previous good controller stays active.
    """

    def __init__(self):
        self._gains_mtime = 0.0
        self._mod_mtime = 0.0
        self.params = self._load_gains()
        self.ctrl = controller_module.DriftController(dict(self.params))
        self._mod_mtime = self._mtime(CONTROLLER_FILE)

    @staticmethod
    def _mtime(path) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _load_gains(self) -> dict:
        import json
        with open(GAINS_PATH) as f:
            params = json.load(f)
        self._gains_mtime = self._mtime(GAINS_PATH)
        return params

    def maybe_reload(self):
        # ---- gains.json (cheap, numeric tuning) ----
        if self._mtime(GAINS_PATH) > self._gains_mtime:
            try:
                self.params = self._load_gains()
                self.ctrl.update_params(dict(self.params))
                print(f"\n[reload] gains.json -> {self.params}")
            except Exception:
                print("\n[reload] gains.json FAILED (keeping old gains):")
                traceback.print_exc()

        # ---- controller.py (control law) ----
        if self._mtime(CONTROLLER_FILE) > self._mod_mtime:
            self._mod_mtime = self._mtime(CONTROLLER_FILE)
            try:
                importlib.reload(controller_module)
                new_ctrl = controller_module.DriftController(dict(self.params))
                # carry over internal state (integrators, etc.)
                for k, v in self.ctrl.__dict__.items():
                    if k != "params":
                        setattr(new_ctrl, k, v)
                self.ctrl = new_ctrl
                print("\n[reload] controller.py reloaded OK")
            except Exception:
                print("\n[reload] controller.py FAILED (keeping old version):")
                traceback.print_exc()

    def __call__(self, state, t, dt):
        return self.ctrl(state, t, dt)


# --------------------------------------------------------------------------- #
# Main SIL loop                                                                #
# --------------------------------------------------------------------------- #
def main():
    from debug_view import DebugView

    sil = BeamNGSIL().open(launch=True)
    controller = HotController()
    print(f"Connected. DT={DT*1000:.1f} ms  ({CONTROL_HZ} Hz control, "
          f"{PHYSICS_HZ} Hz physics)")
    print("Edit gains.json or controller.py and save to tune live. Ctrl-C to stop.")

    if sil.track is not None:
        print(f"Track: {TRACK_MODE}  n={sil.track.n} pts  "
              f"length={sil.track.length:.0f} m  "
              f"half_width={sil.track.half_width:.1f} m")

    view = DebugView(sil.track, update_hz=15.0)

    t = 0.0
    try:
        while True:
            controller.maybe_reload()
            state = sil.get_state()

            # --- off-track / end-of-track handling ---
            if state.get("off_track"):
                e_y = state["track_frame"][0]
                print(f"\n[track] OFF TRACK  e_y={e_y:+.2f} m  t={t:.1f} s")
                if OFF_TRACK_RESET:
                    sil.reset()
                    sil._track_hint = 0
                    t = 0.0
                    continue
                else:
                    break

            if state.get("track_end"):
                print(f"\n[track] reached end of open track  t={t:.1f} s")
                break

            try:
                throttle, brake, delta = controller(state, t, DT)
            except Exception:
                print("\n[controller] runtime error (coasting):")
                traceback.print_exc()
                throttle, brake, delta = 0.0, 0.0, 0.0

            sil.apply_control(throttle, brake, delta)
            sil.step()
            t += DT

            view.update(state, t)

            tf = state.get("track_frame")
            track_str = (
                f"  e_y={tf[0]:+5.2f} e_psi={math.degrees(tf[1]):+5.1f}deg"
                f"  k={tf[2][1]*1000:.1f}‰"
            ) if tf else ""
            print(
                f"t={t:6.2f}  X={state['X']:7.2f} Y={state['Y']:7.2f}  "
                f"v={state['speed']:5.2f} vx={state['vx']:5.2f} vy={state['vy']:5.2f}"
                + track_str,
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        view.close()
        sil.close()


if __name__ == "__main__":
    main()
