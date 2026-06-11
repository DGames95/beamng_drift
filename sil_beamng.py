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
SPAWN_ROT_QUAT = (0.0, 0.0, 0.0, 1.0)   # identity -> facing +x

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
# Interface                                                                    #
# --------------------------------------------------------------------------- #
class BeamNGSIL:
    """Thin wrapper: connect, load empty world, read state, apply control."""

    def __init__(self):
        self.bng: BeamNGpy | None = None
        self.vehicle: Vehicle | None = None
        self._prev = None  # (t, psi, vx_body, vy_body) for finite differencing

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
        self.bng.settings.set_deterministic(PHYSICS_HZ)
        self.bng.scenario.load(scenario)
        self.bng.scenario.start()
        self.bng.control.pause()

        # let the car settle on the ground
        self.bng.control.step(int(PHYSICS_HZ * 1.0))
        self._prime_state()
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

        return {
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
    sil = BeamNGSIL().open(launch=True)
    controller = HotController()
    print(f"Connected. DT={DT*1000:.1f} ms  ({CONTROL_HZ} Hz control, "
          f"{PHYSICS_HZ} Hz physics)")
    print("State keys:", list(sil.get_state().keys()))
    print("Edit gains.json or controller.py and save to tune live. Ctrl-C to stop.")

    t = 0.0
    try:
        while True:
            controller.maybe_reload()
            state = sil.get_state()
            try:
                throttle, brake, delta = controller(state, t, DT)
            except Exception:
                # a runtime error in the control law shouldn't crash the sim
                print("\n[controller] runtime error (coasting):")
                traceback.print_exc()
                throttle, brake, delta = 0.0, 0.0, 0.0
            sil.apply_control(throttle, brake, delta)
            sil.step()
            t += DT

            print(
                f"t={t:6.2f}  X={state['X']:7.2f} Y={state['Y']:7.2f}  "
                f"psi={math.degrees(state['psi']):7.1f}deg  "
                f"vx={state['vx']:6.2f} vy={state['vy']:6.2f}  "
                f"r={state['r']:6.2f}  v={state['speed']:5.2f}",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        sil.close()


if __name__ == "__main__":
    main()
