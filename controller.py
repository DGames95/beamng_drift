"""
Your drift controller lives here.

This file is HOT-RELOADED by sil_beamng.py while the sim runs: save it and
the new control law takes effect on the next tick, without relaunching BeamNG.
If you introduce an error, the sim keeps running on the previous good version
and prints the traceback - fix and save again.

Numeric gains come from gains.json (also reloaded live), so for plain gain
tuning you don't even need to touch this file.

Contract:
    DriftController(params: dict)
    controller(state: dict, t: float, dt: float) -> (throttle, brake, delta)
        state: X, Y, psi, vx, vy, r, ax, ay, speed   (see sil_beamng.get_state)
        delta: front road-wheel angle [rad]
Internal attributes (e.g. integrators) are preserved across hot reloads.
"""

import math


class DriftController:
    def __init__(self, params: dict):
        self.params = params
        # --- internal state (preserved across hot reloads) ---
        self.speed_err_int = 0.0

    def update_params(self, params: dict):
        """Called when gains.json changes; merge new values in."""
        self.params.update(params)

    def __call__(self, state: dict, t: float, dt: float):
        p = self.params

        # ---- longitudinal: hold target speed (PI on throttle) ----
        v = state["speed"]
        e_v = p["target_speed"] - v
        self.speed_err_int += e_v * dt
        # simple anti-windup clamp
        self.speed_err_int = max(-5.0, min(5.0, self.speed_err_int))
        u = p["kp_speed"] * e_v + p["ki_speed"] * self.speed_err_int

        throttle = max(0.0, min(1.0, u))
        brake = max(0.0, min(1.0, -u)) if u < 0 else 0.0

        # ---- lateral: track a target sideslip angle with countersteer ----
        beta = math.atan2(state["vy"], state["vx"]) if v > 0.5 else 0.0
        beta_tgt = math.radians(p["target_beta_deg"])
        # countersteer toward target beta, with yaw-rate damping
        delta = (
            p["kp_beta"] * (beta_tgt - beta)
            - p["kd_r"] * state["r"]
        )
        lim = math.radians(p["delta_limit_deg"])
        delta = max(-lim, min(lim, delta))

        return throttle, brake, delta
