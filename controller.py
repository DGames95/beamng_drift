"""
Vehicle controller — hot-reloaded by sil_beamng.py while the sim runs.

Contract:
    DriftController(params: dict)
    controller(state: dict, t: float, dt: float) -> (throttle, brake, delta)
        state: X, Y, psi, vx, vy, r, ax, ay, speed, track_frame (optional)
        delta: front road-wheel angle [rad]

track_frame is injected by the main loop when a Track is active:
    state["track_frame"] = (e_y, e_psi, kappa_preview, track_idx)

Modes (set "mode" in gains.json):
    "path"   Stanley path-following, no sideslip target (current default)
    "drift"  Sideslip-tracking countersteer — analytic placeholder
    "rl"     Forward pass through a trained stable-baselines3 PPO policy
             (driftRL), consuming state["track_obs"] -> (delta, throttle)

RL mode:
    state["track_obs"] is the 8-vector [vx, vy, r, e_y, e_psi, kappa@0/10/25 m]
    scaled by driftRL's OBS_SCALE (see track.py / drift_env.py).  The policy
    outputs [delta in +-0.5 rad, T in +-1]; T is split into throttle/brake.
    Set "model_path" in gains.json to choose the model (default below).
"""

import math

import numpy as np


DEFAULT_MODEL_PATH = "models/drift_circle/best_model"

# Cache loaded policies by path so PPO.load runs once, not on every hot-reload.
_MODEL_CACHE = {}


def _load_policy(path: str):
    if path not in _MODEL_CACHE:
        # lazy import so "path"/"drift" modes work without sb3/torch installed
        from stable_baselines3 import PPO
        _MODEL_CACHE[path] = PPO.load(path, device="cpu")
    return _MODEL_CACHE[path]


class DriftController:
    def __init__(self, params: dict):
        self.params = params
        # longitudinal integrator
        self.speed_err_int = 0.0
        # last delta for smoothness (unused in path mode but preserved for drift)
        self._last_delta = 0.0

    def update_params(self, params: dict):
        self.params.update(params)

    def __call__(self, state: dict, t: float, dt: float):
        mode = self.params.get("mode", "path")
        if mode == "path":
            return self._path(state, dt)
        elif mode == "rl":
            return self._rl(state)
        else:
            return self._drift(state, dt)

    # ------------------------------------------------------------------ path mode

    def _path(self, state: dict, dt: float):
        p = self.params
        vx = state["vx"]
        speed = state["speed"]

        # --- longitudinal: PI speed hold ---
        e_v = p["target_speed"] - speed
        self.speed_err_int += e_v * dt
        self.speed_err_int = max(-5.0, min(5.0, self.speed_err_int))
        u = p["kp_speed"] * e_v + p["ki_speed"] * self.speed_err_int
        throttle = max(0.0, min(1.0, u))
        brake    = max(0.0, min(1.0, -u)) if u < 0 else 0.0

        # --- lateral: Stanley steering law ---
        # delta = e_psi + arctan(k_e * e_y / max(vx, v_min))
        # k_e  : cross-track gain
        # v_min: softens gain at low speed
        tf = state.get("track_frame")
        if tf is None:
            # no track loaded — go straight
            return throttle, brake, 0.0

        e_y, e_psi, kappa_preview, _idx = tf

        # feedforward curvature: anticipate the upcoming bend
        # kappa_preview[0] is at current position, [1] is 10 m ahead, [2] is 25 m ahead
        kappa_ff = kappa_preview[1] if len(kappa_preview) > 1 else 0.0

        k_e   = p.get("k_stanley", 1.5)
        v_min = p.get("v_min_stanley", 2.0)
        L     = p.get("wheelbase", 2.7)       # [m] etk800 wheelbase (approx)

        # Stanley law on front axle — project e_y to front axle
        # (using simplified: front-axle cross-track ≈ e_y + L/2 * sin(e_psi))
        e_y_front = e_y + (L / 2.0) * math.sin(e_psi)

        delta = e_psi + math.atan2(k_e * e_y_front, max(abs(vx), v_min))

        # feedforward: bicycle model steady-state delta = kappa * L
        delta += p.get("k_ff", 0.6) * kappa_ff * L

        lim = math.radians(p["delta_limit_deg"])
        delta = max(-lim, min(lim, delta))

        self._last_delta = delta
        return throttle, brake, delta

    # ------------------------------------------------------------------ rl mode

    def _rl(self, state: dict):
        """Forward pass through a trained driftRL PPO policy.

        Consumes state["track_obs"] (scaled to match driftRL's OBS_SCALE) and
        returns (throttle, brake, delta).  Action is [delta in +-0.5 rad,
        T in +-1]; T splits into throttle/brake.
        """
        obs = state.get("track_obs")
        if obs is None:
            # no track loaded — nothing to observe, coast straight
            return 0.0, 0.0, 0.0

        policy = _load_policy(self.params.get("model_path", DEFAULT_MODEL_PATH))
        action, _ = policy.predict(
            np.asarray(obs, dtype=np.float32), deterministic=True
        )
        delta = float(np.clip(action[0], -0.5, 0.5))
        T = float(np.clip(action[1], -1.0, 1.0))
        self._last_delta = delta
        return max(0.0, T), max(0.0, -T), delta

    # ------------------------------------------------------------------ drift mode (placeholder)

    def _drift(self, state: dict, dt: float):
        """Sideslip-targeting countersteer — same law as original controller.py.

        RL drop-in: replace this body with a forward pass through a loaded
        SB3 policy, consuming state["track_obs"] built by track.obs().
        """
        p = self.params
        v = state["speed"]

        e_v = p["target_speed"] - v
        self.speed_err_int += e_v * dt
        self.speed_err_int = max(-5.0, min(5.0, self.speed_err_int))
        u = p["kp_speed"] * e_v + p["ki_speed"] * self.speed_err_int
        throttle = max(0.0, min(1.0, u))
        brake    = max(0.0, min(1.0, -u)) if u < 0 else 0.0

        beta = math.atan2(state["vy"], state["vx"]) if v > 0.5 else 0.0
        beta_tgt = math.radians(p.get("target_beta_deg", -5.0))
        delta = (
            p.get("kp_beta", 12.0) * (beta_tgt - beta)
            - p.get("kd_r", 0.15) * state["r"]
        )
        lim = math.radians(p["delta_limit_deg"])
        delta = max(-lim, min(lim, delta))
        return throttle, brake, delta
