"""
Vehicle system identification for drift_env.py parameter extraction.

Runs structured maneuvers in BeamNG.drive on smallgrid and fits a 3-DOF
bicycle model via OLS. Results written to vehicles/<model>[_<tag>].json.

Maneuvers (can be individually skipped with --skip M1,M3,...):
  M1 coast_down      -> C_DRAG  (coast from high speed)
  M2 accel_sweep     -> F_DRIVE_MAX, C_DRAG  (joint fit)
  M3 ug_circle       -> K_us understeer gradient -> LF, LR ratio
  M4 steer_chirp     -> CA_F, CA_R, IZ  (0.1-5 Hz chirp in linear tyre region)
  M5 throttle_chirp  -> validates F_DRIVE_MAX, C_DRAG; excites friction ellipse
  M6 random          -> cross-validation + MU from peak lateral-g

Usage:
    python sysid.py --vehicle etk800 --wheelbase 2.70 --mass 1350
    python sysid.py --vehicle etk800 --wheelbase 2.70 --mass 1350 --skip M3,M5
    python sysid.py --vehicle pickup --wheelbase 3.10 --mass 2100 --tag v2
    python sysid.py --vehicle etk800 --wheelbase 2.70 --mass 1350 --lf 1.18 --lr 1.52

Note on steering calibration: sil_beamng.MAX_STEER_ANGLE must be set to the
correct full-lock road-wheel angle for the chosen vehicle before running M3/M4.
CA_F and CA_R estimates absorb any calibration error, so they remain internally
consistent with drift_env.py as long as the same MAX_STEER_ANGLE is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Patch module-level constants BEFORE importing BeamNGSIL so the class
# constructor picks them up at instantiation time, not at import time.
import sil_beamng as _sil_mod

from sysid_fit import (
    extract_mu,
    fit_bicycle_linear,
    fit_drag,
    fit_drivetrain,
    fit_understeer_gradient,
    lowpass,
    validate_forward_sim,
)

_DT: float = _sil_mod.DT          # 0.02 s  (set from module; do not change)
_CTRL_HZ: float = _sil_mod.CONTROL_HZ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_arr(log: list[dict], *keys: str) -> tuple[np.ndarray, ...]:
    return tuple(np.array([row[k] for row in log]) for k in keys)


def _diff(x: np.ndarray) -> np.ndarray:
    """Central finite difference then 10 Hz low-pass filter."""
    dx = np.gradient(x, _DT)
    return lowpass(dx, cutoff_hz=10.0, fs_hz=_CTRL_HZ)


def _accel_to(sil, target_vx: float, throttle: float = 0.85, timeout_s: float = 30.0):
    """Drive straight until vx reaches target_vx (or timeout)."""
    for _ in range(int(timeout_s / _DT)):
        s = sil.get_state()
        sil.apply_control(throttle, 0.0, 0.0)
        sil.step()
        if s["vx"] >= target_vx:
            return
    print(f"  [warn] accel timeout — reached {s['vx']:.1f} m/s of {target_vx:.1f} m/s")


def _cruise_control(vx: float, vx_target: float, kp: float = 0.15) -> tuple[float, float]:
    err = vx_target - vx
    throttle = float(np.clip(kp * err, 0.0, 0.9))
    brake = float(np.clip(-kp * err, 0.0, 0.4))
    return throttle, brake


# ---------------------------------------------------------------------------
# Maneuvers
# ---------------------------------------------------------------------------

def run_coast_down(
    sil,
    target_speed: float = 33.0,
    stop_speed: float = 5.0,
    passes: int = 2,
) -> list[dict]:
    """M1: accelerate to target_speed then coast; record deceleration."""
    all_log: list[dict] = []
    for p in range(passes):
        sil.reset()
        print(f"  [M1 pass {p+1}/{passes}] accel to {target_speed:.0f} m/s... ",
              end="", flush=True)
        _accel_to(sil, target_speed)
        print("coasting... ", end="", flush=True)
        log = []
        while True:
            s = sil.get_state()
            sil.apply_control(0.0, 0.0, 0.0)
            sil.step()
            log.append({"vx": s["vx"], "ax": s["ax"]})
            if s["vx"] < stop_speed:
                break
        all_log.extend(log)
        print(f"{len(log)} samples.")
    return all_log


def run_accel_sweep(
    sil,
    throttle_levels: tuple = (0.3, 0.5, 0.7, 0.9),
    hold_s: float = 8.0,
) -> list[dict]:
    """M2: hold each throttle level on a straight; record ax vs (T, vx)."""
    all_log: list[dict] = []
    hold_steps = int(hold_s / _DT)
    for T_val in throttle_levels:
        sil.reset()
        print(f"  [M2] T={T_val:.1f} for {hold_s:.0f} s... ", end="", flush=True)
        log = []
        for _ in range(hold_steps):
            s = sil.get_state()
            sil.apply_control(T_val, 0.0, 0.0)
            sil.step()
            log.append({"T": T_val, "vx": s["vx"], "ax": s["ax"]})
        all_log.extend(log)
        print(f"vx={log[-1]['vx']:.1f} m/s  {len(log)} samples.")
    return all_log


def run_ug_circle(
    sil,
    delta_levels_deg: tuple = (5, 12),
    T_start: float = 0.08,
    T_end: float = 0.50,
    settle_s: float = 3.0,
    sweep_s: float = 15.0,
) -> list[dict]:
    """M3: ramp throttle at constant delta to sweep vx across steady-state circles.

    Records (delta_rad, vx, r) every tick during the sweep phase.
    The understeer-gradient fit uses: delta*R = L + K_us*vx^2/g, R = vx/r.
    """
    settle_steps = int(settle_s / _DT)
    sweep_steps = int(sweep_s / _DT)
    all_log: list[dict] = []
    for d_deg in delta_levels_deg:
        sil.reset()
        d_rad = math.radians(d_deg)
        print(f"  [M3] delta={d_deg}° settle {settle_s:.0f} s + sweep {sweep_s:.0f} s... ",
              end="", flush=True)
        # settle at low throttle so the car establishes a circle
        for _ in range(settle_steps):
            s = sil.get_state()
            sil.apply_control(T_start, 0.0, d_rad)
            sil.step()
        # record during throttle ramp
        log = []
        for i in range(sweep_steps):
            T_val = T_start + (T_end - T_start) * i / max(sweep_steps - 1, 1)
            s = sil.get_state()
            sil.apply_control(T_val, 0.0, d_rad)
            sil.step()
            if abs(s["r"]) > 1e-3:   # skip near-zero yaw rate (car still settling)
                log.append({"delta_rad": d_rad, "vx": s["vx"], "r": s["r"]})
        all_log.extend(log)
        vxs = [row["vx"] for row in log]
        print(f"vx {min(vxs):.1f}-{max(vxs):.1f} m/s  {len(log)} samples.")
    return all_log


def run_steer_chirp(
    sil,
    vx_target: float = 15.0,
    amplitude_rad: float = math.radians(5.0),
    f0: float = 0.1,
    f1: float = 5.0,
    chirp_s: float = 40.0,
    runs: int = 2,
) -> list[dict]:
    """M4: frequency-swept sinusoidal steering at constant speed.

    Amplitude is kept small (5°) to stay in the linear tyre region.
    Alternate sign each run to cancel any tyre/road asymmetry.
    """
    n_steps = int(chirp_s / _DT)
    all_log: list[dict] = []
    for run_i in range(runs):
        sil.reset()
        print(f"  [M4 run {run_i+1}/{runs}] accel to {vx_target:.0f} m/s... ",
              end="", flush=True)
        _accel_to(sil, vx_target, throttle=0.5)
        print(f"chirp {chirp_s:.0f} s @ {amplitude_rad:.3f} rad amplitude... ",
              end="", flush=True)
        sign = 1 if run_i % 2 == 0 else -1
        log = []
        for i in range(n_steps):
            t_chirp = i * _DT
            phase = 2.0 * math.pi * (f0 * t_chirp + (f1 - f0) * t_chirp ** 2 / (2.0 * chirp_s))
            delta = sign * amplitude_rad * math.sin(phase)
            s = sil.get_state()
            throttle, brake = _cruise_control(s["vx"], vx_target)
            sil.apply_control(throttle, brake, delta)
            sil.step()
            log.append({
                "vx": s["vx"], "vy": s["vy"], "r": s["r"],
                "ax": s["ax"], "ay": s["ay"],
                "delta": delta, "T": throttle,
            })
        all_log.extend(log)
        print(f"{len(log)} samples.")
    return all_log


def run_throttle_chirp(
    sil,
    vx_init: float = 20.0,
    f0: float = 0.05,
    f1: float = 2.0,
    chirp_s: float = 30.0,
    runs: int = 2,
) -> list[dict]:
    """M5: frequency-swept sinusoidal throttle on a straight (delta=0).

    Excites longitudinal dynamics. At partial steering residual this also
    perturbs the friction-ellipse coupling indirectly.
    """
    n_steps = int(chirp_s / _DT)
    all_log: list[dict] = []
    for run_i in range(runs):
        sil.reset()
        print(f"  [M5 run {run_i+1}/{runs}] accel to {vx_init:.0f} m/s... ",
              end="", flush=True)
        _accel_to(sil, vx_init, throttle=0.7)
        print(f"throttle chirp {chirp_s:.0f} s... ", end="", flush=True)
        log = []
        for i in range(n_steps):
            t_chirp = i * _DT
            phase = 2.0 * math.pi * (f0 * t_chirp + (f1 - f0) * t_chirp ** 2 / (2.0 * chirp_s))
            T_val = float(np.clip(0.5 + 0.4 * math.sin(phase), 0.0, 1.0))
            s = sil.get_state()
            sil.apply_control(T_val, 0.0, 0.0)
            sil.step()
            log.append({
                "T": T_val, "vx": s["vx"], "ax": s["ax"],
                "vy": s["vy"], "r": s["r"], "ay": s["ay"],
            })
        all_log.extend(log)
        print(f"{len(log)} samples.")
    return all_log


def run_random_combined(
    sil,
    duration_s: float = 90.0,
    seed: int = 0,
) -> list[dict]:
    """M6: PRBS-style random steering + throttle for cross-validation.

    Inputs are held for a random duration drawn from U(0.3, 1.5) s.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / _DT)
    sil.reset()
    print(f"  [M6] accel to 12 m/s... ", end="", flush=True)
    _accel_to(sil, 12.0, throttle=0.5)
    print(f"random {duration_s:.0f} s... ", end="", flush=True)

    cur_delta = 0.0
    cur_T = 0.3
    hold_remaining = 0
    log = []
    for _ in range(n_steps):
        if hold_remaining <= 0:
            cur_delta = float(rng.uniform(-0.30, 0.30))
            cur_T = float(rng.uniform(0.10, 0.90))
            hold_remaining = int(rng.uniform(0.3, 1.5) / _DT)
        hold_remaining -= 1
        s = sil.get_state()
        sil.apply_control(cur_T, 0.0, cur_delta)
        sil.step()
        log.append({
            "vx": s["vx"], "vy": s["vy"], "r": s["r"],
            "ax": s["ax"], "ay": s["ay"],
            "delta": cur_delta, "T": cur_T,
        })
    print(f"{len(log)} samples.")
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BeamNG vehicle system identification -> vehicles/<model>.json"
    )
    parser.add_argument("--vehicle",   default="etk800",
                        help="BeamNG vehicle model name (default: etk800)")
    parser.add_argument("--wheelbase", type=float, required=True,
                        help="Wheelbase L = LF+LR [m]")
    parser.add_argument("--mass",      type=float, required=True,
                        help="Vehicle mass [kg]")
    parser.add_argument("--lf",        type=float, default=None,
                        help="Override CoG-to-front distance [m] (skips M3 LF/LR estimate)")
    parser.add_argument("--lr",        type=float, default=None,
                        help="Override CoG-to-rear distance [m]")
    parser.add_argument("--skip",      default="",
                        help="Comma-separated maneuvers to skip, e.g. M3,M5")
    parser.add_argument("--tag",       default="",
                        help="Optional suffix for output filename")
    parser.add_argument("--out-dir",   default="vehicles",
                        help="Directory for JSON output (default: vehicles)")
    args = parser.parse_args()

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    M = float(args.mass)
    L = float(args.wheelbase)

    # Initial LF/LR: use overrides if provided, otherwise assume 50/50
    if args.lf is not None and args.lr is not None:
        LF, LR = float(args.lf), float(args.lr)
        lf_lr_from_spec = True
    else:
        LF, LR = L / 2.0, L / 2.0
        lf_lr_from_spec = args.lf is not None  # partial override → warn

    # Patch sil_beamng globals before instantiating BeamNGSIL
    _sil_mod.VEHICLE_MODEL = args.vehicle
    _sil_mod.TRACK_MODE = None   # sysid uses flat plane, no track object

    from sil_beamng import BeamNGSIL
    sil = BeamNGSIL()
    sil.open(launch=True)

    print(f"\n=== sysid: {args.vehicle}  M={M:.0f} kg  L={L:.3f} m ===")
    print(f"    initial LF={LF:.3f} m  LR={LR:.3f} m  skip={skip or 'none'}\n")

    raw_results: dict = {}
    data_m4: list[dict] = []
    data_m6: list[dict] = []
    all_ay: list[np.ndarray] = []

    # ------------------------------------------------------------------ M1
    if "M1" not in skip:
        print("[M1] Coast-down  ->  C_DRAG")
        data_m1 = run_coast_down(sil)
        vx, ax = _to_arr(data_m1, "vx", "ax")
        raw_results["M1"] = fit_drag(vx, ax, M)
        print(f"     C_DRAG={raw_results['M1']['C_DRAG']:.4f}  "
              f"R2={raw_results['M1']['R2']:.4f}\n")

    # ------------------------------------------------------------------ M2
    if "M2" not in skip:
        print("[M2] Accel sweep  ->  F_DRIVE_MAX, C_DRAG")
        data_m2 = run_accel_sweep(sil)
        T_arr, vx, ax = _to_arr(data_m2, "T", "vx", "ax")
        raw_results["M2"] = fit_drivetrain(T_arr, vx, ax, M)
        print(f"     F_DRIVE_MAX={raw_results['M2']['F_DRIVE_MAX']:.0f} N  "
              f"C_DRAG={raw_results['M2']['C_DRAG']:.4f}  "
              f"R2={raw_results['M2']['R2']:.4f}\n")

    # ------------------------------------------------------------------ M3
    if "M3" not in skip and not lf_lr_from_spec:
        print("[M3] Understeer gradient circle  ->  LF, LR")
        data_m3 = run_ug_circle(sil)
        delta_arr, vx, r = _to_arr(data_m3, "delta_rad", "vx", "r")
        raw_results["M3"] = fit_understeer_gradient(delta_arr, vx, r)
        K_us = raw_results["M3"]["K_us"]
        print(f"     K_us={K_us:.5f} rad·s²/m  "
              f"L_fit={raw_results['M3']['L']:.3f} m (spec={L:.3f})  "
              f"R2={raw_results['M3']['R2']:.4f}")
        print(f"     (LF/LR will be refined after M4 provides CA_F, CA_R)\n")
    elif "M3" in skip and not lf_lr_from_spec:
        print("[M3] skipped — using LF=LR=L/2 as fallback\n")

    # ------------------------------------------------------------------ M4
    if "M4" not in skip:
        print("[M4] Steering chirp  ->  CA_F, CA_R, IZ")
        data_m4 = run_steer_chirp(sil)
        vx4, vy4, r4 = _to_arr(data_m4, "vx", "vy", "r")
        delta4 = np.array([row["delta"] for row in data_m4])
        vy_dot4 = _diff(vy4)
        r_dot4 = _diff(r4)
        mask = vx4 > 3.0
        raw_results["M4"] = fit_bicycle_linear(
            delta4[mask], vx4[mask], vy4[mask], r4[mask],
            vy_dot4[mask], r_dot4[mask], M, LF, LR,
        )
        CA_F = raw_results["M4"]["CA_F"]
        CA_R = raw_results["M4"]["CA_R"]
        IZ   = raw_results["M4"]["IZ"]
        print(f"     CA_F={CA_F:.0f} N/rad  CA_R={CA_R:.0f} N/rad  IZ={IZ:.0f} kg·m²")
        print(f"     R2_vy={raw_results['M4']['R2_vy']:.4f}  "
              f"R2_r={raw_results['M4']['R2_r']:.4f}")

        # Refine LF/LR from K_us (M3) and cornering stiffnesses (M4)
        if not lf_lr_from_spec and "M3" in raw_results and CA_F > 0 and CA_R > 0:
            K_us = raw_results["M3"]["K_us"]
            # K_us*(M/L) = LR/CA_F - LF/CA_R;  LF+LR = L
            # solving: LF = (L/CA_F - K_us*L/M) / (1/CA_F + 1/CA_R)
            rhs = L / CA_F - K_us * L / M
            denom = 1.0 / CA_F + 1.0 / CA_R
            LF_cand = rhs / denom
            LR_cand = L - LF_cand
            if 0.05 * L <= LF_cand <= 0.95 * L:
                LF, LR = LF_cand, LR_cand
                # Refit with refined geometry
                raw_results["M4"] = fit_bicycle_linear(
                    delta4[mask], vx4[mask], vy4[mask], r4[mask],
                    vy_dot4[mask], r_dot4[mask], M, LF, LR,
                )
                CA_F = raw_results["M4"]["CA_F"]
                CA_R = raw_results["M4"]["CA_R"]
                IZ   = raw_results["M4"]["IZ"]
                raw_results["M4"]["LF"] = LF
                raw_results["M4"]["LR"] = LR
                print(f"     refined LF={LF:.3f} m  LR={LR:.3f} m")
                print(f"     re-fit: CA_F={CA_F:.0f}  CA_R={CA_R:.0f}  IZ={IZ:.0f}")
            else:
                print(f"     [warn] K_us gave LF={LF_cand:.3f} m — outside sane range, keeping L/2")

        all_ay.append(np.array([row["ay"] for row in data_m4]))
        print()

    # ------------------------------------------------------------------ M5
    if "M5" not in skip:
        print("[M5] Throttle chirp  ->  validates F_DRIVE_MAX, C_DRAG")
        data_m5 = run_throttle_chirp(sil)
        T_arr, vx, ax = _to_arr(data_m5, "T", "vx", "ax")
        raw_results["M5"] = fit_drivetrain(T_arr, vx, ax, M)
        print(f"     F_DRIVE_MAX={raw_results['M5']['F_DRIVE_MAX']:.0f} N  "
              f"C_DRAG={raw_results['M5']['C_DRAG']:.4f}  "
              f"R2={raw_results['M5']['R2']:.4f}\n")

    # ------------------------------------------------------------------ M6
    if "M6" not in skip:
        print("[M6] Combined random  ->  cross-validation + MU")
        data_m6 = run_random_combined(sil)
        all_ay.append(np.array([row["ay"] for row in data_m6]))
        print()

    sil.close()

    # ------------------------------------------------------------------ MU
    if all_ay:
        mu_result = extract_mu(np.concatenate(all_ay))
    else:
        mu_result = {"MU": 0.9, "ay_percentile": 95}
        print("[warn] no lateral acceleration data — MU defaulting to 0.9")
    MU = mu_result["MU"]
    print(f"MU (95th-pct ay/g) = {MU:.3f}  "
          f"(lower bound — increase if vehicle never reached the friction limit)\n")

    # ------------------------------------------------------------------ assemble params
    # Prefer M2 for drivetrain; fall back through M5 → M1 for individual values
    drv = raw_results.get("M2") or raw_results.get("M5") or {}
    drag_only = raw_results.get("M1") or {}
    lat = raw_results.get("M4") or {}

    C_DRAG      = drv.get("C_DRAG")      or drag_only.get("C_DRAG", 1.0)
    F_DRIVE_MAX = drv.get("F_DRIVE_MAX", 8000.0)
    CA_F        = lat.get("CA_F", 90000.0)
    CA_R        = lat.get("CA_R", 90000.0)
    IZ          = lat.get("IZ",   1800.0)

    drift_env_params = {
        "M":           float(M),
        "IZ":          round(float(IZ), 1),
        "LF":          round(float(LF), 4),
        "LR":          round(float(LR), 4),
        "CA_F":        round(float(CA_F), 1),
        "CA_R":        round(float(CA_R), 1),
        "MU":          round(float(MU), 4),
        "F_DRIVE_MAX": round(float(F_DRIVE_MAX), 1),
        "C_DRAG":      round(float(C_DRAG), 6),
    }

    # ------------------------------------------------------------------ validation
    validation: dict = {}
    if data_m6:
        vx6, vy6, r6 = _to_arr(data_m6, "vx", "vy", "r")
        delta6 = np.array([row["delta"] for row in data_m6])
        T6     = np.array([row["T"]     for row in data_m6])
        validation = validate_forward_sim(
            drift_env_params, delta6, T6,
            float(vx6[0]), float(vy6[0]), float(r6[0]),
            vx6, vy6, r6, _DT,
        )
        print(f"Forward-sim validation RMSE:")
        print(f"  vx={validation['RMSE_vx']:.3f} m/s  "
              f"vy={validation['RMSE_vy']:.3f} m/s  "
              f"r={validation['RMSE_r']:.4f} rad/s\n")

    # ------------------------------------------------------------------ write JSON
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out_dir) / f"{args.vehicle}{tag}.json"

    def _clean(d: dict) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}

    out = {
        "vehicle_model": args.vehicle,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "wheelbase_m": float(L),
        "mass_kg": float(M),
        "maneuvers_run": [m for m in ("M1", "M2", "M3", "M4", "M5", "M6") if m not in skip],
        "raw_fit": {k: _clean(v) for k, v in raw_results.items() if isinstance(v, dict)},
        "mu_estimate": _clean(mu_result),
        "drift_env_params": drift_env_params,
        "validation_rmse": {k: round(v, 5) for k, v in validation.items()},
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Written: {out_path}")
    print("\ndrift_env_params (copy into DriftEnv class attributes):")
    print(json.dumps(drift_env_params, indent=2))


if __name__ == "__main__":
    main()
