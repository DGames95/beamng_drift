"""Sim-to-sim comparison: BeamNG vs the bicycle drift model, same policy & track.

Two questions, from a SINGLE BeamNG rollout of the dr_sim2real policy on a
random track (default seed 55):

  1. CLOSED-LOOP TRAJECTORY / PHASE-PORTRAIT  (like driftRL/evaluate.py)
     Run the same policy in BeamNG AND in the bicycle model on the SAME seed-55
     canonical track and overlay:
        * the (x, y) trajectory in the track's canonical frame
        * the (v_y, r) phase portrait
     so you can see how far the real-physics rollout drifts from the model the
     policy was trained on.

  2. ONE-STEP DYNAMICS DIFF  (the model gap, isolated from integration drift)
     At every BeamNG tick we know the measured body-frame state (vx, vy, r) and
     the command actually applied (delta, T). Feed THAT state and command into
     the bicycle tire/force model and read off its predicted body accelerations.
     Plot, vs time:  throttle, steering, ax (= d vx/dt), ay (= d vy/dt), with
     BeamNG measured against the bicycle one-step prediction. Because the
     bicycle is fed BeamNG's own state each tick, the gap is purely model error,
     not integration divergence.

BeamNG's ax/ay (sil_beamng finite-differences d(body vx)/dt, d(body vy)/dt) and
the bicycle's vx_dot/vy_dot are the time derivative of the SAME body-frame
velocity components (both include the r*v centripetal coupling), so they are
directly comparable.

The bicycle model is a self-contained copy of driftRL/drift_env.py's dynamics
(explicit Euler, dt=0.02 s, tanh-saturating tires + rear friction ellipse) using
the SAME vehicle parameters JSON the training env loads
(driftRL/vehicle_params/etkc_kc8_drift_M.json) and the local track.py geometry +
obs() pipeline (which is kept equal to driftRL's so the policy transfers).

Usage:
    python sim2sim_diff.py                 # launch BeamNG, run seed 55, plot
    python sim2sim_diff.py --seed 55 --seconds 40
    python sim2sim_diff.py --connect       # attach to a running BeamNG
    python sim2sim_diff.py --check         # offline: bicycle rollout + plots only, NO BeamNG
    python sim2sim_diff.py --replay        # re-plot from a saved BeamNG CSV, NO BeamNG

Figures -> report/figures/sim2sim_*.{pdf,png}
Telemetry -> report/sim2sim_beamng_seed<seed>.{csv,npz}
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import matplotlib

from track import Track, DEFAULT_LOOKAHEAD
import test_sim2real_beamng as T2R   # reuse the SIL plumbing (rolling start, anchor, etc.)

FIG_DIR = "report/figures"
OUT_DIR = "report"
VEHICLE_PARAMS_JSON = "driftRL/vehicle_params/etkc_kc8_drift_M.json"

# --- defaults (edit here rather than retyping flags) ---
DEFAULT_MODEL   = "driftRL/models/drift_dr_sim2real/best_model"   # the sim-to-real DR policy
DEFAULT_TRACK   = "circle"   # "random" | "circle"
DEFAULT_SEED    = 55         # random-track layout seed (--track random)
DEFAULT_RADIUS  = 25.0       # circle radius [m]          (--track circle)
DEFAULT_LENGTH  = 600.0      # open-track length [m]      (--track random)
DEFAULT_START_SPEED = 11.0
DEFAULT_SECONDS = 40.0
DEFAULT_VEHICLE = "sunburst2"
DEFAULT_CONFIG  = "drift_pro"

G = 9.81
DT = 0.02   # bicycle integration step [s] (== driftRL DriftEnv.DT, == SIL control DT)


def build_track(args):
    """Canonical track + a (tag, label) used for filenames and plot titles."""
    if args.track == "circle":
        return (Track.circle(radius=args.radius),
                f"circle_r{int(args.radius)}", f"circle R={args.radius:.0f} m")
    return (Track.random_track(np.random.default_rng(args.seed), length=args.length),
            f"random_seed{args.seed}", f"random seed {args.seed}")


# --------------------------------------------------------------------------- #
# Bicycle model — copy of driftRL/drift_env.py dynamics, same parameters       #
# --------------------------------------------------------------------------- #
class BicycleModel:
    """3-DOF single-track model with tanh tires + rear friction ellipse.

    Parameters loaded from the same JSON the training DriftEnv uses, so this
    reproduces exactly the dynamics the policy was trained against.
    """

    def __init__(self, params_json=VEHICLE_PARAMS_JSON):
        with open(params_json) as f:
            d = json.load(f)
        p = d.get("drift_env_params", d)
        self.M, self.IZ = p["M"], p["IZ"]
        self.LF, self.LR = p["LF"], p["LR"]
        self.CA_F, self.CA_R = p["CA_F"], p["CA_R"]
        self.MU = p["MU"]
        self.F_DRIVE_MAX = p["F_DRIVE_MAX"]
        self.C_DRAG = p["C_DRAG"]
        L = self.LF + self.LR
        self.FY_MAX_F = self.MU * self.M * G * self.LR / L
        self.FY_MAX_R = self.MU * self.M * G * self.LF / L

    def accel(self, vx, vy, r, delta, T):
        """Body-frame accelerations (vx_dot, vy_dot, r_dot) for this state+command."""
        vx_safe = max(vx, 0.5)
        alpha_f = np.arctan2(vy + self.LF * r, vx_safe) - delta
        alpha_r = np.arctan2(vy - self.LR * r, vx_safe)

        Fx_r = np.clip(T * self.F_DRIVE_MAX, -self.FY_MAX_R, self.FY_MAX_R)
        fy_max_r_eff = self.FY_MAX_R * np.sqrt(max(1.0 - (Fx_r / self.FY_MAX_R) ** 2, 1e-3))

        Fyf = -self.FY_MAX_F * np.tanh(self.CA_F * alpha_f / self.FY_MAX_F)
        Fyr = -fy_max_r_eff * np.tanh(self.CA_R * alpha_r / fy_max_r_eff)

        Fx = Fx_r - self.C_DRAG * vx * abs(vx)

        vx_dot = (Fx - Fyf * np.sin(delta)) / self.M + r * vy
        vy_dot = (Fyf * np.cos(delta) + Fyr) / self.M - r * vx
        r_dot = (self.LF * Fyf * np.cos(delta) - self.LR * Fyr) / self.IZ
        return float(vx_dot), float(vy_dot), float(r_dot)


# --------------------------------------------------------------------------- #
# Bicycle closed-loop rollout on a fixed canonical track                       #
# --------------------------------------------------------------------------- #
def run_bicycle(policy, model: BicycleModel, track: Track, start_speed, seconds):
    """Roll the policy out in the bicycle model on `track`.

    Deterministic start (vx=start_speed, vy=r=0, aligned at the track start) to
    match how the policy engages in BeamNG. Obs are built with the local
    track.obs() pipeline (== driftRL OBS_SCALE/lookahead) so the policy sees the
    same inputs it would in BeamNG. Logs the same fields as the BeamNG run.
    """
    x, y = float(track.xy[0, 0]), float(track.xy[0, 1])
    psi = float(track.psi[0])
    vx, vy, r = float(start_speed), 0.0, 0.0
    hint = 0
    n = int(seconds / DT)
    log = []
    for k in range(n):
        obs, e_y, e_psi, _, hint = track.obs(x, y, psi, vx, vy, r, hint, DEFAULT_LOOKAHEAD)
        action, _ = policy.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        delta = float(np.clip(action[0], -0.5, 0.5))
        T = float(np.clip(action[1], -1.0, 1.0))

        ax, ay, r_dot = model.accel(vx, vy, r, delta, T)   # accel at current state
        log.append({
            "t": k * DT, "x": x, "y": y, "psi": psi,
            "vx": vx, "vy": vy, "r": r, "beta": math.atan2(vy, max(vx, 0.5)),
            "ax": ax, "ay": ay, "e_y": e_y, "e_psi": e_psi,
            "delta": delta, "throttle": max(0.0, T), "brake": max(0.0, -T), "T": T,
        })

        # global kinematics + explicit Euler (== DriftEnv.step)
        x += DT * (vx * math.cos(psi) - vy * math.sin(psi))
        y += DT * (vx * math.sin(psi) + vy * math.cos(psi))
        psi += DT * r
        vx += DT * ax
        vy += DT * ay
        r += DT * r_dot

        if vx < 1.0 or track.off_track(e_y) or track.at_end(hint):
            print(f"[bicycle] rollout ended at t={k*DT:.1f}s "
                  f"(vx={vx:.1f} e_y={e_y:+.1f} at_end={track.at_end(hint)})")
            break
    return {k: np.array([d[k] for d in log]) for k in log[0]}


# --------------------------------------------------------------------------- #
# BeamNG closed-loop rollout (reuses test_sim2real_beamng plumbing)            #
# --------------------------------------------------------------------------- #
def run_beamng(args, policy, model: BicycleModel, base_track):
    """Run the policy in BeamNG. Returns (log dict, anchor (x0,y0,dpsi)).

    The anchor maps the world-frame BeamNG trajectory back into the track's
    canonical frame so it overlays the bicycle rollout.
    """
    S = T2R.S
    S.VEHICLE_MODEL = args.vehicle
    if args.config and args.config.lower() not in ("stock", "default", "none"):
        S.VEHICLE_CONFIG = f"vehicles/{args.vehicle}/{args.config}.pc"
    else:
        S.VEHICLE_CONFIG = None
    print(f"[beamng] vehicle={S.VEHICLE_MODEL} config={S.VEHICLE_CONFIG}")

    sil = S.BeamNGSIL().open(launch=not args.connect)
    dt = S.DT

    # --- car startup: recover, release handbrake, arcade gearbox (see T2R.run) ---
    try:
        sil.vehicle.recover()
        for _ in range(int(0.5 * S.CONTROL_HZ)):
            sil.step()
        sil.vehicle.set_shift_mode("arcade")
        sil.vehicle.control(throttle=0.0, brake=0.0, parkingbrake=0.0)
        sil.step()
        sil._prime_state()
    except Exception as e:
        print(f"[beamng] startup recover/gearbox setup failed ({e}); continuing")

    if args.start_speed > 0:
        print(f"[beamng] rolling start -> {args.start_speed:.1f} m/s ...", flush=True)
        reached = T2R.rolling_start(sil, args.start_speed)
        print(f"[beamng] rolling start reached {reached:.1f} m/s")

    sign, mean_r = T2R.calibrate_steering_sign(sil)
    if sign is not None and sign != sil.steer_sign:
        print(f"[beamng] WARNING measured steer sign {sign:+.0f} overrides default; using it")
        sil.steer_sign = sign

    target = args.start_speed if args.start_speed > 0 else sil.get_state()["speed"]
    s0 = T2R.settle_straight(sil, target)

    dpsi = s0["psi"] - float(base_track.psi[0])
    sil.track = T2R.anchor_track(base_track, s0["X"], s0["Y"], s0["psi"])
    sil._track_hint = 0
    anchor = (s0["X"], s0["Y"], dpsi)
    print(f"[beamng] engaged X={s0['X']:.1f} Y={s0['Y']:.1f} "
          f"psi={math.degrees(s0['psi']):+.0f}deg speed={s0['speed']:.2f} m/s")

    max_steps = int(args.seconds / dt)
    print(f"[beamng] running {args.seconds:.0f}s ({max_steps} ticks @ {1/dt:.0f} Hz)")

    view = None
    if args.overlay:
        from debug_view import DebugView
        view = DebugView(sil.track, update_hz=15.0)

    log = []
    try:
        for k in range(max_steps):
            t = k * dt
            state = sil.get_state()
            if view is not None:
                view.update(state, t)
            if state.get("off_track"):
                print(f"\n[beamng] OFF TRACK e_y={state['track_frame'][0]:+.2f} at t={t:.1f}s")
                break
            if state.get("track_end"):
                print(f"\n[beamng] reached end of track at t={t:.1f}s")
                break

            obs = state.get("track_obs")
            if obs is None:
                print("[beamng] no track_obs — aborting")
                break

            throttle, brake, delta = T2R.policy_action(policy, obs)
            sil.apply_control(throttle, brake, delta)
            sil.step()

            vx, vy, r = state["vx"], state["vy"], state["r"]
            T = throttle - brake
            ax_b, ay_b, _ = model.accel(vx, vy, r, delta, T)  # bicycle one-step pred
            tf = state["track_frame"]
            log.append({
                "t": t, "X": state["X"], "Y": state["Y"], "psi": state["psi"],
                "vx": vx, "vy": vy, "r": r,
                "beta": math.atan2(vy, vx) if state["speed"] > 0.5 else 0.0,
                "ax": state["ax"], "ay": state["ay"],
                "ax_bike": ax_b, "ay_bike": ay_b,
                "delta": delta, "throttle": throttle, "brake": brake, "T": T,
                "e_y": tf[0], "e_psi": tf[1], "speed": state["speed"],
            })
            print(f"t={t:6.2f} v={state['speed']:5.2f} vx={vx:5.2f} vy={vy:5.2f} "
                  f"beta={math.degrees(log[-1]['beta']):+6.1f} e_y={tf[0]:+5.2f}", end="\r")
    except KeyboardInterrupt:
        print("\n[beamng] interrupted.")
    except Exception as e:
        print(f"\n[beamng] sim ended early ({type(e).__name__}: {e}) — partial telemetry.")
    finally:
        try:
            if view is not None:
                view.close()
            sil.close()
        except Exception:
            pass

    if not log:
        return None, anchor
    return {k: np.array([d[k] for d in log]) for k in log[0]}, anchor


# --------------------------------------------------------------------------- #
# Frame transform + IO                                                         #
# --------------------------------------------------------------------------- #
def world_to_canonical(X, Y, anchor, canon0):
    """Map world-frame BeamNG positions into the canonical track frame.

    Inverse of T2R.anchor_track, which builds
        world = (canon - canon0) @ R(dpsi).T + [x0,y0]
    where canon0 = track.xy[0] (the start sample, = (0,0) for a random track but
    (R,0) for a circle). Inverting:
        canon = (world - [x0,y0]) @ R(dpsi) + canon0.
    """
    x0, y0, dpsi = anchor
    c, s = math.cos(dpsi), math.sin(dpsi)
    R = np.array([[c, -s], [s, c]])
    P = np.stack([np.asarray(X) - x0, np.asarray(Y) - y0], axis=1) @ R
    return P[:, 0] + canon0[0], P[:, 1] + canon0[1]


def save_beamng(blog, anchor, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, f"sim2sim_beamng_{tag}.npz")
    csvp = os.path.join(OUT_DIR, f"sim2sim_beamng_{tag}.csv")
    np.savez(npz, anchor=np.array(anchor, dtype=float), **blog)
    keys = list(blog.keys())
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(blog["t"])):
            w.writerow([blog[k][i] for k in keys])
    print(f"[io] saved telemetry -> {npz}\n[io] saved telemetry -> {csvp}")


def load_beamng(tag):
    npz = os.path.join(OUT_DIR, f"sim2sim_beamng_{tag}.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found — run without --replay first")
    d = np.load(npz)
    anchor = tuple(float(v) for v in d["anchor"])
    blog = {k: d[k] for k in d.files if k != "anchor"}
    return blog, anchor


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def draw_track(ax, track):
    for line in (track.left, track.right):
        pts = np.vstack([line, line[:1]]) if track.closed else line
        ax.plot(pts[:, 0], pts[:, 1], "k-", lw=0.8)
    ax.plot(track.xy[:, 0], track.xy[:, 1], "k--", lw=0.4, alpha=0.5)
    ax.set_aspect("equal")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}", dpi=130)
    print(f"[fig] {FIG_DIR}/{name}.pdf (+ .png)")


def plot_trajectory(blog, bklog, track, anchor, tag, label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_track(ax, track)
    if bklog is not None:
        ax.plot(bklog["x"], bklog["y"], "tab:orange", lw=1.4, label="bicycle model")
    if blog is not None:
        bx, by = world_to_canonical(blog["X"], blog["Y"], anchor, track.xy[0])
        ax.plot(bx, by, "tab:blue", lw=1.4, label="BeamNG")
        ax.scatter(bx[0], by[0], c="g", s=30, zorder=5, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory — BeamNG vs bicycle ({label})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); _save(fig, f"sim2sim_trajectory_{tag}"); plt.close(fig)


def plot_phase(blog, bklog, tag, label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if bklog is not None:
        ax.plot(bklog["vy"], bklog["r"], "tab:orange", lw=0.9, label="bicycle model")
        ax.scatter(bklog["vy"][-1], bklog["r"][-1], c="darkorange", marker="x", zorder=5)
    if blog is not None:
        ax.plot(blog["vy"], blog["r"], "tab:blue", lw=0.9, label="BeamNG")
        ax.scatter(blog["vy"][0], blog["r"][0], c="g", zorder=5, label="start")
        ax.scatter(blog["vy"][-1], blog["r"][-1], c="r", zorder=5, label="end")
    ax.set_xlabel(r"$v_y$ [m/s]"); ax.set_ylabel(r"$r$ [rad/s]")
    ax.set_title(f"Phase portrait — BeamNG vs bicycle ({label})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, f"sim2sim_phase_portrait_{tag}"); plt.close(fig)


def plot_histories(blog, tag, label):
    """Slip-angle + state histories of the BeamNG (sim2real) run, in the exact
    layout of driftRL/evaluate.py diagnostics(): beta; vx/vy; e_y/delta/10*T."""
    import matplotlib.pyplot as plt
    if blog is None:
        print("[fig] no BeamNG telemetry — skipping histories plot")
        return
    t = blog["t"]
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    axs[0].plot(t, np.degrees(blog["beta"])); axs[0].set_ylabel(r"$\beta$ [deg]")
    axs[1].plot(t, blog["vx"], label=r"$v_x$"); axs[1].plot(t, blog["vy"], label=r"$v_y$")
    axs[1].set_ylabel("[m/s]"); axs[1].legend()
    axs[2].plot(t, blog["e_y"], label=r"$e_y$ [m]")
    axs[2].plot(t, np.degrees(blog["delta"]), label=r"$\delta$ [deg]")
    axs[2].plot(t, 10 * blog["T"], label=r"$10\,T$")
    axs[2].set_xlabel("t [s]"); axs[2].legend(ncol=3, fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].set_title(f"Slip angle and state histories — BeamNG ({label})")
    fig.tight_layout(); _save(fig, f"sim2sim_histories_{tag}"); plt.close(fig)


def plot_accel_diff(blog, tag, label):
    """Inputs (throttle, steering) and resulting ax, ay: BeamNG measured vs the
    bicycle model's one-step prediction on BeamNG's own state."""
    import matplotlib.pyplot as plt
    if blog is None:
        print("[fig] no BeamNG telemetry — skipping accel diff plot")
        return
    t = blog["t"]
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(t, blog["T"], "k", lw=1.0)
    axs[0].axhline(0, color="grey", lw=0.5)
    axs[0].set_ylabel("throttle T\n(+thr / -brake)")
    axs[0].set_title(f"BeamNG vs bicycle one-step dynamics ({label})")

    axs[1].plot(t, np.degrees(blog["delta"]), "k", lw=1.0)
    axs[1].set_ylabel(r"steering $\delta$ [deg]")

    axs[2].plot(t, blog["ax"], color="tab:blue", lw=0.8, alpha=0.7, label="BeamNG (measured)")
    axs[2].plot(t, blog["ax_bike"], color="tab:orange", lw=1.2, label="bicycle (predicted)")
    axs[2].set_ylabel(r"$a_x$ [m/s$^2$]"); axs[2].legend(fontsize=8, ncol=2)

    axs[3].plot(t, blog["ay"], color="tab:blue", lw=0.8, alpha=0.7, label="BeamNG (measured)")
    axs[3].plot(t, blog["ay_bike"], color="tab:orange", lw=1.2, label="bicycle (predicted)")
    axs[3].set_ylabel(r"$a_y$ [m/s$^2$]"); axs[3].set_xlabel("t [s]")
    axs[3].legend(fontsize=8, ncol=2)

    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, f"sim2sim_accel_diff_{tag}"); plt.close(fig)

    for comp in ("ax", "ay"):
        err = blog[comp] - blog[comp + "_bike"]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        print(f"[diff] {comp}: RMSE(beamng - bike) = {rmse:.2f} m/s^2, mean bias = {bias:+.2f}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--track", choices=["random", "circle"], default=DEFAULT_TRACK)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="random-track layout seed (--track random)")
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                   help="circle radius [m] (--track circle)")
    p.add_argument("--length", type=float, default=DEFAULT_LENGTH)
    p.add_argument("--start-speed", type=float, default=DEFAULT_START_SPEED)
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    p.add_argument("--vehicle", default=DEFAULT_VEHICLE)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--connect", action="store_true", help="attach to a running BeamNG")
    p.add_argument("--no-overlay", dest="overlay", action="store_false", default=True)
    p.add_argument("--check", action="store_true",
                   help="offline: bicycle rollout + plots only, NO BeamNG")
    p.add_argument("--replay", action="store_true",
                   help="re-plot from saved BeamNG telemetry, NO BeamNG")
    args = p.parse_args()

    matplotlib.use("Agg")

    policy = T2R.load_policy(args.model_path)
    print(f"[main] policy: {args.model_path}  obs{policy.observation_space.shape} "
          f"act{policy.action_space.shape}")

    model = BicycleModel()
    base_track, tag, label = build_track(args)
    print(f"[main] track: {label}  ({base_track.n} pts, "
          f"{'closed' if base_track.closed else f'{base_track.length:.0f} m open'})")

    print("[main] bicycle rollout ...")
    bklog = run_bicycle(policy, model, base_track, args.start_speed, args.seconds)

    if args.check:
        blog, anchor = None, (0.0, 0.0, 0.0)
        print("[main] --check: skipping BeamNG")
    elif args.replay:
        blog, anchor = load_beamng(tag)
        print(f"[main] --replay: loaded {len(blog['t'])} BeamNG ticks")
    else:
        blog, anchor = run_beamng(args, policy, model, base_track)
        if blog is not None:
            save_beamng(blog, anchor, tag)

    plot_trajectory(blog, bklog, base_track, anchor, tag, label)
    plot_phase(blog, bklog, tag, label)
    plot_histories(blog, tag, label)
    plot_accel_diff(blog, tag, label)
    print("[main] done.")


if __name__ == "__main__":
    main()
