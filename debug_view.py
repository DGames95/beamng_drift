"""
Live top-down debug view — runs in a background daemon thread so the SIL
control loop is never blocked waiting for matplotlib/Tk to process events.

Usage:
    from debug_view import DebugView
    view = DebugView(sil.track, update_hz=15.0)   # spawns thread, opens window
    ...
    view.update(state, t_sim)    # non-blocking: just writes to a shared slot
    ...
    view.close()                 # signals thread to exit
"""

import math
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_CAR_LENGTH = 4.5
_CAR_WIDTH  = 2.0


class DebugView:
    def __init__(self, track=None, update_hz: float = 15.0):
        self._track = track
        self._period = 1.0 / max(update_hz, 1.0)

        # shared state: main loop writes, draw thread reads
        self._slot: dict | None = None
        self._t_sim: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True, name="DebugView")
        self._thread.start()

    # ----------------------------------------------------------------- public API (main thread)

    def update(self, state: dict, t_sim: float = 0.0):
        """Drop-in, non-blocking. Called from the control loop."""
        with self._lock:
            self._slot = state
            self._t_sim = t_sim

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ----------------------------------------------------------------- draw thread

    def _run(self):
        self._build()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                state = self._slot
                t_sim = self._t_sim
            if state is not None:
                try:
                    self._draw(state, t_sim)
                except Exception:
                    pass   # never crash the draw thread
            elapsed = time.perf_counter() - t0
            plt.pause(max(self._period - elapsed, 0.001))
        plt.close(self._fig)

    # ----------------------------------------------------------------- build (draw thread)

    def _build(self):
        self._fig, axes = plt.subplots(1, 2, figsize=(13, 7),
                                       gridspec_kw={"width_ratios": [3, 1]})
        self._fig.patch.set_facecolor("#1a1a2e")
        self._fig.canvas.manager.set_window_title("BeamNG SIL — debug view")

        self._ax  = axes[0]
        self._tax = axes[1]
        self._setup_map()
        self._setup_text()
        plt.tight_layout()

    def _setup_map(self):
        ax = self._ax
        ax.set_facecolor("#0d0d1a")
        ax.set_aspect("equal")
        ax.tick_params(colors="#888")
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.set_title("top-down", color="#ccc", fontsize=9)
        ax.set_xlabel("X [m]", color="#888", fontsize=8)
        ax.set_ylabel("Y [m]", color="#888", fontsize=8)
        ax.grid(True, color="#2a2a3e", lw=0.5)

        self._ln_cl,    = ax.plot([], [], color="#3a86ff", lw=1.2, label="centreline", zorder=2)
        self._ln_left,  = ax.plot([], [], color="#ff6b6b", lw=0.8, ls="--", label="boundary", zorder=2)
        self._ln_right, = ax.plot([], [], color="#ff6b6b", lw=0.8, ls="--", zorder=2)

        self._car_patch = plt.Polygon([[0, 0]] * 4, closed=True,
                                      facecolor="#ffd166", edgecolor="#fff", lw=1.0, zorder=5)
        ax.add_patch(self._car_patch)

        self._head_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#06d6a0", lw=1.5), zorder=6)
        self._vel_arrow  = ax.annotate("", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#ffa500", lw=1.5), zorder=6)

        self._nearest_dot, = ax.plot([], [], "o", color="#06d6a0", ms=5, zorder=4)

        self._ot_rect = mpatches.FancyBboxPatch(
            (0.02, 0.88), 0.96, 0.10, boxstyle="round,pad=0.01",
            transform=ax.transAxes, facecolor="#f00", alpha=0.0, zorder=10, clip_on=False)
        ax.add_patch(self._ot_rect)
        self._ot_text = ax.text(0.5, 0.93, "OFF TRACK", transform=ax.transAxes,
                                ha="center", va="center", fontsize=12, fontweight="bold",
                                color="white", alpha=0.0, zorder=11)

        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccc", loc="upper right")

        if self._track is not None:
            t = self._track
            cl = np.vstack([t.xy, t.xy[:1]]) if t.closed else t.xy
            lb = np.vstack([t.left,  t.left[:1]])  if t.closed else t.left
            rb = np.vstack([t.right, t.right[:1]]) if t.closed else t.right
            self._ln_cl.set_data(cl[:, 0], cl[:, 1])
            self._ln_left.set_data(lb[:, 0], lb[:, 1])
            self._ln_right.set_data(rb[:, 0], rb[:, 1])
            pad = 15.0
            ax.set_xlim(cl[:, 0].min() - pad, cl[:, 0].max() + pad)
            ax.set_ylim(cl[:, 1].min() - pad, cl[:, 1].max() + pad)
        else:
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)

    def _setup_text(self):
        ax = self._tax
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        ax.set_title("state", color="#ccc", fontsize=9)
        self._text_obj = ax.text(0.05, 0.97, "", transform=ax.transAxes,
                                 va="top", ha="left", fontsize=9,
                                 color="#e0e0e0", fontfamily="monospace")

    # ----------------------------------------------------------------- draw (draw thread)

    def _draw(self, state: dict, t_sim: float):
        X   = state.get("X",     0.0)
        Y   = state.get("Y",     0.0)
        psi = state.get("psi",   0.0)
        vx  = state.get("vx",    0.0)
        vy  = state.get("vy",    0.0)
        spd = state.get("speed", 0.0)
        r   = state.get("r",     0.0)
        tf        = state.get("track_frame")
        off_track = state.get("off_track", False)

        # car rectangle
        self._car_patch.set_xy(self._car_corners(X, Y, psi))

        # heading arrow
        nose = max(5.0, spd * 0.4)
        self._head_arrow.set_position((X, Y))
        self._head_arrow.xy = (X + nose * math.cos(psi), Y + nose * math.sin(psi))

        # velocity arrow (world frame)
        c, s = math.cos(psi), math.sin(psi)
        vxw, vyw = vx * c - vy * s, vx * s + vy * c
        self._vel_arrow.set_position((X, Y))
        self._vel_arrow.xy = (X + vxw * 0.5, Y + vyw * 0.5)

        # nearest track point
        if tf is not None and self._track is not None:
            nx, ny = self._track.xy[tf[3]]
            self._nearest_dot.set_data([nx], [ny])

        # off-track flash
        a = 0.55 if off_track else 0.0
        self._ot_rect.set_alpha(a)
        self._ot_text.set_alpha(a)

        # pan map if car near edge
        ax = self._ax
        xl, xr = ax.get_xlim()
        yb, yt = ax.get_ylim()
        hw, hh = (xr - xl) / 2, (yt - yb) / 2
        margin = min(hw, hh) * 0.25
        if (X < xl + margin or X > xr - margin or
                Y < yb + margin or Y > yt - margin):
            cx, cy = (xl + xr) / 2, (yb + yt) / 2
            cx += 0.4 * (X - cx)
            cy += 0.4 * (Y - cy)
            ax.set_xlim(cx - hw, cx + hw)
            ax.set_ylim(cy - hh, cy + hh)

        # text panel
        lines = [
            f"t       {t_sim:7.2f} s",
            f"speed   {spd:7.2f} m/s",
            f"vx      {vx:+7.2f}",
            f"vy      {vy:+7.2f}",
            f"r       {math.degrees(r):+7.1f} °/s",
            f"psi     {math.degrees(psi):+7.1f} °",
            f"X       {X:+7.2f} m",
            f"Y       {Y:+7.2f} m",
        ]
        if tf is not None:
            e_y, e_psi, kp, idx = tf
            lines += [
                "",
                "── track ──────────",
                f"e_y     {e_y:+7.3f} m",
                f"e_psi   {math.degrees(e_psi):+7.2f} °",
            ]
            labels = ["@0m ", "@10m", "@25m"]
            for i, lbl in enumerate(labels):
                if i < len(kp):
                    lines.append(f"κ {lbl}  {kp[i]*1000:+7.2f} ‰/m")
            lines.append(f"idx     {idx:7d}")
            if off_track:
                lines += ["", "⚠  OFF TRACK"]

        self._text_obj.set_text("\n".join(lines))
        self._fig.canvas.draw_idle()

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _car_corners(X, Y, psi):
        l, w = _CAR_LENGTH / 2, _CAR_WIDTH / 2
        local = np.array([[ l,  w], [ l, -w], [-l, -w], [-l,  w]], dtype=float)
        c, s = math.cos(psi), math.sin(psi)
        return (local @ np.array([[c, s], [-s, c]])) + np.array([X, Y])
