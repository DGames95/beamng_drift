"""Build the handling-course centerline from a human-driven lap (record_lap.py).

A driven line is already ordered and on-road, so this is simple and robust: trim
the standstill, slice one clean lap between start-line crossings, resample to the
Track's 0.5 m grid, lightly smooth, and save tracks/automation_handling.npz.

Usage:
    python build_lap.py                        # from tracks/handling_lap_raw.npz
    python build_lap.py --raw path.npz --no-close   # open track instead of loop
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automation_track import resample_and_build, save_centerline


def smooth_wrap(a, w, closed):
    w = int(w) | 1
    pad = w // 2
    if closed:
        ap = np.vstack([a[-pad:], a, a[:pad]])
    else:
        ap = np.vstack([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)])
    k = np.ones(w) / w
    return np.stack([np.convolve(ap[:, 0], k, "valid"), np.convolve(ap[:, 1], k, "valid")], 1)


def slice_one_lap(xy, close):
    """Trim leading standstill and (for a loop) slice one lap start->start."""
    d = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    start_move = int(np.searchsorted(cum, 2.0))          # skip first 2 m of jitter
    xy = xy[start_move:]
    if not close:
        return xy, start_move
    start = xy[0]
    dstart = np.hypot(xy[:, 0] - start[0], xy[:, 1] - start[1])
    armed = dstart > 40.0
    # first return to the start zone after leaving it
    for i in range(1, len(xy)):
        if armed[i - 1] and dstart[i] < 8.0:
            return xy[:i], start_move
    return xy, start_move


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="tracks/handling_lap_raw.npz")
    p.add_argument("--out", default="tracks/automation_handling.npz")
    p.add_argument("--no-close", dest="close", action="store_false", default=True,
                   help="build an open track instead of a closed loop")
    p.add_argument("--smooth-m", type=float, default=2.5,
                   help="position smoothing window [m] over the driven line")
    args = p.parse_args()

    d = np.load(args.raw)
    xyz = d["xyz"]
    xy_all = xyz[:, :2]
    print(f"[build] {len(xy_all)} raw samples, {np.hypot(*np.diff(xy_all,axis=0).T).sum():.0f} m driven")

    xy, off = slice_one_lap(xy_all, args.close)
    z = xyz[off:off + len(xy), 2]
    lap_len = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    print(f"[build] using lap slice: {len(xy)} samples, {lap_len:.0f} m, closed={args.close}")

    # resample to ~0.5 m first (raw is time-sampled -> uneven in space), then smooth
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    g = np.arange(0, s[-1], 0.5)
    xr = np.stack([np.interp(g, s, xy[:, 0]), np.interp(g, s, xy[:, 1])], 1)
    zr = np.interp(g, s, z)
    xr = smooth_wrap(xr, max(3, int(args.smooth_m / 0.5)), args.close)

    track = resample_and_build(xr, closed=args.close)
    kmax = np.max(np.abs(track.kappa))
    print(f"[build] Track: {track.n} samples, {track.length:.0f} m, "
          f"|kappa| med={np.median(np.abs(track.kappa)):.4f} "
          f"p95={np.percentile(np.abs(track.kappa),95):.4f} max={kmax:.4f} "
          f"(min R~{1/max(kmax,1e-9):.0f} m)")

    save_centerline(args.out, track.xy, zr[:track.n] if len(zr) >= track.n else
                    np.interp(np.linspace(0, 1, track.n), np.linspace(0, 1, len(zr)), zr),
                    closed=args.close)
    print(f"[build] saved -> {args.out}")

    plt.figure(figsize=(11, 9))
    plt.plot(xy_all[:, 0], xy_all[:, 1], color="0.8", lw=0.6, label="raw driven")
    plt.plot(track.xy[:, 0], track.xy[:, 1], "r-", lw=2, label="centerline")
    plt.plot(track.xy[0, 0], track.xy[0, 1], "gs", ms=10, label="start")
    plt.axis("equal"); plt.legend(); plt.title("handling course from driven lap")
    plt.savefig("tracks/_lap_centerline.png", dpi=90, bbox_inches="tight")
    print("[build] saved tracks/_lap_centerline.png")


if __name__ == "__main__":
    main()
