"""Extract a world-fixed centerline from BeamNG road geometry and turn it into a
`track.Track` the SIL/RL pipeline can consume.

The synthetic tracks (`Track.circle` / `Track.random_track`) are generated in
code and anchored to the car. Real BeamNG roads (e.g. the Automation Test Track
handling course) are fixed in the world, so instead we pull their geometry from
`bng.scenario.get_road_network(include_edges=True)` — which returns each
DecalRoad as a series of (left, middle, right) point triplets — take the
`middle` points as the centerline, resample them onto the same uniform DS grid
`track.py` uses, and compute heading/curvature.

Two-step usage:
    extract (live, once):  summarize_roads() -> pick road id(s) ->
                           centerline_from_roads() -> save_centerline()
    run (offline load):    load_centerline() -> resample_and_build() -> Track

Everything downstream (Track.obs / frame, OBS_SCALE, the policy contract) is
untouched — only the *source* of the centerline changes.
"""

import math
import os

import numpy as np

from track import DS, Track


# --------------------------------------------------------------------------- #
# geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def yaw_to_quat(yaw: float):
    """(x, y, z, w) quaternion for a rotation of `yaw` rad about world +Z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _movavg(a: np.ndarray, w: int, closed: bool) -> np.ndarray:
    """Length-preserving moving average; wraps for closed tracks, edge-pads open."""
    w = int(w) | 1  # force odd so the window is symmetric
    if w <= 1:
        return a
    k = np.ones(w) / w
    pad = w // 2
    if closed:
        ap = np.concatenate([a[-pad:], a, a[:pad]])
    else:
        ap = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    return np.convolve(ap, k, mode="valid")


def resample_and_build(xy, closed: bool = False, ds: float = DS, smooth: int = 5) -> Track:
    """Ordered polyline (arbitrary spacing) -> uniformly-sampled `Track`.

    The repo has no resampling utility; this is it. Resamples x/y onto a uniform
    `ds` grid (== track.DS so Track's index-space lookahead `j = i + int(d/DS)`
    stays valid), derives heading via finite differences, and curvature as the
    wrapped heading rate d(psi)/ds, lightly smoothed to kill the jitter that
    comes from discrete road points.
    """
    xy = np.asarray(xy, dtype=float)
    if closed and not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])

    # cumulative arc-length; drop zero-length segments so interp stays monotone
    seg = np.diff(xy, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    keep = np.concatenate([[True], seglen > 1e-9])
    xy = xy[keep]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    total = float(s[-1])
    if total < ds:
        raise ValueError(f"centerline too short ({total:.2f} m)")

    n = max(4, int(round(total / ds)))
    if closed:
        ds_eff = total / n
        s_new = np.arange(n) * ds_eff        # endpoint excluded (loop)
    else:
        ds_eff = ds
        s_new = np.arange(n) * ds
        s_new = s_new[s_new <= total]        # stay within the polyline
        n = len(s_new)

    pts = np.stack([np.interp(s_new, s, xy[:, 0]),
                    np.interp(s_new, s, xy[:, 1])], axis=1)

    # heading: central difference (forward at open-track ends)
    fwd = np.empty_like(pts)
    if closed:
        fwd = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    else:
        fwd[1:-1] = pts[2:] - pts[:-2]
        fwd[0] = pts[1] - pts[0]
        fwd[-1] = pts[-1] - pts[-2]
    psi = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))

    # curvature = wrapped heading rate
    dpsi = np.diff(psi)
    if closed:
        wrap_last = math.atan2(math.sin(psi[0] - psi[-1]), math.cos(psi[0] - psi[-1]))
        dpsi = np.append(dpsi, wrap_last)
    else:
        dpsi = np.append(dpsi, dpsi[-1])
    kappa = _movavg(dpsi / ds_eff, smooth, closed)

    return Track(pts, psi, kappa, closed)


# --------------------------------------------------------------------------- #
# road-network extraction                                                      #
# --------------------------------------------------------------------------- #
def _edge_middles(edges) -> np.ndarray:
    """(N,3) 'middle' points from a road's edge list (dict or (l,m,r) tuples)."""
    pts = []
    for e in edges:
        m = e["middle"] if isinstance(e, dict) else e[1]
        pts.append((float(m[0]), float(m[1]), float(m[2])))
    return np.asarray(pts, dtype=float)


def _road_edges(bng, road_id, road_data=None):
    """Edge list for a road, from the network payload or a direct query."""
    if isinstance(road_data, dict) and road_data.get("edges"):
        return road_data["edges"]
    return bng.scenario.get_road_edges(road_id)


def summarize_roads(bng, drivable_only: bool = True):
    """List every road with enough metadata to identify the handling course.

    Returns a list of dicts: id, n_points, length, first_xy, last_xy, bbox,
    drivability — sorted longest-first (the serpentine handling road is long).
    """
    net = bng.scenario.get_road_network(include_edges=True, drivable_only=drivable_only)
    out = []
    for rid, rd in net.items():
        try:
            mid = _edge_middles(_road_edges(bng, rid, rd))
        except Exception:
            continue
        if len(mid) < 3:
            continue
        xy = mid[:, :2]
        z = mid[:, 2]
        length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
        # windiness + corner-alternation: how much the heading turns per metre,
        # and how often the turn direction flips (the handling serpentine is high
        # on both); plus elevation drop (its corners are downhill).
        seg = np.diff(xy, axis=0)
        hd = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        dhd = np.diff(hd)
        total_turn = float(np.sum(np.abs(dhd)))
        sign_flips = int(np.sum(np.diff(np.sign(dhd)) != 0)) if len(dhd) > 1 else 0
        out.append({
            "id": rid,
            "n_points": int(len(mid)),
            "length": round(length, 1),
            "turn_per_m": round(total_turn / max(length, 1.0), 4),
            "sign_flips": sign_flips,
            "z_drop": round(float(z.max() - z.min()), 1),
            "first_xy": [round(float(xy[0, 0]), 1), round(float(xy[0, 1]), 1)],
            "last_xy": [round(float(xy[-1, 0]), 1), round(float(xy[-1, 1]), 1)],
            "bbox": [round(float(xy[:, 0].min()), 1), round(float(xy[:, 1].min()), 1),
                     round(float(xy[:, 0].max()), 1), round(float(xy[:, 1].max()), 1)],
            "drivability": (rd.get("drivability") if isinstance(rd, dict) else None),
        })
    out.sort(key=lambda r: r["length"], reverse=True)
    return out


def _stitch(segs):
    """Concatenate road segments end-to-end, reversing any that run backwards."""
    segs = [np.asarray(s, float) for s in segs if len(s) > 1]
    if not segs:
        raise ValueError("no usable road segments")
    result = segs[0].copy()
    remaining = segs[1:]
    while remaining:
        end = result[-1, :2]
        best = None  # (dist, index, reverse)
        for i, s in enumerate(remaining):
            d_start = float(np.hypot(*(s[0, :2] - end)))
            d_end = float(np.hypot(*(s[-1, :2] - end)))
            for d, rev in ((d_start, False), (d_end, True)):
                if best is None or d < best[0]:
                    best = (d, i, rev)
        _, i, rev = best
        s = remaining.pop(i)
        result = np.vstack([result, s[::-1] if rev else s])
    return result


def centerline_from_roads(bng, road_ids):
    """Stitched centerline for one or more DecalRoad ids -> (xy (N,2), z (N,))."""
    net = bng.scenario.get_road_network(include_edges=True, drivable_only=False)
    segs = [_edge_middles(_road_edges(bng, rid, net.get(rid))) for rid in road_ids]
    stitched = _stitch(segs)
    return stitched[:, :2], stitched[:, 2]


# --------------------------------------------------------------------------- #
# persistence                                                                 #
# --------------------------------------------------------------------------- #
def save_centerline(path, xy, z, closed=False):
    """Save centerline (world x/y, z, and whether it's a loop) to an .npz."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(path, xy=np.asarray(xy, dtype=float), z=np.asarray(z, dtype=float),
             closed=np.asarray(bool(closed)))


def load_centerline(path):
    """Load a centerline -> (xy (N,2), z (N,), closed bool)."""
    d = np.load(path)
    closed = bool(d["closed"]) if "closed" in d.files else False
    return d["xy"], d["z"], closed
