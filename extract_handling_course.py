"""Extract the Automation Test Track handling-course centerline from BeamNG.

`get_road_network` needs a running level, so this launches BeamNG once on the
`automation_test_track` map, queries its DecalRoads, and either:

  * DISCOVERY (default): prints/saves a summary of every drivable road so you
    can identify the handling course (it's a long serpentine on the WEST side of
    the map — look for a long road whose bbox sits at low/negative X), or
  * EXTRACT (--road-id ...): stitches the chosen road(s), resamples to the
    Track's DS grid, saves the centerline to tracks/automation_handling.npz, and
    prints the suggested spawn pose for test_sim2real_beamng.py.

Usage:
    python extract_handling_course.py                       # discover roads
    python extract_handling_course.py --road-id <id> [--road-id <id> ...]
    python extract_handling_course.py --road-id <id> --closed   # if it loops

Then run the policy on it:
    python test_sim2real_beamng.py --track automation
"""

import argparse
import json

import numpy as np

import sil_beamng as S
from automation_track import (
    _edge_middles,
    _road_edges,
    centerline_from_roads,
    resample_and_build,
    save_centerline,
    summarize_roads,
    yaw_to_quat,
)

LEVEL = "automation_test_track"
DEFAULT_OUT = "tracks/automation_handling.npz"
# The in-game handling circuit's race definition (closed loop of gate nodes).
RACE_JSON = (r"E:\SteamLibrary\steamapps\common\BeamNG.drive\gameplay\missions"
             r"\automation_test_track\timeTrial\003-Handling\race.race.json")
# A spot to drop the throwaway car so the scenario loads. get_road_network
# queries the level, not the car, so this only needs to be somewhere valid-ish;
# the car may settle oddly and that's fine for extraction.
SPAWN = (0.0, 0.0, 100.0)


def _open_level(connect):
    """Load the automation_test_track level via the existing SIL and return it."""
    S.MAP = LEVEL
    S.TRACK_MODE = None            # don't build a synthetic track we won't use
    S.SPAWN_POS = SPAWN
    S.SPAWN_ROT_QUAT = (0.0, 0.0, 0.0, 1.0)
    print(f"[extract] loading level '{LEVEL}' ...", flush=True)
    return S.BeamNGSIL().open(launch=not connect)


def list_scenarios(sil):
    """Print scenarios/missions on this level (to locate the handling course)."""
    bng = sil.bng
    try:
        scns = bng.scenario.get_level_scenarios(LEVEL)
    except Exception as e:
        print(f"[extract] get_level_scenarios failed ({type(e).__name__}: {e})")
        scns = []
    print(f"\n[extract] {len(scns)} scenario(s) on '{LEVEL}':")
    for s in scns:
        name = getattr(s, "name", "?")
        path = getattr(s, "path", "?")
        print(f"    name={name!s:<40} path={path}")
    # deep-scan the level's mission tree on disk for anything 'handling'
    import glob
    import os as _os
    for root in (S.BNG_HOME, S.BNG_USER):
        if not root:
            continue
        hits = glob.glob(_os.path.join(root, "**", "*handling*"), recursive=True)
        for h in hits[:20]:
            print(f"    [disk:{_os.path.basename(root)}] {h}")


def discover(sil, out_json):
    try:
        list_scenarios(sil)
    except Exception as e:
        print(f"[extract] scenario listing skipped ({type(e).__name__}: {e})")
    roads = summarize_roads(sil.bng, drivable_only=True)
    print(f"\n[extract] {len(roads)} drivable road(s), longest first:\n")
    for r in roads[:40]:
        print(f"  id={r['id']!s:<12} len={r['length']:>7.1f}m  turn/m={r['turn_per_m']:>5.3f}  "
              f"flips={r['sign_flips']:>3}  zdrop={r['z_drop']:>5.1f}  bbox={r['bbox']}")

    # The handling course is a WEST-side (min-X < 0) serpentine with downhill
    # corners: rank plausible-length roads by windiness * elevation drop.
    def score(r):
        west = r["bbox"][0] < 0
        good_len = 300.0 <= r["length"] <= 2500.0
        return (r["turn_per_m"] * (1.0 + r["z_drop"])) if (west and good_len) else -1.0

    ranked = sorted(roads, key=score, reverse=True)
    print("\n[extract] most handling-course-like (west, windy, downhill):\n")
    for r in ranked[:8]:
        if score(r) < 0:
            break
        print(f"  id={r['id']!s:<12} len={r['length']:>7.1f}m  turn/m={r['turn_per_m']:>5.3f}  "
              f"flips={r['sign_flips']:>3}  zdrop={r['z_drop']:>5.1f}  "
              f"first={r['first_xy']}  bbox={r['bbox']}")

    with open(out_json, "w") as f:
        json.dump(roads, f, indent=2, default=str)
    print(f"\n[extract] full summary -> {out_json}")
    print("[extract] pick the handling-course id(s) and re-run with --road-id ...")


def load_gates(path):
    """Ordered (N,2) gate xy of a closed race path, following startNode+segments."""
    d = json.load(open(path))
    nodes = {n["oldId"]: n["pos"] for n in d["pathnodes"]}
    nxt = {s["from"]: s["to"] for s in d["segments"]}
    order, cur = [], d["startNode"]
    for _ in range(len(nodes)):
        order.append(cur)
        cur = nxt.get(cur)
        if cur is None:
            break
    xy = np.array([nodes[i][:2] for i in order], dtype=float)
    return xy, order


def identify(sil, race_json, thresh=14.0):
    """Find which DecalRoads carry the race circuit (roads passing near its gates)."""
    gates_xy, order = load_gates(race_json)
    print(f"[extract] {len(gates_xy)} gate(s) from {race_json}")
    net = sil.bng.scenario.get_road_network(include_edges=True, drivable_only=False)
    rows = []
    for rid, rd in net.items():
        try:
            mid = _edge_middles(_road_edges(sil.bng, rid, rd))
        except Exception:
            continue
        if len(mid) < 2:
            continue
        xy = mid[:, :2]
        # nearest road point to each gate
        dmin = [float(np.min(np.hypot(xy[:, 0] - g[0], xy[:, 1] - g[1]))) for g in gates_xy]
        cover = int(sum(d < thresh for d in dmin))
        if cover:
            length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
            rows.append((cover, rid, length, float(mid[:, 2].mean()),
                         [round(d, 1) for d in dmin]))
    rows.sort(reverse=True)
    print(f"\n[extract] roads covering >=1 gate (within {thresh:.0f} m):\n")
    for cover, rid, length, meanz, dmin in rows:
        print(f"  id={rid!s:<12} gates={cover:>2}/{len(gates_xy)}  len={length:>7.1f}m  "
              f"z~{meanz:>6.1f}  gate_dists={dmin}")
    print("\n[extract] extract them with:  --road-id " +
          " --road-id ".join(str(r[1]) for r in rows[:6]) + " --closed")


def dump_candidate_roads(sil, race_json, out, thresh=14.0):
    """Save raw geometry of every road passing near a gate, for offline rebuild."""
    gates_xy, order = load_gates(race_json)
    net = sil.bng.scenario.get_road_network(include_edges=True, drivable_only=False)
    saved = {"gates_xy": gates_xy}
    kept = 0
    for rid, rd in net.items():
        try:
            mid = _edge_middles(_road_edges(sil.bng, rid, rd))
        except Exception:
            continue
        if len(mid) < 2:
            continue
        xy = mid[:, :2]
        dmin = [float(np.min(np.hypot(xy[:, 0] - g[0], xy[:, 1] - g[1]))) for g in gates_xy]
        if sum(d < thresh for d in dmin) >= 1:
            saved[f"road_{str(rid).replace('.', '_')}"] = mid  # (N,3)
            kept += 1
    np.savez(out, **saved)
    print(f"[extract] dumped {kept} candidate road(s) + gates -> {out}")


def extract(sil, road_ids, out, closed):
    xy, z = centerline_from_roads(sil.bng, road_ids)
    print(f"[extract] stitched {len(xy)} raw points from road(s) {road_ids}")

    # sanity-build a Track so we fail here (not at run time) if geometry is bad
    track = resample_and_build(xy, closed=closed)
    print(f"[extract] resampled Track: {track.n} samples, "
          f"~{track.length:.0f} m, closed={track.closed}")

    save_centerline(out, xy, z)
    x0, y0 = float(track.xy[0, 0]), float(track.xy[0, 1])
    z0 = float(z[0])
    psi0 = float(track.psi[0])
    print(f"\n[extract] saved centerline -> {out}")
    print(f"[extract] suggested spawn:  pos=({x0:.2f}, {y0:.2f}, {z0 + 0.3:.2f})  "
          f"yaw={np.degrees(psi0):+.1f} deg  quat={tuple(round(q, 4) for q in yaw_to_quat(psi0))}")
    print("[extract] run it:  python test_sim2real_beamng.py --track automation"
          + (" --closed" if closed else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--road-id", dest="road_ids", action="append", default=None,
                   help="DecalRoad id to extract (repeat to stitch several). "
                        "Omit for discovery mode.")
    p.add_argument("--identify", nargs="?", const=RACE_JSON, default=None,
                   help="find the DecalRoads carrying a race circuit (default: the "
                        "handling-circuit race file)")
    p.add_argument("--dump", nargs="?", const=RACE_JSON, default=None,
                   help="dump raw geometry of all gate-adjacent roads for offline "
                        "rebuild (default: the handling-circuit race file)")
    p.add_argument("--dump-out", default="tracks/circuit_roads.npz",
                   help="output path for --dump")
    p.add_argument("--out", default=DEFAULT_OUT, help="centerline .npz output path")
    p.add_argument("--closed", action="store_true", help="treat the course as a loop")
    p.add_argument("--connect", action="store_true",
                   help="attach to an already-running BeamNG instead of launching")
    p.add_argument("--roads-json", default="tracks/roads_summary.json",
                   help="where to write the discovery summary")
    args = p.parse_args()

    sil = _open_level(args.connect)
    try:
        if args.road_ids:
            extract(sil, args.road_ids, args.out, args.closed)
        elif args.dump:
            dump_candidate_roads(sil, args.dump, args.dump_out)
        elif args.identify:
            identify(sil, args.identify)
        else:
            discover(sil, args.roads_json)
    finally:
        sil.close()


if __name__ == "__main__":
    main()
