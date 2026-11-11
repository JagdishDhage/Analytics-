"""Command-line interface.

    analyze   <video> --profile p.json --out events.jsonl
    calibrate <video> [--interactive | --template base.json | --preview p.json]
    truth     --labels labels.jsonl --events events.jsonl [--md report.md]
    synth     --out clip.mp4 --labels labels.jsonl        (test footage)
    report    --events e1.jsonl e2.jsonl ... --team TX --out report.md   (phase 2)

mapanalyze <map video> --profile mp.json --tracks t.jsonl --map-events me.jsonl
mapcalibrate <map video> [--template base.json | --preview mp.json | --pois maps/x.json]
maptruth   --labels l.jsonl --tracks t.jsonl --profile mp.json
mapsynth   --out clip.mp4 --labels l.jsonl                   (test footage)
mapreport  --tracks t1.jsonl ... --map-events me1.jsonl ... --team TX   (phase 3)

dashboard  --events e1.jsonl ... [--tracks t1.jsonl ...] --team TX --out dash.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_video
    from .profile import Profile
    prof = Profile.load(args.profile)
    events, stats = analyze_video(args.video, prof, out_path=args.out,
                                  our_team=args.team, workers=args.workers,
                                  verbose=not args.quiet)
    if not args.quiet:
        print(json.dumps(stats.as_dict(), indent=2))
        print(f"wrote {args.out} ({len(events)} events)")
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats.as_dict(), indent=2) + "\n")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from . import calibrate as cal
    if args.preview:
        out = cal.preview(args.video, args.preview, args.preview_png, at=args.at)
        print(json.dumps(out, indent=2, default=str))
        return 0
    if args.template:
        cal.template(args.video, args.out, args.template, our_team=args.team)
        return 0
    cal.interactive(args.video, args.out, base=args.base, at=args.at, our_team=args.team)
    return 0


def cmd_truth(args: argparse.Namespace) -> int:
    from .truth import render_report, report_json, score
    rep = score(args.labels, args.events, tolerance=args.tolerance, grace=args.grace,
                confidence_threshold=args.confidence_threshold)
    md = render_report(rep, title=args.title)
    if args.md:
        Path(args.md).write_text(md + "\n")
        print(f"wrote {args.md}")
    if args.json:
        Path(args.json).write_text(report_json(rep) + "\n")
        print(f"wrote {args.json}")
    if not args.md and not args.json:
        print(md)
    g = rep.gate()
    print(f"GATE: {'PASSED' if g['passed'] else 'NOT MET'} "
          f"(precision {g['kill_knock_precision']:.4f}, recall {g['kill_knock_recall']:.4f}, "
          f"duplicates {g['duplicates']}, phantoms {g['phantoms']}, "
          f"teams_alive {g['teams_alive_accuracy']})")
    return 0 if g["passed"] else 2


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import write_clip
    info = write_clip(args.out, args.labels, width=args.width, height=args.height,
                      fps=args.fps, duration=args.duration, seed=args.seed,
                      noise=args.noise, our_team=args.team, map_name=args.map,
                      n_teams=args.teams)
    print(json.dumps(info, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report
    table = None
    if args.points_table:
        table = json.loads(Path(args.points_table).read_text())
    md = build_report(args.events, team=args.team, block_name=args.block,
                      placement_points=table,
                      confidence_threshold=args.confidence_threshold,
                      asked_for=args.asked_for or [])
    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


def cmd_mapanalyze(args: argparse.Namespace) -> int:
    from .map_analyze import analyze_map_video
    from .map_profile import MapProfile
    prof = MapProfile.load(args.profile)
    recs, evs, stats = analyze_map_video(args.video, prof, tracks_out=args.tracks,
                                         events_out=args.map_events, our_team=args.team,
                                         workers=args.workers, verbose=not args.quiet)
    if not args.quiet:
        print(json.dumps(stats.as_dict(), indent=2))
        print(f"wrote {args.tracks} ({len(recs)} records) and {args.map_events} "
              f"({len(evs)} events)")
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats.as_dict(), indent=2) + "\n")
    return 0


def cmd_maptruth(args: argparse.Namespace) -> int:
    from .map_profile import MapProfile
    from .map_truth import render_map_report, score
    prof = MapProfile.load(args.profile)
    rep = score(args.labels, args.tracks, prof, tolerance=args.tolerance,
                identity_tol_m=args.identity_tolerance_m)
    md = render_map_report(rep, title=args.title)
    if args.md:
        Path(args.md).write_text(md)
        print(f"wrote {args.md}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep.as_dict(), indent=2) + "\n")
        print(f"wrote {args.json}")
    if not args.md and not args.json:
        print(md)
    return 0


def cmd_mapsynth(args: argparse.Namespace) -> int:
    from .map_synth import write_map_clip
    info = write_map_clip(args.out, args.labels, width=args.width, height=args.height,
                          fps=args.fps, duration=args.duration, seed=args.seed,
                          noise=args.noise, our_team=args.team, map_name=args.map,
                          n_teams=args.teams, sample_fps=args.sample_fps,
                          map_area=args.map_area)
    print(json.dumps(info, indent=2))
    return 0


def cmd_mapcalibrate(args: argparse.Namespace) -> int:
    from . import map_calibrate as mc
    if args.preview:
        print(json.dumps(mc.preview(args.video, args.preview, args.preview_png, at=args.at),
                         indent=2, default=str))
        return 0
    if args.pois:
        mc.pick_pois(args.video, args.profile, args.pois, at=args.at)
        return 0
    if args.template:
        mc.template(args.video, args.out, args.template, map_name=args.map)
        return 0
    mc.interactive(args.video, args.out, base=args.base, at=args.at, map_name=args.map)
    return 0


def cmd_mapreport(args: argparse.Namespace) -> int:
    from .map_report import build_map_report
    md = build_map_report(args.tracks, args.map_events, team=args.team,
                          hud_events_paths=args.hud_events,
                          block_name=args.block,
                          confidence_threshold=args.confidence_threshold,
                          time_offsets=args.time_offsets)
    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import build_model, write_dashboard
    table = json.loads(Path(args.points_table).read_text()) if args.points_table else None
    model = build_model(args.events, team=args.team, tracks_paths=args.tracks,
                        map_events_paths=args.map_events, block_name=args.block,
                        placement_points=table,
                        confidence_threshold=args.confidence_threshold)
    out = write_dashboard(args.out, model)
    size = Path(out).stat().st_size
    print(f"wrote {out} ({size/1024:.0f} KB, {len(model['matches'])} matches)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bgmi-scrim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="extract events from a spectator VOD")
    a.add_argument("video")
    a.add_argument("--profile", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--team", default=None, help="our team tag (defaults to the profile's)")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--stats", default=None, help="write run stats JSON here")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="write or verify a profile for a video")
    c.add_argument("video")
    c.add_argument("--out", default="profile.json")
    c.add_argument("--base", default=None, help="start the picker from this profile")
    c.add_argument("--template", default=None, help="scale this profile to the video instead")
    c.add_argument("--preview", default=None, help="draw this profile's boxes on a frame")
    c.add_argument("--preview-png", default="profile_preview.png")
    c.add_argument("--at", type=float, default=60.0, help="frame time to use, seconds")
    c.add_argument("--team", default=None)
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("truth", help="score events against hand labels")
    t.add_argument("--labels", required=True)
    t.add_argument("--events", required=True)
    t.add_argument("--tolerance", type=float, default=2.5, help="event match window, seconds")
    t.add_argument("--grace", type=float, default=0.75,
                   help="frame-field transition tolerance, seconds")
    t.add_argument("--confidence-threshold", type=float, default=0.6)
    t.add_argument("--md", default=None)
    t.add_argument("--json", default=None)
    t.add_argument("--title", default="Phase 1 accuracy report")
    t.set_defaults(func=cmd_truth)

    s = sub.add_parser("synth", help="render synthetic HUD footage with labels (testing)")
    s.add_argument("--out", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--width", type=int, default=1920)
    s.add_argument("--height", type=int, default=1080)
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--duration", type=float, default=150.0)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--noise", type=float, default=0.0)
    s.add_argument("--team", default="TX")
    s.add_argument("--map", default="ERANGEL")
    s.add_argument("--teams", type=int, default=25)
    s.set_defaults(func=cmd_synth)

    ma = sub.add_parser("mapanalyze", help="phase 3: extract positions from a map stream")
    ma.add_argument("video")
    ma.add_argument("--profile", required=True)
    ma.add_argument("--tracks", required=True, help="per-sample positions (JSONL)")
    ma.add_argument("--map-events", required=True, help="derived map events (JSONL)")
    ma.add_argument("--team", default=None)
    ma.add_argument("--workers", type=int, default=4)
    ma.add_argument("--stats", default=None)
    ma.add_argument("--quiet", action="store_true")
    ma.set_defaults(func=cmd_mapanalyze)

    mt = sub.add_parser("maptruth", help="score map tracks against labelled positions")
    mt.add_argument("--labels", required=True)
    mt.add_argument("--tracks", required=True)
    mt.add_argument("--profile", required=True)
    mt.add_argument("--tolerance", type=float, default=0.35)
    mt.add_argument("--identity-tolerance-m", type=float, default=80.0)
    mt.add_argument("--md", default=None)
    mt.add_argument("--json", default=None)
    mt.add_argument("--title", default="Phase 3 map-stream accuracy")
    mt.set_defaults(func=cmd_maptruth)

    ms = sub.add_parser("mapsynth", help="render synthetic map footage with labels (testing)")
    ms.add_argument("--out", required=True)
    ms.add_argument("--labels", required=True)
    ms.add_argument("--width", type=int, default=1920)
    ms.add_argument("--height", type=int, default=1080)
    ms.add_argument("--fps", type=float, default=30.0)
    ms.add_argument("--sample-fps", type=float, default=None,
                    help="render only the frames an analyzer samples (smaller file)")
    ms.add_argument("--duration", type=float, default=420.0)
    ms.add_argument("--seed", type=int, default=5)
    ms.add_argument("--noise", type=float, default=0.0)
    ms.add_argument("--team", default="TX")
    ms.add_argument("--map", default="ERANGEL")
    ms.add_argument("--teams", type=int, default=25)
    ms.add_argument("--map-area", type=int, nargs=4, default=None)
    ms.set_defaults(func=cmd_mapsynth)

    mc = sub.add_parser("mapcalibrate", help="write or verify a map-stream profile")
    mc.add_argument("video")
    mc.add_argument("--out", default="map_profile.json")
    mc.add_argument("--base", default=None)
    mc.add_argument("--template", default=None)
    mc.add_argument("--preview", default=None)
    mc.add_argument("--preview-png", default="map_preview.png")
    mc.add_argument("--profile", default=None, help="profile to use with --pois")
    mc.add_argument("--pois", default=None, help="write/extend this POI table by clicking")
    mc.add_argument("--map", default=None, help="map name (sets the world size)")
    mc.add_argument("--at", type=float, default=90.0)
    mc.set_defaults(func=cmd_mapcalibrate)

    db = sub.add_parser("dashboard", help="one self-contained HTML dashboard from the logs")
    db.add_argument("--events", nargs="+", required=True,
                    help="phase-1 events.jsonl, one per match, in match order")
    db.add_argument("--tracks", nargs="*", default=None,
                    help="phase-3 tracks.jsonl per match, same order")
    db.add_argument("--map-events", nargs="*", default=None,
                    help="phase-3 map events per match, same order")
    db.add_argument("--team", required=True)
    db.add_argument("--block", default="scrim block")
    db.add_argument("--points-table", default=None)
    db.add_argument("--confidence-threshold", type=float, default=0.6)
    db.add_argument("--out", default="dashboard.html")
    db.set_defaults(func=cmd_dashboard)

    mr = sub.add_parser("mapreport", help="phase 3: map report (drops, rotations, zones)")
    mr.add_argument("--tracks", nargs="+", required=True)
    mr.add_argument("--map-events", nargs="+", required=True)
    mr.add_argument("--team", required=True)
    mr.add_argument("--hud-events", nargs="*", default=None,
                    help="phase-1 events.jsonl per match, to place fights on the map")
    mr.add_argument("--time-offsets", nargs="*", type=float, default=None,
                    help="seconds to add to HUD timestamps per match if the two "
                         "recordings do not start together")
    mr.add_argument("--block", default="scrim block")
    mr.add_argument("--confidence-threshold", type=float, default=0.6)
    mr.add_argument("--out", default=None)
    mr.set_defaults(func=cmd_mapreport)

    r = sub.add_parser("report", help="phase 2: team preparation report from event logs")
    r.add_argument("--events", nargs="+", required=True)
    r.add_argument("--team", required=True)
    r.add_argument("--block", default="scrim block")
    r.add_argument("--out", default=None)
    r.add_argument("--points-table", default=None,
                   help="JSON file mapping placement -> points (organiser rule)")
    r.add_argument("--confidence-threshold", type=float, default=0.6)
    r.add_argument("--asked-for", nargs="*", default=None,
                   help="metrics the coach asked for, to be answered or declared unmeasurable")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
