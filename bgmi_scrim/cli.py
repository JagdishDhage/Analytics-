"""Command-line interface.

    analyze   <video> --profile p.json --out events.jsonl
    calibrate <video> [--interactive | --template base.json | --preview p.json]
    truth     --labels labels.jsonl --events events.jsonl [--md report.md]
    synth     --out clip.mp4 --labels labels.jsonl        (test footage)
    report    --events e1.jsonl e2.jsonl ... --team TX --out report.md   (phase 2)
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
    md = build_report(args.events, team=args.team, block_name=args.block,
                      asked_for=args.asked_for or [])
    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"wrote {args.out}")
    else:
        print(md)
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

    r = sub.add_parser("report", help="phase 2: team preparation report from event logs")
    r.add_argument("--events", nargs="+", required=True)
    r.add_argument("--team", required=True)
    r.add_argument("--block", default="scrim block")
    r.add_argument("--out", default=None)
    r.add_argument("--asked-for", nargs="*", default=None,
                   help="metrics the coach asked for, to be answered or declared unmeasurable")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
