"""Command-line interface. `ask` (the agent) arrives in Milestone 3."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .aoi import get_aoi, list_aois
from .config import get_settings
from .pipeline import first_clear_observation, write_outputs
from .tools.compare import compare_counts
from .tools.imagery import search_scenes


def _cmd_aois(_: argparse.Namespace) -> int:
    for a in list_aois():
        print(f"{a.id:18} {a.name}  bbox={a.bbox.as_tuple()}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    aoi = get_aoi(args.aoi)
    scenes = search_scenes(aoi.bbox, args.start, args.end, max_cloud=args.max_cloud)
    for s in scenes:
        print(f"{s.datetime:%Y-%m-%d %H:%M}  cloud={s.cloud_cover:5.1f}%  {s.id}")
    print(f"{len(scenes)} scene(s)", file=sys.stderr)
    return 0


def _observe(args: argparse.Namespace, start: str, end: str):
    aoi = get_aoi(args.aoi)
    obs, rejected = first_clear_observation(aoi, start, end, max_scene_cloud=args.max_cloud,
                                            max_aoi_cloud=args.max_aoi_cloud)
    for r in rejected:
        print(f"  rejected {r.scene.id}: {r.reason}", file=sys.stderr)
    if obs is None:
        print(f"No usable scene for {aoi.id} between {start} and {end}.", file=sys.stderr)
    return obs


def _cmd_detect(args: argparse.Namespace) -> int:
    obs = _observe(args, args.start, args.end)
    if obs is None:
        return 2
    paths = write_outputs(obs, args.out or get_settings().output_dir)
    r = obs.result
    print(f"{r.aoi_id} {r.datetime:%Y-%m-%d}: {r.count} large vessel(s) "
          f"[AOI cloud {r.aoi_cloud_fraction:.0%}, scene {r.scene_id}]")
    for note in r.notes:
        print(f"  note: {note}")
    print(f"  wrote {paths['png']} and {paths['json']}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    before = _observe(args, args.before_start, args.before_end)
    after = _observe(args, args.after_start, args.after_end)
    if before is None or after is None:
        return 2
    out = args.out or get_settings().output_dir
    write_outputs(before, out)
    write_outputs(after, out)
    cmp = compare_counts(before.result, after.result)
    print(json.dumps(cmp.model_dump(mode="json"), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ask-a-satellite", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("aois", help="list known areas of interest").set_defaults(func=_cmd_aois)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--aoi", required=True, help="AOI id, e.g. port-everglades")
        sp.add_argument("--max-cloud", type=float, default=30, help="max scene cloud cover %%")

    s = sub.add_parser("search", help="list candidate Sentinel-2 scenes")
    common(s)
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.set_defaults(func=_cmd_search)

    def observe(sp: argparse.ArgumentParser) -> None:
        common(sp)
        sp.add_argument("--max-aoi-cloud", type=float, default=0.1, help="max AOI cloud fraction 0-1")
        sp.add_argument("--out", help="output directory (default: $ASK_SAT_OUTPUT_DIR or outputs/)")

    d = sub.add_parser("detect", help="count large vessels in the newest clear scene in a date range")
    observe(d)
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.set_defaults(func=_cmd_detect)

    c = sub.add_parser("compare", help="compare vessel counts between two date ranges")
    observe(c)
    for k in ("before-start", "before-end", "after-start", "after-end"):
        c.add_argument(f"--{k}", required=True)
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
