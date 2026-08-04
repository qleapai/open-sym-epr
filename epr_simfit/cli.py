"""Command-line entry points."""

from __future__ import annotations

import argparse

from .demo_data import generate_examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pbn-epr")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="generate synthetic demo ASC files")
    demo.add_argument("--out", default="examples/generated_demo")
    args = parser.parse_args(argv)
    if args.command == "demo":
        paths = generate_examples(args.out)
        print(f"Generated {len(paths)} demo spectra in {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
