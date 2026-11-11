"""Offline training measurements and reports."""

import argparse
import json
import sys
from pathlib import Path


def output(**fields):
    # Quoted JSON strings are also TOON scalars; no encoder dependency is needed.
    for key, value in fields.items():
        print(f"{key}: {json.dumps(value, ensure_ascii=False)}")


class Parser(argparse.ArgumentParser):
    def error(self, message):
        output(error=message, help=f"Run {self.prog} --help for valid arguments")
        raise SystemExit(2)


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--version", "-v", "-V", action="version", version="0.1.0")
    commands = parser.add_subparsers(dest="action", parser_class=Parser)
    report = commands.add_parser(
        "report",
        help="Render exactly nine measurements",
        epilog=(
            "Example: training-harness report --selection config/results.json --readme README.md"
        ),
    )
    report.add_argument("logs", nargs="*", type=Path, help="JSONL files (omit for empty results)")
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--prices", type=Path, default=Path("config/prices.json"))
    report.add_argument("--readme", type=Path, help="Replace only the marked README table")
    args = parser.parse_args(argv)
    try:
        from .report import METRICS, load_runs, render_table, result_values, update_readme

        if args.action is None:
            output(bin=str(Path(sys.argv[0]).resolve()), description=__doc__)
            values = result_values({}, {}, {})
        else:
            selection = json.loads(args.selection.read_text())
            prices = json.loads(args.prices.read_text())
            values = result_values(load_runs(args.logs), selection, prices)
            if args.readme:
                update_readme(args.readme, render_table(values))
                output(status="updated", readme=str(args.readme.resolve()))
                return 0
        print("measurements[9]{metric,result}:")
        for metric, value in zip(METRICS, values, strict=True):
            print(f"  {json.dumps(metric, ensure_ascii=False)},{json.dumps(value)}")
        output(help="training-harness report --help")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        output(error=str(error), help=f"Run {parser.prog} {args.action or ''} --help")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
