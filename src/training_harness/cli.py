"""Offline reports and an opt-in launcher sweep. Never provisions resources."""

import argparse
import json
import os
import shlex
import subprocess
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


def sweep(args):
    from .recorder import BACKENDS
    from .report import MISSING, load_runs, scaling, throughput

    if args.backend not in BACKENDS:
        raise ValueError("backend must be ddp, ray, or jax")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("sweep needs a launcher command after --")
    output_dir = args.output.resolve()
    if args.execute and output_dir.exists():
        raise ValueError("sweep output already exists; use a new output directory")
    plans = []
    for size in (1, 2, 4, 8):
        run_id = f"{args.run_prefix}-{size}"
        log = output_dir / f"{size}.jsonl"
        replacements = {"world_size": str(size), "run_id": run_id, "log": str(log)}
        argv = [part.format_map(replacements) for part in command]
        plans.append((size, run_id, log, argv))
    print("plans[4]{world_size,run_id,command}:")
    for size, run_id, _log, argv in plans:
        print(f"  {size},{json.dumps(run_id)},{json.dumps(shlex.join(argv))}")
    if not args.execute:
        output(
            status="dry-run; no commands executed", help="Add --execute to run on your allocation"
        )
        return
    output_dir.mkdir(parents=True)
    mapping = {}
    for size, run_id, log, argv in plans:
        env = os.environ | {
            "HARNESS_RUN_ID": run_id,
            "HARNESS_LOG": str(log),
            "HARNESS_WORLD_SIZE": str(size),
            "HARNESS_BACKEND": args.backend,
            "HARNESS_GPU_TYPE": args.gpu_type,
            "HARNESS_WORKLOAD": args.workload,
        }
        result = subprocess.run(argv, env=env, check=False, stdout=sys.stderr)
        if result.returncode:
            raise ValueError(f"launcher exited {result.returncode} at {size} GPUs; logs kept")
        if not log.exists():
            raise ValueError(f"launcher did not write {log}; integrate Run hooks first")
        mapping[size] = run_id
    runs = load_runs([plan[2] for plan in plans])
    by_size = {size: runs[run_id] for size, run_id in mapping.items()}
    for events in by_size.values():
        for key in ("backend", "gpu_type", "workload"):
            if events[0][key] != getattr(args, key):
                raise ValueError(f"launcher log disagrees with sweep {key}")
    efficiencies = scaling(by_size)
    rows = ["| GPUs | Tokens/s | Scaling efficiency |", "| --- | --- | --- |"]
    for size, events in by_size.items():
        tps = throughput(events)
        efficiency = efficiencies[size]
        tps_text = f"{tps:.2f}" if tps is not None else MISSING
        efficiency_text = f"{efficiency:.2%}" if efficiency is not None else MISSING
        rows.append(f"| {size} | {tps_text} | {efficiency_text} |")
    (output_dir / "scaling.md").write_text("\n".join(rows) + "\n")
    (output_dir / "sweep.json").write_text(json.dumps(mapping, indent=2) + "\n")
    output(status="complete", report=str(output_dir / "scaling.md"))


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
    sweep_parser = commands.add_parser(
        "sweep",
        help="Plan or execute 1,2,4,8-GPU measurements",
        epilog=(
            "Example: training-harness sweep --output runs/s1 --gpu-type L4 "
            "--workload gpt124m-v1 -- srun --gpus={world_size} python train.py"
        ),
    )
    sweep_parser.add_argument("--backend", choices=["ddp", "ray", "jax"], default="ddp")
    sweep_parser.add_argument("--run-prefix", default="scale")
    sweep_parser.add_argument("--gpu-type", required=True)
    sweep_parser.add_argument("--workload", required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument(
        "--execute", action="store_true", help="Actually run the supplied argv"
    )
    sweep_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.action == "sweep":
            sweep(args)
            return 0
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
