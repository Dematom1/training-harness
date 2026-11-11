"""Derive results from explicit run selections; absent evidence stays absent."""

import os
import statistics
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

from .recorder import finite, read_events

MISSING = "not measured yet"
START = "<!-- harness-results:start -->"
END = "<!-- harness-results:end -->"
METRICS = (
    "Tokens/s on 8 GPUs",
    "Scaling efficiency 1→8 GPUs",
    "Profiled bottleneck before/after",
    "Preemption recovery minutes (Slurm requeue, median of 3)",
    "Max steps lost",
    "Preemptions survived",
    "Cost per run (estimated GPU-only USD)",
    "Ray tokens/s",
    "JAX tokens/s",
)


def load_runs(paths: list[Path]) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    seen = set()
    for event in read_events(paths):
        identity = event["event_id"]
        if identity in seen:
            raise ValueError("duplicate event/log input")
        seen.add(identity)
        runs[event["run_id"]].append(event)
    for events in runs.values():
        # Preserve append order: a signal can be recorded after its latched timestamp.
        first = events[0]
        for event in events:
            for key in ("backend", "world_size", "gpu_type", "workload", "variant", "git_sha"):
                if event[key] != first[key]:
                    raise ValueError(f"run {first['run_id']} changes {key}")
            if event["event"] == "step":
                finite(event["seconds"], "step seconds", positive=True)
                finite(event["tokens"], "step tokens", positive=True)
            if event["event"] == "phase":
                finite(event["seconds"], "phase seconds", positive=True)
    return dict(runs)


def throughput(events: list[dict]) -> float | None:
    steps = [e for e in events if e["event"] == "step" and not e["warmup"]]
    if not steps:
        return None
    return sum(e["tokens"] for e in steps) / sum(e["seconds"] for e in steps)


def compatible(left: list[dict], right: list[dict], *, variant: bool = True) -> None:
    keys = ["backend", "workload", "gpu_type"] + (["variant", "git_sha"] if variant else [])
    for key in keys:
        if left[0][key] != right[0][key]:
            raise ValueError(f"comparison requires matching {key}")


def scaling(runs: dict[int, list[dict]]) -> dict[int, float | None]:
    baseline = runs.get(1, [])
    base_tps = throughput(baseline)
    result = {}
    for size, events in sorted(runs.items()):
        if events and events[0]["world_size"] != size:
            raise ValueError("scaling selection has wrong world size")
        if events and baseline:
            compatible(baseline, events)
        tps = throughput(events)
        result[size] = tps / (size * base_tps) if base_tps and tps else None
    return result


def recoveries(events: list[dict]) -> list[dict]:
    """Match one signal to one later restored Slurm attempt's first step."""
    pending = None
    restored = {}
    samples = []
    for event in events:
        kind = event["event"]
        if kind == "preemption":
            pending = event
        elif kind == "checkpoint_restore":
            restored[event["attempt_id"]] = event["step"]
        elif kind == "step" and pending is not None:
            step = restored.get(event["attempt_id"])
            if (
                pending["signal"] != 0
                and event["attempt_id"] != pending["attempt_id"]
                and step is not None
                and event["slurm_job_id"]
                and event["slurm_job_id"] == pending["slurm_job_id"]
                and event["restart_count"] > pending["restart_count"]
            ):
                seconds = event["timestamp"] - pending["timestamp"]
                finite(seconds, "recovery seconds")
                samples.append(
                    dict(
                        timestamp=event["timestamp"],
                        minutes=seconds / 60,
                        steps_lost=max(0, pending["step"] - step),
                    )
                )
                pending = None
    return samples


def cost(events: list[dict], prices: dict) -> float | None:
    if not events:
        return None
    if prices.get("currency", "USD") != "USD":
        raise ValueError("cost reporting requires USD prices")
    quote = prices.get("gpus", {}).get(events[0]["gpu_type"], {})
    rate = quote.get("usd_per_gpu_hour")
    if rate is None or not quote.get("source") or not quote.get("as_of"):
        return None
    finite(rate, "GPU hourly price", positive=True)
    date.fromisoformat(quote["as_of"])
    starts = {e["attempt_id"]: e for e in events if e["event"] == "attempt_start"}
    ends = {e["attempt_id"]: e for e in events if e["event"] == "attempt_end"}
    if not starts or starts.keys() != ends.keys():
        return None
    seconds = 0.0
    for attempt, start in starts.items():
        elapsed = ends[attempt]["timestamp"] - start["timestamp"]
        finite(elapsed, "attempt duration")
        seconds += elapsed
    return seconds / 3600 * events[0]["world_size"] * rate


def result_values(runs: dict[str, list[dict]], selection: dict, prices: dict) -> list[str]:
    keys = {
        "ddp_1",
        "ddp_8",
        "profile_before",
        "profile_after",
        "bottleneck",
        "recovery",
        "ray",
        "jax",
    }
    if not isinstance(selection, dict) or set(selection) - keys:
        raise ValueError("selection must be an object with known keys from config/results.json")
    for key, value in selection.items():
        if key == "recovery":
            if not isinstance(value, list) or any(not isinstance(s, str) or not s for s in value):
                raise ValueError("recovery selection must be a list of run IDs")
        elif value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} selection must be a run ID/name or null")
    if not isinstance(prices, dict) or not isinstance(prices.get("gpus", {}), dict):
        raise ValueError("prices must be an object with a gpus object")
    if any(not isinstance(quote, dict) for quote in prices.get("gpus", {}).values()):
        raise ValueError("each GPU price must be an object")

    def selected(key: str, backend: str | None = None, size: int | None = None) -> list[dict]:
        run_id = selection.get(key)
        if run_id is None:
            return []
        if run_id not in runs:
            raise ValueError(f"selected run {run_id!r} has no log")
        events = runs[run_id]
        if backend and events[0]["backend"] != backend:
            raise ValueError(f"{key} must use backend {backend}")
        if size and events[0]["world_size"] != size:
            raise ValueError(f"{key} must use {size} GPUs")
        return events

    def tps(events):
        value = throughput(events)
        return f"{value:,.2f} ({events[0]['run_id']})" if value is not None else MISSING

    one = selected("ddp_1", "ddp", 1)
    eight = selected("ddp_8", "ddp", 8)
    efficiency = scaling({1: one, 8: eight}).get(8)
    values = [tps(eight), f"{efficiency:.2%}" if efficiency is not None else MISSING]

    before = selected("profile_before")
    after = selected("profile_after")
    name = selection.get("bottleneck")
    profile = MISSING
    if before and after and name:
        compatible(before, after, variant=False)
        if before[0]["world_size"] != after[0]["world_size"]:
            raise ValueError("profile comparison requires matching world_size")
        durations = []
        for events in (before, after):
            traces = {e["profile_id"] for e in events if e["event"] == "profile"}
            phases = [
                e["seconds"]
                for e in events
                if e["event"] == "phase" and e["name"] == name and e["profile_id"] in traces
            ]
            durations.append(statistics.mean(phases) * 1000 if phases else None)
        if all(d is not None for d in durations):
            profile = f"{name}: {durations[0]:.2f} → {durations[1]:.2f} ms/phase (mean)"
    values.append(profile)

    samples = []
    recovery_reference = None
    for run_id in selection.get("recovery", []):
        if run_id not in runs:
            raise ValueError(f"recovery run {run_id!r} has no log")
        events = runs[run_id]
        if events[0]["backend"] != "ddp":
            raise ValueError("Slurm recovery measurements require ddp runs")
        if recovery_reference:
            compatible(recovery_reference, events)
            if recovery_reference[0]["world_size"] != events[0]["world_size"]:
                raise ValueError("recovery measurements require matching world_size")
        recovery_reference = events
        samples.extend(recoveries(events))
    if len(set(selection.get("recovery", []))) != len(selection.get("recovery", [])):
        raise ValueError("duplicate recovery selection")
    latest_three = sorted(samples, key=lambda sample: sample["timestamp"])[-3:]
    recovery = (
        f"{statistics.median(s['minutes'] for s in latest_three):.2f}"
        if len(latest_three) == 3
        else MISSING
    )
    values.extend(
        [
            recovery,
            str(max(s["steps_lost"] for s in samples)) if samples else MISSING,
            str(len(samples)) if samples else MISSING,
        ]
    )
    estimate = cost(eight, prices)
    if estimate is not None:
        quote = prices["gpus"][eight[0]["gpu_type"]]
        values.append(
            f"${estimate:.4f} ({eight[0]['run_id']}; {quote['as_of']}; {quote['source']})"
        )
    else:
        values.append(MISSING)
    for backend in ("ray", "jax"):
        events = selected(backend, backend, 8)
        if eight and events and events[0]["workload"] != eight[0]["workload"]:
            raise ValueError(f"{backend} must match the eight-GPU DDP workload")
        values.append(tps(events))
    return values


def render_table(values: list[str]) -> str:
    def escape(value):
        return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    rows = ["| Measurement | Result |", "| --- | --- |"]
    rows.extend(
        f"| {metric} | {escape(value)} |" for metric, value in zip(METRICS, values, strict=True)
    )
    return "\n".join(rows)


def update_readme(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError("README needs exactly one ordered harness-results marker pair")
    start = text.index(START) + len(START)
    end = text.index(END)
    updated = text[:start] + "\n" + table + "\n" + text[end:]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
