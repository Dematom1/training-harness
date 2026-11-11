"""Single-coordinator append-only records. Framework imports are opt-in."""

import json
import math
import os
import signal
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

BACKENDS = {"ddp", "ray", "jax"}


def finite(value: float, name: str, *, positive: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def read_events(paths: list[Path]) -> list[dict]:
    events = []
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    if not line.endswith("\n"):
                        raise ValueError(
                            "unterminated record; preserve and repair a copy before resume"
                        )
                    event = json.loads(line)
                    if event["schema"] != 1:
                        raise ValueError("unsupported schema")
                    for key in (
                        "run_id",
                        "attempt_id",
                        "event_id",
                        "event",
                        "backend",
                        "workload",
                        "gpu_type",
                        "variant",
                        "git_sha",
                    ):
                        if not isinstance(event[key], str) or not event[key]:
                            raise ValueError(f"invalid {key}")
                    finite(event["timestamp"], "timestamp")
                    if type(event["world_size"]) is not int or event["world_size"] < 1:
                        raise ValueError("invalid world_size")
                    if type(event["restart_count"]) is not int or event["restart_count"] < 0:
                        raise ValueError("invalid restart_count")
                    if (
                        not isinstance(event["slurm_job_id"], str)
                        or event["backend"] not in BACKENDS
                    ):
                        raise ValueError("invalid Slurm job ID or backend")
                    if event["event"] == "step":
                        if type(event["step"]) is not int or event["step"] < 1:
                            raise ValueError("invalid step")
                        if type(event["tokens"]) is not int or event["tokens"] < 1:
                            raise ValueError("invalid tokens")
                        if type(event["warmup"]) is not bool or not math.isfinite(event["loss"]):
                            raise ValueError("invalid warmup or loss")
                        finite(event["seconds"], "step seconds", positive=True)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid run record: {error}"
                    ) from error
                events.append(event)
    return events


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class Run:
    """All ranks call hooks; only rank zero writes. Use as a context manager.

    Save callbacks must durably publish a checkpoint, including optimizer, RNG,
    and data cursor, before returning its path. Load callbacks return its step.
    Hooks never synchronize workers unless the caller supplies that operation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        backend: str,
        world_size: int,
        gpu_type: str,
        workload: str,
        variant: str = "baseline",
        rank: int = 0,
        git_sha: str | None = None,
        barrier: Callable[[], None] | None = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
        if (
            type(world_size) is not int
            or world_size < 1
            or type(rank) is not int
            or not 0 <= rank < world_size
        ):
            raise ValueError("world_size must be positive and rank must be in [0, world_size)")
        if not all(isinstance(s, str) and s.strip() for s in (run_id, gpu_type, workload, variant)):
            raise ValueError("run_id, gpu_type, workload, and variant must be nonempty strings")
        self.path = Path(path)
        self.rank = rank
        self.barrier = barrier
        self.last_step = 0
        self.pending: tuple[int, float, int] | None = None
        self.closed = False
        self.restored = False
        self.profile_id: str | None = None
        self.annotation = None
        self.metadata = dict(
            schema=1,
            run_id=run_id,
            attempt_id=uuid.uuid4().hex,
            git_sha=git_sha or git_revision(),
            backend=backend,
            world_size=world_size,
            gpu_type=gpu_type,
            workload=workload,
            variant=variant,
            slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
            restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        )
        if self.metadata["restart_count"] < 0:
            raise ValueError("SLURM_RESTART_COUNT must be nonnegative")
        self.history = read_events([self.path]) if self.path.exists() else []
        for event in self.history:
            for key in (
                "run_id",
                "backend",
                "world_size",
                "gpu_type",
                "workload",
                "variant",
                "git_sha",
            ):
                if event[key] != self.metadata[key]:
                    raise ValueError(f"existing log has different {key}; use a separate log")
        # Every reader must finish its immutable startup snapshot before rank zero appends.
        if barrier is not None:
            barrier()
        if rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._emit("attempt_start")

    def _emit(self, event: str, *, timestamp: float | None = None, **fields) -> None:
        if self.closed:
            raise ValueError("run is closed")
        if self.rank != 0:
            return
        record = self.metadata | dict(
            event=event,
            event_id=uuid.uuid4().hex,
            timestamp=time.time() if timestamp is None else timestamp,
            **fields,
        )
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if event != "step" and event != "phase":
                os.fsync(stream.fileno())

    def on_step(
        self, step: int, tokens: int, loss: float, seconds: float, *, warmup: bool = False
    ) -> None:
        """Record globally summed tokens and slowest-worker elapsed seconds."""
        if type(step) is not int or step <= self.last_step:
            raise ValueError("step must increase from the last completed/restored step")
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("tokens must be a positive global token count")
        finite(seconds, "seconds", positive=True)
        if type(warmup) is not bool or type(loss) not in (int, float) or not math.isfinite(loss):
            raise ValueError("loss must be finite and warmup must be a bool")
        if self.metadata["restart_count"] > 0 and not self.restored:
            raise ValueError("restore a checkpoint before recording a requeued step")
        self._emit(
            "step",
            step=step,
            tokens=tokens,
            loss=loss,
            seconds=seconds,
            tokens_per_second=tokens / seconds,
            warmup=warmup,
        )
        self.last_step = step

    def checkpoint(self, save: Callable[[int], str]) -> str:
        """Call on all ranks; save returns the same durable path on each rank."""
        path = save(self.last_step)
        if not isinstance(path, str) or not path:
            raise ValueError("save must return a nonempty checkpoint path")
        self._emit("checkpoint_save", step=self.last_step, path=path)
        return path

    def restore(self, load: Callable[[str], int]) -> int:
        checkpoints = [e for e in self.history if e["event"] == "checkpoint_save"]
        if not checkpoints:
            raise ValueError("no recorded checkpoint to restore")
        checkpoint = checkpoints[-1]
        step = load(checkpoint["path"])
        if step != checkpoint["step"]:
            raise ValueError("restored step differs from the recorded checkpoint")
        self._emit("checkpoint_restore", step=step, path=checkpoint["path"])
        self.last_step = step
        self.restored = True
        return step

    @contextmanager
    def phase(self, name: str, *, synchronize: Callable[[], None]) -> Iterator[None]:
        """Time a named phase. Supply device completion, not just host enqueue."""
        if not name.strip():
            raise ValueError("phase name must be nonempty")
        synchronize()
        start = time.perf_counter()
        with self.annotation(name) if self.annotation else nullcontext():
            yield
            synchronize()
        self._emit(
            "phase",
            name=name,
            seconds=time.perf_counter() - start,
            profile_id=self.profile_id,
        )

    @contextmanager
    def profile(self, directory: str | Path) -> Iterator[None]:
        """Opt-in coordinator trace. Synchronize devices within the caller's block."""
        if self.rank != 0:
            yield
            return
        if self.profile_id is not None:
            raise ValueError("profile contexts cannot nest")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.profile_id = uuid.uuid4().hex
        try:
            if self.metadata["backend"] == "jax":
                import jax.profiler

                self.annotation = jax.profiler.TraceAnnotation
                with jax.profiler.trace(str(directory)):
                    yield
            else:
                import torch

                self.annotation = torch.profiler.record_function
                activities = [torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                    yield
                profiler.export_chrome_trace(str(directory / "trace.json"))
            self._emit("profile", path=str(directory.resolve()), profile_id=self.profile_id)
        finally:
            self.profile_id = None
            self.annotation = None

    @contextmanager
    def signals(self) -> Iterator[None]:
        """Handlers only latch state; checkpoint later at a collective-safe point."""

        def latch(signum, frame):
            if self.pending is None:
                self.pending = (signum, time.time(), self.last_step)

        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
        try:
            for sig in previous:
                signal.signal(sig, latch)
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def poll_preemption(
        self,
        save: Callable[[int], str],
        *,
        agree: Callable[[bool], bool] = bool,
    ) -> None:
        """All ranks poll together. agree must OR the request across workers."""
        if not agree(self.pending is not None):
            return
        signum, timestamp, step = self.pending or (0, time.time(), self.last_step)
        self._emit(
            "preemption",
            timestamp=timestamp,
            signal=signum,
            step=self.last_step,
            signal_step=step,
        )
        self.checkpoint(save)
        self._emit("requeue_requested", step=self.last_step)
        raise SystemExit(75)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        status = "completed" if exc is None else "failed"
        if isinstance(exc, SystemExit) and exc.code == 75:
            status = "preempted"
        self._emit("attempt_end", status=status, step=self.last_step)
        # Do not let a peer's exit 75 make srun kill the writer before its durable end record.
        if status == "preempted" and self.barrier is not None:
            self.barrier()
        self.closed = True
