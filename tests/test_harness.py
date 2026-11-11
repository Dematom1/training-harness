"""Offline evidence only. Synthetic timings must never enter README results."""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from training_harness import Run
from training_harness.cli import main
from training_harness.recorder import read_events
from training_harness.report import (
    END,
    MISSING,
    START,
    cost,
    load_runs,
    recoveries,
    render_table,
    result_values,
    scaling,
    throughput,
    update_readme,
)


class DummyModel:
    """A scalar CPU model used only here to exercise user-owned checkpoints."""

    def __init__(self, path):
        self.path = path
        self.weight = 0.0
        self.step = 0

    def train_step(self):
        loss = (self.weight - 1.0) ** 2
        self.weight -= 0.1 * 2 * (self.weight - 1.0)
        self.step += 1
        return loss

    def save(self, step):
        assert step == self.step
        # Immutable step paths keep the last acknowledged checkpoint recoverable.
        path = self.path.with_name(f"{self.path.stem}-{step}.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"step": step, "weight": self.weight}))
        temporary.replace(path)
        return str(path)

    def load(self, path):
        state = json.loads(Path(path).read_text())
        self.weight, self.step = state["weight"], state["step"]
        return self.step


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_hook(self, run_id="test", **kwargs):
        options = (
            dict(
                run_id=run_id,
                backend="ddp",
                world_size=1,
                gpu_type="test-gpu",
                workload="synthetic-v1",
                git_sha="test-sha",
            )
            | kwargs
        )
        return Run(self.root / f"{run_id}.jsonl", **options)

    def events(self, run_id="test"):
        return read_events([self.root / f"{run_id}.jsonl"])

    def measured(self, run_id, size=1, backend="ddp", tokens=100, seconds=1, variant="baseline"):
        with self.run_hook(run_id, world_size=size, backend=backend, variant=variant) as run:
            run.on_step(1, 10000, 1, 1, warmup=True)
            run.on_step(2, tokens, 0.5, seconds)
        return self.events(run_id)

    def test_dummy_model_and_append_only_resume(self):
        model = DummyModel(self.root / "model.json")
        with self.run_hook() as run:
            run.on_step(1, 32, model.train_step(), 2)
            saved = run.checkpoint(model.save)
        prefix = (self.root / "test.jsonl").read_bytes()
        expected_weight = model.weight
        model.weight = 99
        with self.run_hook() as run:
            self.assertEqual(run.restore(model.load), 1)
            self.assertEqual(model.weight, expected_weight)
            run.on_step(2, 32, model.train_step(), 1)
        self.assertTrue((self.root / "test.jsonl").read_bytes().startswith(prefix))
        self.assertTrue(Path(saved).exists())
        steps = [e for e in self.events() if e["event"] == "step"]
        self.assertEqual([s["tokens_per_second"] for s in steps], [16, 32])
        self.assertEqual(steps[0]["git_sha"], "test-sha")
        self.assertAlmostEqual(throughput(self.events()), 64 / 3)

    def test_scaling_weighted_math_and_warmup(self):
        runs = {
            size: self.measured(str(size), size, tokens=tokens)
            for size, tokens in [(1, 100), (2, 190), (4, 360), (8, 720)]
        }
        self.assertEqual(scaling(runs), {1: 1, 2: 0.95, 4: 0.9, 8: 0.9})
        with self.run_hook("weighted") as run:
            run.on_step(1, 100, 1, 1)
            run.on_step(2, 100, 1, 3)
        self.assertEqual(throughput(self.events("weighted")), 50)
        self.assertIsNone(scaling({1: [], 8: runs[8]})[8])

    def test_comparisons_reject_different_workloads(self):
        one = self.measured("one")
        eight = self.measured("eight", 8)
        eight[0]["workload"] = "different"
        with self.assertRaisesRegex(ValueError, "workload"):
            scaling({1: one, 8: eight})

    def test_rank_one_never_writes_but_calls_checkpoint_callback(self):
        callback = Mock(return_value="durable/path")
        with self.run_hook(world_size=2, rank=1) as run:
            run.on_step(1, 10, 1, 1)
            run.checkpoint(callback)
        callback.assert_called_once_with(1)
        self.assertFalse((self.root / "test.jsonl").exists())

    def test_validation_and_failed_checkpoint_is_not_recorded(self):
        with self.run_hook() as run:
            for params in [(1, 0, 1, 1), (1, 1, float("nan"), 1), (1, 1, 1, 0)]:
                with self.assertRaises(ValueError):
                    run.on_step(*params)
            run.on_step(1, 1, 1, 1)
            with self.assertRaises(ValueError):
                run.on_step(1, 1, 1, 1)
            with self.assertRaises(OSError):
                run.checkpoint(Mock(side_effect=OSError("disk full")))
        self.assertNotIn("checkpoint_save", [e["event"] for e in self.events()])
        with self.assertRaisesRegex(ValueError, "backend"):
            self.run_hook(backend="unknown")
        with self.assertRaisesRegex(ValueError, "different workload"):
            self.run_hook(workload="other")
        (self.root / "bad.jsonl").write_text('{"schema":1')
        with self.assertRaisesRegex(ValueError, "invalid run record"):
            read_events([self.root / "bad.jsonl"])

    def test_signal_safe_point_requeue_and_three_recoveries(self):
        model = DummyModel(self.root / "model.json")
        for restart in range(4):
            os.environ["SLURM_RESTART_COUNT"] = str(restart)
            now = 1000 + restart * 120
            try:
                with patch("training_harness.recorder.time.time", return_value=now):
                    with self.run_hook() as run, run.signals():
                        if restart:
                            run.restore(model.load)
                        run.on_step(model.step + 1, 32, model.train_step(), 1)
                        if restart < 3:
                            signal.raise_signal(signal.SIGUSR1 if restart != 1 else signal.SIGTERM)
                            # A latched signal does not checkpoint inside the handler.
                            self.assertFalse(
                                any(
                                    e["event"] == "preemption"
                                    for e in self.events()
                                    if e["attempt_id"] == run.metadata["attempt_id"]
                                )
                            )
                            run.poll_preemption(model.save)
            except SystemExit as error:
                self.assertEqual(error.code, 75)
        samples = recoveries(self.events())
        self.assertEqual(len(samples), 3)
        self.assertEqual([sample["minutes"] for sample in samples], [2, 2, 2])
        values = result_values({"test": self.events()}, {"recovery": ["test"]}, {})
        self.assertEqual(values[3:6], ["2.00", "0", "3"])
        self.assertEqual(len({e["attempt_id"] for e in self.events()}), 4)

    def test_steps_lost_when_signal_checkpoint_fails_and_old_state_restored(self):
        model = DummyModel(self.root / "model.json")
        with self.assertRaises(OSError):
            with self.run_hook() as run, run.signals():
                run.on_step(1, 10, model.train_step(), 1)
                run.checkpoint(model.save)
                run.on_step(2, 10, model.train_step(), 1)
                signal.raise_signal(signal.SIGUSR1)
                run.poll_preemption(Mock(side_effect=OSError("interrupted write")))
        os.environ["SLURM_RESTART_COUNT"] = "1"
        with self.run_hook() as run:
            run.restore(model.load)
            run.on_step(2, 10, model.train_step(), 1)
        samples = recoveries(self.events())
        self.assertEqual(samples[0]["steps_lost"], 1)
        # A second post-resume step must not count another survival.
        self.assertEqual(len(samples), 1)
        self.assertEqual(
            result_values({"test": self.events()}, {"recovery": ["test"]}, {})[3], MISSING
        )

    def test_requeue_requires_restore_and_actual_slurm_restart(self):
        self.measured("test")
        os.environ["SLURM_RESTART_COUNT"] = "1"
        with self.run_hook() as run:
            with self.assertRaisesRegex(ValueError, "restore"):
                run.on_step(1, 1, 1, 1)
        self.assertEqual(recoveries(self.events()), [])

    def test_collective_stop_and_original_signal_handlers_restored(self):
        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(SystemExit):
            with self.run_hook() as run, run.signals():
                run.poll_preemption(lambda step: "checkpoint", agree=lambda local: True)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        self.assertEqual([e for e in self.events() if e["event"] == "preemption"][0]["signal"], 0)

    def test_phase_and_opt_in_torch_profile(self):
        profiler = Mock()
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                profile=Mock(return_value=nullcontext(profiler)),
                record_function=Mock(side_effect=lambda name: nullcontext()),
            ),
        )
        for run_id, variant, duration in [("before", "baseline", 2), ("after", "optimized", 1)]:
            with self.run_hook(run_id, variant=variant) as run:
                with patch.dict(sys.modules, {"torch": torch}):
                    with run.profile(self.root / run_id):
                        sync = Mock()
                        with patch(
                            "training_harness.recorder.time.perf_counter", side_effect=[0, duration]
                        ):
                            with run.phase("input", synchronize=sync):
                                pass
                        self.assertEqual(sync.call_count, 2)
        runs = {name: self.events(name) for name in ("before", "after")}
        selection = dict(profile_before="before", profile_after="after", bottleneck="input")
        self.assertEqual(
            result_values(runs, selection, {})[2], "input: 2000.00 → 1000.00 ms/phase (mean)"
        )
        profiler.export_chrome_trace.assert_called()
        no_trace = [e for e in runs["before"] if e["event"] != "profile"]
        self.assertEqual(result_values(runs | {"before": no_trace}, selection, {})[2], MISSING)

    def test_jax_trace_hook_is_lazy(self):
        profiler = SimpleNamespace(
            trace=Mock(return_value=nullcontext()), TraceAnnotation=lambda name: nullcontext()
        )
        jax = SimpleNamespace(profiler=profiler)
        with patch.dict(sys.modules, {"jax": jax, "jax.profiler": profiler}):
            with self.run_hook(backend="jax") as run, run.profile(self.root / "trace"):
                pass
        profiler.trace.assert_called_once_with(str(self.root / "trace"))
        self.assertIn("profile", [e["event"] for e in self.events()])

    def test_cost_uses_completed_attempts_and_dated_price_only(self):
        with patch("training_harness.recorder.time.time", side_effect=[1000, 4600]):
            with self.run_hook(world_size=8):
                pass
        events = self.events()
        prices = {
            "gpus": {
                "test-gpu": {
                    "usd_per_gpu_hour": 2,
                    "source": "test://quote",
                    "as_of": "2025-01-01",
                }
            }
        }
        self.assertEqual(cost(events, prices), 16)
        self.assertIsNone(cost(events[:-1], prices))
        self.assertIsNone(cost(events, {}))
        prices["gpus"]["test-gpu"]["as_of"] = None
        self.assertIsNone(cost(events, prices))

    def test_exact_nine_metric_readme_and_missing_data(self):
        values = result_values({}, {}, {})
        table = render_table(values)
        self.assertEqual(table.count(MISSING), 9)
        self.assertEqual(len(table.splitlines()), 11)
        readme = self.root / "README.md"
        readme.write_text(f"intro\n{START}\nold\n{END}\noutro\n")
        update_readme(readme, table)
        self.assertEqual(readme.read_text(), f"intro\n{START}\n{table}\n{END}\noutro\n")
        update_readme(readme, table)
        self.assertEqual(readme.read_text().count(MISSING), 9)
        readme.write_text("no markers")
        with self.assertRaises(ValueError):
            update_readme(readme, table)
        with self.assertRaisesRegex(ValueError, "no log"):
            result_values({}, {"ddp_8": "missing"}, {})

    def test_ray_jax_and_explicit_run_selection(self):
        runs = {
            "eight": self.measured("eight", 8, tokens=720),
            "one": self.measured("one"),
            "ray": self.measured("ray", 8, "ray", tokens=600),
            "jax": self.measured("jax", 8, "jax", tokens=650),
        }
        values = result_values(runs, dict(ddp_1="one", ddp_8="eight", ray="ray", jax="jax"), {})
        self.assertEqual(values[:2], ["720.00 (eight)", "90.00%"])
        self.assertEqual(values[-2:], ["600.00 (ray)", "650.00 (jax)"])
        selection = self.root / "selection.json"
        selection.write_text(json.dumps(dict(ddp_1="one", ddp_8="eight", ray="ray", jax="jax")))
        prices = self.root / "prices.json"
        prices.write_text("{}")
        readme = self.root / "README.md"
        readme.write_text(f"{START}\n{END}\n")
        with redirect_stdout(io.StringIO()):
            status = main(
                [
                    "report",
                    *[str(self.root / f"{name}.jsonl") for name in runs],
                    "--selection",
                    str(selection),
                    "--prices",
                    str(prices),
                    "--readme",
                    str(readme),
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("90.00%", readme.read_text())
        self.assertEqual(readme.read_text().count(MISSING), 5)
        with self.assertRaisesRegex(ValueError, "backend"):
            result_values(runs, {"ray": "jax"}, {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_runs([self.root / "one.jsonl"] * 2)

    def test_cli_home_and_rejects_unneeded_sweep(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main([]), 0)
        self.assertIn("measurements[9]", stream.getvalue())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["sweep"])
        self.assertEqual(error.exception.code, 2)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--unknown"])
        self.assertEqual(error.exception.code, 2)

    def test_preemption_exit_barrier_follows_durable_end_record(self):
        calls = []

        def barrier():
            calls.append([e["event"] for e in self.events()] if calls else [])

        with self.assertRaises(SystemExit):
            with self.run_hook(barrier=barrier) as run, run.signals():
                signal.raise_signal(signal.SIGUSR1)
                run.poll_preemption(lambda step: "checkpoint")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][-1], "attempt_end")

    def test_invalid_configs_and_unterminated_records_fail_closed(self):
        for selection in ([], {"ddp8": "typo"}, {"recovery": None}):
            with self.assertRaises(ValueError):
                result_values({}, selection, {})
        with self.assertRaises(ValueError):
            result_values({}, {}, {"gpus": []})
        self.measured("test")
        log = self.root / "test.jsonl"
        log.write_text(log.read_text().rstrip("\n"))
        with self.assertRaisesRegex(ValueError, "unterminated"):
            self.run_hook()

    def test_resume_in_a_new_process(self):
        script = self.root / "restart.py"
        script.write_text("""import os, signal, sys
from training_harness import Run
path = sys.argv[1]
with Run(path, run_id="process", backend="ddp", world_size=1,
         gpu_type="cpu-test", workload="test") as run, run.signals():
    if os.environ["SLURM_RESTART_COUNT"] != "0":
        run.restore(lambda path: 1)
        run.on_step(2, 10, 1, 1)
    else:
        run.on_step(1, 10, 1, 1)
        signal.raise_signal(signal.SIGUSR1)
        run.poll_preemption(lambda step: "synthetic-checkpoint")
""")
        log = self.root / "process.jsonl"
        first = subprocess.run([sys.executable, str(script), str(log)], check=False)
        self.assertEqual(first.returncode, 75)
        second = subprocess.run(
            [sys.executable, str(script), str(log)],
            env=os.environ | {"SLURM_RESTART_COUNT": "1"},
            check=False,
        )
        self.assertEqual(second.returncode, 0)
        self.assertEqual(len(recoveries(read_events([log]))), 1)


if __name__ == "__main__":
    unittest.main()
