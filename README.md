# training-harness

Measurement hooks for a user-owned 124M GPT training loop on PyTorch DDP with
Slurm, Ray Train, and JAX. This repository does not implement a model or a trainer.

The intended workload pretrains on a public FineWeb-Edu sample of about 2.5B
tokens, then tunes on sanitized Data First Jobs postings for listing quality.
Infrastructure lives separately in the default-off GCP training project. This
package never provisions resources or reads cloud credentials.

## Results

No GPU, cloud, or real preemption measurements have run. CPU tests use synthetic
values and never populate this table. Unavailable evidence stays `not measured yet`.

<!-- harness-results:start -->
| Measurement | Result |
| --- | --- |
| Tokens/s on 8 GPUs | not measured yet |
| Scaling efficiency 1→8 GPUs | not measured yet |
| Profiled bottleneck before/after | not measured yet |
| Preemption recovery minutes (Slurm requeue, median of 3) | not measured yet |
| Max steps lost | not measured yet |
| Preemptions survived | not measured yet |
| Cost per run (estimated GPU-only USD) | not measured yet |
| Ray tokens/s | not measured yet |
| JAX tokens/s | not measured yet |
<!-- harness-results:end -->

The [hook and measurement reference](docs/hooks.md) defines timing, recovery,
pricing, and backend contracts. Cost is an estimate, not a cloud bill. The price
table deliberately has no rates or quote dates until a real regional quote exists.

## Run offline checks

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
```

Runtime code uses the standard library. PyTorch, Ray, and JAX belong to your
training environment and are not installed by this package. Tests use a tiny CPU
model and fake profiler modules, including a new-process requeue simulation.
There is no hosted CI workflow or automatic cloud execution.

## Integrate your loop

Use `Run` on each worker. Supply the distributed operations described in
[backend adapters](docs/hooks.md#backend-adapters). This sketch calls functions
owned by your loop, not functions supplied by this repository.

```python
from training_harness import Run

with Run(
    log_path, run_id=run_id, backend="ddp", world_size=gpu_count,
    gpu_type=gpu_type, workload=workload_id, rank=rank, barrier=barrier,
) as run, run.signals():
    if restart_count:
        run.restore(load_checkpoint)
    for step in your_steps():
        # The loop times a device-complete step, sums tokens across workers,
        # and takes the maximum worker duration before it calls on_step.
        tokens, loss, seconds = your_timed_step(step)
        run.on_step(step, tokens, loss, seconds, warmup=step <= warmup_steps)
        run.poll_preemption(save_checkpoint, agree=any_worker_requested_stop)
```

## Compare runs

Run the training loop separately at each GPU count and record each run with
`Run`. Select the resulting run IDs in `config/results.json` as described below;
the report computes scaling efficiency from the 1- and 8-GPU measurements.
See [`templates/slurm.sbatch`](templates/slurm.sbatch) for requeue jobs.

## Publish measured results

1. Copy actual run IDs into `config/results.json`. Keep unknown selections null.
2. Fill `config/prices.json` with a GPU-only regional price, source URL, and
   `as_of` date in `YYYY-MM-DD` format. Keep rates null until you have a quote.
3. Pass all selected JSONL logs to the report command.

```sh
uv run training-harness report runs/scaling-v1/*.jsonl runs/recovery/*.jsonl \
  --selection config/results.json --prices config/prices.json --readme README.md
```

The command replaces only the marked table. Missing selections are unmeasured.
A selected run without a log is an error. With the committed empty selection,
this command regenerates the current table without any training run.

```sh
uv run training-harness report --selection config/results.json --readme README.md
```
