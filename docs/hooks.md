# Hook and measurement reference

## Ownership and log format

The caller owns distributed initialization, the model, the optimizer, data loading,
checkpoint serialization, and resource scheduling. `Run` records observations.
It does not infer GPU counts or multiply rank-zero tokens by the number of GPUs.

`Run(path, *, run_id, backend, world_size, gpu_type, workload, variant="baseline",
rank=0, git_sha=None, barrier=None)` accepts `ddp`, `ray`, or `jax`. `world_size`
is the total number of GPUs, including all hosts. `rank` is the global worker
rank, not the local device index. For single-process multi-device JAX, rank is zero.
GPU type must match the price-table key. `workload` identifies the model, corpus
snapshot, sequence length, precision, global tokens per optimizer step, warmup,
and measurement window. Keep those fixed across comparisons. `variant` names an
implementation change. The report checks these labels, not the underlying model.

Use a new run ID and log for each experiment, but keep both stable across its
requeues. All ranks must see the same durable shared filesystem. A multiprocess
adapter must pass `barrier` so every rank reads the prior log before rank zero
appends the new attempt. Only one rank-zero writer may own a log at a time.
Do not overlap restarted attempts or concatenate copies of the same log.

Each line is a JSON object with schema version 1. Every event carries these fields.

| Field | Meaning |
| --- | --- |
| `event`, `event_id` | Event name and unique record ID |
| `run_id`, `attempt_id` | Stable logical run and fresh process attempt |
| `timestamp` | UTC Unix seconds from the coordinator's wall clock |
| `git_sha` | Explicit revision, or `git rev-parse HEAD` at launch, or `unknown` |
| `backend`, `world_size`, `gpu_type` | Backend and total accelerator allocation |
| `workload`, `variant` | Comparison labels |
| `slurm_job_id`, `restart_count` | `SLURM_JOB_ID` and `SLURM_RESTART_COUNT`, empty or zero outside Slurm |

Event-specific fields are as follows.

| Event | Fields |
| --- | --- |
| `attempt_start` | No extra fields |
| `step` | `step`, global `tokens`, `loss`, `seconds`, `tokens_per_second`, `warmup` |
| `checkpoint_save`, `checkpoint_restore` | `step`, durable `path` |
| `phase` | `name`, synchronized `seconds`, `profile_id` or null |
| `profile` | Completed trace directory `path`, `profile_id` |
| `preemption` | Received `signal`, completed `step` at polling, `signal_step` at receipt |
| `requeue_requested` | `step` after the signal checkpoint succeeds |
| `attempt_end` | `status` as completed, preempted, or failed, and `step` |

The preemption timestamp is signal receipt, not the later safe-point poll.
Signal zero means another worker requested the coordinated stop. That timestamp
is the coordinator's poll time and does not qualify for a recovery sample.
Deliver the Slurm signal to every rank for a signal-to-recovery measurement.
Clocks on replacement nodes must be synchronized.
Lifecycle events use flush and fsync. Step and phase events flush without fsync.
An abrupt host loss can lose unflushed filesystem data. A malformed or truncated
line is an error, never silently skipped or rewritten. Preserve the damaged log
and recover from a verified copy before appending again.

## Step and checkpoint hooks

`on_step(step, tokens, loss, seconds, *, warmup=False)` records a completed optimizer
step. Steps increase within each attempt from the last restored step. `tokens`
counts actual non-padding training tokens globally, across gradient accumulation
and every worker. `seconds` covers data loading, forward, backward, optimizer,
and training collectives for that step. Complete asynchronous device work before
taking the end timestamp. Reduce elapsed time with MAX across workers. Reduce
local token counts with SUM and loss with your token-weighted mean. Every rank
calls the hook with these global values, but only rank zero writes.

Throughput is `sum(tokens) / sum(seconds)` over non-warmup steps, not a mean of
per-step rates. It excludes checkpoint time, queue delay, recovery, and recorder
reductions outside the timed interval. Replayed steps count as processed work.
It is steady-state throughput, not unique-token progress per end-to-end hour.
Use unprofiled runs for throughput because profiler overhead changes timing.

`checkpoint(save)` calls `save(last_step)` on every rank. The callback returns
the same nonempty path after all ranks finish a durable save. It must save the
model, optimizer, scheduler, RNG, sampler or data cursor, and completed step.
Use immutable versioned paths, write temporary files, fsync, atomically publish,
and complete a worker barrier before returning. Never overwrite the last
acknowledged checkpoint in place. A failed save produces no checkpoint event.

`restore(load)` chooses the latest successful save from the startup snapshot.
It calls `load(path)` on every worker. The callback must restore the full state
and return its completed step. A mismatch fails rather than silently counting a
recovery. Call it before any requeued step. A requeue without a recorded durable
checkpoint fails. If the log lost its last save event, the preceding versioned
checkpoint remains the acknowledged recovery point.

## Backend adapters

These are contracts for the caller's adapters, not trainer implementations.
No backend is launched or initialized by `Run`.

### PyTorch DDP under Slurm

Initialize the process group and bind one visible GPU per worker. Derive global
rank from `SLURM_PROCID`, local rank from `SLURM_LOCALID`, and worker count from
`SLURM_NTASKS` for the supplied one-GPU-per-task template. Set rendezvous address
and port for the allocation in your launcher. The template does not configure
your network or process group. Under `torchrun`, use its `RANK`, `LOCAL_RANK`, and
`WORLD_SIZE` instead. Do not use Slurm worker counts as GPU counts for other layouts.

Pass `torch.distributed.barrier` into `Run`. At each step, synchronize CUDA before
and after the timed region. All-reduce token count with SUM and elapsed seconds
with MAX. The stop agreement callback must all-reduce an integer flag with MAX
and return a bool on every worker. Call it on every step, even without a signal.
Checkpoint callbacks must finish all DDP collective work before return and exit.
On a coordinated preemption, `Run` also calls the supplied barrier after rank zero
flushes its end record, before any worker exits and triggers srun peer termination.

### Ray Train

Create `Run` inside the caller's `TorchTrainer` worker function, not on the Ray
driver. Use `ray.train.get_context().get_world_rank()` and `.get_world_size()` for
a one-GPU-per-worker allocation. Pass the PyTorch barrier and the same timing,
token, and stop reductions as DDP. Set `backend="ray"`. The log and checkpoints
must use shared durable storage reachable by replacement workers.

The caller may publish its saved directory through `ray.train.Checkpoint` and
`ray.train.report`. All workers must participate in those operations as required
by Ray. Ray retries must reuse the run ID and restore hook. They do not count as
Slurm recovery samples without Slurm job and increasing restart-count evidence.
Use a separate Ray throughput run with the same workload and GPU count as DDP.

### JAX

Set `backend="jax"`, `world_size=jax.device_count()`, and `rank=jax.process_index()`.
For multihost JAX, initialize distributed execution first and pass a barrier using
`jax.experimental.multihost_utils.sync_global_devices` with a fixed label. For a
single process, no barrier is needed. Block on the returned device arrays with
`jax.block_until_ready` before taking elapsed time. Include the compiled update,
not just dispatch time. Mark JIT compilation and warmup steps as warmup.

Sum local token counts and take MAX of elapsed seconds over processes with
multihost collectives. If a token count is already global, do not sum it again.
The stop agreement callback must OR requests across processes. Sharded save and
restore callbacks must complete on all processes before the coordinator logs
success. The caller chooses its checkpoint library. Nothing here requires Orbax.
Use a separate JAX throughput run with the same workload and GPU count as DDP.

## Profiling one bottleneck

`profile(directory)` lazily imports `torch.profiler` for DDP and Ray, or
`jax.profiler` for JAX. Run it on all workers if the block contains collectives.
Only the coordinator captures a trace. PyTorch writes `trace.json` with CPU and
available CUDA activities. JAX writes its native trace files to the directory.
Use a new directory per capture. Real GPU profiler support belongs to the caller's
framework installation. The CPU tests fake the profiler interfaces, not GPU traces.

`phase(name, *, synchronize)` measures completed device work with a monotonic
clock. Inside a profile context it also creates a PyTorch `record_function` or
JAX `TraceAnnotation` range with that name. For CUDA use `torch.cuda.synchronize`.
For JAX supply a callback that blocks on the phase's current device outputs.
An empty callback is appropriate only for synchronous CPU work.

```python
with run.profile(trace_directory):
    with run.phase("input-transfer", synchronize=synchronize_device):
        your_input_transfer()
```

Choose the bottleneck from the trace, record a before run, change one thing, and
record an after run. Select `profile_before`, `profile_after`, and the exact
`bottleneck` name in the results config. The comparison reports mean milliseconds
per named phase before and after. It only accepts phase samples linked to a
successfully captured profile. Both runs need matching backend, GPU type,
workload, and world size. Their implementation variants may differ. This is a
coordinator phase measurement, not a claim about all-rank critical-path time.
Keep batch size, warmup, and sample window equal, and inspect both saved traces.

## Slurm requeue and recovery

[`templates/slurm.sbatch`](../templates/slurm.sbatch) requests `--requeue` and a
120-second USR1 notice to job steps. It never runs without your explicit
submission. Adapt its partition, GPU type, topology, time limit, paths, and
rendezvous configuration to your site. Slurm and GCP must actually support
requeue; the template cannot enable that infrastructure policy.

`signals()` installs SIGTERM and SIGUSR1 handlers on the main thread and restores
the previous handlers on exit. A handler only latches the first signal.
`poll_preemption(save, *, agree=bool)` runs at the next safe point. It logs the
signal, saves a checkpoint, records a requeue request, and exits with code 75.
The batch script interprets only code 75 as a request for `scontrol requeue`.
Other failures remain failures. The default `agree=bool` is for one worker only.
Use a collective OR for distributed jobs. Do not swallow `SystemExit(75)`.

Allow more notice than a worst-case step plus checkpoint takes. SIGKILL, a dead
VM, or a network stall offers no safe checkpoint guarantee. SIGTERM handling
requires an actual grace period. Slurm can also requeue through its configured
preemption policy. The log distinguishes a request from a proven requeue.

A recovery sample requires a preemption event followed by a different attempt,
the same nonempty Slurm job ID, an increased `SLURM_RESTART_COUNT`, a successful
restore, and the first completed post-resume step. Recovery minutes run from
signal receipt to that step's `on_step` timestamp, including checkpoint, queue,
startup, restore, and first-step time. A local process restart alone is not a
real Slurm measurement.

Select the recovery run IDs in `recovery`. Use one run with three requeues or
three comparable trials. The report uses the latest three successful recovery
samples for the median. Fewer than three leaves the median unmeasured. It counts
all matched recoveries as preemptions survived. Max steps lost is the largest
`max(0, completed_step_before_restart - restored_step)` across those recoveries.
Failed saves can therefore expose rollback to an older acknowledged checkpoint.
Zero lost steps is possible after a successful signal checkpoint. These counts
cover logged completed steps, not an in-flight optimizer step or unseen host loss.

## Scaling and prices

Run the training loop separately at each GPU count and select run IDs through
`config/results.json`. The report computes `tps_N / (N * tps_1)`; a missing baseline
stays unmeasured. The 1-GPU comparison is an instrumented baseline, not a theoretical
GPU specification. Scaling comparisons require matching workload, variant, GPU
type, backend, and git revision. A logical run must keep its git revision across
restarts; use a new run ID for a code change.

Cost selects the `ddp_8` run. It sums completed attempt intervals from
`attempt_start` to `attempt_end`, including failed or preempted attempts, then
multiplies wall hours by GPU count and `usd_per_gpu_hour`. Requeue downtime is
excluded because no attempt owns GPUs then. Startup before the hook, teardown
after it, idle allocation time outside it, CPU, memory, storage, and networking
are excluded. An unfinished attempt makes the full-run estimate unmeasured.

Each GPU key in `config/prices.json` records a nullable hourly USD rate, a source
URL, `as_of` quote date, region, and purchase model. Replace nulls using the
actual region and Spot SKU. The renderer requires a positive rate, source, and
valid date. It displays the quote source and date alongside the estimate.
Rates are not fetched or refreshed automatically. All attempts in a logical run
use that one quoted rate, so the result remains an estimate even for variable
Spot prices. No rates are fabricated in the committed config.

## Offline verification limits

`uv run python -m unittest discover -s tests -v` exercises the scalar dummy model,
append-only restore, signal latching, checkpoint failure, repeated requeues,
new-process resume, weighted throughput, price math, backend labels, profiler
hooks, and nine-row rendering. Synthetic durations and prices
exist only in tests. Actual DDP reductions, Ray retries, JAX device completion,
GPU traces, and Slurm site policy require the caller's real training environment.
No CPU test result is evidence of tokens per second on a GPU.
