---
name: alignment-pipeline
description: Decide which workflow a bitwise-alignment problem needs and run it from start to finish. For pre-training (two runs that should agree, a loss spike after a change, a resume that drifts, suspected SDC) the steps are preflight, choosing a reference, capturing with OpGuard or graph probes, finding the first divergence, fixing, confirming and adding a regression entry. For RL train–inference (rollout and trainer log-probs differ) the steps are auditing the model, making each op batch-invariant, checking the whole model at L1–L3 and L5, checking sampling, checking L4 against the engine, recording online, splitting the gap into staleness and numeric error, localizing offline and adding a regression entry. Use this skill first; it calls the more specific skills.
---

# Alignment pipeline

This skill only routes the work. For each step it says when to start, which
tool or skill does the work, what evidence the step must leave behind, and
when to stop. The work itself and its rules are in the linked skills and
handbook parts, so they are not repeated here.

Equality always means bitwise equality. There is no tolerance, and nothing is
rounded to force two results to agree. Every claim names the GPU and the
software versions it was shown on, or says "untested".

## Which workflow to run

| What the user has | Workflow |
|---|---|
| Two training runs that should agree, or a spike or drift that appeared after an upgrade, a migration, a config change or a resume | A. Pre-training |
| A machine that may be silently corrupting data | A, step A0 only |
| Rollout log-probs that differ from the trainer's, or an RL run that has silently gone off-policy | B. RL train–inference |
| A single operator that has no batch-invariant implementation | Go straight to [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) |
| A framework migration, a checkpoint conversion, precomputed reference log-probs, or another pair of runs that should agree | [Part 5](../../handbook/5-scenarios/README.md); for conversions, [tools/ckpt_compare.py](../../tools/ckpt_compare.py) |
| A new stack (torch, Triton, NCCL, driver, CUDA, GPU) | Run the suite first, as described in [regression](../../handbook/4-operations/regression.md) |

Before you start either workflow on a new machine or stack, look up what has
been measured there in [hardware coverage](../../handbook/4-operations/hardware.md).

## A. Pre-training

| Step | Start when | Tool or skill | Evidence to leave | Stop when |
|---|---|---|---|---|
| A0. SDC | A device is suspected, or a long run is about to start | [tools/sdc_scan.py](../../tools/sdc_scan.py) (`--quick` first, then fleet and golden); inside the job, `sdc_guard` and `cross_rank_check` ([case](../../handbook/1-pretraining/sdc.md)) | The scan's result JSON, which names the device, the op and the repeat | A device disagrees with the rest of its group (rule out software differences first), or every device passes |
| A1. Preflight | Always, on both runs | [tools/determinism_preflight.py](../../tools/determinism_preflight.py): run `report` on every rank after the restore, then `diff(a, b, declared=[...])` | Both reports and the diff, in which only the declared difference remains | An undeclared difference shows up that makes a bitwise comparison meaningless; fix the setting and run again |
| A1'. Data | The preflight is clean | [tools/data_fingerprint.py](../../tools/data_fingerprint.py): on both runs, call `Recorder(dir, rank)` and `record(step, batch)` where batches leave the loader, then run `compare` (with `--global` when the DP sizes differ) ([page](../../handbook/1-pretraining/data-alignment.md)) | Both record directories and the compare result | A record differs; fix it according to its class (order, samples, packing, masks) and compare again |
| A2. Reference | The preflight is clean | [Part 1, step 1](../../handbook/1-pretraining/README.md#step-1-pick-a-reference-run) | A written note of the one difference under test | — |
| A3. Capture | The run is eager | Capture with OpGuard through [tools/opguard_bridge.py](../../tools/opguard_bridge.py) `run_pair`, then check `coverage(run)` | Both traces and the coverage report | The coverage misses the region you care about; fix the wrapping first |
| A3'. Capture | The run uses `torch.compile` or CUDA graphs | [tools/graph_probe.py](../../tools/graph_probe.py) with `isolate=True` ([case](../../handbook/1-pretraining/capture-compiled-code.md)) | The probe traces, and proof that compiling with and without the probes gives the same bits | The probes change the compiled bits; isolate them |
| A4. First divergence | You have two traces | `opguard_bridge.first_divergence` | The facts at the pivot: the operator, the source line, the call, the fields that differ, and any unaligned events before it | No event differs and the final parameters are equal, which means the runs agree |
| A5. Cause | You have a pivot | [pretraining-alignment](../pretraining-alignment/SKILL.md) (steps 5–6) together with OpGuard's harness skills; `suggest()` gives leads | A hypothesis, tested with the smallest intervention that can confirm it | The intervention removes the difference at the pivot |
| A6. Fix | The cause is confirmed | For a setting or a piece of state, [training step](../../handbook/1-pretraining/training-step.md) and [DP and FSDP-style sharding (emulated: flat gradient, reduce-scatter, sharded AdamW)](../../handbook/1-pretraining/data-parallel.md); for a kernel, [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | The change and its cost (measured after a warm-up, with synchronization, as a median) | — |
| A7. Confirm | A fix is in place | Capture the pair of runs again | No event differs, none is unaligned, and the final parameters are bit-identical | The fix is confirmed; otherwise go back to A4 |
| A8. Regression | The fix is confirmed | An example that takes `--quick`/`--json` and reports in the `bi-results-v1` format, plus one entry in [suite/manifest.json](../../suite/manifest.json) ([regression](../../handbook/4-operations/regression.md#how-to-add-a-check-or-an-example)) | The example, its result file and a baseline | The suite passes on the stack |

## B. RL train–inference

The order matters, because each level assumes that the levels before it hold.
Skip a step only when its start condition is false, and say that you skipped
it.

| Step | Start when | Tool or skill | Evidence to leave | Stop when |
|---|---|---|---|---|
| B1. Audit | You have the trainer's model | Run [tools/bi_audit.py](../../tools/bi_audit.py) on the model | The operator inventory and an L2 verdict for each module | Every module is L2, or the modules that are not are listed |
| B2. Per-op L2 | An operator is not L2 | [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | The operator's verification matrix and its cost | Every operator passes or has a documented fallback |
| B3. Whole model L1–L3 | Every op is L2 | The method of the [whole model case](../../handbook/2-rl/whole-model.md), in every serving mode and including activation gradients | Logprob equality in each mode | L1–L3 hold in every mode the engine uses |
| B4. L5 | The trainer and the engine use different TP, DP or EP degrees | [whole-model-tp](../../handbook/2-rl/parallelism.md), [training-dp-fsdp](../../handbook/1-pretraining/data-parallel.md), recipe 10 | Equality across degrees, in the forward and the backward | Results are equal at every degree in use |
| B5. Sampling | Tokens are sampled, replayed, preempted or speculated | [tools/bi_sampling.py](../../tools/bi_sampling.py) ([case](../../handbook/2-rl/sampling.md)) | Replay checks after recompute and rollback, with other values of n, with speculative chunks and in a new process | Tokens replay bitwise |
| B6. L4 | B3 holds on each side | [engine-parity](../engine-parity/SKILL.md) ([vLLM case](../../handbook/2-rl/engine-parity-vllm.md)); for LoRA, the [LoRA case](../../handbook/2-rl/lora.md) | Both rounding sequences, and equality per op and per layer | 0 logprobs differ, or every remaining difference is a recorded decision with its cost |
| B7. Online records | The RL job is running | OpGuard's `opguard_harness.rl_records.RecordWriter`, which records rollout batches, trainer log-probs, forwards, weight syncs and snapshots ([part 2](../../handbook/2-rl/README.md#synchronous-and-asynchronous-runs)) | Record files for each rank | — |
| B8. Split | You have records | Run `rl.weight_sync_check` first, then `rl.decompose` | The unmatched weight rows (there should be none), `numeric` and `staleness` for each token, and the tokens generated under mixed weight versions set aside | `numeric` is exactly zero, bit for bit; the infrastructure is cleared and the whole gap is staleness |
| B9. Localize | `numeric` ≠ 0 | OpGuard's `rl-alignment` skill (`harness/skills/rl-alignment`): `rl.replay_request`, a recapture with `OPGUARD_XOR_ROWS=1`, then `vectors.*`; see the [example](../../handbook/2-rl/rl-loop.md) | The divergent op with evidence from the other side, and its cause class | A cause has been tested with the smallest change; then go back to B2, B4, B5 or B6 for the fix |
| B10. Regression | A fix is in place | The same as A8 | The example and its baseline | The suite passes on the stack |

## When to stop and ask

- A fix would cost more than the user agreed to. Report the measured cost,
  together with the cheaper levels that already hold.
- A difference is algorithmic, meaning there are two exact forms of the same
  math. That is a choice, not a bug: report both forms and their costs.
- The engine or the kernel is closed and has no knob. Report the fallback
  options (batch-invariant-kernel, step 2).
- A required regression check fails on a new stack. That is a finding to
  report, not something to tune away.
