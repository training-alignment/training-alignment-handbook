# 2. Train–inference alignment in RL

## Where we are

In RL, the rollout engine (vLLM, SGLang) samples tokens and reports their
logprobs, and the trainer (FSDP, Megatron, TorchTitan) recomputes the same
logprobs to form the policy-gradient ratio. Both sides run the same policy, but
with different code, kernels, batching, decode modes and parallelism. Their
logprobs therefore differ even when the weights are identical, and the run
quietly becomes off-policy although nobody chose that. This part teaches how
to make the two sides agree bit for bit, and how to find what broke when they
do not. It assumes you know the invariance levels L1–L5 from
[Part 0](../0-foundations/README.md).

Treat parity first as a tool for debugging and auditing: during a short window,
the infrastructure must produce a gap of exactly zero. Whether to keep it on in
production is a separate decision about cost.

## The workflow

<figure class="fx" data-fig="rl-workflow" aria-label="The RL train–inference workflow" markdown>
<div class="fx-stage" markdown>
<div class="fx-phases" style="--cols: minmax(0, 1fr) minmax(0, 0.75fr) minmax(0, 1.25fr)" markdown>
<div class="fx-phase" markdown="span"><span class="fx-label">Make each side invariant</span><span class="fx-seq is-col"><span class="fx-step" data-hot="0">[1. Audit the operators](whole-model.md)</span><span class="fx-step" data-hot="1">[2. The whole model](whole-model.md)<span class="fx-note">L1–L3</span></span><span class="fx-step" data-hot="2">[3. Parallel degrees](parallelism.md)<span class="fx-note">L5</span></span><span class="fx-step" data-hot="3">[4. Sampling](sampling.md)</span><span class="fx-step" data-hot="4">[5. LoRA](lora.md)</span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">Make the two sides agree</span><span class="fx-seq is-col"><span class="fx-step" data-hot="5 9">[6. Engine parity](engine-parity-vllm.md)<span class="fx-note">L4</span></span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">Watch the running loop</span><span class="fx-seq is-col"><span class="fx-step" data-hot="6">[7. Record online](rl-loop.md)</span><span class="fx-step" data-hot="7">[8. Split the gap](rl-loop.md)<span class="fx-note">into staleness and numeric error</span></span><span class="fx-fork is-pair"><span class="fx-leg" data-hot="8"><span class="fx-when">numeric ≠ 0</span><span class="fx-step to-bad" data-hot="8">[9. Localize offline](rl-loop.md)<span class="fx-note" data-hot="9">then back to step 6</span></span></span><span class="fx-leg" data-hot="10"><span class="fx-when">numeric = 0</span><span class="fx-step to-ok" data-hot="10">[10. Keep as a regression check](../4-operations/regression.md)</span></span></span></span></div>
</div>
</div>
<ol class="fx-say">
<li>1. Audit the operators and list the modules whose output depends on the batch or on the serving mode.</li>
<li>2. Make the whole model on one side give a sequence the same bits in every batch and every serving mode (L1–L3).</li>
<li>3. Get the same bits at every tensor-parallel degree, and when prefill and decode run on separate machines (L5).</li>
<li>4. Make every sampled token replayable, even after batching, preemption, n > 1 and speculative decoding.</li>
<li>5. Make LoRA adapters round the same way in a training batch and in a serving batch that mixes many adapters.</li>
<li>6. Make the trainer's logprobs equal to the engine's (L4).</li>
<li>7. Record the running loop online.</li>
<li>8. Split every token's gap into staleness and numeric error.</li>
<li>If numeric ≠ 0, step 9 localizes the error offline, down to one operator.</li>
<li>Then go back to step 6 with the cause it found.</li>
<li data-hold="2">If numeric = 0, step 10 keeps the result as a baseline that every new version must reproduce.</li>
</ol>
<figcaption>Figure: steps 1–5 make each side invariant on its own, step 6 makes the two sides agree, and steps 7–10 watch the running loop. Each step links to its page.</figcaption>
</figure>

The [walkthrough](walkthrough-verl.md) at the end of this part follows these
steps on one real job. The [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md) sends
a question to the step that answers it.

## The idea: make each side stable before comparing the two

Two logprobs can differ for two reasons. Either one side is unstable on its
own, or both sides are stable but round differently. Fix them in that order.
If the engine's logprob for a token changes when another request joins the
batch, no trainer can ever match it. That is why steps 2–5 make each side
invariant on its own, and only step 6 compares the two.

Some differences are not kernel bugs at all. They come from two forms of the
same mathematics that are both exact but round differently: chunked and
recurrent linear attention, absorbed and explicit MLA, split-KV decode and
single-pass prefill. Tuning a kernel cannot remove such a difference. Pick one
form, use it on both sides, and accept its cost;
[Part 3](../3-kernels/README.md) shows how to write the kernels.

## The idea: split the gap before you debug it

In an asynchronous loop, the rollout used policy version v, while the trainer
has already moved on to version c. The gap you observe mixes two things, and
the two can be separated exactly:

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

Here is a tiny example. A token was sampled at version 3, and the trainer is
now at version 5. Recompute the token's logprob on the trainer with the
version 3 weights, and compare:

| Token | logp_rollout(3) | logp_trainer(3) | logp_trainer(5) | gap | numeric | staleness |
|---|---|---|---|---|---|---|
| A | -1.20 | -1.20 | -1.35 | 0.15 | 0 | 0.15 |
| B | -1.20 | -1.21 | -1.35 | 0.15 | 0.01 | 0.14 |

Both tokens show the same gap. For token A, all of it is policy lag, which is
exactly what importance sampling is meant to correct, so the infrastructure is
cleared. For token B, part of the gap is an infrastructure fault that no
correction should absorb, and that part is worth localizing. Only `numeric` is
the infrastructure's to fix, and you can only measure it by replaying the
token at the rollout's own version.

## Synchronous and asynchronous runs

- **Synchronous runs (staleness 0).** Here `numeric` is the whole gap. With
  parity on both sides, the target is a difference of exactly 0 for every
  token over a short on-policy window. If the gap is 0 and training is still
  broken, the problem lies in the data or the algorithm.
- **Asynchronous runs (staleness > 0).** Record the loop online, split each
  token's gap, and localize only the tokens whose `numeric` is nonzero. A token
  generated across a weight update (with the KV cache kept) belongs to two
  versions at once and cannot be split; set such tokens aside.

## What can go wrong

!!! warning "Comparing the two sides before each one is stable"
    If the trainer is compared with an engine that is not batch-invariant,
    every change to the trainer looks partly right, because the engine's own
    variation is as large as the gap. Check the engine against itself first.

!!! warning "Correcting a numeric error as if it were staleness"
    Importance-sampling corrections absorb any gap, including one caused by an
    infrastructure fault. The fault then keeps training the policy without
    anyone noticing. Split the gap at the rollout's version before applying
    any correction.

!!! warning "Trusting the version label"
    A rollout batch labelled with one version may actually have run on the
    weights of another. Its numeric error then looks real but does not
    reproduce on replay. Record a fingerprint of the weights that each batch
    actually used.

## Summary

- First make each side invariant on its own (L1–L3, L5, sampling, LoRA), then
  make the two sides agree (L4).
- Some mismatches come from two exact forms of the same mathematics; choose
  one form and use it on both sides.
- Split every gap into staleness and numeric error. Only the numeric error is
  an infrastructure fault.
- Localize a nonzero numeric error offline by following the data flow, and
  once it is fixed, keep the case as a regression baseline.

## Go deeper

- Step pages: [whole model](whole-model.md), [parallelism](parallelism.md),
  [sampling](sampling.md), [LoRA](lora.md),
  [engine parity](engine-parity-vllm.md), [RL loop](rl-loop.md).
- [Walkthrough](walkthrough-verl.md): every step of this part on one real job,
  verl with FSDP2 and a vLLM rollout.
- Results reported by other groups (IsoExec, verl-omni, TorchTitan RL) and a
  worked example that localizes a single element: [report](../../examples/cross_engine/REPORT.md#context-moved-from-the-part-2-overview);
  the sources are listed in [references](../references.md).
- Skills: [alignment-pipeline](../../skills/alignment-pipeline/SKILL.md),
  [engine-parity](../../skills/engine-parity/SKILL.md),
  [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md);
  OpGuard's [`rl-alignment` skill](https://github.com/ziming-zh/OpGuard/tree/main/harness/skills/rl-alignment) (`harness/skills/rl-alignment` in the OpGuard
  repository) and its protocol, `docs/RLAlignment.md`.
