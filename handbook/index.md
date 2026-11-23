---
hide:
  - toc
---

# Training Alignment Handbook

How to make two training or inference runs agree bit for bit, and how to find
the first place where they stop agreeing.

## The three workflows

The handbook teaches three workflows. Two of them start from a symptom. In
the first, two pre-training runs that should agree do not (part 1). In the
second, the rollout engine and the trainer give different log-probabilities
for the same tokens (part 2). Both can end at an operator whose result depends
on the other rows in its batch. When that happens, they hand the operator to
the third workflow, which makes it batch-invariant (part 3).
[Part 0](0-foundations/README.md) teaches the ideas that all three rest on.

<figure class="flows" aria-label="The three workflows" markdown>
<p class="flows-base" markdown>All three build on [Part 0: Foundations](0-foundations/README.md), which defines the invariance levels L1–L5 and explains where and how to compare two runs.</p>
<div class="flow-lane" data-lane="pre" data-side="above" markdown>
<p class="flow-head" markdown><span class="flow-part">Part 1</span> [Pre-training](1-pretraining/README.md) <span class="flow-goal">Find where two runs stop agreeing</span></p>
<ol class="flow-steps" style="--steps: 7" markdown>
<li markdown="span">[Preflight](1-pretraining/determinism-and-preflight.md)<span class="flow-desc">Turn on the settings that make a run repeatable, and check that both runs really use them.</span></li>
<li markdown="span">[Data alignment](1-pretraining/data-alignment.md)<span class="flow-desc">Check that every rank of both runs gets the same batches at every step.</span></li>
<li class="flow-handoff" markdown="span">[Localize with OpGuard](1-pretraining/localize-with-opguard.md)<span class="flow-desc">Capture both runs, find the first operator where they differ, trace back from it to the cause, then fix the cause and confirm the fix.</span></li>
<li markdown="span">[Training step](1-pretraining/training-step.md)<span class="flow-desc">If the cause lies in the backward pass, the optimizer, the loss or a resume, make that part repeat bit for bit.</span></li>
<li markdown="span">[Data parallel](1-pretraining/data-parallel.md)<span class="flow-desc">If the cause is a gradient that ranks sum together or shard, make the sum independent of how the work is split.</span></li>
<li markdown="span">[Compiled code](1-pretraining/capture-compiled-code.md)<span class="flow-desc">If the job runs `torch.compile` or CUDA graphs, take fingerprints inside the compiled code, which an eager capture cannot see.</span></li>
<li markdown="span">[SDC](1-pretraining/sdc.md)<span class="flow-desc">Once software is ruled out, find the hardware that silently returns wrong values.</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="kernel" data-delay="1" data-target markdown>
<p class="flow-head" markdown><span class="flow-part">Part 3</span> [Batch-invariant kernels](3-kernels/README.md) <span class="flow-goal">Make one operator batch-invariant</span></p>
<ol class="flow-steps" markdown>
<li markdown="span">[GEMM and quantized GEMM](3-kernels/gemm-and-quantized.md)<span class="flow-desc">Use the same tile and the same reduction order at every batch size.</span></li>
<li markdown="span">[Attention](3-kernels/attention.md)<span class="flow-desc">Split the keys the same way in prefill, in decode and in the backward pass.</span></li>
<li markdown="span">[MLA and sparse attention](3-kernels/mla-and-sparse.md)<span class="flow-desc">Do the same when the keys are compressed or picked by top-k selection.</span></li>
<li markdown="span">[Linear attention](3-kernels/linear-attention.md)<span class="flow-desc">Update the recurrent state in the same chunk order every time.</span></li>
<li markdown="span">[MoE](3-kernels/moe.md)<span class="flow-desc">Make routing, the expert GEMMs and the combine independent of which tokens share a batch.</span></li>
<li markdown="span">[Collectives](3-kernels/collectives.md)<span class="flow-desc">Add up the partial results of all GPUs in one fixed order.</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="rl" data-side="below" markdown>
<p class="flow-head" markdown><span class="flow-part">Part 2</span> [RL train–inference](2-rl/README.md) <span class="flow-goal">Make the rollout engine and the trainer agree</span></p>
<ol class="flow-steps" markdown>
<li class="flow-handoff" markdown="span">[Whole model](2-rl/whole-model.md)<span class="flow-desc">Make one side give a sequence the same logits whether it runs alone, in a batch or in any serving mode.</span></li>
<li markdown="span">[Parallelism](2-rl/parallelism.md)<span class="flow-desc">Keep that true with tensor parallelism, and when prefill and decode run on separate servers.</span></li>
<li markdown="span">[Sampling](2-rl/sampling.md)<span class="flow-desc">Make sure the sampled tokens can be reproduced from the logits and a seed.</span></li>
<li markdown="span">[LoRA](2-rl/lora.md)<span class="flow-desc">Keep each adapter's result unchanged when one batch mixes several adapters.</span></li>
<li class="flow-handoff" markdown="span">[Engine parity](2-rl/engine-parity-vllm.md)<span class="flow-desc">Make a real inference engine round at the same points as the trainer, so that the two sides agree.</span></li>
<li markdown="span">[RL loop](2-rl/rl-loop.md)<span class="flow-desc">Record each rollout, tell staleness apart from a real mismatch, and localize any gap offline.</span></li>
<li markdown="span">[Walkthrough](2-rl/walkthrough-verl.md)<span class="flow-desc">Follow every step on one real job: verl with FSDP2 and a vLLM rollout.</span></li>
</ol>
</div>
<p class="flows-next" markdown>Afterwards, [Part 4: Operations](4-operations/README.md) checks the results again whenever the software stack or the hardware changes.</p>
<figcaption><span class="flow-key"></span>A step drawn with a dashed border hands the operator it found to part 3. Hover over a step, or give it keyboard focus, to read what it does.</figcaption>
</figure>

Not sure where to start? The [alignment-pipeline skill](../skills/alignment-pipeline/SKILL.md)
asks you a few questions and sends you to the right workflow. The terms used
throughout the handbook are defined in the [glossary](glossary.md).

## Part 1: when two pre-training runs stop agreeing

Start with the [overview](1-pretraining/README.md). The agent skill for this
part is [pretraining-alignment](../skills/pretraining-alignment/SKILL.md).

1. [Determinism and preflight](1-pretraining/determinism-and-preflight.md):
   turn on the settings that make a run repeatable, and check that both runs
   really use them.
2. [Data alignment](1-pretraining/data-alignment.md): check that every rank of
   both runs gets the same batches at every step.
3. [Localize with OpGuard](1-pretraining/localize-with-opguard.md): capture
   both runs, find the first operator where they differ, and trace back from
   it to the cause. Then fix the cause and confirm the fix.

After that, read the page for the place the first divergence points to:

- [Training step](1-pretraining/training-step.md) shows how to make the
  backward pass, the optimizer, the loss and a resume from a checkpoint
  repeat bit for bit.
- [Data parallel](1-pretraining/data-parallel.md) shows how to keep gradients
  equal when ranks sum them together or shard them.
- [Compiled code and CUDA graphs](1-pretraining/capture-compiled-code.md)
  shows how to take fingerprints inside `torch.compile` and CUDA graphs, where
  an eager capture cannot see.
- [SDC](1-pretraining/sdc.md) shows how to find hardware that silently returns
  wrong values.

## Part 2: when the rollout engine and the trainer disagree

Start with the [overview](2-rl/README.md). The agent skills for this part are
[engine-parity](../skills/engine-parity/SKILL.md) and OpGuard's `rl-alignment`.

1. [Whole model](2-rl/whole-model.md): make one side give a sequence the same
   logits whether it runs alone, in a batch or in any serving mode.
2. [Parallelism](2-rl/parallelism.md): keep that true with tensor parallelism,
   and when prefill and decode run on separate servers.
3. [Sampling](2-rl/sampling.md): make sure the sampled tokens can be
   reproduced from the logits and a seed.
4. [LoRA](2-rl/lora.md): keep each adapter's result unchanged when one batch
   mixes several adapters.
5. [Engine parity](2-rl/engine-parity-vllm.md): make a real inference engine
   round at the same points as the trainer, so that the two sides agree.
6. [RL loop](2-rl/rl-loop.md): record each rollout, tell staleness apart from
   a real mismatch, and localize any gap offline.
7. [Walkthrough](2-rl/walkthrough-verl.md): follow every step on one real
   job, verl with FSDP2 and a vLLM rollout.

## Part 3: making one operator batch-invariant

Start with [the standard](3-kernels/README.md) that every invariant operator
must meet. The agent skill for this part is
[batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md).

1. [GEMM and quantized GEMM](3-kernels/gemm-and-quantized.md): use the same
   tile and the same reduction order at every batch size.
2. [Attention](3-kernels/attention.md): split the keys the same way in
   prefill, in decode and in the backward pass.
3. [MLA and sparse attention](3-kernels/mla-and-sparse.md): do the same when
   the keys are compressed or picked by top-k selection.
4. [Linear attention](3-kernels/linear-attention.md): update the recurrent
   state in the same chunk order every time.
5. [MoE](3-kernels/moe.md): make routing, the expert GEMMs and the combine
   independent of which tokens share a batch.
6. [Collectives](3-kernels/collectives.md): add up the partial results of all
   GPUs in one fixed order.

## Part 5: the same method in other scenarios

The same method also works for other pairs of runs. See the
[overview](5-scenarios/README.md), and the pages on
[framework migration](5-scenarios/framework-migration.md),
[checkpoint conversion](5-scenarios/checkpoint-conversion.md) and
[offline post-training](5-scenarios/offline-post-training.md).

## Part 4, the skills and the reference pages

- [Part 4: Operations](4-operations/README.md) shows how to check the results
  again after a [stack change](4-operations/regression.md) and on
  [new hardware](4-operations/hardware.md).
- The skills are [alignment-pipeline](../skills/alignment-pipeline/SKILL.md),
  [pretraining-alignment](../skills/pretraining-alignment/SKILL.md),
  [engine-parity](../skills/engine-parity/SKILL.md),
  [batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md) and
  [handbook-writing](../skills/handbook-writing/SKILL.md).
- Reference pages: [References](references.md) ·
  [Glossary](glossary.md) · [Audit](../AUDIT.md) · [Tools](../tools/) ·
  [Acknowledgments](acknowledgments.md)

Every claim on a page links to the report that holds its measurements and
names the stack it was measured on.

## Citation {#citation}

If this handbook, its tools or its skills help your work, please cite it:

```bibtex
@misc{zhou2026trainingalignment,
  title        = {Training Alignment Handbook},
  author       = {Zhou, Ziming and Xu, Peiyu},
  year         = {2026},
  howpublished = {\url{https://training-alignment.github.io/training-alignment-handbook/}},
  note         = {Code and reports: \url{https://github.com/training-alignment/training-alignment-handbook}}
}
```
