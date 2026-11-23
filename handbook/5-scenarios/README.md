# 5. Other alignment scenarios

## Where we are

Parts 1 to 4 taught one method through its workflows: finding where two
pre-training runs part ([part 1](../1-pretraining/README.md)), making a rollout
engine and a trainer agree ([part 2](../2-rl/README.md)), making one operator
invariant ([part 3](../3-kernels/README.md)), and keeping the results true when
the stack changes ([part 4](../4-operations/README.md)). Many other needs are
the same question asked about a different pair of programs: a framework
migration, a checkpoint conversion, or a reference model run in an inference
engine. This part shows how to apply the method you already have to seven such
scenarios. Three of them have their own page, and the other four get a short
section on this page.

## The idea: say what you compare, on which inputs, and what should hold

Every scenario has four parts: a reference, a suspect, the inputs that both
receive, and a claim about which outputs should be equal and how strictly.
Once you have written those four down, one of the existing workflows applies.
Here are a few pairs:

| Pair | Same inputs | What should be equal | What to expect |
|---|---|---|---|
| A checkpoint and its converted copy | none, since nothing is computed | every row of every weight | bitwise equality, always |
| One model in two training frameworks | the data, the weights and the seed | every boundary that both have | bitwise equality, but only after every rounding point is aligned |
| A trainer and a precomputed reference logprob | the same token ids and weights | each token's logprob | bitwise equality once you have engine parity |
| One request sent alone, and the same request inside a busy server | the request | its tokens and logprobs | bitwise equality once serving is batch-invariant |
| The same job on two GPU generations | everything | nothing at the bit level | the bits will differ, so localize instead |

The last row matters as much as the others. Where bitwise equality is not
promised, the goal is to find the first operator whose bits differ and make a
decision about it, not to reach equality at any cost. So the rule is: **before
comparing anything, state which two programs you compare, on which inputs, and
what you claim should hold.** This is the controlled experiment of
[part 1](../1-pretraining/README.md#the-idea-two-runs-one-difference), with the
claim written out explicitly.

## The scenarios

| Scenario | Workflow to reuse | Tools and skills | What is specific to it |
|---|---|---|---|
| [Framework migration and library upgrades](framework-migration.md) | [part 1](../1-pretraining/README.md), with the other stack as the reference, plus the rounding-point method of [engine parity](../2-rl/engine-parity-vllm.md) | [determinism_preflight.py](../../tools/determinism_preflight.py), [opguard_bridge.py](../../tools/opguard_bridge.py), [engine-parity](../../skills/engine-parity/SKILL.md) | The two programs' traces do not line up, and each difference is either a bug or a choice |
| [Checkpoint conversion and resharding](checkpoint-conversion.md) | none, since nothing is computed: first a copy check, then one forward | [ckpt_compare.py](../../tools/ckpt_compare.py) | Each row keeps its values when the tensor is repacked; whether rows ended up in the right place needs a forward |
| [Offline post-training](offline-post-training.md): reference logprobs for SFT and DPO, teacher logits for distillation | [part 2](../2-rl/README.md) at L4, without sampling | [engine-parity](../../skills/engine-parity/SKILL.md), OpGuard's RL records | Which tokens are scored, chat templates, padding |
| [Reward models and evaluation](#reward-models-and-evaluation) | [whole model](../2-rl/whole-model.md), [sampling](../2-rl/sampling.md), [regression](../4-operations/regression.md) | [bi_audit.py](../../tools/bi_audit.py), [bi_sampling.py](../../tools/bi_sampling.py) | When a score is near a threshold, one bit can change the decision |
| [Deterministic serving](#deterministic-serving) | [whole model](../2-rl/whole-model.md) on the engine side; [part 3](../3-kernels/README.md) | [bi_audit.py](../../tools/bi_audit.py), [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md) | Every serving mode a request can run into |
| [Speculative decoding and MTP](#speculative-decoding-and-mtp) | [sampling](../2-rl/sampling.md), [whole model](../2-rl/whole-model.md) at L3 | [bi_sampling.py](../../tools/bi_sampling.py) | Verification processes a chunk of tokens, while decoding processes one |
| [Hardware migration and mixed fleets](#hardware-migration-and-mixed-fleets) | [hardware](../4-operations/hardware.md), [localize with OpGuard](../1-pretraining/localize-with-opguard.md) | [sdc_scan.py](../../tools/sdc_scan.py), [bi_suite.py](../../tools/bi_suite.py) | Bitwise equality is not expected across GPU generations |

## Reward models and evaluation

Here you want a reward model to give a (prompt, response) pair the same score
in any batch, and an evaluation to give the same number when it is run again.
A reward model is a whole model with a scalar head, so its output depends on
the batch and the mode in the same ways as a policy's does. What differs is how
the output is used. A score is compared with a threshold, ranked against other
responses, or averaged into a pass rate. A score one bit away from a threshold
can therefore change a decision, and a changed decision changes the training
signal or the number you report.

How to apply the existing pages:

1. **Make the reward model invariant** like any other model. Audit its operators
   with [bi_audit.py](../../tools/bi_audit.py), and compare one pair scored
   alone with the same pair scored inside batches, padded and packed
   ([whole model](../2-rl/whole-model.md)).
2. **Match the engine, if there is one.** If the score is computed in an
   engine and recomputed by the trainer, you are at L4: follow [engine parity](../2-rl/engine-parity-vllm.md).
3. **Make sampling replayable.** If the evaluation samples, make its tokens replayable from the logits and a
   seed ([sampling](../2-rl/sampling.md), [bi_sampling.py](../../tools/bi_sampling.py)).
4. **Keep a fixed evaluation set** and its scores as a
   [regression](../4-operations/regression.md) entry.

You are done when every pair gets the same score bits at every batch size, and
a rerun with a different batch composition gives the same evaluation result.

Not yet shown: no reward model or evaluation harness has been measured here.

## Deterministic serving

Here you want a request's tokens and logprobs to depend neither on which other
requests share its batch (L2) nor on how the server chunks, caches or decodes
it (L3). This is the engine side of the [whole-model step](../2-rl/whole-model.md), as
a user experiences it: two identical requests at temperature 0 get different
answers because the server happened to be busy in different ways. The causes
and the fixes are the same.

How to apply the existing pages:

1. **Audit the served model**, and test every mode a request can run into: a
   decode batch that keeps changing, chunked prefill, a prefix-cache hit, and
   preemption ([whole model](../2-rl/whole-model.md)).
2. **Replace every operator that varies** with the batch by an invariant kernel
   ([part 3](../3-kernels/README.md)), and keep the sampler replayable
   ([sampling](../2-rl/sampling.md)).
3. **Measure the cost** at your decode batch sizes. If it is too high, offer the
   invariant path only to the requests that need it, not to all of them.

You are done when each request's logprobs are bit-equal whether it is served
alone or under load, in every mode the server uses.

Not yet shown: a production server under real traffic, and serving across
several nodes.

## Speculative decoding and MTP

Here you want a decoder that uses speculation or multi-token prediction (MTP)
to emit exactly the tokens and logprobs that the target model emits on its own.
Verification runs the target model over several draft tokens at once, much
like a short prefill, while plain decoding runs one token at a time. If the
two paths round differently, an accepted token's logprob differs from the one
plain decoding would give, and the acceptance test itself can flip. The draft
model's own numerics affect only speed, not the result, as long as acceptance
compares against the target model's own sample.

How to apply the existing pages:

1. **Make the target model's verification forward equal its decode forward.** A
   chunk against a single token is the L3 case of the
   [whole model](../2-rl/whole-model.md), and attention must split the keys
   the same way in prefill and decode ([attention](../3-kernels/attention.md)).
2. **Make acceptance replayable.** When drafts are verified by exact equality,
   speculative decoding gives the same tokens as decoding without a draft
   ([sampling](../2-rl/sampling.md), [bi_sampling.py](../../tools/bi_sampling.py)).
3. **Treat MTP heads that serve as drafts in the same way.** An MTP loss used in
   pre-training belongs to the training step instead
   ([part 1](../1-pretraining/README.md)).

You are done when the tokens and the logprob bits are the same with and
without speculation, at every draft length.

Not yet shown: a real engine's speculative decoding, and MTP heads.

## Hardware migration and mixed fleets

Here you want to know which results carry over to new GPUs, and, when two
machines disagree, to find where they part. Bitwise equality is not expected across GPU generations, because libraries
choose different kernels and accumulation orders for each architecture
([hardware](../4-operations/hardware.md)). The goal therefore changes from
equality to localization. The first operator whose bits differ should be a
library GEMM or a reduction whose order the library chooses; anything else is
a finding.

How to apply the existing pages:

1. **Qualify the new hardware**: run the regression suite against the committed
   baselines, and run the SDC scan across groups of devices
   ([hardware](../4-operations/hardware.md),
   [bi_suite.py](../../tools/bi_suite.py), [sdc_scan.py](../../tools/sdc_scan.py)).
2. **Compare devices bit for bit** within one GPU generation and one
   software stack. A disagreement there is a fault ([SDC](../1-pretraining/sdc.md)).
3. **Localize across generations.** Capture both runs with OpGuard on the same inputs and
   take the first divergence ([localize with OpGuard](../1-pretraining/localize-with-opguard.md)).
   Then classify it: either a kernel choice that depends on the architecture,
   which is expected, or anything else, which you should report.
4. **Handle RL across generations.** With the trainer and the engine on
   different GPU generations, only
   kernels whose reduction order is written into their source code can carry
   over, and even that is untested across architectures. Otherwise, measure
   the numeric gap and record it as a decision, the way
   [engine parity](../2-rl/engine-parity-vllm.md) handles differences that
   neither side can remove.

You are done when every difference between generations is classified, and the
expected ones are recorded together with the stack they belong to.

Not yet shown: the handbook's kernels on other GPU architectures, and an RL
run that mixes GPU generations.

## What can go wrong

!!! warning "Expecting bitwise equality where none is promised"
    Two GPU generations, or the kernels of two different libraries, are not
    built to agree to the last bit. Chasing equality there spends your time on
    differences that are expected. State the claim first, and where the claim
    is not equality, localize and decide.

!!! warning "Letting a tolerance stand in for a bit check"
    Outputs that are "close enough" can hide faults that happen to be small.
    For example, a cast that rounds small weights can leave every logit
    unchanged on a few inputs ([checkpoint conversion](checkpoint-conversion.md)).
    Check copies and recomputations bit for bit, and use a tolerance only for
    pairs that cannot agree.

!!! warning "Comparing two different computations"
    A reference logprob scored over different tokens, a chat template applied
    on one side only, or a score that went through different post-processing
    all make the numbers differ although nothing is wrong numerically. Before
    comparing values, check that both sides computed the same thing.

## Summary

- Every scenario consists of a pair of programs, the inputs they both
  receive, and a claim. Write those down first.
- A copy must be bitwise equal. Two programs can be made bitwise equal by
  aligning their rounding points. Across two GPU generations you localize the
  differences instead of removing them.
- The existing workflows apply unchanged: preflight, capture, first
  divergence, rounding points, decision, regression.
- Each section states what is not yet shown; treat those items as open.

## Go deeper

- Pages: [framework migration](framework-migration.md),
  [checkpoint conversion](checkpoint-conversion.md),
  [offline post-training](offline-post-training.md).
- [Report: checkpoint conversion](../../examples/ckpt_conversion/REPORT.md),
  with the planted faults and what each check saw.
- Skills: [alignment-pipeline](../../skills/alignment-pipeline/SKILL.md),
  [engine-parity](../../skills/engine-parity/SKILL.md),
  [pretraining-alignment](../../skills/pretraining-alignment/SKILL.md).
