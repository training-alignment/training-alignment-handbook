# Matching vLLM bit for bit (L4)

## Where we are

After the previous steps, each side is invariant on its own: the
[whole model](whole-model.md) gives the same bits in every serving mode and at
every [parallel degree](parallelism.md), the [sampler](sampling.md) is
replayable, and [LoRA](lora.md) adapters round the same way in every batch. In
RL, though, the engine is usually not your code. It is vLLM or SGLang, with
its own model code, kernels and compiler. This step makes the trainer's
logprobs equal to the engine's, bit for bit (L4).

## The idea

Two sides that are each invariant can still disagree, because they round at
different places. A **rounding point** is any place where a value is cast to a
narrower type, or where several operations are grouped under a single
rounding. Here is a tiny example from RMSNorm, where x̂ is the normalized row
in fp32 and w is the bf16 weight:

| Side | What it computes | Roundings |
|---|---|---|
| A | bf16(x̂) · w, then cast to bf16 | two: first x̂ is cast, then the product |
| B | bf16(x̂ · w) | one: the product is formed in fp32 and then cast |

Both are correct, yet they give different bits on many elements. A model has
dozens of such points: how the projections are packed, where a bias is added,
the norm, the RoPE table and its formula, the attention kernel, the lm_head
and the log_softmax. Every point upstream of the logits has to match before a
single logprob becomes equal. That is why the number of differing tokens can
stay close to the total through several correct fixes and then drop to zero
with the last one.

The method follows from this:

1. **Check the engine against itself first.** If the engine's logprob for a
   token changes when other requests join the batch, no trainer can match it.
2. **Write down both rounding sequences**, one line per rounding point, in
   order.
3. **Align one rounding point at a time** and measure after each change.
   Watch how far into the model the first differing layer moves, not only how
   many tokens differ.
4. **Prefer calling the engine's own kernel** from the trainer to writing your
   own version. The same source code does not give the same bits unless the
   tile shapes, the reduction order and the compiler match as well.

The general principle is the same as for kernels: equal results need the same
arithmetic, not just the same formula.

## The step

The [engine-parity skill](../../skills/engine-parity/SKILL.md) runs these
actions as a workflow.

1. **Take an inventory of both forwards**, to know what actually runs on each
   side. From the source code and from the loaded model, list the packed and
   fused modules, the custom operators, whether a compiler and CUDA graphs are
   used, which attention backend was chosen, and every switch that changes
   kernels (skill section 1). Print the module tree and the backend from the
   running engine; do not copy them from the configuration you passed in.
2. **Write down both rounding sequences** in the table in
   [rounding points](../../skills/engine-parity/references/rounding-points.md),
   marking each entry as "read from source" or "measured". The result is a
   checklist in which every difference between the two sides is a line with a
   name.
3. **Check the engine against itself**, so that it reaches L1 and L2 before
   you compare it with any trainer. Run the same batch twice, and run each
   prompt alone and in batches of several sizes; if the engine has a
   batch-invariant mode, switch it on and read which operators that mode
   covers on your GPU. Every prompt must get the same logprobs alone and in a
   batch. If not, report it as variation inside the engine and fix it there.
4. **Align the trainer one rounding point at a time** until the logprobs of
   the sampled tokens reach L4. Feed the engine's own tokens through the
   trainer with teacher forcing, compare the fp32 logprob bits with no
   tolerance, change one rounding point, and count the differing tokens
   again; [trainer.py](../../examples/cross_engine/trainer.py) and
   [cross_engine.py](../../examples/cross_engine/cross_engine.py) serve as
   templates. You are done when no token differs; a leave-one-out pass then
   shows which points matter.
5. **Localize whatever still differs** down to the first operator that
   differs. Replay operators, feeding every captured engine input to each
   trainer candidate, and compare the residual stream layer by layer; for a
   compiled engine, use fingerprints at operator boundaries instead of hooks
   ([RL loop](rl-loop.md)). The result is a named operator, with evidence from
   both sides.
6. **Give the matched forward a correct backward.** Logprobs must match bit
   for bit, but gradients only need to be correct, so the backward does not
   have to reproduce the bits of the forward kernels. Give small operators
   hand-written fp32 derivatives, and recompute attention and MoE in PyTorch
   on the saved inputs to get their gradients, while the forward keeps the
   values of the engine's kernels. You are done when every parameter gradient
   passes relative L2 ≤ 1e-3 against an independent fp32 reference; the
   [report](../../examples/verl_fsdp2_vllm/REPORT.md#gradient-check) is a
   worked example.

| Approach | What it does | Cost class | When to choose it |
|---|---|---|---|
| Move the trainer onto the engine's kernels | The trainer calls the engine's own GEMM, norm, attention and log_softmax | moderate (under 2x) to large on the engine side, from its invariant mode; free for the trainer's forward | The engine is fixed and the trainer is yours; the trainer also needs a correct backward for these kernels, which does not have to reproduce the forward's bits |
| Move the engine onto the trainer's rounding | Unfuse and reorder the engine's operators so that its casts follow the trainer; use invariant operators on both sides | moderate (under 2x) to large on rollout throughput (reported) | The trainer is fixed and you can patch the engine |
| One model definition for both sides | The engine keeps only cache management and shares the forward, so only the kernels remain to be aligned | large (2x or more) on trainer throughput for some architectures (reported) | You can change both frameworks |
| An execution contract | Pin every choice that affects rounding on both sides, check it with a hash, and use parallel-invariant kernels | moderate (under 2x) (reported) | Several frameworks must stay aligned over time |

What has been shown: a trainer that calls the engine's own kernels matched a
real engine's logprobs for the sampled tokens bit for bit, across batch sizes.
This was shown on one model and one GPU type, in eager mode, for the forward
only ([report](../../examples/cross_engine/REPORT.md#results)). Inside a full
RL loop in a real framework, a sharded trainer that calls the engine's kernels
also matched a rollout running with CUDA graphs and the compiler off, bit for
bit over a short synchronous run, on a real MoE checkpoint with attention sinks
and sliding windows, with a backward that passes the gradient check on a
shortened version of the model; this was shown on one model and one GPU type,
in bf16, with each rollout replica on one GPU
([report](../../examples/verl_fsdp2_vllm/REPORT.md)). Results that others
reported for the other approaches are collected in the
[report](../../examples/cross_engine/REPORT.md#three-ways-to-reach-parity). Not
yet shown: an engine compiled by its compiler, tensor parallelism at L4, and
quantized weights.

## What can go wrong

!!! warning "The engine's batch-invariant mode may not cover every operator on every GPU"
    Depending on the GPU architecture, a mode flag may switch some operators
    to invariant kernels and leave others on a library GEMM. The engine then
    still varies with the batch, and you end up tuning the trainer against
    noise. Read the code path for your GPU, and check the engine's L2
    yourself.

!!! warning "Comparing the wrong logprob"
    An engine may report logprobs after applying temperature, penalties or
    other processing, or it may compute the log_softmax on a different device
    from yours. The comparison then shows a gap that is not in the model.
    First check that the reported logprob equals the log_softmax of the
    captured logits, computed the way the engine computes it.

!!! warning "Hooks do not fire inside compiled graphs"
    A compiler fuses norms, RoPE and activations into neighbouring operators,
    and the engine may turn its custom operators off when it compiles.
    Capture at the module level sees nothing there. Use fingerprints at
    operator boundaries, or match the engine in eager mode first.

!!! warning "An unchanged token count does not mean the fix failed"
    Every rounding point upstream of the logits must match before any token
    becomes equal. If you judge each fix by the token count, you will throw
    away correct fixes. Track the first differing layer after each change.

!!! warning "A revert that changes nothing proves little"
    In a leave-one-out pass, a point whose revert changes no token may still
    matter for other prompts or at later positions; a RoPE table, for example,
    can differ only beyond a certain position. Write "not shown to matter
    here", not "irrelevant".

!!! warning "A sharding wrapper can cast fp32 buffers"
    A mixed-precision or sharding wrapper may cast an fp32 table, such as the
    RoPE cos/sin table, to bf16 when the table crosses the boundary of a
    sharded module. Almost every logprob then differs, although each kernel
    is right. Pass positions across the boundary and build the table
    inside the layer, or check the dtypes of buffers at each boundary.

!!! warning "Padded vocabulary columns"
    The lm_head can have more columns than the tokenizer has tokens, and the
    engine may mask the extra columns before the log_softmax while the trainer
    does not, or the reverse. Most logprobs then differ. Normalize over the
    same columns as the engine.

## Summary

- Two sides that are each invariant can still disagree; they must also round
  at the same places.
- Check the engine against itself first. An engine that does not reach L2
  cannot be matched.
- Write down both rounding sequences, align one point at a time, and track the
  first layer that differs.
- Have the trainer call the engine's own kernels rather than rewriting them.
- Logprobs must match bit for bit, while the backward only needs correct
  gradients.

## Go deeper

- [Lab report](../../examples/cross_engine/REPORT.md): one real engine and
  model, the table of steps, the leave-one-out pass, the engine's variation
  against itself, which operators its invariant mode covers on which GPU, the
  cost, and results reported by other groups.
- [Full RL loop report](../../examples/verl_fsdp2_vllm/REPORT.md): a full RL
  loop with a sharded trainer, an MoE checkpoint, a CUDA-graph rollout and a
  gradient check; the [walkthrough](walkthrough-verl.md) follows it stage by
  stage.
- Skill: [engine-parity](../../skills/engine-parity/SKILL.md) and
  [rounding points](../../skills/engine-parity/references/rounding-points.md).
- Code: [examples/cross_engine/](../../examples/cross_engine/).
- Next step: [RL loop](rl-loop.md).
