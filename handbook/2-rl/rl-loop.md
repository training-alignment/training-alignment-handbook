# Monitoring the RL loop

## Where we are

After [engine parity](engine-parity-vllm.md), the trainer and the engine agree
on a fixed set of prompts. A running RL loop adds things that test did not
have: new weights are pushed at every version, asynchronous rollouts run at
older versions, and the batch changes at every step. This step keeps watch
over the loop. You record cheaply while it runs, split every gap into
staleness and numeric error, and localize offline only the gaps that are real
infrastructure faults. These are steps 7–9 of the [RL workflow](README.md).

## The idea

As the [workflow page](README.md) showed, a gap splits exactly into two
parts:

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

Three conditions decide whether you can trust the split, and they also set the
order of the steps:

1. **The weights must be the ones the records claim.** If the engine holds a
   wrong row after a weight push, every comparison at that version is
   meaningless. Check the weights before looking at any logprob.
2. **Each token must belong to one version.** If new weights arrive while a
   sequence is still being generated and its KV cache is kept, the sequence
   contains tokens from two policies. It cannot be replayed under either of
   them, so set it aside.
3. **A divergence needs evidence from both sides.** When you follow a nonzero
   numeric error through the model, a row that has no partner on the other
   side is only a gap in the record, usually a step that the other side fused
   away. It is not a divergence. A divergence is the same input row giving a
   different output row on the two sides.

Localization follows the data flow, not the order of events in the trace.
Start from a point where both sides agree, such as the embedding output, and
follow each tensor to the operator that consumes it, until you reach the first
pair with equal inputs and different outputs.

<figure class="fx" data-fig="rl-triage" aria-label="Triage of a gap in a running RL loop">
<div class="fx-stage">
<div class="fx-dflow">
<span class="fx-node" data-hot="0">Records from the running loop</span>
<span class="fx-vl" data-hot="1"></span>
<span class="fx-node is-decision" data-hot="1">Do the weights match?</span><span class="fx-hl" data-hot="2">no</span><span class="fx-node to-bad" data-hot="2">Fix the weight sync and stop</span>
<span class="fx-vl" data-hot="3">yes</span>
<span class="fx-node" data-hot="3 4">Split each token's gap at the rollout's version<span class="fx-swap is-block" data-at="3"><span class="fx-split" data-to="4"><span class="is-whole">gap 0.15</span></span><span class="fx-split" data-at="4" style="--a:1;--b:14"><span class="is-a">numeric 0.01</span><span class="is-b">staleness 0.14</span></span></span></span>
<span class="fx-vl" data-hot="5"></span>
<span class="fx-node is-decision" data-hot="5">numeric = 0?</span><span class="fx-hl" data-hot="5">yes</span><span class="fx-node to-ok" data-hot="5">The infrastructure is cleared<span class="fx-note">the gap is policy lag only</span></span>
<span class="fx-vl" data-hot="6">no</span>
<span class="fx-node" data-hot="6">Recapture both sides and follow the data flow</span>
<span class="fx-vl" data-hot="7"></span>
<span class="fx-node is-key" data-hot="7">Classify the cause and test the smallest fix</span>
</div>
</div>
<ol class="fx-say">
<li>The records come from the running loop, with a fingerprint of the weights each batch used.</li>
<li>First ask whether the engine held the weights that the trainer had at that version.</li>
<li>If it did not, every comparison at that version is meaningless, so fix the weight sync and stop.</li>
<li>If it did, take each token's gap, here 0.15.</li>
<li data-hold="2">Replaying the token at the rollout's own version splits the gap exactly into a numeric error of 0.01 and a staleness of 0.14.</li>
<li data-hold="1.5">If the numeric error were 0, the whole gap would be policy lag and the infrastructure would be cleared.</li>
<li data-hold="1.5">Here it is 0.01, so recapture both sides and follow the data flow, not the order of the trace.</li>
<li data-hold="2">Finally, classify the cause and test the smallest fix that removes it.</li>
</ol>
<figcaption>Figure: check the weights first, then split every gap at the rollout's version, and localize only a numeric error that is not zero.</figcaption>
</figure>

## The step

OpGuard's RL harness implements every action below. Records are written with
`opguard_harness.rl_records.RecordWriter`. The actions (`rl.*` on record files
and `vectors.*` on captures) report facts only: they return
`verdict: not_evaluated` and apply no tolerance. The protocol and every field
are described in OpGuard's `docs/RLAlignment.md`, and the OpGuard skill
`rl-alignment` (`harness/skills/rl-alignment`) runs the steps as an agent
workflow.

1. **Record online** enough, at low cost, to split and replay any sampled
   batch later. `RecordWriter` writes five kinds of record: `rollout_batch`
   holds the token and prompt ids, the positions, each token's policy
   version, the bits of the rollout logprobs, the batch composition, and a
   fingerprint of the weights the engine held; `forward` holds each forward
   and its row layout; `weight_sync` holds per-row weight fingerprints from
   both sides at every sync; `snapshot` says where each version's weights are
   kept; and when staleness is zero, `trainer_logprob` is recorded as well.
   The summary of the records must list every type, and versions are
   snapshotted only at sampled intervals, so their cost stays bounded.
2. **Check the weights** to confirm that the engine ran the weights the
   trainer pushed. `rl.weight_sync_check` compares the row fingerprints as
   multisets, so a different packing, for example a fused QKV, does not count
   as a difference. Every compared version should be clean; otherwise the
   check names the faulty parameter and row on each side.
3. **Split the gap** into the numeric error and the staleness of every token.
   `rl.decompose` computes the split in exact arithmetic, flags tokens that
   span two versions, and reports which recorded weight fingerprint each
   batch matches. For older versions, `rl.replay_request` builds a replay (the
   snapshot of version v, the same tokens, and either the original batch
   composition or a batch-invariant one) for your own workload to run
   offline. The numeric error should be zero for every token that belongs to
   a single version; otherwise you get the list of tokens where it is not.
4. **Localize offline** to the first operator where the numeric error
   appears. Recapture both sides with per-vector fingerprints (OpGuard
   `OPGUARD_XOR_ROWS=1`), list the forwards with `rl.forwards`, and query
   with `vectors.describe`, `vectors.lookup`, `vectors.coverage` and
   `vectors.compare`, starting from an anchor where both sides agree and
   following the data flow. For an engine that runs compiled code or CUDA
   graphs, take the fingerprints with
   [tools/graph_probe.py](../../tools/graph_probe.py)
   (see [capturing torch.compile and CUDA graph code](../1-pretraining/capture-compiled-code.md)).
   You are done when you have a paired candidate: the same input row gives a
   different output row on the two sides.
5. **Classify the cause and test it.** Decide which class the cause belongs
   to: a rounding point or fusion order, a dependence on the batch, an
   algorithmic choice, the parallel layout, a different logprob formula, or
   version bookkeeping and weight sync. Then make the smallest change that
   brings the numeric error at that operator to zero, and replay; the cause
   is confirmed when the replay with the change equals the other side.
6. **Keep the fix** by adding the check, together with its baseline, to the
   [regression](../4-operations/regression.md) suite. Every later version
   must then reproduce the result.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Parity on, synchronous, for a short window | A gap of exactly 0 for every token, or a failure that you can localize | the cost of the parity configuration, for the window only | Before a long run, or to clear the infrastructure of suspicion |
| Records and fingerprints at every step, snapshots at sampled versions | The split for any recorded batch, computed offline | small (a few percent) without snapshots; snapshots grow with the model size | Every production run you may need to audit |
| Offline replay and recapture | The localization of one token's numeric error | seconds per localization, outside the training path | Only for tokens with a nonzero numeric error |

What has been shown: in a small loop with batch-invariant operator sets on
both sides, the numeric error was exactly zero, bit for bit, on every token
that belonged to a single version, in both synchronous and asynchronous runs,
so staleness accounted for the whole gap. The weight check named a planted bad
row even through a fused repacking. Planted code faults were localized to
their operators in a known-answer demonstration, whose data-flow order was
written from the model code rather than discovered
([report](../../examples/rl_loop/REPORT.md)). Not yet shown: a large model,
TP/EP on either side, CUDA graphs, and a production engine inside the loop.

## What can go wrong

!!! warning "Engine flags are only claims"
    The flags recorded with a rollout may say batch-invariant while a stock
    operator set actually ran. If you trust them, you search on the wrong
    side. Treat flags as claims and let the numeric split decide.

!!! warning "A wrong version label looks like a numeric error"
    A batch labelled with one version may have run on the weights of another.
    Online, its numeric error is nonzero, yet a replay at the recorded version
    equals the trainer and finds no differing row. Compare the batch's weight
    fingerprint with every synced version, and replay at the version that
    matches.

!!! warning "Pairing events by order fails silently when shapes differ"
    One trainer prefill and many engine decode forwards call functions a
    different number of times, so a diff that pairs events by their order can
    report nothing at all. Silence there is not evidence. Compare per-vector
    fingerprints over declared token dimensions instead.

!!! warning "Zero fingerprints do not show agreement"
    Some rows legitimately XOR to zero, for instance rows made of two
    identical halves, as at position 0 under grouped-query attention. If you
    count them as equal, the walk stops too early. Treat agreement that rests
    only on zero fingerprints as no evidence.

!!! warning "A fused producer has no partner"
    The output row of a fused projection is wider than any single trainer row,
    so a walk that requires equal widths finds no pair. Compare it at its
    first consumer after the split.

!!! warning "A replay reproduces only under a stated batch composition"
    A replayed rollout gives its original numbers only with the recorded batch
    composition, or with batch-invariant kernels. Without either, the replay
    differs for reasons that have nothing to do with the fault. Say which of
    the two you used.

## Summary

- Record cheaply online, and snapshot versions only as often as you can
  afford.
- Check the weights first, then split every gap, and set aside tokens that
  span two versions.
- A numeric error of zero clears the infrastructure for those batches; the
  staleness goes to the importance-sampling correction.
- Localize a nonzero numeric error offline by following the data flow, with
  evidence from both sides, and confirm the cause with an intervention.

## Go deeper

- [Lab report](../../examples/rl_loop/REPORT.md): the loop, every scenario, the
  planted faults, what the harness found and how it was fixed, and the cost.
- Code: [rl_core.py](../../examples/rl_loop/rl_core.py),
  [rl_loop.py](../../examples/rl_loop/rl_loop.py),
  [tools/opguard_bridge.py](../../tools/opguard_bridge.py),
  [tools/graph_probe.py](../../tools/graph_probe.py).
- Localizing a single element against a real engine:
  [report](../../examples/cross_engine/REPORT.md#worked-example-one-element-at-q_norm).
- OpGuard: `docs/RLAlignment.md` and the `rl-alignment` skill.
- Next: [walkthrough](walkthrough-verl.md), every step of this part on one
  real job; then [regression](../4-operations/regression.md).
