# Training Alignment Handbook

How to make two training or inference runs agree bit for bit, and how to find
the first place where they stop agreeing.

**[Read the handbook](https://training-alignment.github.io/training-alignment-handbook/)** ·
**[中文版](https://training-alignment.github.io/training-alignment-handbook/zh/)** ·
[Agent skills](#use-it-with-an-agent) ·
[OpGuard](https://github.com/ziming-zh/OpGuard)

[![site](https://github.com/training-alignment/training-alignment-handbook/actions/workflows/pages.yml/badge.svg)](https://github.com/training-alignment/training-alignment-handbook/actions/workflows/pages.yml)

## What it is for

- **Pre-training.** Two runs that should agree do not: after an upgrade, a
  config change, a resume, or on a suspect machine. Find the first operator
  where they part, then the cause.
- **RL train–inference.** The rollout engine and the trainer give different
  log-probabilities for the same tokens. Make them agree, or measure exactly
  how much of the gap is staleness and how much is numeric error.
- **One operator.** A kernel's result depends on what else is in its batch.
  Make it batch-invariant and prove it.

Each part is a workflow written for engineers, and the same workflow is
packaged as an agent skill. Every claim links to the experiment that
measured it.

## Contents

**[0. Foundations](handbook/0-foundations/README.md)**: why the order of a
sum changes its bits, the five levels of invariance, and how to compare two
runs bitwise.

**[1. Pre-training alignment](handbook/1-pretraining/README.md)**: from a
symptom to the first divergence, its cause and a regression check.
- [Determinism and preflight](handbook/1-pretraining/determinism-and-preflight.md): turn on and verify the settings that make a run repeatable
- [Data alignment](handbook/1-pretraining/data-alignment.md): every rank gets the same batches at every step
- [Localize with OpGuard](handbook/1-pretraining/localize-with-opguard.md): capture both runs and find the first operator that differs
- [Training step](handbook/1-pretraining/training-step.md): backward, loss, optimizer and resume that repeat bit for bit
- [Data parallelism and FSDP](handbook/1-pretraining/data-parallel.md): gradients that stay equal when ranks sum or shard them
- [Compiled code and CUDA graphs](handbook/1-pretraining/capture-compiled-code.md): fingerprints where an eager capture cannot see
- [Silent data corruption](handbook/1-pretraining/sdc.md): find GPUs that return wrong values

**[2. Train–inference alignment in RL](handbook/2-rl/README.md)**: make each
side invariant, make both sides agree, then watch the running loop.
- [Whole-model invariance](handbook/2-rl/whole-model.md): one sequence, the same logits alone, in a batch and in every serving mode
- [Any tensor-parallel degree](handbook/2-rl/parallelism.md): the same result at different TP sizes and with prefill and decode split
- [Replayable sampling](handbook/2-rl/sampling.md): sampled tokens that can be reproduced from the logits and a seed
- [LoRA](handbook/2-rl/lora.md): one adapter in training, many mixed in one serving batch
- [Matching vLLM](handbook/2-rl/engine-parity-vllm.md): a trainer and a real engine that agree bit for bit
- [Monitoring the RL loop](handbook/2-rl/rl-loop.md): records, the staleness split and offline localization

**[3. Batch-invariant kernels](handbook/3-kernels/README.md)**: the standard
an invariant operator must meet, then one page per operator class.
- [GEMM and quantized GEMM](handbook/3-kernels/gemm-and-quantized.md) ·
  [Softmax attention](handbook/3-kernels/attention.md) ·
  [MLA and sparse attention](handbook/3-kernels/mla-and-sparse.md) ·
  [Linear attention](handbook/3-kernels/linear-attention.md) ·
  [MoE](handbook/3-kernels/moe.md) ·
  [Collectives](handbook/3-kernels/collectives.md)

**[4. Operations](handbook/4-operations/README.md)**: keep the results true on
your machines.
- [Regression checks](handbook/4-operations/regression.md) after an upgrade or on new hardware ·
  [Hardware coverage](handbook/4-operations/hardware.md)

**[5. Other alignment scenarios](handbook/5-scenarios/README.md)**: the same
method for other pairs of runs.
- [Framework migration](handbook/5-scenarios/framework-migration.md) ·
  [Checkpoint conversion](handbook/5-scenarios/checkpoint-conversion.md) ·
  [Offline post-training](handbook/5-scenarios/offline-post-training.md)

Reference: [Glossary](handbook/glossary.md) · [Not yet shown](handbook/open-items.md) ·
[References](handbook/references.md) · [Audit](AUDIT.md) ·
[Acknowledgments](handbook/acknowledgments.md)

## Use it with an agent

Each workflow is also a skill: a folder with a `SKILL.md` that tells an agent
which steps to take, which tool runs each step and what evidence to keep.
Start from `alignment-pipeline`, which routes a question to the others.

| Skill | Use it when |
|---|---|
| [alignment-pipeline](skills/alignment-pipeline/SKILL.md) | You are not sure which workflow applies |
| [pretraining-alignment](skills/pretraining-alignment/SKILL.md) | Two training runs should agree and do not |
| [engine-parity](skills/engine-parity/SKILL.md) | A trainer and an inference engine must give the same log-probabilities |
| [batch-invariant-kernel](skills/batch-invariant-kernel/SKILL.md) | An operator has no batch-invariant implementation |
| [handbook-writing](skills/handbook-writing/SKILL.md) | You are adding a page to this handbook |
| `investigate-first-diff`, `trace-upstream`, `validate-hypothesis`, `rl-alignment` | Localizing with OpGuard; these live in [OpGuard](https://github.com/ziming-zh/OpGuard)'s `harness/skills/` |

For Claude Code, link both sets of skills into `~/.claude/skills/`:

```bash
git clone https://github.com/training-alignment/training-alignment-handbook
git clone https://github.com/ziming-zh/OpGuard
mkdir -p ~/.claude/skills
for d in training-alignment-handbook/skills/*/ OpGuard/harness/skills/*/; do ln -s "$(realpath "$d")" ~/.claude/skills/; done
```

Keep both repositories checked out: the skills call the tools below and
OpGuard's harness. The RL steps need OpGuard's `rl.*` and `vectors.*`
actions, which are in this OpGuard repository and not yet upstream.

## Tools

| Tool | What it does |
|---|---|
| [bi_check.py](tools/bi_check.py) | Bitwise invariance checks for one operator; `--demo` runs them on stock PyTorch ops |
| [bi_kernels.py](tools/bi_kernels.py) | Batch-invariant reference kernels for GEMM and attention, in Triton |
| [bi_audit.py](tools/bi_audit.py) | Lists a model's operators and says which modules are batch-variant |
| [bi_sampling.py](tools/bi_sampling.py) | A reference sampler whose tokens can be replayed |
| [determinism_preflight.py](tools/determinism_preflight.py) | Reads back a job's determinism settings and diffs two runs |
| [data_fingerprint.py](tools/data_fingerprint.py) | Fingerprints every batch at the data loader and classifies the first difference |
| [opguard_bridge.py](tools/opguard_bridge.py) | OpGuard's fingerprint formats, protocol and harness from the handbook's code |
| [graph_probe.py](tools/graph_probe.py) | Fingerprints inside `torch.compile` code and CUDA graphs |
| [ckpt_compare.py](tools/ckpt_compare.py) | Compares two checkpoints with different layouts, row by row |
| [sdc_scan.py](tools/sdc_scan.py) | Scans GPUs for silent data corruption |
| [bi_report.py](tools/bi_report.py), [bi_suite.py](tools/bi_suite.py) | The result format and the regression suite that re-runs every example |

## Repository layout

```text
handbook/   the book: one folder per part, English .md and Chinese .zh.md
skills/     agent skills for each workflow
tools/      the tools above
examples/   one folder per experiment: code, results-*.json and REPORT.md
suite/      regression manifest, baselines and measured coverage
scripts/    site build hooks and checks
```

To build the site locally: `pip install mkdocs-material mkdocs-static-i18n jieba`,
then `mkdocs serve` ([details](handbook/4-operations/README.md#building-the-site)).

## Citation

If this handbook, its tools or its skills help your work, please cite it.
GitHub's "Cite this repository" button reads the same entry from
[CITATION.cff](CITATION.cff).

```bibtex
@misc{zhou2026trainingalignment,
  title        = {Training Alignment Handbook},
  author       = {Zhou, Ziming and Xu, Peiyu},
  year         = {2026},
  howpublished = {\url{https://training-alignment.github.io/training-alignment-handbook/}},
  note         = {Code and reports: \url{https://github.com/training-alignment/training-alignment-handbook}}
}
```

## Status and license

This is a working draft. Numbers quoted from other reports cite their source;
numbers measured here link to the report that measured them, and what is not
yet shown is listed in [Not yet shown](handbook/open-items.md).

Licensed under the [Apache License 2.0](LICENSE).
