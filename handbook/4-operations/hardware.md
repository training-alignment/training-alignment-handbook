# Hardware coverage

Every bitwise claim in this handbook holds on the stack it was shown on. When
you move to other GPUs, another driver or other libraries, the
[regression suite](regression.md) tells you which claims still hold. This page
explains which properties of a stack can change the bits, why two GPU
generations are not expected to agree, and how to qualify new hardware.

## The idea

Suppose you add 4,096 numbers on a GPU. On a GPU with many SMs the library
may split them into many partial sums, and on a GPU with fewer SMs into fewer.
Each split rounds in different places, so the total can differ in the last
bit even though the source code and the inputs are the same. The rule is:
**anything that decides how a reduction is split, ordered or rounded is part
of the stack, and a bitwise result belongs to that stack.** This is the
principle behind batch invariance, applied one level up: there the batch
chose the kernel, and here the machine does.

## Which properties of a stack can change the bits

| Property | How it can change the bits |
|---|---|
| Architecture | Each generation has its own matrix instructions, and nothing guarantees that they use the same accumulation order or intermediate precision. Libraries ship different kernels for each architecture, and some kernels and number formats exist only on newer generations. Triton compiles for each architecture separately, so the same source becomes different machine code. |
| SM count | Libraries choose kernels, tiles and split-K from the SM count, and reduction and random-number kernels size their grids from it. Two GPUs of the same architecture with different SM counts can therefore disagree. |
| Driver | In the measured scan, two driver versions with the same libraries gave the same bits. The driver is still part of the baseline key, because it ships parts of CUDA. |
| CUDA build and libraries | cuBLAS, cuDNN and NCCL can choose different kernels, algorithms and protocols in each version, and torch can change which backend an operator uses (SDPA's, for example). |
| Interconnect | NCCL chooses its algorithm (Ring or Tree) and its protocol from the topology and the message size, and the algorithm decides the order of a collective's sum. |

What can carry over to other hardware is a kernel whose reduction order is
written in its source and does not depend on the device. The handbook's
Triton kernels gave the same bits on two GPUs of one generation with very
different SM counts, while stock GEMMs, large reductions and random-number
streams did not. Whether those kernels give the same bits on another
architecture is **untested**. Measured: see
[measured coverage](../../suite/COVERAGE.md#same-bits-on-two-gpus-of-one-generation).

Across generations, you should not expect bitwise agreement. One reported
example: the DeepSeek-V3 technical report (arXiv 2412.19437) states that FP8
GEMMs on newer tensor cores accumulate with limited precision, which is why it
promotes the partial sums to CUDA cores at a fixed interval. So verify each
invariance level (L1–L5) on each stack, and claim equality across generations
only after you have measured it.

## How to qualify new hardware

You want to know which claims hold on the new stack and which stock
behaviour differs. You are done when a reviewed baseline for the new stack is
committed and every difference is either a stock result recorded as specific
to that stack or a reported finding.

1. **Run the regression suite in full mode** with `--vs-committed`, as the
   [new hardware section of the regression page](regression.md#new-hardware)
   describes ([tools/bi_suite.py](../../tools/bi_suite.py)).
2. **Run the SDC scan** with `--cross-group-info` against the committed golden
   files ([tools/sdc_scan.py](../../tools/sdc_scan.py)). It reports, for
   information only, which operator families give the same bits as the stacks
   measured before.
3. **Sort the differences.** Every required check that fails is a finding, so
   report it. Every documented stock failure that now passes is a change:
   record it on its topic page and name the new stack.
4. **Record the baseline**, and add the stack to
   [measured coverage](../../suite/COVERAGE.md#what-was-measured).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| The suite in quick mode | every claim, checked at reduced sizes | small | a first look, or a minor library update |
| The suite in full mode with `--vs-committed` | the sizes the pages were written from, and what has moved | large | new GPUs, a new architecture, or before a release |
| The SDC scan across groups | which operator families match the measured stacks bit for bit | small | new GPUs, to see what can be compared across machines |

## What has not been tested on other hardware {#untested}

Two things must be kept apart here. **OpGuard itself** (capture,
fingerprints and the comparison that finds the first divergence) is used in
production LLM training on the platforms below, as its authors report
([OpGuard paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)).
**This handbook's own measurements** (the batch-invariant kernels and fixes,
the fixed collective trees and the engine-parity setup) come from one GPU
generation, connected over PCIe inside a node and over plain Ethernet between
two nodes. The engine-parity setup has also been shown on another GPU
generation, with the engine's own kernels, in the verl × FSDP2 × vLLM case
([report](../../examples/verl_fsdp2_vllm/REPORT.md)); the handbook's own
kernels have still been measured on one generation only. These measurements
have not yet been shown on these platforms:

| Area | OpGuard | This handbook's kernels, fixes and setups |
|---|---|---|
| Other GPU architectures, other accelerators | used in production (reported) | untested |
| FP8 tensor-core GEMM and its promotion interval, NVFP4 | used in production (reported) | untested (the FP8 scales were emulated) |
| Newer attention kernels (FlashAttention 3, FlashMLA, FlashInfer) on other GPU architectures | used in production (reported) | untested |
| NVLink, NVSwitch, NVLS; RDMA, InfiniBand, RoCE | used in production (reported) | untested; the fixed collective trees were shown over PCIe and Ethernet only |
| More than two nodes | used in production (reported) | untested; a pinned Tree that held on two nodes may not hold on three |
| Other torch, Triton, NCCL and CUDA versions | used in production (reported) | untested, except vLLM's own stack in the cross-engine case and the verl and vLLM stack in the verl × FSDP2 × vLLM case |

So on such a platform, localize with OpGuard as usual, but re-verify every
invariance claim of this handbook with the suite before you rely on it. The
concrete stacks for each result file are in
[measured coverage](../../suite/COVERAGE.md).

## What can go wrong

!!! warning "Assuming that the bits carry over"
    A new GPU looks like a faster version of the old one, but its library
    picks different kernels, so a claim carried over without measuring may be
    false on it. Run the suite before you rely on any bitwise claim.

!!! warning "Comparing machines on stock operators"
    Two machines of different types that run stock GEMMs or random-number
    kernels can disagree without any bug anywhere. Compare across machines
    only on operators whose order is fixed in their source, or compare each
    machine with itself.

!!! warning "Hiding a failure on new hardware with a tolerance"
    A required check that fails on new hardware is a finding about that
    hardware or its libraries. Hiding it with a tolerance makes every later
    claim on that stack untrustworthy, so report it.

## Summary

- A bitwise result belongs to its stack: the architecture, SM count, driver,
  CUDA build, libraries and interconnect.
- Do not expect bitwise agreement across GPU generations; measure it before
  you claim it.
- Only kernels whose reduction order is fixed in their source can carry over.
- Qualify new hardware with the suite in full mode and the SDC scan, and then
  record a baseline.

## Go deeper

- [Measured coverage](../../suite/COVERAGE.md): the stacks that were measured,
  the result files, the baselines and the comparison across GPUs.
- [Regression](regression.md): how to run the suite and read its report.
- [SDC scan](../1-pretraining/sdc.md): the fingerprint scan across machines.
