# Not yet shown

Everything the handbook claims was measured on a small set of machines and
stacks. This page lists what it has not shown, so you can tell whether your
setup is covered. The pages themselves carry a one-line "not yet shown" note
where a gap changes what you would do.

OpGuard itself is used in production training on large clusters
([OpGuard paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)).
The gaps below are about this handbook's own kernels, fixes and cases.

## Hardware and precision

- **Other GPU generations and other accelerators.** Every result ran on a few
  NVIDIA GPU types; see [hardware coverage](4-operations/hardware.md#untested).
  Kernels can pick different tiles on other generations, so re-run the
  regression suite before relying on a result.
- **FP8 and NVFP4 tensor-core GEMMs.** The FP8 scales so far were emulated.
- **FlashAttention 3, FlashMLA and FlashInfer** on GPU generations other than
  the ones measured.

## Scale and interconnect

- **More than two nodes, and fast interconnects** (NVLink, NVSwitch, NVLS,
  InfiniBand, RoCE). All multi-GPU results ran over PCIe inside a node and
  over plain Ethernet between two nodes. Collective algorithms change with the
  fabric, so the fixed reduction trees need checking there.
- **Real `torch.distributed.fsdp`.** The [data-parallel case](1-pretraining/data-parallel.md)
  emulates FSDP (flat gradient, reduce-scatter, sharded AdamW). bf16
  gradients, mixed precision and loss scaling are untested. The [verl case](2-rl/walkthrough-verl.md) does use real FSDP2, but
  only checks logprobs, not a pre-training step.
- **OpGuard on multi-rank pre-training.** The pre-training cases captured one
  rank; captures across ranks and the CUPTI call-site backend are not shown.
- **DeepSpeed ZeRO and Megatron**, including Megatron's custom all-reduce.

## Engines and RL frameworks

- **SGLang at L4.** Not run.
- **vLLM beyond the verl case.** A real MoE checkpoint matched at L4 with the
  rollout on one GPU per replica and the compiler off
  ([report](../examples/verl_fsdp2_vllm/REPORT.md)). Not shown: tensor or
  expert parallelism, prefix caching, speculative decoding, LoRA, and vLLM's
  default compiled mode.
- **Localizing inside compiled graphs.** OpGuard sees Python call sites, so
  a difference inside a torch.compile or CUDA-graph region can only be
  localized after reproducing it in eager mode.
- **Online records from real engines.** The [RL loop](2-rl/rl-loop.md) records
  have only been written from a toy engine, not from vLLM, SGLang, verl or
  OpenRLHF.
- **The kernel cases against a trainer.** MoE, MLA, sparse and linear
  attention are invariant on their own side; none has been matched against a
  trainer at L4.
- **Real weights in the whole-model case.** One real architecture was checked
  only with random weights, and only for a few layers
  ([report](../examples/whole_model/REPORT.md)).
- **Replay in the original batch.** RL replays have only been run with
  batch-invariant kernels, and batches that mix policy versions have not been
  captured again.

## Compiled code and backward passes

- **torch.compile and CUDA graphs** for kernels measured only in eager mode:
  attention backward, paged and varlen attention, linear attention, LoRA, MLA,
  sampling, the whole model and TP.
- **Backward passes not yet run:** MoE under expert parallelism, TP parameter
  gradients, and the KV-cache path.

## Methods still to design

- **A blind trial.** Every end-to-end case so far planted a known cause. No
  agent has yet been run through the workflow without being told the answer.
- **Fused optimizers.** The exact summation order of fused AdamW and SGD is
  not known; our models of it come close but are not exact
  ([training step](1-pretraining/training-step.md)).
- **OpGuard coverage gaps.** GEMMs inside `torch.nn.Linear` are not captured,
  and some model code cannot be wrapped
  ([pre-training case](1-pretraining/localize-with-opguard.md)).
- **Kernels not built:** a backward for recurrent GDN, a backward for paged
  attention, and fixed-size split-KV for long decodes.
- **MoE routing rules** for capacity-based dropping and exact score ties.

## Checks of this repository not yet re-run

- The regression suite has only quick-mode baselines; its full mode and its
  CI workflow have never run. Some failures of stock kernels are claimed for
  one GPU type only ([measured coverage](../suite/COVERAGE.md)).
- A few GPU re-runs listed in the [audit](../AUDIT.md#5-gpu-re-run) are still
  pending.
