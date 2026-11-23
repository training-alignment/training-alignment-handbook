# 参考文献

本手册用到了下面这些来源。标 * 的链接来自我们的工作笔记，发布前应再核对一遍。更完整、分好类的
论文、工具和博客列表见兄弟仓库 [Awesome Bitwise Alignment](https://github.com/training-alignment/awesome-bitwise-alignment)。

## 工具与方法

- OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM
  Training, OSDI '26. https://www.usenix.org/conference/osdi26/presentation/zhou-ziming
- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025)。文章指出，推理之所以
  不确定，是因为缺乏 batch invariance，并给出了不变的 RMSNorm、矩阵乘法和注意力 kernel。
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ *
- `batch_invariant_ops`（Thinking Machines）。https://github.com/thinking-machines-lab/batch_invariant_ops *
- PyTorch 关于可复现性的说明。https://pytorch.org/docs/stable/notes/randomness.html
- cuBLAS 关于结果可复现性的说明。https://docs.nvidia.com/cuda/cublas/#results-reproducibility *

## 强化学习的训推对齐

- IsoExec: Unified Execution to Eliminate Trainer-Inference Mismatch in SkyRL
  （vLLM blog，2026-08-21）。https://vllm.ai/blog/2026-08-21-isoexec
- Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven
  （vLLM blog，2026-10-14）。https://vllm.ai/blog/2026-10-14-train-inference-consistency
- Yichuan Wang et al., Closing the Training–Inference Mismatch for Linear
  Attention in RL（异步强化学习，GDN）。https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/
- verl-omni issue #637：用 HF 的数值实现，让 rollout 和 actor 逐 bit 一致的一种模式。
  https://github.com/verl-project/verl-omni/issues/637
- Yingru Li et al., When Speed Kills Stability: Demystifying RL Collapse from the
  Training–Inference Mismatch. https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

## 本仓库里的测量

每个结果文件都放在它的代码旁边，对应的报告写明了测量所用的软件栈。所有结果文件，以及每个文件
背后的硬件和软件，列在[测得的覆盖范围](../suite/COVERAGE.md#result-files)里；哪些因素会让结果在
不同软件栈之间变化，见[硬件覆盖](4-operations/hardware.md)。
