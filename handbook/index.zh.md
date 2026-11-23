---
hide:
  - toc
---

# 训练对齐手册

怎样让两次训练或推理的结果逐 bit 一致；不一致时，怎样找到最先出现差异的地方。

## 三条工作流程

本手册讲三条工作流程。前两条从症状出发：一条处理两次本该一致的预训练跑出了不同结果（第 1 部分），另一条处理 rollout 引擎和 trainer 对同样的 token 算出了不同的 logprob（第 2 部分）。这两条流程查到最后，原因常常是某个算子算一行时会受同一个 batch 里其他行的影响。这时就把这个算子交给第三条流程，把它改成 batch-invariant 的实现，也就是结果不随 batch 变化（第 3 部分）。三条流程共同依赖的基本概念放在[第 0 部分](0-foundations/README.md)。

<figure class="flows" aria-label="三条工作流程" markdown>
<p class="flows-base" markdown>三条流程都以[第 0 部分：基础](0-foundations/README.md)为前提。那里定义了 L1–L5 五个不变性级别，也讲了在哪里、用什么方法比较两次运行。</p>
<div class="flow-lane" data-lane="pre" data-side="above" markdown>
<p class="flow-head" markdown><span class="flow-part">第 1 部分</span> [预训练](1-pretraining/README.md) <span class="flow-goal">找到两次运行从哪里开始不一致</span></p>
<ol class="flow-steps" style="--steps: 7" markdown>
<li markdown="span">[确定性与预检](1-pretraining/determinism-and-preflight.md)<span class="flow-desc">打开让运行可重复的设置，并确认两次运行真的都用上了。</span></li>
<li markdown="span">[数据对齐](1-pretraining/data-alignment.md)<span class="flow-desc">确认两次运行的每个 rank 在每一步拿到的 batch 完全相同。</span></li>
<li class="flow-handoff" markdown="span">[用 OpGuard 定位](1-pretraining/localize-with-opguard.md)<span class="flow-desc">采集两次运行，找到第一个结果不同的算子，从它往回追到原因，修复后再确认。</span></li>
<li markdown="span">[训练 step](1-pretraining/training-step.md)<span class="flow-desc">如果原因在反向传播、优化器、loss 或恢复训练里，就让这一部分逐 bit 可重复。</span></li>
<li markdown="span">[数据并行](1-pretraining/data-parallel.md)<span class="flow-desc">如果原因是跨 rank 求和或分片的梯度，就让求和结果不受任务划分方式的影响。</span></li>
<li markdown="span">[torch.compile 与 CUDA graph](1-pretraining/capture-compiled-code.md)<span class="flow-desc">如果训练任务用了 `torch.compile` 或 CUDA graph，就在编译后的代码内部取 fingerprint，因为 eager 采集看不到那里。</span></li>
<li markdown="span">[SDC](1-pretraining/sdc.md)<span class="flow-desc">排除了软件原因之后，找出悄悄返回错误数值的硬件。</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="kernel" data-delay="1" data-target markdown>
<p class="flow-head" markdown><span class="flow-part">第 3 部分</span> [batch-invariant kernel](3-kernels/README.md) <span class="flow-goal">让一个算子不受 batch 影响</span></p>
<ol class="flow-steps" markdown>
<li markdown="span">[GEMM 与量化 GEMM](3-kernels/gemm-and-quantized.md)<span class="flow-desc">无论 batch size 多大，都用同一种 tile 和同一个归约顺序。</span></li>
<li markdown="span">[注意力](3-kernels/attention.md)<span class="flow-desc">prefill、decode 和反向传播都按同一种方式切分 key。</span></li>
<li markdown="span">[MLA 与稀疏注意力](3-kernels/mla-and-sparse.md)<span class="flow-desc">key 经过压缩或由 top-k 选出时，也照此处理。</span></li>
<li markdown="span">[线性注意力](3-kernels/linear-attention.md)<span class="flow-desc">每次都按同一个分块顺序更新循环状态。</span></li>
<li markdown="span">[MoE](3-kernels/moe.md)<span class="flow-desc">让路由、expert GEMM 和 combine 的结果与哪些 token 同在一个 batch 无关。</span></li>
<li markdown="span">[集合通信](3-kernels/collectives.md)<span class="flow-desc">按一个固定的顺序把各 GPU 的部分结果加起来。</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="rl" data-side="below" markdown>
<p class="flow-head" markdown><span class="flow-part">第 2 部分</span> [强化学习训推](2-rl/README.md) <span class="flow-goal">让 rollout 引擎和 trainer 结果一致</span></p>
<ol class="flow-steps" markdown>
<li class="flow-handoff" markdown="span">[整个模型](2-rl/whole-model.md)<span class="flow-desc">同一端算一条序列，不管单独算、和别的序列拼成 batch 算，还是换一种推理模式，logits 都相同。</span></li>
<li markdown="span">[并行](2-rl/parallelism.md)<span class="flow-desc">用 TP 切分模型，或者把 prefill 和 decode 放到不同的服务器上，这一点也要成立。</span></li>
<li markdown="span">[采样](2-rl/sampling.md)<span class="flow-desc">保证采样出的 token 能用 logits 和种子重新算出来。</span></li>
<li markdown="span">[LoRA](2-rl/lora.md)<span class="flow-desc">一个 batch 里混用多个 adapter 时，每个 adapter 的结果保持不变。</span></li>
<li class="flow-handoff" markdown="span">[引擎一致性](2-rl/engine-parity-vllm.md)<span class="flow-desc">让真实的推理引擎在和 trainer 相同的地方做舍入，两端结果才能一致。</span></li>
<li markdown="span">[RL 循环](2-rl/rl-loop.md)<span class="flow-desc">记录每次 rollout，区分 staleness 和真正的不一致，并离线定位差异。</span></li>
<li markdown="span">[实战](2-rl/walkthrough-verl.md)<span class="flow-desc">在一个真实的训练任务上走完每一步：verl 加 FSDP2，rollout 用 vLLM。</span></li>
</ol>
</div>
<p class="flows-next" markdown>之后，每当软件栈或硬件发生变化，[第 4 部分：运维](4-operations/README.md)会重新检查这些结果。</p>
<figcaption><span class="flow-key"></span>虚线框的步骤会把它找到的算子交给第 3 部分。把鼠标移到某一步上，或用键盘把焦点移过去，就能看到这一步做什么。</figcaption>
</figure>

不确定从哪里开始？[alignment-pipeline skill](../skills/alignment-pipeline/SKILL.md)
会问你几个问题，然后把你引到合适的流程。全书用到的术语都收在[术语表](glossary.md)里。

## 第 1 部分：两次预训练的结果不一致

先读[概览](1-pretraining/README.md)。这一部分对应的 agent skill 是
[pretraining-alignment](../skills/pretraining-alignment/SKILL.md)。

1. [确定性与预检](1-pretraining/determinism-and-preflight.md)：打开让运行可重复的设置，并确认两次运行真的都用上了。
2. [数据对齐](1-pretraining/data-alignment.md)：确认两次运行的每个 rank 在每一步拿到的 batch 完全相同。
3. [用 OpGuard 定位](1-pretraining/localize-with-opguard.md)：采集两次运行，找到第一个结果不同的算子，再从它往回追到原因。然后修复原因，并确认修复有效。

接下来，首个分歧点指向哪里，就去读对应的那一页：

- [训练 step](1-pretraining/training-step.md)讲怎样让反向传播、优化器、loss 计算以及从 checkpoint 恢复训练都逐 bit 可重复。
- [数据并行](1-pretraining/data-parallel.md)讲梯度在多个 rank 之间求和或分片时，怎样让结果保持不变。
- [torch.compile 与 CUDA graph](1-pretraining/capture-compiled-code.md)讲怎样在 `torch.compile` 和 CUDA graph 内部取 fingerprint（指纹），因为 eager 采集看不到那里。
- [SDC](1-pretraining/sdc.md)讲怎样找出悄悄返回错误数值的硬件。

## 第 2 部分：rollout 引擎和 trainer 结果不一致

先读[概览](2-rl/README.md)。这一部分对应的 agent skill 是
[engine-parity](../skills/engine-parity/SKILL.md) 和 OpGuard 的 `rl-alignment`。

1. [整个模型](2-rl/whole-model.md)：同一端算一条序列，不管单独算、和别的序列拼成 batch 算，还是换一种推理模式，logits 都相同。
2. [并行](2-rl/parallelism.md)：用 TP 切分模型，或者把 prefill 和 decode 放到不同的服务器上，这一点也要成立。
3. [采样](2-rl/sampling.md)：保证采样出的 token 能用 logits 和种子重新算出来。
4. [LoRA](2-rl/lora.md)：一个 batch 里混用多个 adapter 时，让每个 adapter 的结果保持不变。
5. [引擎一致性](2-rl/engine-parity-vllm.md)：让真实的推理引擎在和 trainer 相同的地方做舍入，两端结果才能一致。
6. [RL 循环](2-rl/rl-loop.md)：记录每次 rollout，区分 staleness（策略滞后）和真正的不一致，并离线定位差异。
7. [实战](2-rl/walkthrough-verl.md)：在一个真实的训练任务上走完每一步，verl 加 FSDP2，rollout 用 vLLM。

## 第 3 部分：让一个算子不受 batch 影响

先读[标准](3-kernels/README.md)，它规定了一个不变的算子要满足什么。这一部分对应的 agent skill 是
[batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md)。

1. [GEMM 与量化 GEMM](3-kernels/gemm-and-quantized.md)：无论 batch size 多大，都用同一种 tile 和同一个归约顺序。
2. [注意力](3-kernels/attention.md)：prefill、decode 和反向传播都按同一种方式切分 key。
3. [MLA 与稀疏注意力](3-kernels/mla-and-sparse.md)：key 经过压缩或由 top-k 选出时，也照此处理。
4. [线性注意力](3-kernels/linear-attention.md)：每次都按同一个分块顺序更新循环状态。
5. [MoE](3-kernels/moe.md)：让路由、expert GEMM 和 combine 的结果与哪些 token 同在一个 batch 无关。
6. [集合通信](3-kernels/collectives.md)：按一个固定的顺序把各 GPU 的部分结果加起来。

## 第 5 部分：把同一套方法用到其他场景

同一套方法也适用于其他需要两两比较的场景。见[概览](5-scenarios/README.md)，以及[框架迁移](5-scenarios/framework-migration.md)、[checkpoint 转换](5-scenarios/checkpoint-conversion.md)和[离线后训练](5-scenarios/offline-post-training.md)这几页。

## 第 4 部分、skill 与参考资料

- [第 4 部分：运维](4-operations/README.md)讲在[软件栈变化](4-operations/regression.md)之后、
  以及[换了新硬件](4-operations/hardware.md)时，怎样重新检查结果。
- 所有 skill：[alignment-pipeline](../skills/alignment-pipeline/SKILL.md)、
  [pretraining-alignment](../skills/pretraining-alignment/SKILL.md)、
  [engine-parity](../skills/engine-parity/SKILL.md)、
  [batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md) 和
  [handbook-writing](../skills/handbook-writing/SKILL.md)。
- 参考资料：[待办事项](open-items.md) · [参考文献](references.md) ·
  [术语表](glossary.md) · [审计](../AUDIT.md) · [工具](../tools/) ·
  [致谢](acknowledgments.md)

每一页上的结论都链接到对应的报告。报告里有测量结果，也写明了测量所用的软件栈。

## 引用 {#citation}

如果这本手册、其中的工具或 skill 对你的工作有帮助，请引用：

```bibtex
@misc{zhou2026trainingalignment,
  title        = {Training Alignment Handbook},
  author       = {Zhou, Ziming and Xu, Peiyu},
  year         = {2026},
  howpublished = {\url{https://training-alignment.github.io/training-alignment-handbook/}},
  note         = {Code and reports: \url{https://github.com/training-alignment/training-alignment-handbook}}
}
```
