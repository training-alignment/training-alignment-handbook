# 2. RL 训推对齐

## 我们在哪里

在强化学习中，rollout 引擎（vLLM、SGLang）负责采样 token，并给出每个 token 的
logprob；trainer（FSDP、Megatron、TorchTitan）则重新计算同样的 logprob，
用来算策略梯度里的比率。两端跑的是同一个策略，但代码、kernel、batching 方式、decode
模式和并行方式都不一样。所以即使权重完全相同，两端的 logprob 也会不同，训练就在没人
打算这样做的情况下悄悄变成了 off-policy。这一部分讲两件事：怎样让两端逐 bit 一致，以及不一致时
怎样找到出错的地方。阅读前需要先了解[第 0 部分](../0-foundations/README.md)介绍的
不变性级别 L1–L5。

逐 bit 一致首先是一个调试和审计工具：在一段较短的时间窗口内，基础设施带来的差距
必须恰好为零。生产环境里要不要一直开着，是另一个需要权衡代价的决定。

## 工作流程

<figure class="fx" data-fig="rl-workflow" aria-label="强化学习训推对齐的工作流程" markdown>
<div class="fx-stage" markdown>
<div class="fx-phases" style="--cols: minmax(0, 1fr) minmax(0, 0.75fr) minmax(0, 1.25fr)" markdown>
<div class="fx-phase" markdown="span"><span class="fx-label">先让每一端各自不变</span><span class="fx-seq is-col"><span class="fx-step" data-hot="0">[1. 审查算子](whole-model.md)</span><span class="fx-step" data-hot="1">[2. 整个模型](whole-model.md)<span class="fx-note">L1–L3</span></span><span class="fx-step" data-hot="2">[3. TP 规模](parallelism.md)<span class="fx-note">L5</span></span><span class="fx-step" data-hot="3">[4. 采样](sampling.md)</span><span class="fx-step" data-hot="4">[5. LoRA](lora.md)</span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">再让两端一致</span><span class="fx-seq is-col"><span class="fx-step" data-hot="5 9">[6. 引擎一致性](engine-parity-vllm.md)<span class="fx-note">L4</span></span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">监控运行中的训练循环</span><span class="fx-seq is-col"><span class="fx-step" data-hot="6">[7. 在线记录](rl-loop.md)</span><span class="fx-step" data-hot="7">[8. 拆分差距](rl-loop.md)<span class="fx-note">拆成 staleness 和数值误差</span></span><span class="fx-fork is-pair"><span class="fx-leg" data-hot="8"><span class="fx-when">numeric ≠ 0</span><span class="fx-step to-bad" data-hot="8">[9. 离线定位](rl-loop.md)<span class="fx-note" data-hot="9">然后回到第 6 步</span></span></span><span class="fx-leg" data-hot="10"><span class="fx-when">numeric = 0</span><span class="fx-step to-ok" data-hot="10">[10. 保留为回归检查](../4-operations/regression.md)</span></span></span></span></div>
</div>
</div>
<ol class="fx-say">
<li>1. 审查算子，列出输出会随 batch 或推理模式变化的模块。</li>
<li>2. 让一端的整个模型对同一条序列，不管放在什么 batch 里、用什么推理模式，都算出逐 bit 一致的结果（L1–L3）。</li>
<li>3. 换 TP 规模，结果仍逐 bit 一致；prefill 和 decode 分到不同机器上跑时也一样（L5）。</li>
<li>4. 让每个采样出的 token 都能重放，经过 batching、抢占、n > 1 和 speculative decoding 之后也不例外。</li>
<li>5. 训练时一个 batch 只有一个 adapter，推理时一个 batch 混着多个 adapter，要让 LoRA 在这两种情况下舍入完全相同。</li>
<li>6. 让 trainer 算出的 logprob 与引擎的相等（L4）。</li>
<li>7. 在线记录正在运行的循环。</li>
<li>8. 把每个 token 的差距拆成 staleness 和数值误差两部分。</li>
<li>如果 numeric ≠ 0，第 9 步离线定位这个误差，一直定位到具体的算子。</li>
<li>然后带着找到的原因回到第 6 步。</li>
<li data-hold="2">如果 numeric = 0，第 10 步把结果保留为基线，此后每个新版本都必须复现它。</li>
</ol>
<figcaption>图：第 1–5 步让每一端各自满足不变性，第 6 步让两端一致，第 7–10 步监控正在运行的训练循环。每一步都链接到对应的页面。</figcaption>
</figure>

这一部分最后的[实战](walkthrough-verl.md)在一个真实的训练任务上走了一遍这些步骤。
遇到问题时，[alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md)
会告诉你该看哪一步。

## 思路：先让每一端自己稳定，再比较两端

两个 logprob 不同，原因只有两种：要么某一端自己就不稳定，要么两端都稳定，只是舍入的
方式不同。修的时候要按这个顺序来。如果另一个请求加入 batch 后，引擎给某个 token 的
logprob 就变了，那么任何 trainer 都不可能对上它。所以第 2–5 步先让每一端各自满足不变性，
到第 6 步才比较两端。

有些差异根本不是 kernel 的 bug，而是同一个数学计算有两种写法：两种都是精确的，只是
舍入方式不同。例如线性注意力的分块形式和递归形式、MLA 的吸收式和显式写法、
split-KV 的 decode 和单遍的 prefill。这类差异靠调 kernel 是消除不了的，只能选定一种写法，
两端都用它，并接受它的代价。kernel 怎么写，见[第 3 部分](../3-kernels/README.md)。

## 思路：调试之前先把差距拆开

在异步训练循环中，rollout 用的是策略版本 v，而 trainer 已经前进到了版本 c。观察到的
差距里混着两样东西，而且可以精确地拆开：

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

举个小例子。某个 token 是在版本 3 时采样的，trainer 现在已经到了版本 5。用版本 3 的权重
在 trainer 上重新计算这个 token 的 logprob，再做比较：

| Token | logp_rollout(3) | logp_trainer(3) | logp_trainer(5) | gap | numeric | staleness |
|---|---|---|---|---|---|---|
| A | -1.20 | -1.20 | -1.35 | 0.15 | 0 | 0.15 |
| B | -1.20 | -1.21 | -1.35 | 0.15 | 0.01 | 0.14 |

两个 token 的差距一样大。A 的差距全部来自策略滞后，这正是重要性采样要校正的部分，
所以基础设施没有问题。B 的差距里有一部分是基础设施故障，任何校正都不应该把它吸收掉，
值得定位的正是这一部分。只有 `numeric` 归基础设施负责修复，而且只有在 rollout 自己的
版本上重放，才能测出它。

## 同步运行与异步运行

- **同步运行（staleness 为 0）。** 这时 `numeric` 就是全部差距。两端都做到逐 bit 一致后，
  目标是在一段较短的 on-policy 窗口里，每个 token 的差都恰好为 0。如果差距已经是 0，
  训练却仍然有问题，那么问题出在数据或算法上。
- **异步运行（staleness > 0）。** 在线记录训练循环，拆分每个 token 的差距，只定位
  `numeric` 不为零的 token。如果一个 token 的生成跨过了一次权重更新（KV cache 被保留了
  下来），它就同时属于两个版本，无法拆分，先把这类 token 放到一边。

## 可能出错的地方

!!! warning "两端还没各自稳定就开始比较"
    如果拿 trainer 去和一个不是 batch-invariant 的引擎比较，trainer 的每次改动看起来都
    像是对了一部分，因为引擎自身的波动和差距一样大。先拿引擎和它自己比较。

!!! warning "把数值误差当成 staleness 来校正"
    重要性采样的校正会吸收任何差距，包括基础设施故障造成的差距。这样一来，故障会在
    没有人察觉的情况下持续影响策略的训练。在应用任何校正之前，先在 rollout 的版本上
    拆分差距。

!!! warning "轻信版本标签"
    一个标着某个版本的 rollout batch，实际运行时用的可能是另一个版本的权重。这时它的
    数值误差看起来是真的，重放时却不会复现。要为每个 batch 实际用到的权重记录一个
    fingerprint。

## 小结

- 先让每一端各自满足不变性（L1–L3、L5、采样、LoRA），再让两端一致（L4）。
- 有些不一致来自同一个数学计算的两种精确写法，这时选定一种，两端都用它。
- 把每个差距拆成 staleness 和数值误差。只有数值误差才是基础设施故障。
- 不为零的数值误差，要沿着数据流离线定位；修好之后，把这个案例保存为回归基线。

## 深入阅读

- 各步骤页面：[整个模型](whole-model.md)、[并行](parallelism.md)、
  [采样](sampling.md)、[LoRA](lora.md)、
  [引擎一致性](engine-parity-vllm.md)、[RL 循环](rl-loop.md)。
- [实战](walkthrough-verl.md)：在一个真实的训练任务上走完这一部分的每一步，verl 加 FSDP2，rollout 用 vLLM。
- 其他团队报告的结果（IsoExec、verl-omni、TorchTitan RL），以及一个定位到单个元素的
  完整实例：[报告](../../examples/cross_engine/REPORT.md#context-moved-from-the-part-2-overview)；
  来源列在[参考文献](../references.md)中。
- Skill：[alignment-pipeline](../../skills/alignment-pipeline/SKILL.md)、
  [engine-parity](../../skills/engine-parity/SKILL.md)、
  [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)；
  OpGuard 的 [`rl-alignment` skill](https://github.com/ziming-zh/OpGuard/tree/main/harness/skills/rl-alignment)（OpGuard 仓库中的 `harness/skills/rl-alignment`）
  及其协议 `docs/RLAlignment.md`。
