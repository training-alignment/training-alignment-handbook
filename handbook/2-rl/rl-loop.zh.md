# 监控 RL 训练循环

## 我们在哪里

完成[引擎一致性](engine-parity-vllm.md)之后，trainer 和引擎在一组固定的 prompt 上已经
一致。而一个正在运行的 RL 训练循环，会带来那次测试里没有的情况：每个版本都要推送新权重，
异步的 rollout 用的是旧版本，batch 每一步都在变。这一步负责监控整个循环：运行时以很低的
成本做记录，把每个差距拆成 staleness 和数值误差，只对真正属于基础设施故障的差距做离线
定位。这对应 [RL 工作流程](README.md)的第 7–9 步。

## 思路

[第 2 部分的概览页](README.md)已经说明，差距可以精确地拆成两部分：

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

这个拆分是否可信，取决于三个条件，它们也决定了后面步骤的顺序：

1. **权重必须是记录里声称的那一份。** 如果一次权重推送之后，引擎里有一行权重是错的，
   那么这个版本上的所有比较都没有意义。在看任何 logprob 之前，先检查权重。
2. **每个 token 只能属于一个版本。** 如果一条序列还在生成时新权重就到了，而它的
   KV cache 又被保留下来，这条序列就含有来自两个策略的 token。它在哪个版本下都无法
   重放，只能先放到一边。
3. **认定分歧需要两端的证据。** 沿着模型追查一个不为零的数值误差时，如果某一行在另一端
   找不到对应的行，那只是记录里的一个空缺，通常是另一端把这一步融合掉了，并不是分歧。
   所谓分歧，是指同样的输入行在两端给出了不同的输出行。

定位要沿着数据流进行，而不是按 trace 里事件的先后顺序。从两端一致的地方出发，例如
embedding 的输出，跟着每个张量走到使用它的那个算子，直到找到第一对输入相等、输出不同
的算子。

<figure class="fx" data-fig="rl-triage" aria-label="在运行中的强化学习循环里，怎样分诊一个差距">
<div class="fx-stage">
<div class="fx-dflow">
<span class="fx-node" data-hot="0">运行中的循环留下的记录</span>
<span class="fx-vl" data-hot="1"></span>
<span class="fx-node is-decision" data-hot="1">权重是否一致？</span><span class="fx-hl" data-hot="2">否</span><span class="fx-node to-bad" data-hot="2">修复权重同步，到此为止</span>
<span class="fx-vl" data-hot="3">是</span>
<span class="fx-node" data-hot="3 4">在 rollout 的版本上拆分每个 token 的差距<span class="fx-swap is-block" data-at="3"><span class="fx-split" data-to="4"><span class="is-whole">gap 0.15</span></span><span class="fx-split" data-at="4" style="--a:1;--b:14"><span class="is-a">numeric 0.01</span><span class="is-b">staleness 0.14</span></span></span></span>
<span class="fx-vl" data-hot="5"></span>
<span class="fx-node is-decision" data-hot="5">numeric = 0？</span><span class="fx-hl" data-hot="5">是</span><span class="fx-node to-ok" data-hot="5">基础设施没有问题<span class="fx-note">差距只来自策略滞后</span></span>
<span class="fx-vl" data-hot="6">否</span>
<span class="fx-node" data-hot="6">重新采集两端，沿数据流追查</span>
<span class="fx-vl" data-hot="7"></span>
<span class="fx-node is-key" data-hot="7">判定原因类别，检验最小的修复</span>
</div>
</div>
<ol class="fx-say">
<li>记录来自正在运行的训练循环，其中附有每个 batch 实际所用权重的 fingerprint。</li>
<li>先确认：引擎当时持有的权重，是不是 trainer 在那个版本上的权重？</li>
<li>如果不是，这个版本上的所有比较都没有意义，修好权重同步就可以停下了。</li>
<li>如果是，就取每个 token 的差距，这里是 0.15。</li>
<li data-hold="2">在 rollout 自己的版本上重放这个 token，就能把差距精确拆成数值误差 0.01 和 staleness 0.14。</li>
<li data-hold="1.5">如果数值误差是 0，整个差距就只是策略滞后，基础设施没有问题。</li>
<li data-hold="1.5">这里它是 0.01，所以要重新采集两端，沿着数据流追查，而不是按 trace 的顺序。</li>
<li data-hold="2">最后判定原因属于哪一类，并检验能消除它的最小修复。</li>
</ol>
<figcaption>图：先检查权重，再在 rollout 的版本上拆分每个差距，只对不为零的数值误差做定位。</figcaption>
</figure>

## 步骤

下面的每个动作，OpGuard 的 RL harness 都已经实现。记录由
`opguard_harness.rl_records.RecordWriter` 写入。各个动作（作用于记录文件的 `rl.*`，
以及作用于采集结果的 `vectors.*`）只报告事实：它们返回 `verdict: not_evaluated`，
不使用任何容差。协议和每个字段的含义见 OpGuard 的 `docs/RLAlignment.md`；OpGuard 的
`rl-alignment` skill（`harness/skills/rl-alignment`）让 agent 按这些步骤执行。

1. **在线记录**，以很低的成本记下足够的信息，以后能拆分并重放任何一个抽样到的 batch。`RecordWriter`
   写入五类记录：`rollout_batch` 包括 token 和 prompt 的 id、位置、每个 token 的策略版本、rollout logprob
   的原始 bit、batch 组成，以及引擎当时所持权重的 fingerprint；`forward` 记录每一次前向及其行布局；
   `weight_sync` 记录每次同步时两端逐行的权重 fingerprint；`snapshot` 记录各版本权重保存在哪里；
   staleness 为零时，还会记录 `trainer_logprob`。记录摘要里要列出每一类记录；版本快照只按抽样保存，成本有上限。
2. **检查权重**，确认引擎运行的正是 trainer 推送的权重。`rl.weight_sync_check` 把逐行 fingerprint
   当作多重集合来比较，所以打包方式不同（例如融合的 QKV）不会被算作差异。每个比较过的版本都应该没有问题；
   否则它会在两端分别指出出错的参数和行。
3. **拆分差距**，得到每个 token 的数值误差和 staleness。`rl.decompose` 用精确算术完成拆分，标出跨越两个版本的
   token，并说明每个 batch 与哪一份记录下来的权重 fingerprint 相符。对于较旧的版本，`rl.replay_request`
   会构造重放请求（版本 v 的快照、相同的 token、原来的 batch 组成或 batch-invariant 的组成），交给你自己的
   训练任务离线运行。每个只属于一个版本的 token，数值误差都应为零；否则会得到一份数值误差不为零的 token 清单。
4. **离线定位**数值误差最先出现的那个算子。用逐向量 fingerprint 重新采集两端（OpGuard `OPGUARD_XOR_ROWS=1`），
   用 `rl.forwards` 列出各次前向，再用 `vectors.describe`、`vectors.lookup`、`vectors.coverage` 和
   `vectors.compare` 查询，从两端一致的锚点出发，沿数据流往下走。如果引擎用了 torch.compile 或 CUDA graph，
   就用 [tools/graph_probe.py](../../tools/graph_probe.py)
   取 fingerprint（见[torch.compile 与 CUDA graph](../1-pretraining/capture-compiled-code.md)）。
   找到一对候选就完成了：同样的输入行在两端给出了不同的输出行。
5. **判定原因类别，并检验它。** 先判断原因属于哪一类：舍入点或融合顺序、对 batch 的依赖、算法上的选择、
   并行布局、logprob 公式不同，还是版本记账或权重同步出了问题。然后做一个最小的改动，让这个算子处的数值误差
   变为零，再重放；加上改动后的重放结果等于另一端的结果，原因就确认了。
6. **把修复保留下来**：把这项检查连同它的基线加入[回归](../4-operations/regression.md)套件，
   之后的每个版本都必须复现这个结果。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 开启一致性、同步运行、只跑一段短窗口 | 每个 token 的差距恰好为 0，或者得到一个可以定位的失败 | 一致性配置本身的代价，只在窗口内付出 | 开始长时间运行之前，或者要为基础设施排除嫌疑时 |
| 每一步都记录并取 fingerprint，按抽样保存版本快照 | 可以离线拆分任何一个记录过的 batch | 不含快照时为小（百分之几）；快照的代价随模型规模增长 | 每一次你可能需要审计的生产运行 |
| 离线重放并重新采集 | 定位单个 token 的数值误差 | 每次定位只需几秒，而且不在训练路径上 | 只用于数值误差非零的 token |

已证明：在一个小训练循环中，两端都使用 batch-invariant 算子集时，无论同步还是异步运行，
每个只属于一个版本的 token，数值误差都逐 bit 为零，因此全部差距都来自 staleness。即使经过融合后的重新
打包，权重检查也指出了人为植入的错误行。人为植入的代码故障，在一次已知答案的演示中被
定位到了对应的算子；这次演示所用的数据流顺序是根据模型代码写出来的，而不是自动发现的
（[报告](../../examples/rl_loop/REPORT.md)）。尚未证明：大模型、任何一端使用 TP/EP、
CUDA graph，以及在循环中使用生产引擎。

## 可能出错的地方

!!! warning "引擎的标志只是一种声明"
    rollout 记录下来的标志可能写着 batch-invariant，实际运行的却是原生算子集。如果相信
    这些标志，就会到错误的一端去找。要把标志当作声明，由数值拆分来下结论。

!!! warning "版本标签出错，看起来就像数值误差"
    一个标着某个版本的 batch，实际用的可能是另一个版本的权重。在线看，它的数值误差
    不为零；可是在记录的版本上重放，结果却和 trainer 相等，也找不到任何不同的行。
    要把这个 batch 的权重 fingerprint 和每次同步的版本逐一比较，再在相符的版本上重放。

!!! warning "形状不同时，按顺序配对事件会什么也报告不出来"
    一次 trainer 的 prefill 和多次引擎的 decode 前向，调用函数的次数不同，所以按先后
    顺序配对事件的 diff 可能什么差异都报告不出来。这种情况下没有报告并不是证据。
    应当在声明好的 token 维度上比较逐向量 fingerprint。

!!! warning "fingerprint 为零不代表两端一致"
    有些行的 XOR 本来就是零，例如由两个相同的半段组成的行，分组查询注意力在位置 0 处
    就会出现这种情况。如果把它们算作相等，追查就会过早停止。只靠为零的 fingerprint 得到的一致，
    要视为没有证据。

!!! warning "融合后的生产者找不到对应项"
    融合投影输出的一行，比 trainer 中任何单独一行都宽，所以要求宽度相等的追查找不到
    配对。应当在它被拆开之后的第一个使用者那里进行比较。

!!! warning "重放只有在说明了 batch 组成时才能复现"
    重放的 rollout 只有在使用记录下来的 batch 组成，或者使用 batch-invariant kernel 时，才能得到
    原来的数值。两者都不满足时，重放结果会因为与故障无关的原因而不同。要说明你用的是
    哪一种。

## 小结

- 运行时以低成本在线记录；版本快照只按你负担得起的频率保存。
- 先检查权重，再拆分每个差距；跨越两个版本的 token 先放到一边。
- 数值误差为零，就为这些 batch 排除了基础设施的嫌疑；staleness 交给重要性采样去校正。
- 不为零的数值误差，要沿数据流离线定位，要有两端的证据，并通过干预确认原因。

## 深入阅读

- [实验报告](../../examples/rl_loop/REPORT.md)：训练循环本身、每个场景、植入的故障、
  harness 发现了什么以及如何修复，还有代价。
- 代码：[rl_core.py](../../examples/rl_loop/rl_core.py)、
  [rl_loop.py](../../examples/rl_loop/rl_loop.py)、
  [tools/opguard_bridge.py](../../tools/opguard_bridge.py)、
  [tools/graph_probe.py](../../tools/graph_probe.py)。
- 针对真实引擎、定位到单个元素的例子：
  [报告](../../examples/cross_engine/REPORT.md#worked-example-one-element-at-q_norm)。
- OpGuard：`docs/RLAlignment.md` 和 `rl-alignment` skill。
- 下一步：[实战](walkthrough-verl.md)，在一个真实的训练任务上走完这一部分的每一步；
  然后是[回归](../4-operations/regression.md)。
