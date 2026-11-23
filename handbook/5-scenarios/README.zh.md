# 5. 其他对齐场景

## 我们在哪里

第 1 到第 4 部分通过几条工作流程讲了同一套方法：找到两次预训练从哪里开始不一致
（[第 1 部分](../1-pretraining/README.md)），让 rollout 引擎和 trainer 逐 bit 一致
（[第 2 部分](../2-rl/README.md)），让单个算子满足不变性
（[第 3 部分](../3-kernels/README.md)），以及在软件栈变化时让这些结论继续成立
（[第 4 部分](../4-operations/README.md)）。很多别的需求其实是同一个问题，只是换了一对程序来问：
迁移训练框架、转换 checkpoint、在推理引擎里跑参考模型，等等。这一部分讲怎样把已经掌握的方法
用到七个这样的场景上。其中三个有单独的页面，另外四个在本页各占一小节。

## 思路：先说清比较什么、用什么输入、要求什么成立

每个场景都由四样东西组成：一个参考方，一个被怀疑有问题的一方（suspect），两边收到的输入，
以及一个要求，即哪些输出应该相等、相等到什么程度。把这四样写下来以后，就能直接套用已有的某条工作流程。
下面是几个例子：

| 比较的两方 | 相同的输入 | 什么应该相等 | 预期结果 |
|---|---|---|---|
| 一个 checkpoint 和它转换后的副本 | 没有，因为什么都不用计算 | 每个权重的每一行 | 永远逐 bit 一致 |
| 同一个模型在两个训练框架里 | 数据、权重、种子 | 两边都有的每一个边界 | 逐 bit 一致，但前提是所有舍入点都已对齐 |
| trainer 和预先算好的参考 logprob | 相同的 token id 和权重 | 每个 token 的 logprob | 做到引擎一致性之后逐 bit 一致 |
| 单独发出的一个请求，和繁忙服务器里的同一个请求 | 这个请求 | 它的 token 和 logprob | 推理服务做到 batch-invariant 之后逐 bit 一致 |
| 同一个训练任务跑在两代 GPU 上 | 全部相同 | 不要求逐 bit 一致 | 结果会不同，所以改为定位 |

最后一行和前面几行一样重要。在没有承诺逐 bit 一致的地方，目标是找到第一个出现差异的算子，
并对它做出决定，而不是不惜代价追求相等。所以规则是：**比较任何东西之前，先写清楚比较哪两个
程序、用什么输入、要求什么成立。**这其实就是[第 1 部分](../1-pretraining/README.md#the-idea-two-runs-one-difference)
里的对照实验，只是把要求明确地写了出来。

## 各个场景

| 场景 | 复用哪条工作流程 | 工具和 skill | 这个场景特有的问题 |
|---|---|---|---|
| [框架迁移与库升级](framework-migration.md) | [第 1 部分](../1-pretraining/README.md)，以另一套软件栈为参考；再加上[引擎一致性](../2-rl/engine-parity-vllm.md)里对齐舍入点的方法 | [determinism_preflight.py](../../tools/determinism_preflight.py)、[opguard_bridge.py](../../tools/opguard_bridge.py)、[engine-parity](../../skills/engine-parity/SKILL.md) | 两个程序的 trace 对不齐；每一处差异要么是 bug，要么是一种选择 |
| [checkpoint 转换与重新切分](checkpoint-conversion.md) | 不需要计算：先检查副本，再跑一次前向 | [ckpt_compare.py](../../tools/ckpt_compare.py) | 张量重新打包后，每一行的值保持不变；行有没有放对位置，要靠前向来检查 |
| [离线后训练](offline-post-training.md)：SFT 和 DPO 用的参考 logprob，蒸馏用的 teacher logits | [第 2 部分](../2-rl/README.md)的 L4，但没有采样 | [engine-parity](../../skills/engine-parity/SKILL.md)、OpGuard 的 RL 记录 | 对哪些 token 打分、对话模板、padding |
| [奖励模型与评测](#reward-models-and-evaluation) | [整个模型](../2-rl/whole-model.md)、[采样](../2-rl/sampling.md)、[回归](../4-operations/regression.md) | [bi_audit.py](../../tools/bi_audit.py)、[bi_sampling.py](../../tools/bi_sampling.py) | 分数接近阈值时，差一个 bit 就可能改变判定 |
| [确定性的推理服务](#deterministic-serving) | 引擎端的[整个模型](../2-rl/whole-model.md)；[第 3 部分](../3-kernels/README.md) | [bi_audit.py](../../tools/bi_audit.py)、[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md) | 一个请求可能遇到的每一种推理模式 |
| [speculative decoding 与 MTP](#speculative-decoding-and-mtp) | [采样](../2-rl/sampling.md)、L3 下的[整个模型](../2-rl/whole-model.md) | [bi_sampling.py](../../tools/bi_sampling.py) | 验证一次处理一段 token，decode 一次只处理一个 |
| [硬件迁移与混合集群](#hardware-migration-and-mixed-fleets) | [硬件](../4-operations/hardware.md)、[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md) | [sdc_scan.py](../../tools/sdc_scan.py)、[bi_suite.py](../../tools/bi_suite.py) | 不同代的 GPU 之间，本来就不指望逐 bit 一致 |

## 奖励模型与评测

这里要做到：奖励模型给一个（prompt，response）对打分，放在任何 batch 里分数都相同；评测重跑
一次，得到的数字也相同。奖励模型就是一个带标量输出头的完整模型，所以它的输出和策略模型一样，会随 batch 和推理模式变化。
不同的是输出的用法：分数要和阈值比较，要和其他回答排序，或者被平均成通过率。所以一个离阈值
只差一个 bit 的分数就可能改变判定，判定一变，训练信号或者你报告的数字也就跟着变了。

怎样套用已有的页面：

1. **让奖励模型满足不变性**，和对待任何模型一样：用 [bi_audit.py](../../tools/bi_audit.py)
   审查它的算子，并把一个样本单独打分的结果，与它放进 batch、做 padding 或 packing 之后的结果
   比较（[整个模型](../2-rl/whole-model.md)）。
2. **如果有引擎，就和它对齐。**如果分数是在引擎里算、再由 trainer 重算的，那就是 L4 的问题，按[引擎一致性](../2-rl/engine-parity-vllm.md)来做。
3. **让采样可以重放。**如果评测需要采样，就要让 token 能从 logits 和种子重放出来
   （[采样](../2-rl/sampling.md)、[bi_sampling.py](../../tools/bi_sampling.py)）。
4. **保留一个固定的评测集**，把它和它的分数作为一个[回归](../4-operations/regression.md)条目保存下来。

每个样本的分数在各种 batch size 下都逐 bit 一致，换一种 batch 组成重跑，评测结果也相同，
就算完成了。

尚未证明：这里还没有测过任何奖励模型或评测框架。

## 确定性的推理服务

这里要做到：一个请求的 token 和 logprob，既不受同一个 batch 里其他请求的影响（L2），也不受
服务器怎样切块、缓存或 decode 它的影响（L3）。这其实就是[整个模型](../2-rl/whole-model.md)那一步里的引擎端，只是换成了用户的视角：两个完全
相同的请求在 temperature 0 下得到了不同的回答，原因是服务器当时的负载不一样。原因和修法都相同。

怎样套用已有的页面：

1. **审查部署的模型**，并测试一个请求可能遇到的每种模式：不断变化的 decode batch、chunked prefill、
   命中 prefix cache、被抢占（[整个模型](../2-rl/whole-model.md)）。
2. **把每个随 batch 变化的算子换成不变的 kernel**（[第 3 部分](../3-kernels/README.md)），
   并让采样器保持可重放（[采样](../2-rl/sampling.md)）。
3. **测量代价**，用你实际的 decode batch size。如果代价太高，就只让需要的请求走不变的路径，
   而不是所有请求都走。

每个请求不管是单独处理还是在高负载下处理，logprob 都逐 bit 一致，并且服务器用到的每种模式
都是如此，就算完成了。

尚未证明：真实流量下的生产服务器，以及跨多个节点的推理服务。

## speculative decoding 与 MTP

这里要做到：用了 speculative decoding 或多 token 预测（MTP）之后，输出的 token 和 logprob
与目标模型自己 decode 时完全相同。验证阶段让目标模型一次处理好几个草稿 token，很像一次短的 prefill；普通的 decode 则一次只处理
一个 token。如果这两条路径的舍入方式不同，被接受的 token 的 logprob 就会和普通 decode 给出的
不一样，连接受与否的判断本身都可能翻转。至于草稿模型自己的数值，只要接受时比较的是目标模型
自己的采样结果，就只影响速度，不影响结果。

怎样套用已有的页面：

1. **让目标模型验证时的前向，等于它 decode 时的前向。**一段 token 对一个 token，正是
   [整个模型](../2-rl/whole-model.md)里 L3 的情形；注意力在 prefill 和 decode 中必须用同一种
   方式切分 key（[注意力](../3-kernels/attention.md)）。
2. **让接受过程可以重放。**用精确相等来验证草稿时，speculative decoding 给出的 token 与不用草稿、直接 decode
   的结果完全相同（[采样](../2-rl/sampling.md)、[bi_sampling.py](../../tools/bi_sampling.py)）。
3. **当作草稿用的 MTP 头，按同样的方式处理。**预训练里的 MTP loss 则属于训练 step 那一部分
   （[第 1 部分](../1-pretraining/README.md)）。

不管用不用 speculative decoding、草稿长度是多少，token 和 logprob 都逐 bit 一致，就算完成了。

尚未证明：真实引擎里的 speculative decoding，以及 MTP 头。

## 硬件迁移与混合集群

这里要做到：知道哪些结论能带到新 GPU 上；两台机器结果不一致时，找到它们从哪里开始分开。
不同代的 GPU 之间，本来就不指望逐 bit 一致，因为库会针对每种架构选择不同的 kernel 和累加顺序
（[硬件](../4-operations/hardware.md)）。所以目标从“相等”变成了“定位”。第一个出现差异的算子，
应当是库里的 GEMM，或者由库决定顺序的归约；如果是别的算子，那就是一项值得报告的发现。

怎样套用已有的页面：

1. **给新硬件做验收**：用已经提交的基线跑一遍回归套件，再在各组设备之间做 SDC 扫描
   （[硬件](../4-operations/hardware.md)、[bi_suite.py](../../tools/bi_suite.py)、[sdc_scan.py](../../tools/sdc_scan.py)）。
2. **设备之间逐 bit 比较**，限于同一代 GPU、同一套软件栈之内；这里出现不一致，就是故障
   （[SDC](../1-pretraining/sdc.md)）。
3. **跨代时做定位。**用 OpGuard 在相同的输入上采集两次运行，找出首个分歧点
   （[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md)），再给它分类：要么是随架构而变的
   kernel 选择，属于预期之内；要么是别的原因，需要报告出来。
4. **处理跨代的 RL。**如果 RL 里的 trainer 和引擎跑在不同代的 GPU 上，只有归约顺序写死在源码里的 kernel 才可能
   保持一致，而且这一点在不同架构之间还没有测试过。否则就测量数值差距，把它记录为一项决定，
   就像[引擎一致性](../2-rl/engine-parity-vllm.md)处理两端都无法消除的差异那样。

不同代之间的每一处差异都已经分类，预期之内的差异连同它所属的软件栈一起记录在案，就算完成了。

尚未证明：手册里的 kernel 在其他 GPU 架构上的表现，以及混用不同代 GPU 的 RL 训练。

## 可能出错的地方

!!! warning "在没有承诺的地方期待逐 bit 一致"
    两代 GPU，或者两个不同库的 kernel，本来就不是按最后一个 bit 都一致来设计的。在那里追求
    相等，时间都会花在预期之内的差异上。先写下要求；如果要求不是相等，就去定位，然后做决定。

!!! warning "用容差代替逐 bit 检查"
    “足够接近”的输出，可能掩盖恰好很小的故障。例如一次只把小权重舍入掉的类型转换，在少量输入
    上可能一个 logit 都不改变（[checkpoint 转换](checkpoint-conversion.md)）。副本和重算都要
    逐 bit 检查；容差只用在本来就不可能一致的两方之间。

!!! warning "比较的是两个不同的计算"
    如果参考 logprob 打分的是另一批 token、只有一端套用了对话模板，或者分数经过了不同的后处理，
    数字就会不同，但数值计算上并没有错。比较数值之前，先确认两端算的是同一件事。

## 小结

- 每个场景都由比较的两个程序、两边收到的输入和一个要求组成，要先把它们写下来。
- 副本必须逐 bit 一致；两个不同的程序可以通过对齐舍入点做到逐 bit 一致；两代 GPU 之间只做定位，
  不追求相等。
- 已有的工作流程可以原样套用：预检、采集、找首个分歧点、对齐舍入点、做决定、加入回归。
- 每一节都写明了尚未证明的部分，把它们当作还没解决的问题。

## 深入阅读

- 页面：[框架迁移](framework-migration.md)、
  [checkpoint 转换](checkpoint-conversion.md)、
  [离线后训练](offline-post-training.md)。
- [报告：checkpoint 转换](../../examples/ckpt_conversion/REPORT.md)，
  包括植入的故障，以及每项检查看到了什么。
- Skill：[alignment-pipeline](../../skills/alignment-pipeline/SKILL.md)、
  [engine-parity](../../skills/engine-parity/SKILL.md)、
  [pretraining-alignment](../../skills/pretraining-alignment/SKILL.md)。
