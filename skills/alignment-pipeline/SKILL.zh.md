---
name: alignment-pipeline
description: 判断一个逐 bit 对齐问题该走哪条流程，并把它从头做到尾。预训练场景（两次本应一致的运行、改动后出现 loss spike、恢复训练后结果漂移、怀疑有 SDC）依次是：预检，选定参考运行，用 OpGuard 或 graph probe 采集，找首个分歧点，修复，确认，加回归条目。RL 训推场景（rollout 和 trainer 的 logprob 对不上）依次是：审计模型，让每个算子都 batch-invariant，检查整个模型的 L1–L3 和 L5，检查采样，对照引擎检查 L4，在线记录，把差距拆成 staleness 和数值误差，离线定位，加回归条目。先用这个 skill，它会再调用更具体的 skill。
---

# 对齐流程

这个 skill 只负责分流。每一步都写明四件事：什么时候开始，用哪个工具或 skill 来做，
做完要留下什么证据，什么时候停下。具体怎么做、要守哪些规则，都写在链接的 skill
和 handbook 各部分里，这里不再重复。

这里说的相等一律是逐 bit 一致。不设容差，也不靠舍入把结果凑成相等。
每个结论都要写明是在哪种 GPU、哪些软件版本上得到的；没测过就写“未测试”。

## 先判断走哪条流程

| 用户手上的情况 | 流程 |
|---|---|
| 两次训练本应一致；或者升级、迁移、改配置、恢复训练之后，出现了 spike 或漂移 | A. 预训练 |
| 怀疑某台机器在悄悄算错数据 | 走 A，只做 A0 这一步 |
| rollout 的 logprob 和 trainer 的对不上；或者 RL 训练已经悄悄变成了 off-policy | B. RL 训推 |
| 某个算子还没有 batch-invariant 实现 | 直接用 [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) |
| 框架迁移、checkpoint 转换、预先算好的参考 logprob，或者其他需要对齐的一对运行 | 见[第 5 部分](../../handbook/5-scenarios/README.md)；checkpoint 转换用 [tools/ckpt_compare.py](../../tools/ckpt_compare.py) |
| 换了新的软硬件栈（torch、Triton、NCCL、驱动、CUDA、GPU） | 先跑一遍回归套件，做法见 [regression](../../handbook/4-operations/regression.md) |

在新机器或新栈上开始任何一条流程之前，先到[硬件覆盖](../../handbook/4-operations/hardware.md)
里看看那里已经测过什么。

## A. 预训练

| 步骤 | 何时开始 | 工具或 skill | 要留下的证据 | 何时停下 |
|---|---|---|---|---|
| A0. SDC | 怀疑某个设备有问题，或者一次长时间训练即将开始 | [tools/sdc_scan.py](../../tools/sdc_scan.py)（先跑 `--quick`，再跑 fleet 和 golden）；训练任务内部用 `sdc_guard` 和 `cross_rank_check`（[案例](../../handbook/1-pretraining/sdc.md)） | 扫描结果 JSON，里面写明是哪个设备、哪个算子、第几次重复 | 某个设备和同组其他设备的结果不一致（先排除软件上的差异），或者全部通过 |
| A1. 预检 | 每次都做，两次运行都要做 | [tools/determinism_preflight.py](../../tools/determinism_preflight.py)：恢复状态之后在每个 rank 上运行 `report`，再用 `diff(a, b, declared=[...])` 比较 | 两份报告和 diff 结果，其中除了已声明的那处差异，没有别的差异 | 出现了未声明的差异，逐 bit 比较因此失去意义；先改好设置，再重跑 |
| A1'. 数据 | 预检没有问题 | [tools/data_fingerprint.py](../../tools/data_fingerprint.py)：两次运行都在 batch 离开 loader 的地方调用 `Recorder(dir, rank)` 和 `record(step, batch)`，然后运行 `compare`（DP 规模不同时加 `--global`）（[页面](../../handbook/1-pretraining/data-alignment.md)） | 两份记录目录和 compare 的结果 | 发现某条记录不同；按它的类别（order、samples、packing、masks）修好，再比较一次 |
| A2. 参考运行 | 预检没有问题 | [第 1 部分第 1 步](../../handbook/1-pretraining/README.md#step-1-pick-a-reference-run) | 把要测试的那唯一一处差异写下来 | — |
| A3. 采集 | 运行在 eager 模式下 | 通过 [tools/opguard_bridge.py](../../tools/opguard_bridge.py) 的 `run_pair` 用 OpGuard 采集，再检查 `coverage(run)` | 两份 trace 和覆盖报告 | 覆盖报告缺了你要看的那段代码；先把包装修好 |
| A3'. 采集 | 运行用了 `torch.compile` 或 CUDA graph | [tools/graph_probe.py](../../tools/graph_probe.py)，并加上 `isolate=True`（[案例](../../handbook/1-pretraining/capture-compiled-code.md)） | probe 的 trace；以及带 probe 和不带 probe 各编译一次、结果逐 bit 一致的证明 | probe 改变了编译后的结果；把 probe 隔离开 |
| A4. 首个分歧点 | 拿到了两份 trace | `opguard_bridge.first_divergence` | 分歧点的信息：哪个算子、哪行代码、哪次调用、哪些字段不同，以及在它之前有哪些没对齐的事件 | 没有任何事件不同，最终参数也相等，说明两次运行是一致的 |
| A5. 找原因 | 找到了分歧点 | [pretraining-alignment](../pretraining-alignment/SKILL.md)（第 5–6 步），配合 OpGuard 的 harness skill；用 `suggest()` 找线索 | 一个假设，并且已经用能验证它的最小干预检验过 | 这个干预消除了分歧点上的差异 |
| A6. 修复 | 原因已经确认 | 设置或状态的问题，见[训练 step](../../handbook/1-pretraining/training-step.md)和 [DP 与 FSDP 风格的切分（模拟实现：扁平梯度、reduce-scatter、按 shard 存放的 AdamW）](../../handbook/1-pretraining/data-parallel.md)；kernel 的问题，用 [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | 改动本身和它的代价（先预热、做同步，再取中位数） | — |
| A7. 确认 | 已经做了修复 | 把这对运行重新采集一次 | 没有不同的事件，也没有没对齐的事件，最终参数逐 bit 一致 | 确认完成；否则回到 A4 |
| A8. 回归 | 修复已经确认 | 写一个支持 `--quick`/`--json`、按 `bi-results-v1` 格式输出结果的示例，并在 [suite/manifest.json](../../suite/manifest.json) 里加一条（[regression](../../handbook/4-operations/regression.md#how-to-add-a-check-or-an-example)） | 示例、它的结果文件和一个基线 | 回归套件在这个栈上通过 |

## B. RL 训推

顺序很重要，因为每一级都以前面几级成立为前提。只有某一步的开始条件不满足时，
才可以跳过这一步，而且要说明跳过了。

| 步骤 | 何时开始 | 工具或 skill | 要留下的证据 | 何时停下 |
|---|---|---|---|---|
| B1. 审计 | 拿到了 trainer 的模型 | 在模型上运行 [tools/bi_audit.py](../../tools/bi_audit.py) | 算子清单，以及每个模块的 L2 判定 | 每个模块都满足 L2，或者已经列出了不满足的模块 |
| B2. 单个算子的 L2 | 有算子不满足 L2 | [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | 这个算子的验证矩阵和代价 | 每个算子都通过，或者都有记录在案的回退方案 |
| B3. 整个模型的 L1–L3 | 所有算子都满足 L2 | 按[整模型案例](../../handbook/2-rl/whole-model.md)的方法做，覆盖每种推理模式，也覆盖激活梯度 | 每种模式下的 logprob 都相等 | 引擎用到的每种模式都满足 L1–L3 |
| B4. L5 | trainer 和引擎的 TP、DP 或 EP 规模不同 | [whole-model-tp](../../handbook/2-rl/parallelism.md)、[training-dp-fsdp](../../handbook/1-pretraining/data-parallel.md)、recipe 10 | 换并行规模后结果仍然相等，前向和反向都要 | 用到的每种规模下结果都相等 |
| B5. 采样 | token 会经过采样、重放、抢占或 speculative decoding | [tools/bi_sampling.py](../../tools/bi_sampling.py)（[案例](../../handbook/2-rl/sampling.md)） | 在重算、回滚、改变 n、speculative decoding 分块和换新进程之后，重放的结果 | token 能逐 bit 重放出来 |
| B6. L4 | 两端各自都满足 B3 | [engine-parity](../engine-parity/SKILL.md)（[vLLM 案例](../../handbook/2-rl/engine-parity-vllm.md)）；LoRA 见 [LoRA 案例](../../handbook/2-rl/lora.md) | 两端的舍入序列，以及逐算子、逐层的比较结果 | 0 个 logprob 不同，或者剩下的每处差异都是记录在案的决定，并写明了代价 |
| B7. 在线记录 | RL 训练任务正在运行 | OpGuard 的 `opguard_harness.rl_records.RecordWriter`，它记录 rollout batch、trainer 的 logprob、前向、权重同步和快照（[第 2 部分](../../handbook/2-rl/README.md#synchronous-and-asynchronous-runs)） | 每个 rank 的记录文件 | — |
| B8. 拆分 | 拿到了记录 | 先运行 `rl.weight_sync_check`，再运行 `rl.decompose` | 对不上的权重行（应当没有）；每个 token 的 `numeric` 和 `staleness`；跨多个权重版本生成的 token 单独放到一边 | `numeric` 逐 bit 为零，说明基础设施没有问题，差距全部来自 staleness |
| B9. 定位 | `numeric` ≠ 0 | OpGuard 的 `rl-alignment` skill（`harness/skills/rl-alignment`）：先 `rl.replay_request`，再用 `OPGUARD_XOR_ROWS=1` 重新采集，然后用 `vectors.*`；见[示例](../../handbook/2-rl/rl-loop.md) | 出现分歧的算子、来自另一端的证据，以及原因属于哪一类 | 已经用最小的改动验证了原因；然后回到 B2、B4、B5 或 B6 去修复 |
| B10. 回归 | 已经做了修复 | 和 A8 一样 | 示例和基线 | 回归套件在这个栈上通过 |

## 什么时候停下来问用户

- 修复的代价超出了用户同意的范围。报告测得的代价，并说明哪些更便宜的级别已经成立。
- 差异来自算法，也就是同一个数学有两种精确的写法。这是一种选择，不是 bug；
  把两种写法和各自的代价都报告出来。
- 引擎或 kernel 是闭源的，没有可调的开关。报告有哪些回退方案（batch-invariant-kernel 的第 2 步）。
- 新栈上某项必需的回归检查没有通过。这是要报告的发现，不能靠调参把它消掉。
