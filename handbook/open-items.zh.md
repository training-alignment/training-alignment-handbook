# 尚未验证

手册里的结论都是在少数几种机器和软件栈上测出来的。这一页列出还没验证的部分，方便你判断自己的环境在不在覆盖范围里。
各页正文里，凡是会改变你做法的缺口，都有一行"尚未证明"。

OpGuard 本身已经在大规模集群的生产训练中使用
（[OpGuard 论文](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)）。
下面的缺口指的是本手册自己的 kernel、修复和案例。

## 硬件与精度

- **其他代际的 GPU 和其他加速器。** 所有结果都在少数几种 NVIDIA GPU 上测得，见
  [硬件覆盖](4-operations/hardware.md#untested)。换一代 GPU，kernel 可能选不同的 tile，
  所以在依赖某个结果之前，先跑一遍回归检查。
- **FP8 和 NVFP4 的 tensor-core GEMM。** 目前的 FP8 scale 都是模拟的。
- **FlashAttention 3、FlashMLA 和 FlashInfer**，在测过的 GPU 之外的代际上。

## 规模与互联

- **两个以上的节点，以及高速互联**（NVLink、NVSwitch、NVLS、InfiniBand、RoCE）。
  所有多 GPU 结果都是节点内走 PCIe、两个节点之间走普通以太网。集合通信的算法会随网络变化，
  固定的归约树需要在这些网络上重新检查。
- **真正的 `torch.distributed.fsdp`。** [数据并行的案例](1-pretraining/data-parallel.md)
  是模拟的 FSDP（扁平梯度、reduce-scatter、切分的 AdamW）。bf16 梯度、混合精度和 loss scaling
  都没测过。[verl 案例](2-rl/walkthrough-verl.md)用的是真正的 FSDP2，但只检查了 logprob，
  没有检查预训练的一步。
- **多 rank 预训练上的 OpGuard。** 预训练案例只采集了一个 rank；跨 rank 的采集和 CUPTI
  调用点后端都还没验证。
- **DeepSpeed ZeRO 和 Megatron**，包括 Megatron 自己的 all-reduce。

## 引擎与 RL 框架

- **SGLang 的 L4。** 还没跑。
- **verl 案例之外的 vLLM。** 一个真实的 MoE checkpoint 在每个副本一个 GPU、关掉编译器的条件下
  达到了 L4（[报告](../examples/verl_fsdp2_vllm/REPORT.md)）。没验证的有：张量并行或 expert 并行、
  prefix caching、speculative decoding、LoRA，以及 vLLM 默认的编译模式。
- **在编译后的 graph 里定位。** OpGuard 看到的是 Python 调用点，所以 torch.compile 或
  CUDA graph 区域里的差异，只能先在 eager 模式下复现出来再定位。
- **真实引擎写出的在线记录。** [RL 训练循环](2-rl/rl-loop.md)的记录目前只从一个玩具引擎写出过，
  还没有从 vLLM、SGLang、verl 或 OpenRLHF 写出过。
- **kernel 案例对 trainer 的比较。** MoE、MLA、稀疏注意力和线性注意力各自满足不变性，
  但都还没有和某个 trainer 对到 L4。
- **整个模型案例里的真实权重。** 一种真实架构只用随机权重检查过，而且只检查了几层
  （[报告](../examples/whole_model/REPORT.md)）。
- **在原始 batch 里重放。** RL 的重放只在 batch-invariant kernel 下跑过；混合了多个策略版本的
  batch 还没有重新采集过。

## 编译代码与反向

- **torch.compile 和 CUDA graph**，针对只在 eager 模式下测过的 kernel：attention 反向、
  paged 和 varlen attention、线性注意力、LoRA、MLA、采样、整个模型和 TP。
- **还没跑过的反向：** expert 并行下的 MoE、TP 的参数梯度，以及 KV cache 路径。

## 还需要设计的方法

- **盲测。** 目前每个端到端案例都事先埋好了已知原因，还没有在不告诉 agent 答案的情况下
  让它走完整条流程。
- **融合优化器。** 融合版 AdamW 和 SGD 的精确求和顺序还不清楚；我们对它的模拟很接近，但不精确
  （[训练 step](1-pretraining/training-step.md)）。
- **OpGuard 的覆盖缺口。** `torch.nn.Linear` 里的 GEMM 采集不到，有些模型代码也包装不了
  （[预训练案例](1-pretraining/localize-with-opguard.md)）。
- **还没写的 kernel：** 循环形式 GDN 的反向、paged attention 的反向，以及长 decode 用的固定大小
  split-KV。
- **MoE 路由规则**：按 capacity 丢 token 的规则，以及分数完全相同时怎么打破平局。

## 本仓库还没重跑的检查

- 回归检查只有 quick 模式的基线；full 模式和 CI 工作流都还没跑过。原生 kernel 的一些失败
  只在一种 GPU 上得到确认（[实测覆盖](../suite/COVERAGE.md)）。
- [审计](../AUDIT.md#5-gpu-re-run)里列出的几项 GPU 重跑还没完成。
