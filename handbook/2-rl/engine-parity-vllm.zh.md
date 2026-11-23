# 与 vLLM 逐 bit 一致（L4）

## 我们在哪里

经过前面几步，每一端都已经各自满足不变性：[整个模型](whole-model.md)在每种推理模式下、
每种 [TP 规模](parallelism.md)下都算出逐 bit 一致的结果，[采样器](sampling.md)采出的
token 可以重放，[LoRA](lora.md) adapter 在任何 batch 里的舍入方式也都相同。不过在 RL 中，
推理引擎通常不是你自己的代码，而是 vLLM 或 SGLang，有自己的模型代码、kernel 和编译器。
这一步要让 trainer 算出的 logprob 与引擎的逐 bit 一致（L4）。

## 思路

两端即使各自都不变，结果仍然可能不同，因为它们舍入的地方不一样。**舍入点**指的是把一个值
转换成更窄的类型的地方，或者把几步运算合并成一次舍入的地方。看一个 RMSNorm 的小例子：
x̂ 是 fp32 的归一化结果，w 是 bf16 的权重。

| 哪一端 | 算的是什么 | 舍入几次 |
|---|---|---|
| A | bf16(x̂) · w，结果再转成 bf16 | 两次：先转换 x̂，再转换乘积 |
| B | bf16(x̂ · w) | 一次：在 fp32 里相乘，最后转换一次 |

两种写法都没错，但很多元素的结果不再逐 bit 一致。一个模型里有几十个这样的点：投影怎样
打包、bias 在哪里加、norm、RoPE 表和公式、注意力 kernel、lm_head、log_softmax。
logits 上游的每一个点都对上之后，才会有哪怕一个 logprob 相等。所以即使前面几次修复都是
对的，结果不同的 token 也可能一直接近全部，直到修好最后一处才一下子降到零。

做法就由此而来：

1. **先拿引擎和它自己比较。** 如果其他请求加入 batch 后，引擎给某个 token 的 logprob
   就变了，那么任何 trainer 都不可能对上它。
2. **把两端的舍入序列都写下来**，按先后顺序，每个舍入点一行。
3. **一次只对齐一个舍入点**，每改一处就测一次。要看第一个出现差异的层往模型深处推进了
   多少，而不只看有多少 token 不同。
4. **trainer 里尽量直接调用引擎自己的 kernel**，而不是自己重写一份。源码相同并不意味着
   结果逐 bit 一致，除非 tile 形状、归约顺序和编译器也都相同。

背后的通用原则和写 kernel 时一样：要结果相同，需要的是运算过程相同，而不只是公式相同。

## 步骤

[engine-parity skill](../../skills/engine-parity/SKILL.md) 把下面这些动作串成一个工作流程来执行。

1. **盘点两端的前向**，弄清每一端实际运行的是什么。从源码和加载后的模型里列出：打包和融合过的模块、
   自定义算子、是否用了编译器和 CUDA graph、实际选中的注意力后端，以及每一个会改变 kernel 的开关
   （skill 第 1 节）。模块树和后端要从正在运行的引擎里打印出来，而不是照抄你传进去的配置。
2. **写下两端的舍入序列**，填进[舍入点](../../skills/engine-parity/references/rounding-points.md)里的表，
   每一条都标明是“根据源码阅读”还是“测量所得”。最后得到一份检查清单，两端之间的每一处差异都是表里有名字的一行。
3. **拿引擎和它自己比较**，让引擎在和任何 trainer 比较之前先达到 L1 和 L2。同一个 batch 跑两次；
   每个 prompt 单独跑一次，再放进几种不同大小的 batch 里跑。如果引擎有 batch-invariant 模式，就打开它，
   并查清楚这个模式在你的 GPU 上实际覆盖了哪些算子。每个 prompt 单独跑和放进 batch 跑，logprob 都要相同；
   如果不同，就把它作为引擎内部的波动报告出来，并在引擎里修。
4. **一次一个舍入点地对齐 trainer**，直到采样出的 token 的 logprob 达到 L4。用 teacher forcing
   把引擎自己生成的 token 送进 trainer，逐 bit 比较 fp32 的 logprob，不设任何容差；改一个点，再重新数一遍。
   [trainer.py](../../examples/cross_engine/trainer.py) 和
   [cross_engine.py](../../examples/cross_engine/cross_engine.py) 可以当模板用。没有任何 token 不同时就完成了；
   之后再做一遍逐一回退（每次只撤销一个点），就能看出哪些点真正有影响。
5. **定位剩下的差异**，找到第一个仍然不同的算子。做算子重放（把采集到的每个引擎输入，分别喂给 trainer
   的每个候选实现），并逐层比较残差流；如果引擎经过了编译，就在算子边界上取 fingerprint，代替 hook
   （[RL 循环](rl-loop.md)）。结果是一个具体的算子，并有来自两端的证据。
6. **给对齐后的前向配一个正确的反向。** logprob 必须逐 bit 一致，梯度只需要正确，所以反向不必和
   前向 kernel 逐 bit 一致。小算子手写 fp32 导数；注意力和 MoE 的梯度，则用保存下来的输入在
   PyTorch 里重算得到，而前向仍然用引擎 kernel 算出的值。每个参数的梯度与一个独立的 fp32 参考
   相比，相对 L2 误差都不超过 1e-3，就完成了；[报告](../../examples/verl_fsdp2_vllm/REPORT.md#gradient-check)
   是一个完整的例子。

| 途径 | 做什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 让 trainer 改用引擎的 kernel | trainer 直接调用引擎自己的 GEMM、norm、注意力和 log_softmax | 引擎端因为要开不变模式，代价为中等（不到 2 倍）到大；trainer 端的前向免费 | 引擎不能改，trainer 由你掌控；trainer 还需要为这些 kernel 配一个正确的反向，但反向不必和前向逐 bit 一致 |
| 让引擎改用 trainer 的舍入方式 | 拆开并重排引擎的算子，让类型转换的位置跟 trainer 一样；两端都用不变的算子 | rollout 吞吐量的代价为中等（不到 2 倍）到大（报告值） | trainer 不能改，而你可以给引擎打补丁 |
| 两端共用一份模型定义 | 引擎只保留缓存管理，前向由两端共用，剩下要对齐的只有 kernel | 某些架构上 trainer 吞吐量的代价为大（2 倍或更多）（报告值） | 两个框架你都能改 |
| 制定一份执行契约 | 把两端所有影响舍入的选择都固定下来，用哈希检查；使用不受并行方式影响的 kernel | 中等（不到 2 倍）（报告值） | 多个框架需要长期保持对齐 |

已证明：trainer 调用引擎自己的 kernel 后，在不同的 batch size 下，它算出的采样 token
logprob 与一个真实引擎逐 bit 一致。这是在一个模型、一种 GPU 型号、eager 模式下得到的，
只验证了前向（[报告](../../examples/cross_engine/REPORT.md#results)）。在一个真实框架的完整
RL 循环里，参数切成 shard 的 trainer 调用引擎的 kernel 后，在一次较短的同步训练中也与 rollout
逐 bit 一致。这里的 rollout 使用 CUDA graph 并关闭了编译器，模型是一个带 attention sink 和
滑动窗口的真实 MoE checkpoint，反向也在缩短版的模型上通过了梯度检查。这同样只在一个模型、
一种 GPU 型号上验证过，精度为 bf16，每个 rollout 副本只用一张 GPU
（[报告](../../examples/verl_fsdp2_vllm/REPORT.md)）。其他途径的结果来自
别的团队，汇总在[报告](../../examples/cross_engine/REPORT.md#three-ways-to-reach-parity)里。
尚未证明：用自带编译器编译过的引擎、L4 下的 TP，以及量化权重。

## 可能出错的地方

!!! warning "引擎的 batch-invariant 模式不一定在每种 GPU 上覆盖所有算子"
    一个模式开关可能只把一部分算子换成不变的 kernel，另一部分仍然走 GEMM 库，具体取决于
    GPU 架构。这时引擎仍然随 batch 变化，你调 trainer 时追的其实是噪声。要读你的 GPU
    实际走的代码路径，并亲自检查引擎是否达到 L2。

!!! warning "比较的 logprob 不是同一个东西"
    引擎报告的 logprob 可能已经经过 temperature、惩罚项或其他处理，也可能是在和你不同的
    设备上算的 log_softmax。这样比较出来的差距并不在模型里。先确认：按引擎的算法对采集到的
    logits 做 log_softmax，结果确实等于引擎报告的 logprob。

!!! warning "hook 在编译后的图里不会触发"
    编译器会把 norm、RoPE 和激活函数融合进相邻的算子，引擎在编译时也可能关掉自己的
   自定义算子。这些地方在模块级采集里什么都看不到。要么在算子边界上取 fingerprint，
    要么先对齐 eager 模式下的引擎。

!!! warning "不同 token 的数量没变，不代表这次修复失败"
    logits 上游的每个舍入点都对上之后，才会有 token 相等。如果按 token 数来评判每次修复，
    就会把正确的修复也丢掉。每次改动之后，改为跟踪第一个出现差异的层。

!!! warning "撤销后没有变化，说明不了多少"
    在逐一回退中，撤销某个点后没有任何 token 改变，这个点仍可能在别的 prompt 上、
    或者在更靠后的位置上有影响（RoPE 表可能只在某个位置之后才不同）。要说“这里没有
    显示出影响”，不要说“无关”。

!!! warning "sharding 包装层可能转换 fp32 buffer"
    一张 fp32 的表（例如 RoPE 的 cos/sin 表）跨过 shard 模块的边界时，混合精度或 sharding
    包装层可能把它转成 bf16。这时虽然每个 kernel 都没错，几乎每个 logprob 仍然都不同。
    要么把位置传过边界，在层内部再构造这张表；要么在每个边界上检查 buffer 的 dtype。

!!! warning "词表的填充列"
    lm_head 的列数可能比 tokenizer 的 token 数多；引擎可能在 log_softmax 之前把多出来的列
    屏蔽掉，而 trainer 没有，或者反过来。这时大多数 logprob 都会不同。做归一化时，要和引擎
    用同样的列。

## 小结

- 两端即使各自不变，结果也可能不同；它们还必须在相同的地方舍入。
- 先拿引擎和它自己比较，因为达不到 L2 的引擎根本无法对齐。
- 写下两端的舍入序列，一次对齐一个点，并跟踪第一个出现差异的层。
- 在 trainer 里直接调用引擎自己的 kernel，不要自己重写。
- logprob 必须逐 bit 一致，而反向只需要梯度正确。

## 深入阅读

- [实验报告](../../examples/cross_engine/REPORT.md)：一个真实的引擎和模型、各步骤的
  结果表、逐一回退的结果、引擎自身的波动、它的不变模式在哪种 GPU 上覆盖哪些算子、代价，
  以及其他团队报告的结果。
- [完整 RL 循环的实验报告](../../examples/verl_fsdp2_vllm/REPORT.md)：一个完整的 RL 循环，
  其中有参数切成 shard 的 trainer、MoE checkpoint、使用 CUDA graph 的 rollout，以及梯度检查；
  [实战](walkthrough-verl.md)一页按阶段走了一遍。
- Skill：[engine-parity](../../skills/engine-parity/SKILL.md) 和
  [舍入点](../../skills/engine-parity/references/rounding-points.md)。
- 代码：[examples/cross_engine/](../../examples/cross_engine/)。
- 下一步：[RL 循环](rl-loop.md)。
