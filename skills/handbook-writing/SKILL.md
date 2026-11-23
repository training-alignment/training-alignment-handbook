---
name: handbook-writing
description: Write or rewrite a page of the Training Alignment Handbook as a teaching page about a workflow, so an engineer understands why each step exists and how to do it, without reading experiment logs. Use for any new page, any rewrite, and before adding a measured result.
---

# Writing a handbook page

The handbook teaches three workflows: finding where two pre-training runs part
(part 1), making a rollout engine and a trainer agree (part 2), and making one
operator invariant (part 3). Part 0 teaches the ideas all three rest on. Each
page is one step or one idea of a workflow. Experiments are evidence and live
in reports, not on the page.

## Who reads it

An ML systems engineer who knows PyTorch, GPUs and RL training and has a
problem now. They read a page to learn what to do next and why.

## How a page is built (the d2l order)

1. **Where we are.** One or two sentences linking to the previous step: what
   the reader has now and what is still wrong. The first page of a part says
   what the whole workflow achieves and lists its steps.
2. **The idea, by intuition first.** Explain the mechanism with a tiny example
   or a picture before any rule: three numbers added in two orders; one row in
   a batch of 1 and of 64; a decode step against a prefill. Then state the
   rule in one sentence and name the general principle it instances.
3. **The step.** What to do, in order, as a short numbered list. Each step
   is a bold imperative followed by two or three sentences of prose: what to
   do and with which tool or skill (linked), and how you know it worked. Say
   why only when the reason is not obvious from the step itself. Do not split
   every step into "Goal / How / Success" labels: the goal usually repeats
   the step's name, and the uniform labels make a page read like a form.
   Options as a small table: option, what it gives, cost class, when to
   choose it.
4. **What can go wrong.** Pitfalls as `!!! warning` admonitions: one sentence
   of cause, one of consequence, one of what to do.
5. **Summary.** Three to five bullets the reader should remember.
6. **Go deeper.** The report with the measurements, the code, the skill.

## What the page does not contain

- **No specific experiments.** No GPU model, driver, library version, model
  size, batch size sweep, timing, percentage of elements, or count of checks.
  Those are in `examples/<name>/REPORT.md`. The page states the general
  behaviour ("cuBLAS can pick a different kernel when the number of rows
  changes, so the same row can round differently in a different batch") and
  links the report as evidence ("measured: see report").
- No description of how an agent built something, no retries, no file-by-file
  inventory.
- No code identifiers in explanatory sentences. Tool and skill names appear
  only in "The step" and "Go deeper".

Cost is described by class, not by number: "free", "small (a few percent)",
"moderate (under 2x)", "large (2x or more)", and the report has the numbers.
A claim of what holds (bitwise equal, invariant at L2) may be stated, with a
link to its evidence and the conditions it was shown under in plain words
("shown on one GPU type", "shown up to 16 GPUs on two nodes").

## Sentences

- One idea per paragraph; start with it.
- Define each term at first use or link the glossary; use the same word for
  the same thing everywhere.
- Prefer a figure (Mermaid) or a small table to a paragraph of structure.
- No sales words, no "simply", no hedging chains.

## Natural language

Write as a careful engineer explains to a colleague, not as a summary of
notes. Read every title and sentence aloud; if a reader would have to decode
it, rewrite it.

- **Titles are short and plain**, like entries in a book's table of
  contents: a noun phrase of a few words (at most about six English words or
  fifteen Chinese characters), not a sentence and not a compressed formula.
  The page's opening paragraph says the rest. "LoRA: one adapter per
  training batch, many per serving batch" and "LoRA: matching training and
  serving when one batch mixes many adapters" both become "LoRA with
  mixed-adapter batches"; "Getting the same result at any tensor-parallel
  degree (L5)" becomes "Any tensor-parallel degree (L5)". Section headings
  follow the same rule.
- **No fragments, no stacked noun phrases, no colon chains.** Full sentences
  with a subject and a verb. Avoid "X: Y, Z" headlines inside prose.
- **Do not narrate the page or coach the reader.** No "this page will walk
  you through", "the way a textbook…", "worth reading slowly", "the subtle
  part", "the question for the rest of the page". State the case and the
  result; the reader is a colleague, not a student.
- **Say who does what.** "The engine groups the tokens that share an adapter"
  rather than "adapter grouping by the engine".
- **One term per thing, explained once.** Do not mix "bits", "位" and
  "比特" for the same idea; pick the glossary word.
- **Generic section names stay plain** ("Where we are", "The idea", "The
  steps"), but the content under them must read as connected prose, with
  transitions ("so", "because", "this means").
- **Chinese is written as Chinese**: natural word order, no translated
  English syntax, no literal renderings such as "位都相同", "两侧相遇的位置",
  "以相同方式舍入". Prefer "结果逐位一致", "在哪一步把两部分加起来",
  "在同一个地方做舍入". Keep the glossary's English terms (rollout, trainer,
  kernel, logprob, …) but explain them in Chinese around them.

### Chinese terms: keep the English words Chinese engineers use

On zh pages keep conventional English terms instead of coined translations.
Translate only words that are ordinary Chinese (确定性, 不变性, 归约, 舍入, 算子,
采样, 梯度, 优化器, 精度, 并行度). "Job" is 训练任务/任务, not 作业; "side" is
rollout 端 / trainer 端 / 两端, not 侧.

| Keep in English | Do not write |
|---|---|
| bit（"逐 bit 一致"） | 位、比特（"位都相同"） |
| batch、token、kernel、GEMM、tile、split-K | 批次、令牌、内核、通用矩阵乘 |
| prefill、decode、KV cache | 预填充、解码、键值缓存 |
| TP、DP、EP、PP、SP，以及 rank、shard | 张量并行规模下、秩 |
| rollout、trainer、logprob、staleness | 推演、训练器、对数概率、陈旧度 |
| torch.compile、CUDA graph、eager | 编译代码、CUDA 图、急切模式 |
| checkpoint、dtype、bf16、fp32 | 检查点、数据类型 |
| all-reduce、reduce-scatter、all-to-all | 全规约、规约散射 |
| LoRA、adapter、MoE、expert、top-k | 适配器、专家混合 |
| fingerprint、harness、skill、agent、SDC | 指纹（可在首次出现时括注"指纹"） |

Rewrites of titles that read as machine translation:

| Machine-translated, or too long | Short, natural Chinese |
|---|---|
| 并行度：每个 TP 大小下位都相同（L5） | 任意 TP 规模（L5） |
| 作业运行编译代码时 | torch.compile 与 CUDA graph |
| LoRA：训练时只有一个 adapter，推理时一个 batch 混着多个 adapter | LoRA：多 adapter 混合 batch |
| 让单侧不变：整个模型 | 整个模型的不变性（L1–L3） |

## Honesty

- Never weaken a result in words: bitwise means bitwise; a known-answer
  demonstration is called one; what was emulated is called emulated.
- Untested matters stay visible: a short "Not yet shown" line where it changes
  what a reader would do, and the full list in open-items.

## Chinese pages

Write the Chinese page as a page, not a sentence-by-sentence translation: same
sections, links and code; natural Chinese order; the glossary's English terms
kept. Short sentences.

## Checklist

- [ ] The page opens by saying where the reader is in the workflow.
- [ ] The mechanism is explained with a tiny example before any rule.
- [ ] No hardware model, version, size, timing or percentage on the page.
- [ ] Every action names its tool or skill and its success condition.
- [ ] Every claim of what holds links to its evidence.
- [ ] en and zh have the same sections, links and code (`scripts/check_i18n.py`).
- [ ] `mkdocs build --strict` passes.
