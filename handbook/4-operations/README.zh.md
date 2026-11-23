# 4. 运维

在你现在这套软件栈上，第 1–3 部分的流程都已经走通：设置检查过了，算子是不变的，两端的结果
也一致。这一部分讲软件栈发生变化时，比如库升级、驱动更新或者换了新的 GPU，怎样让这些结果
继续成立。

## 思路

逐 bit 一致的结果只属于测出它的那套软件栈。同一份源码换一个库版本，可能会选中另一个 kernel，
而另一个 kernel 可能按另一种顺序把同样的数加起来。所以软件栈每变一次，都要把每个判定重新检查
一遍，而不是默认它仍然成立。

<figure class="fx" data-fig="ops-loop" aria-label="软件栈变化后重新检查结果" markdown>
<div class="fx-stage" markdown>
<div class="fx-seq" style="--n:3" markdown="span"><span class="fx-node" data-hot="0">软件栈变了</span><span class="fx-node" data-hot="1">[运行回归套件](regression.md)</span><span class="fx-node is-decision" data-hot="2">与这套软件栈的基线比较</span></div>
<div class="fx-fork is-end" style="--n:3" markdown="span"><span class="fx-leg" data-hot="3"><span class="fx-when">某条结论不再成立</span><span class="fx-node to-bad" data-hot="3">这是一个发现：报告并修复</span></span><span class="fx-leg" data-hot="4"><span class="fx-when">某个原生算子的结果变了</span><span class="fx-node to-key" data-hot="4">更新主题页，记录新的基线</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">与基线相同</span><span class="fx-node to-ok" data-hot="5">完成</span></span></div>
</div>
<ol class="fx-say">
<li>软件栈变了：库升级、驱动更新，或者换了新的 GPU。</li>
<li>运行回归套件，把每一项验证过的检查重跑一遍。</li>
<li>把每个结果和这套软件栈的基线比较。</li>
<li data-hold="1.5">如果某条结论不再成立，这就是一个发现，要报告并修复。</li>
<li data-hold="1.5">如果某个原生算子的结果变了，就更新对应的主题页，并记录新的基线。</li>
<li data-hold="2">如果都和基线相同，就完成了，直到下一次变化。</li>
</ol>
<figcaption>图：逐 bit 一致的结果只属于测出它的那套软件栈，所以软件栈每变一次，都要重新检查一次。</figcaption>
</figure>

## 步骤

| 步骤 | 页面 | 工具 |
|---|---|---|
| 把每一项验证过的检查重跑一遍，看哪些回归了、哪些变了 | [回归](regression.md) | [tools/bi_suite.py](../../tools/bi_suite.py) |
| 了解在哪套软件栈上测过什么，以及换新硬件时该做什么 | [硬件覆盖](hardware.md) | [tools/bi_suite.py](../../tools/bi_suite.py)、[tools/sdc_scan.py](../../tools/sdc_scan.py) |

## 构建网站

本手册同时也是一个中英双语、书籍风格的网站，用 MkDocs Material 和 mkdocs-static-i18n 根据
[mkdocs.yml](../../mkdocs.yml) 构建。

- **布局。** Markdown 文件仍然是源文件，在 GitHub 上照常可读。网站从 `handbook/`（`docs_dir`）
  构建：`handbook/index.md` 是网站首页，仓库里的 `README.md` 仍然是 GitHub 上的首页。
- **中文。** 每个页面 `page.md` 旁边放着它的译文 `page.zh.md`（后缀模式），网站把译文发布在
  `/zh/` 下。术语的选择见[术语表](../glossary.md)。
  [scripts/check_i18n.py](../../scripts/check_i18n.py) 会逐对比较中英页面：表格和行数、数字、
  链接目标、代码块、行内代码和标题。
- **Skill。** 构建时，[scripts/mkdocs_hooks.py](../../scripts/mkdocs_hooks.py) 把
  `skills/**/*.md` 复制到 `handbook/skills/`（git 会忽略这个目录），所以 skill 也成了网站上的页面。
- **指向代码的链接。** 同一个 hook 会改写指向仓库文件（`tools/`、`examples/`、`suite/`、
  `AUDIT.md`）的链接。如果设置了 `HANDBOOK_SOURCE_URL`（例如设成
  `https://github.com/ORG/REPO/blob/main`），链接就指向那里；如果没有设置，这些文件会被复制到
  网站的 `source/` 下，每个目录还会生成一个文件列表。

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install mkdocs==1.6.1 mkdocs-material==9.7.7 mkdocs-static-i18n==1.3.1 jieba
.venv-docs/bin/mkdocs build --strict              # site/ ; must end with no warnings
python3 scripts/check_i18n.py                     # every page pair
python3 scripts/check_links.py                    # Markdown links in the repo and hrefs in site/
.venv-docs/bin/mkdocs serve -a 127.0.0.1:8765     # preview: / and /zh/
```
