# 4. Operations

You now have a stack on which the workflows of parts 1–3 hold: the settings
have been checked, the operators are invariant, and the two sides agree. This
part shows how to keep it that way when the stack changes under you, for
example after a library upgrade, a driver update or a move to new GPUs.

## The idea

A bitwise result belongs to the stack it was measured on. The same source
code on another library version can pick a different kernel, and a different
kernel can add the same numbers in a different order. So each time the stack
changes, you re-check every verdict instead of assuming that it still holds.

<figure class="fx" data-fig="ops-loop" aria-label="Re-checking the results when the stack changes" markdown>
<div class="fx-stage" markdown>
<div class="fx-seq" style="--n:3" markdown="span"><span class="fx-node" data-hot="0">The stack changes</span><span class="fx-node" data-hot="1">[Run the regression suite](regression.md)</span><span class="fx-node is-decision" data-hot="2">Compare with this stack's baseline</span></div>
<div class="fx-fork is-end" style="--n:3" markdown="span"><span class="fx-leg" data-hot="3"><span class="fx-when">a claim no longer holds</span><span class="fx-node to-bad" data-hot="3">It is a finding: report it and fix it</span></span><span class="fx-leg" data-hot="4"><span class="fx-when">a stock result changed</span><span class="fx-node to-key" data-hot="4">Update the topic page and record a new baseline</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">same as the baseline</span><span class="fx-node to-ok" data-hot="5">Done</span></span></div>
</div>
<ol class="fx-say">
<li>The stack changes: a library upgrade, a driver update or new GPUs.</li>
<li>Run the regression suite, which repeats every verified check.</li>
<li>Compare each result with the baseline for this stack.</li>
<li data-hold="1.5">If a claim no longer holds, that is a finding. Report it and fix it.</li>
<li data-hold="1.5">If a result of a stock operator changed, update the topic page and record a new baseline.</li>
<li data-hold="2">If everything matches the baseline, you are done until the next change.</li>
</ol>
<figcaption>Figure: a bitwise result belongs to the stack it was measured on, so every change of the stack is followed by a re-check.</figcaption>
</figure>

## The steps

| Step | Page | Tool |
|---|---|---|
| Re-run every verified check, and read what regressed or changed | [Regression](regression.md) | [tools/bi_suite.py](../../tools/bi_suite.py) |
| Find out what was measured on which stack, and what to do on new hardware | [Hardware coverage](hardware.md) | [tools/bi_suite.py](../../tools/bi_suite.py), [tools/sdc_scan.py](../../tools/sdc_scan.py) |

## Building the site

The handbook is also a book-style website in English and Chinese. It is built
with MkDocs Material and mkdocs-static-i18n from [mkdocs.yml](../../mkdocs.yml).

- **Layout.** The Markdown files remain the source and stay readable on
  GitHub. The site is built from `handbook/` (`docs_dir`).
  `handbook/index.md` is the site's home page, and the repository's
  `README.md` remains the front page on GitHub.
- **Chinese.** Each page `page.md` has its translation `page.zh.md` next to it
  (suffix mode), and the site serves the translation under `/zh/`. The term
  choices are in the [glossary](../glossary.md).
  [scripts/check_i18n.py](../../scripts/check_i18n.py) compares every pair of
  pages: tables and their rows, numbers, link targets, code blocks, inline
  code and headings.
- **Skills.** At build time,
  [scripts/mkdocs_hooks.py](../../scripts/mkdocs_hooks.py) copies
  `skills/**/*.md` into `handbook/skills/` (which git ignores), so the skills
  become pages too.
- **Links to code.** The same hook rewrites links to repository files
  (`tools/`, `examples/`, `suite/`, `AUDIT.md`). If `HANDBOOK_SOURCE_URL` is
  set (for example to `https://github.com/ORG/REPO/blob/main`), the links
  point there. If it is not set, the files are copied into the site under
  `source/`, and each directory gets a file listing.

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install mkdocs==1.6.1 mkdocs-material==9.7.7 mkdocs-static-i18n==1.3.1 jieba
.venv-docs/bin/mkdocs build --strict              # site/ ; must end with no warnings
python3 scripts/check_i18n.py                     # every page pair
python3 scripts/check_links.py                    # Markdown links in the repo and hrefs in site/
.venv-docs/bin/mkdocs serve -a 127.0.0.1:8765     # preview: / and /zh/
```
