# -*- coding: utf-8 -*-
"""Update README / CHANGELOG / CITATION / FINDINGS for 0.18.0.0."""
import io

ROOT = r'D:\开源系列\RawHebb'


def load(p):
    return io.open(p, encoding='utf-8').read()


def save(p, t):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)


def rep(t, old, new, count=1):
    if old not in t:
        raise SystemExit('MISSING: ' + old[:90])
    return t.replace(old, new, count)


# ---------------------------------------------------------------- README
p = ROOT + r'\README.md'
t = load(p)
t = rep(t, 'current release **0.17.0.0**', 'current release **0.18.0.0**')
t = rep(t, 'RawHebb+bio+0.17.0.0.py     single-file application, current',
        'RawHebb+bio+0.17.0.0.py     single-file application (kept for history)\n'
        'RawHebb+bio+0.18.0.0.py     single-file application, current')
t = rep(t, 'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0.py  the exact v139 branch that became 0.17.0.0',
        'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0.py  the exact v139 branch that became 0.17.0.0\n'
        'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16+0.9.254.0.py  int16 1M count table (identical numerics)\n'
        'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384+0.9.258.0.py  the exact v140 branch that became 0.18.0.0')

t = rep(t, '| metric | **0.17.0.0** | 0.16.0.0 | bigram (unigram-prior, same window) | bigram add-1 | unigram add-1 | old 0.6.0.0* |',
        '| metric | **0.18.0.0** | 0.17.0.0 | bigram (unigram-prior, same window) | bigram add-1 | unigram add-1 | old 0.6.0.0* |')
t = rep(t, '| cross-entropy | **3.141** | 3.150 |', '| cross-entropy | **3.054** | 3.141 |')
t = rep(t, '| top-1 | **41.59 %** | 41.35 % |', '| top-1 | **41.86 %** | 41.59 % |')
t = rep(t, '| top-5 | **63.86 %** | 63.84 % |', '| top-5 | **64.23 %** | 63.86 % |')

old_par = t[t.index('The model is **better than a window-matched bigram'):t.index('> **Naming correction (2026-10-20).**')]
new_par = '''The model is **better than a window-matched bigram on CE, top-1 and top-5**, with no
back-propagation anywhere. 0.18.0.0 changes only the training window: the online counts and state
now see the **whole dialogue** (capped at 384 tokens) instead of the last 128, while evaluation
still scores exactly the same last-128-token windows. The corpus averages 472 tokens per dialogue
(median 511), so 0.18 learns from about four times more context per dialogue with no new
parameters. It improves 0.17.0.0 by **0.088 CE, 0.27 points top-1 and 0.44 points top-5**.
**Seed robustness:** three seeds give CE 3.0543 / 3.0554 / 3.0561 (mean **3.0553**), top-1 41.86 /
41.65 / 41.71 % (mean **41.74 %**) and top-5 64.23 / 64.27 / 64.19 % (mean **64.23 %**); the
0.17.0.0 means are 3.1434 / 41.48 % / 63.79 %. Training at 512 tokens (the complete dialogue) is
tied on CE (mean 3.0542) but lower on top-1 (41.43 %) and about 30 % slower, so 384 is shipped.
The screen trajectory 128 -> 192 -> 256 -> 384 -> 512 is 3.5864 -> 3.5137 -> 3.4891 -> 3.4529 ->
3.4503. The 1M trigram table stored as int16 is numerically identical to float32 (full seed43
3.1411 vs 3.1411) at half the memory; a 2M int16 table fits, but it scores 3.5869 on the screen
versus 3.5864 at 1M, so the bucket axis is saturated at 1M. Hidden width stays at 432; hidden
400/428/436/448 are flat and 464 is worse.

'''
t = t.replace(old_par, new_par, 1)

t = rep(t, '> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 / 0.15.0.0 / 0.16.0.0 / 0.17.0.0',
        '> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 / 0.15.0.0 / 0.16.0.0 / 0.17.0.0 / 0.18.0.0')

anchor17 = t[t.index('| **0.17.0.0** | **widens the trigram count table to 1M buckets**'):]
end = anchor17.index('\n') + 1
row17 = anchor17[:end]
row18 = ('| **0.18.0.0** | **trains on the whole dialogue (up to 384 tokens) while scoring the same '
         'last-128-token windows**. Full corpus seed43: CE **3.0543**, top-1 **41.86 %**, top-5 '
         '**64.23 %**. Three seeds: 3.0543 / 3.0554 / 3.0561 (mean **3.0553**), top-1 41.86 / '
         '41.65 / 41.71 % (mean **41.74 %**), top-5 64.23 / 64.27 / 64.19 % (mean **64.23 %**); '
         'the 0.17 means are 3.1434 / 41.48 % / 63.79 %. |\n')
t = t.replace(row17, row17 + row18, 1)

t = rep(t, '**必须说清楚的现状（0.17.0.0）**：全量 2850 条 / 1 epoch 后，验证 **CE 3.1411 / top-1 41.59% /',
        '**必须说清楚的现状（0.18.0.0）**：全量 2850 条 / 1 epoch 后，验证 **CE 3.0543 / top-1 41.86% /')
t = rep(t, 'top-5 63.86%**（tri 桶 1M + tri 先验 2 + hidden432 + 行归一化 past-context bag 先验；\n0.16 是 3.1496/41.35/63.84）。',
        'top-5 64.23%**（完整对话训练 ≤384 token + tri 桶 1M int16 + tri 先验 2 + hidden432 +\n行归一化 past-context bag 先验；0.17 是 3.1411/41.59/63.86）。')

old_cn = t[t.index('0.17.0.0 在 0.16 上只把 **tri 计数桶 256k→1M**'):t.index('0.16.0.0 在 0.15 上只动三个纯计数量')]
new_cn = '''0.18.0.0 只改训练窗口：在线计数与状态从「对话最后 128 token」改为「完整对话（上限 384）」，
评测口径完全不变（仍取每条对话最后 128 token，18 900 个 assistant token）。语料平均 472
token/对话（中位 511），所以 0.18 每条对话多看约 4 倍上下文，没有新增参数。相对 0.17：
CE -0.088、top-1 +0.27、top-5 +0.44。三种子 CE 3.0543/3.0554/3.0561（均值 **3.0553**），
top-1 41.86/41.65/41.71%（均值 **41.74%**），top-5 64.23/64.27/64.19%（均值 **64.23%**）；
0.17 均值 3.1434/41.48%/63.79%。训练 512（完整对话）CE 均值 3.0542 打平，但 top-1 41.43%
且慢约 30%，故发布 384。int16 计数表与 float32 数值一致（1M 全量均 3.1411），内存减半；
2M int16 筛选 3.5869 未超过 1M 的 3.5864，桶轴到顶。见 `docs/FINDINGS.md`。

'''
t = t.replace(old_cn, new_cn, 1)
save(p, t)
print('README ok')

# ---------------------------------------------------------------- CHANGELOG
p = ROOT + r'\CHANGELOG.md'
t = load(p)
t = rep(t, '## [0.17.0.0] - current', '## [0.17.0.0]')
entry = '''## [0.18.0.0] - current
* **Full-dialogue training window.** The online counts and membrane state now see the whole
  dialogue (capped at 384 tokens) instead of its last 128 tokens; evaluation is unchanged and
  still scores the last-128-token windows, so the comparison is exact. On this corpus dialogues
  average 472 tokens (median 511), so each one contributes about four times more context. No new
  parameters, no new constants, no back-propagation.
* Full corpus seed43: CE **3.0543**, top-1 **41.86 %**, top-5 **64.23 %** (0.17: 3.1411 /
  41.59 % / 63.86 %), i.e. **-0.088 CE, +0.27 top-1, +0.44 top-5**.
* **Seeds:** CE 3.0543 / 3.0554 / 3.0561 (mean **3.0553**), top-1 41.86 / 41.65 / 41.71 % (mean
  **41.74 %**), top-5 64.23 / 64.27 / 64.19 % (mean **64.23 %**); the 0.17 means are 3.1434 /
  41.48 % / 63.79 %.
* Training at 512 tokens (the complete dialogue) ties on CE (mean 3.0542) but gives top-1 41.43 %
  and is about 30 % slower, so 384 is shipped. Screen trajectory 128 -> 192 -> 256 -> 384 -> 512:
  3.5864 -> 3.5137 -> 3.4891 -> 3.4529 -> 3.4503.
* **int16 count table validated:** a 1M int16 table reproduces float32 exactly (full seed43
  3.1411 / 41.59 % / 63.86 %) at half the memory. With int16 a 2M table fits, but it scores 3.5869
  on the screen versus 3.5864 at 1M, so the count-bucket axis is saturated at 1M.

'''
t = rep(t, '# Changelog\n', '# Changelog\n')
i = t.index('## [0.17.0.0]')
t = t[:i] + entry + t[i:]
save(p, t)
print('CHANGELOG ok')

# ---------------------------------------------------------------- CITATION
p = ROOT + r'\CITATION.cff'
t = load(p)
t = rep(t, 'version: 0.17.0.0', 'version: 0.18.0.0')
t = rep(t, '''150-dialogue split (full corpus, one epoch) the current release reaches CE 3.1411,
  top-1 41.59% and top-5 63.86% (three-seed mean 3.1434 / 41.48% / 63.79%), beating a''',
        '''150-dialogue split (full corpus, one epoch) the current release reaches CE 3.0543,
  top-1 41.86% and top-5 64.23% (three-seed mean 3.0553 / 41.74% / 64.23%), beating a''')
save(p, t)
print('CITATION ok')

# ---------------------------------------------------------------- FINDINGS
p = ROOT + r'\docs\FINDINGS.md'
t = load(p)
v140 = '''
### v140 - full-dialogue training, int16 counts, and the 2M ceiling (0.18.0.0)

The corpus tokeniser caps a dialogue at 511 tokens; measured over all 2 850 training dialogues the
lengths are mean 472.1, median 511, and only 0.6 % / 1.4 % / 4.5 % / 14.2 % are <=128 / 192 / 256 /
384 tokens. The online learner had always been fed `ids[-128:]`, i.e. roughly the last quarter of
each dialogue. v140 changes only that training window; `evaluate` still slices the same last 128
tokens, so every number stays comparable.

Screens (500 dialogues, `--val-max 50`, 1M int16 table, hidden 432, trigram prior 2, row-normalised
bag, Jeffreys 0.5): train length 128 -> 3.5864, 192 -> 3.5137, 256 -> 3.4891, 384 -> 3.4529,
512 -> 3.4503. The monotone gain is the extra context actually entering the counts; 512 is the full
dialogue and saturates.

Full corpus seed43, train 384 / eval 128: **3.0543 / 41.86 / 64.23**; train 512: 3.0535 / 41.59 /
64.15. Three seeds of 384: 3.0543 / 3.0554 / 3.0561 (mean **3.0553**), top-1 41.86 / 41.65 / 41.71 %
(mean **41.74 %**), top-5 64.23 / 64.27 / 64.19 % (mean **64.23 %**). Three seeds of 512: 3.0535 /
3.0549 / 3.0541 (mean 3.0542), top-1 41.59 / 41.36 / 41.33 % (mean 41.43 %), top-5 64.15 / 64.08 /
64.21 % (mean 64.15 %). CE is tied within one seed standard deviation (about 0.0008); 384 is
+0.31 top-1, +0.08 top-5 and about 30 % faster (1 442-2 306 s versus about 3 000 s), so 384 is
**0.18.0.0**. Relative to 0.17 (mean 3.1434 / 41.48 / 63.79) that is -0.088 CE, +0.26 top-1,
+0.44 top-5 from a pure training-window change with no new parameters.

**int16 counts.** Storing the 1M trigram table as int16 reproduces float32 exactly at full scale:
seed43 3.1411 / 41.59 / 63.86 (float32) versus 3.1411 / 41.59 / 63.86 (int16), and the 500 screen
3.5864 versus 3.5863. This halves the table (6.2 GB checkpoint instead of 12.4 GB) and makes
second-pass evaluation possible via `_v162_eval_ckpt.py`, which is how seed45 of the 512 run was
recovered after its PTY buffer was cleared. The 2M int16 table (11.55 GB) trains, but its screen is
3.5869 / 38.13 / 59.49 versus 3.5864 / 38.30 / 59.43 at 1M, so the bucket axis is **saturated**, not
merely resource-limited; 1M stays.

**Operational notes.** D: filled up again during the first 1M int16 full run and `torch.save` failed
after training (`inline_container.cc:672`); 51.2 GB of screen checkpoints were removed. The PTY
registry was cleared twice more by the terminal UI, killing running runs; the final protocol is one
full run at a time over PTY, and detached `Start-Process` is not used. Large int16 checkpoints can
be re-scored with `_v162_eval_ckpt.py` (mmap + filtered `load_state_dict`), which avoids the
double-table OOM of `_v122_eval.py`.
'''
t = t.rstrip('\n') + '\n' + v140
save(p, t)
print('FINDINGS ok')
