# -*- coding: utf-8 -*-
"""Docs for 0.20.0.0 (312-faithful shell; full-window evaluation)."""
import io

ROOT = r'D:\开源系列\RawHebb'


def load(p):
    return io.open(p, encoding='utf-8').read()


def save(p, t):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)


def rep(t, old, new, n=1):
    if old not in t:
        raise SystemExit('MISSING: ' + old[:90])
    return t.replace(old, new, n)


# ---------------- README
p = ROOT + r'\README.md'
t = load(p)
t = rep(t, 'current release **0.19.0.0**', 'current release **0.20.0.0**')
t = rep(t, 'RawHebb+bio+0.19.0.0.py     single-file application, current',
        'RawHebb+bio+0.19.0.0.py     single-file application (superseded: menu NameError)\n'
        'RawHebb+bio+0.20.0.0.py     single-file application, current')
t = rep(t, 'RawHebb+shell19+0.9.264.0.py  the exact v141 shell branch that became 0.19.0.0',
        'RawHebb+shell19+0.9.264.0.py  v141 shell branch (superseded by 0.20.0.0)\n'
        'RawHebb+shell20+0.9.265.0.py  the exact v142 shell branch that became 0.20.0.0')

t = rep(t, '| metric | **0.19.0.0** | 0.18.0.0 |', '| metric | **0.20.0.0** (full window) | 0.18.0.0 (last-128) |')
t = rep(t, '| cross-entropy | **3.054** | 3.054 |', '| cross-entropy | **3.355** | 3.054 |')
t = rep(t, '| top-1 | **41.86 %** | 41.86 % |', '| top-1 | **38.06 %** | 41.86 % |')
t = rep(t, '| top-5 | **64.23 %** | 64.23 % |', '| top-5 | **60.60 %** | 64.23 % |')

anchor = 'The model is **better than a window-matched bigram on CE, top-1 and top-5**'
add20 = '''0.20.0.0 is the first **usable** shell. 0.19.0.0 crashed on menu item 1 (a helper was
defined inside a replaced block) and its data layer was only an approximation of 3120012. 0.20
ports the actual 3120012 chain: `render_messages` semantics (`<|bos|>`, system/user/tool/
assistant segments, `reasoning_content` + `tool_calls`, `<|eos|>`), `_truncate_pairs` (keep the
system prompt and the newest complete user/assistant pairs, budget in real tokens), a record-level
10 % validation sample, and validation on the **same window as training** instead of a separate
last-128 slice. `.txt` remains unsupervised pretraining (all tokens, shuffled windows, no validation
by default). The model recipe is unchanged, so the numbers in the two columns above use different
protocols and must not be compared: 0.18.0.0 was scored on the last 128 tokens, 0.20.0.0 on the
full training windows with the same 0.18 checkpoint (166 632 scored positions).

'''
t = rep(t, anchor, add20 + anchor)

t = rep(t, '> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 / 0.15.0.0 / 0.16.0.0 / 0.17.0.0 / 0.18.0.0 / 0.19.0.0',
        '> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 / 0.15.0.0 / 0.16.0.0 / 0.17.0.0 / 0.18.0.0 / 0.19.0.0 / 0.20.0.0')

row19 = t[t.index('| **0.19.0.0** | **shell overhaul'):]
row19 = row19[:row19.index('\n') + 1]
row20 = ('| **0.20.0.0** | **usable shell: the 3120012 data chain adapted to the online Hebbian core**. '
         '`render_messages` + `_truncate_pairs` semantics, record-level 10 % validation split, and '
         'validation on the same window as training (full `SEQ_LEN`, 166 632 scored positions). '
         'Same 0.18 checkpoint: CE **3.3545** / top-1 **38.06 %** / top-5 **60.60 %** - a different '
         'protocol from the last-128 numbers, not an improvement or regression. Fixes the 0.19 menu '
         'crash and adds online learning. |\n')
t = t.replace(row19, row20 + row19, 1)

t = rep(t, '**必须说清楚的现状（0.19.0.0）**：模型与 0.18 相同（0.19 只修外壳）。全量 2850 条 / 1 epoch 后，验证 **CE 3.0543 / top-1 41.86% /',
        '**必须说清楚的现状（0.20.0.0）**：0.20 只改外壳与数据口径，模型配方同 0.18，未重新训练。'
        '同一份 0.18 权重在 312 同窗口径（完整训练窗口，166 632 个计分位置）下是 '
        '**CE 3.3545 / top-1 38.06% / top-5 60.60%**；旧的 last-128 口径数字（CE 3.0543 / top-1 41.86% /')

old_cn = t[t.index('0.19.0.0 不改模型数学，只修外壳'):t.index('0.18.0.0 只改训练窗口：')]
new_cn = '''0.20.0.0 是把 3120012 的壳子真正移植过来：渲染（system/user/tool/assistant、reasoning、
tool_calls）、`_truncate_pairs`（保留 system + 最近完整问答对，按真实 token 计预算）、
按记录随机抽 10% 验证集、验证与训练同窗。0.19 的菜单 NameError 已修；`.txt` 预训练
（全 token、窗口打乱、默认不验证）与 `.jsonl` 对话（assistant mask）分开。模型配方不变，
因此 0.20 的指标与旧版 last-128 口径不可比：同一份 0.18 权重在 166 632 个位置上是
**CE 3.3545 / top-1 38.06% / top-5 60.60%**，旧口径为 3.0543 / 41.86% / 64.23%。

'''
t = t.replace(old_cn, new_cn, 1)
save(p, t)
print('README ok')

# ---------------- CHANGELOG
p = ROOT + r'\CHANGELOG.md'
t = load(p)
i = t.index('## [0.19.0.0]')
entry = '''## [0.20.0.0] - current
* **Usable shell: the 3120012 data chain ported to the RawHebb core.** 0.19.0.0 crashed on menu
  item 1 because a helper was defined inside a replaced block; that is fixed, and the approximate
  data layer is replaced by an actual port of the 3120012 logic.
* **Rendering/mask:** `<|bos|>` + tools + system/user/tool/assistant segments + `<|eos|>`;
  assistant `reasoning_content`, `content` and `tool_calls` are trained, everything else is -100.
* **Context truncation:** `_truncate_pairs` semantics - keep the leading system prompt and the
  newest complete user/assistant pairs, budgeted in real tokens (including role/end/reasoning/tool
  calls), never cutting a sentence in the middle.
* **Validation:** a record-level random 10 % sample by default (fixed seed) or an explicit val
  file; validation uses the **same `SEQ_LEN` window as training**, not a separate last-128 slice.
  `.txt` pretraining remains all-token with shuffled windows and no validation by default.
* **No new benchmark claim:** the model recipe is unchanged and was not retrained. The same 0.18
  checkpoint scores CE **3.3545** / top-1 **38.06 %** / top-5 **60.60 %** over 166 632 positions
  under the full-window protocol; these numbers are not comparable to the last-128 numbers of
  0.15-0.18.
* **Online learning** (menu 13) requires a trained checkpoint and supports file streaming or
  in-chat updates.

'''
t = t[:i] + entry + t[i:]
save(p, t)
print('CHANGELOG ok')

# ---------------- CITATION
p = ROOT + r'\CITATION.cff'
t = load(p)
t = rep(t, 'version: 0.19.0.0', 'version: 0.20.0.0')
t = rep(t, '''150-dialogue split (full corpus, one epoch) the current release reaches CE 3.0543,
  top-1 41.86% and top-5 64.23% (three-seed mean 3.0553 / 41.74% / 64.23%), beating a''',
        '''150-dialogue split (full corpus, one epoch). The 0.18 checkpoint reaches CE 3.0543,
  top-1 41.86% and top-5 64.23% on the last-128-token protocol; 0.20.0.0 scores the same
  checkpoint on full training windows (CE 3.3545 / 38.06% / 60.60%, 166 632 positions), so the
  two protocols are not directly comparable. The model beats a''')
save(p, t)
print('CITATION ok')

# ---------------- FINDINGS
p = ROOT + r'\docs\FINDINGS.md'
t = load(p)
v142 = '''
### v142 - the 3120012 shell, actually ported (0.20.0.0)

0.19.0.0 was not usable: menu item 1 raised `NameError: select_dataset_file` because the helper
block was inserted inside the section that a later replacement removed. More importantly, its data
layer only imitated 3120012. v142 ports the real chain.

**Rendering and mask.** `render_messages` semantics are reproduced token by token: `<|bos|>`, an
optional tools header (`<|user|>[工具]…<|end|>`), system (`<|user|>[系统]…<|end|>`), user, tool
results, and assistant (`<|assistant|>` + `reasoning_content` + `content` + `tool_calls` +
`<|end|>`), followed by `<|eos|>`. Only assistant content, its `<|end|>` and the final `<|eos|>`
are trained; everything else is -100. A synthetic test gives 28 tokens for a short system/user/
assistant turn and decodes the trained positions as `想一下你好！` (reasoning + answer).

**Context.** `_truncate_pairs` is ported: keep the leading consecutive system messages, then walk
backwards and keep complete (user, assistant) pairs until the token budget (+2 for bos/eos) is
reached; the last pair is always accepted. Token cost is computed with the actual rendered segment
(role + content + end + reasoning + tool_calls), not characters. A 20-pair dialogue truncated to
120 tokens keeps the system prompt and pairs 17-19 and drops pair 0.

**Split and validation.** `.jsonl` validation is now a record-level random sample (`random.sample`,
10 %, fixed seed) or an explicit val file. Validation uses the same processing and the same
`SEQ_LEN` window as training; `EVAL_LEN=0` disables the old last-128 slice. `.txt` is pretraining:
whole-file tokenisation into `SEQ_LEN+1` windows, every position scored, windows shuffled by the
fixed seed, no validation unless a `.txt` val file is given.

**Measured.** A 40-dialogue smoke run with no val file produced 36 train / 4 val records and full
window validation (4 358 positions, CE 4.2928). Evaluating the 0.18 seed43 checkpoint with the new
protocol gives CE **3.3545**, top-1 **38.06 %**, top-5 **60.60 %** over 166 632 positions. The old
last-128 protocol gave 3.0543 / 41.86 / 64.23 on 19 050 positions. These are different token sets
and different difficulty; 0.20 is a shell release, not a model improvement. A menu run in a clean
directory containing only a `.txt` file auto-trained a vocabulary, listed the dataset, trained two
windows and returned to the menu without error.

**Online learning.** Requires a checkpoint. From random initialisation the first Newton step is
ill-conditioned (||W_out|| 52 -> 11813 after two dialogues); with the 0.18 checkpoint the same two
dialogues move it 0.633 -> 0.634.
'''
t = t.rstrip('\n') + '\n' + v142
save(p, t)
print('FINDINGS ok')

# ---------------- HANDOVER
p = r'D:\explore\wav\HANDOVER.md'
t = load(p)
t = rep(t, '**当前最优（2026-10-22 v141）**：`RawHebb+bio+0.19.0.0.py`\n（= 0.18 模型配方 + 外壳工程；分支 `RawHebb+shell19+0.9.264.0.py`）',
        '**当前可用（2026-10-22 v142）**：`RawHebb+bio+0.20.0.0.py`\n（= 0.18 模型配方 + 3120012 数据壳；分支 `RawHebb+shell20+0.9.265.0.py`；0.19 有菜单 NameError，已废弃）')
t = rep(t, '**0.19.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.19.0.0>（commit `e3d39d5`））',
        '**0.19.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.19.0.0>（commit `e3d39d5`，已废弃）、\n**0.20.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.20.0.0>）')
t = rep(t, '**2026-10-22 后续（v141）**',
        '''**2026-10-22 后续（v142）**：**0.20.0.0 可用版**。0.19 的菜单在干净目录点 1 会 `NameError`
（helper 被替换块吃掉）；数据层也只是近似 3120012。0.20 直接移植 312 的渲染（system/tool/
reasoning/tool_calls）、`_truncate_pairs`（保留 system + 最近完整问答对，按真实 token 计预算）、
按记录抽 10% 验证集、验证与训练同窗（`EVAL_LEN=0`）。`.txt` 预训练全 token、窗口打乱、默认不验证。
同一份 0.18 权重在新口径 166 632 位置上 **CE 3.3545/38.06/60.60**，旧 last-128 是
3.0543/41.86/64.23，两套不可比。干净目录菜单训练 2 窗口通过。

**2026-10-22 后续（v141）**''')
save(p, t)
print('HANDOVER ok')

# ---------------- memory
p = r'D:\explore\wav\.workbuddy\memory\2026-10-22.md'
t = load(p)
t = t.rstrip('\n') + '''

## v142 / 0.20.0.0（同日追加，可用版）

- 0.19 实测在干净目录菜单 1 直接 `NameError`（helpers 被 SECTION 10 替换吃掉）；已修。
- 用户要求“把 312 的壳子拔下来修剪干净适配核心”，不要再拿旧 benchmark 当挡箭牌。
- 0.20 直接移植 312 数据链：
  - render_messages 语义：bos + tools + system/user/tool/assistant + eos；
    assistant 的 reasoning_content/content/tool_calls 参与训练，其余 -100；
  - `_truncate_pairs`：system 保留 + 最近完整问答对，按真实 token（role/end/reasoning/tools）计预算；
  - jsonl 验证按记录随机抽 10%（seed 固定），也可外部 val；
  - 验证与训练同窗（EVAL_LEN=0），旧的 last-128 只作为可选截断；
  - txt 预训练：整文件窗口、全 token、窗口按 seed 打乱、默认不验证。
- 不再宣称 0.18 指标：同一 0.18 权重在新口径 166632 位置 **3.3545/38.06/60.60**；
  旧 last-128 3.0543/41.86/64.23，两者不可比。
- 测试：合成数据（system/tools/reasoning/截断）、干净目录菜单训练 2 窗口、jsonl 自动切分
  36/4、txt 预训练、在线学习（需已训练权重）。
- 文件：`RawHebb+bio+0.20.0.0.py`，分支 `RawHebb+shell20+0.9.265.0.py`，
  生成 `_v164_build20.py`、测试 `_v164_data_test.py`。
'''
save(p, t)
print('memory ok')
