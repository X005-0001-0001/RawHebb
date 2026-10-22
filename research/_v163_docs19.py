# -*- coding: utf-8 -*-
"""Docs for 0.19.0.0 (shell overhaul; same model numbers as 0.18)."""
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


# README
p = ROOT + r'\README.md'
t = load(p)
t = rep(t, 'current release **0.18.0.0**', 'current release **0.19.0.0**')
t = rep(t, 'RawHebb+bio+0.18.0.0.py     single-file application, current',
        'RawHebb+bio+0.18.0.0.py     single-file application (kept for history)\n'
        'RawHebb+bio+0.19.0.0.py     single-file application, current')
t = rep(t, 'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384+0.9.258.0.py  the exact v140 branch that became 0.18.0.0',
        'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384+0.9.258.0.py  the exact v140 branch that became 0.18.0.0\n'
        'RawHebb+shell19+0.9.264.0.py  the exact v141 shell branch that became 0.19.0.0')

t = rep(t, '| metric | **0.18.0.0** | 0.17.0.0 |', '| metric | **0.19.0.0** | 0.18.0.0 |')
t = rep(t, '| cross-entropy | **3.054** | 3.141 |', '| cross-entropy | **3.054** | 3.054 |')
t = rep(t, '| top-1 | **41.86 %** | 41.59 % |', '| top-1 | **41.86 %** | 41.86 % |')
t = rep(t, '| top-5 | **64.23 %** | 63.86 % |', '| top-5 | **64.23 %** | 64.23 % |')

anchor = 'The model is **better than a window-matched bigram on CE, top-1 and top-5**'
add19 = '''The model is **better than a window-matched bigram on CE, top-1 and top-5**, with no
back-propagation anywhere. 0.19.0.0 changes **no model math**; it is the 0.18 recipe with a repaired
shell. The vocabulary now has a full lifecycle (train / save / load / metadata validation / fallback)
instead of a hard-wired file, and there are two data pipelines: `.txt` is unsupervised pretraining
(all tokens are targets, no validation by default), `.jsonl` is dialogue (assistant mask, validation
split by dialogue at 10% or an explicit val file). Dialogues longer than the training window keep
the system prompt and the newest complete user/assistant pairs instead of being cut mid-sentence.
A new online-learning menu branch updates the model sample by sample (file streaming or during
chat). Loading the 0.18 checkpoint reproduces CE 3.0543 / top-1 41.86 % / top-5 64.23 % exactly.

'''
t = rep(t, anchor, add19 + anchor)

t = rep(t, '> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 / 0.15.0.0 / 0.16.0.0 / 0.17.0.0 / 0.18.0.0',
        '> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 / 0.15.0.0 / 0.16.0.0 / 0.17.0.0 / 0.18.0.0 / 0.19.0.0')

row18 = ('| **0.18.0.0** | **trains on the whole dialogue (up to 384 tokens) while scoring the same '
         'last-128-token windows**. Full corpus seed43: CE **3.0543**, top-1 **41.86 %**, top-5 '
         '**64.23 %**. Three seeds: 3.0543 / 3.0554 / 3.0561 (mean **3.0553**), top-1 41.86 / '
         '41.65 / 41.71 % (mean **41.74 %**), top-5 64.23 / 64.27 / 64.19 % (mean **64.23 %**); '
         'the 0.17 means are 3.1434 / 41.48 % / 63.79 %. |\n')
row19 = ('| **0.19.0.0** | **shell overhaul; model identical to 0.18.0.0**. Full-dialogue training, '
         'same CE **3.0543** / top-1 **41.86 %** / top-5 **64.23 %** (three-seed mean 3.0553 / '
         '41.74 % / 64.23 %). Restores the vocabulary lifecycle, `.txt` pretraining / `.jsonl` '
         'dialogue pipelines, message-aware truncation, and an online-learning menu branch; '
         'loading the 0.18 checkpoint is verified bit-identical. |\n')
if row18 not in t:
    raise SystemExit('row18 missing')
t = t.replace(row18, row19 + row18, 1)

t = rep(t, '**必须说清楚的现状（0.18.0.0）**：全量 2850 条 / 1 epoch 后，验证 **CE 3.0543 / top-1 41.86% /',
        '**必须说清楚的现状（0.19.0.0）**：模型与 0.18 相同（0.19 只修外壳）。全量 2850 条 / 1 epoch 后，验证 **CE 3.0543 / top-1 41.86% /')

cn_anchor = '0.18.0.0 只改训练窗口：'
cn19 = '''0.19.0.0 不改模型数学，只修外壳：词表恢复「训练 / 保存 / 加载 / 元数据校验 / 缺失回退」；
数据分两条管线——`.txt` 是无监督预训练（整文件切窗口，全部 token 计损，默认不验证），
`.jsonl` 是对话（只学 assistant，验证集按对话随机切 10%，也可显式给 val 文件）；
超长对话保留 system + 最近完整问答对，不在句子中间硬切。菜单新增「在线学习」：
文件增量逐条更新 / 对话中每轮更新。加载 0.18 权重复测仍是 **CE 3.0543 / top-1 41.86% /
top-5 64.23%**，证明外壳改动没有动数值。

'''
t = rep(t, cn_anchor, cn19 + cn_anchor)
save(p, t)
print('README ok')

# CHANGELOG
p = ROOT + r'\CHANGELOG.md'
t = load(p)
i = t.index('## [0.18.0.0]')
entry = '''## [0.19.0.0] - current
* **Shell overhaul; the model recipe is unchanged.** Loading the 0.18.0.0 checkpoint reproduces
  CE **3.0543** / top-1 **41.86 %** / top-5 **64.23 %** exactly (verified with
  `research/_v162_eval_ckpt.py`), so the numbers below are identical to 0.18.
* **Vocabulary lifecycle restored** (from 3120012elite): ByteLevel BPE can be trained from the
  corpus, saved as `tokenizer.bin` + `tokenizer.json`, loaded with metadata validation, and falls
  back to the base vocabulary when present. Rebuilt vocabularies are detected on checkpoint load
  and refused explicitly instead of silently skipping most tables.
* **Two data pipelines** matching 3120012: `.txt` = unsupervised pretraining (all tokens, windows
  of `SEQ_LEN+1`, no validation by default); `.jsonl` = dialogue (assistant mask, validation split
  by dialogue at 10 % with a fixed seed, or an explicit val file).
* **Message-aware truncation:** dialogues longer than `SEQ_LEN` keep the system prompt and the
  newest complete user/assistant pairs (`_truncate_pairs` behaviour), instead of cutting a sentence
  in the middle.
* **Online learning menu branch:** stream a `.txt`/`.jsonl` incrementally (one update per sample,
  no shuffle/validation) or learn during chat (one update per exchange). Requires a trained
  checkpoint so the Newton head does not start from an ill-conditioned random state.
* **Config cleanup:** removed six never-read placebo constants (`BATCH_SIZE`, `LEARNING_RATE`,
  `WEIGHT_DECAY`, `GRAD_CLIP_NORM`, `MEM_MOD_DECAY`, `ADAPT_TAU_MS`), rewrote the configuration
  with sourced/measured comments, and added dataset selection, input validation and a tokenizer
  submenu to the menu.

'''
t = t[:i] + entry + t[i:]
save(p, t)
print('CHANGELOG ok')

# CITATION
p = ROOT + r'\CITATION.cff'
t = load(p)
t = rep(t, 'version: 0.18.0.0', 'version: 0.19.0.0')
save(p, t)
print('CITATION ok')

# FINDINGS
p = ROOT + r'\docs\FINDINGS.md'
t = load(p)
v141 = '''
### v141 - shell overhaul: vocabulary lifecycle, two data pipelines, online branch (0.19.0.0)

0.19.0.0 changes no model mathematics. The release file is generated from the 0.9.264 shell branch;
loading the 0.18 seed43 checkpoint with it reproduces **3.0543 / 41.86 / 64.23** exactly, and the
tokenizer produces bit-identical `chat_to_ids` for the eval slice of all 200 checked val dialogues.

The shell had lost three pieces of the 3120012 design. First, `RawHebbTokenizer` only loaded two
hard-wired files; it now trains ByteLevel BPE from the corpus, saves `tokenizer.bin` +
`tokenizer.json`, validates metadata, falls back to the base vocabulary, and refuses checkpoints
whose vocab/hidden/layer metadata disagree instead of silently skipping most tables through
`strict=False`. Second, the data layer only understood `.jsonl` and had no dialogue split unless an
external val file was supplied. Training now has two explicit pipelines: `.txt` is unsupervised
pretraining (whole-file tokenisation, non-overlapping `SEQ_LEN+1` windows, every position scored,
no validation by default, matching 3120012); `.jsonl` is dialogue (assistant mask, validation split
by dialogue at 10 % with seed 43, or an explicit val file). Dialogues longer than 384 tokens are
truncated at message-pair boundaries, keeping the system prompt and the newest complete
user/assistant pairs rather than cutting mid-sentence. Third, the menu lost dataset selection and
had no online branch; the new menu restores file selection and validated inputs, adds a tokenizer
submenu, and adds item 13: online learning by streaming a file (one local update per sample) or
during chat (one update per exchange). Online learning requires a trained checkpoint; starting it
from random initialisation makes the first Newton step explode (‖W_out‖ 52 -> 11813 after two
dialogues), whereas a loaded model moves only 0.633 -> 0.634 on the same two dialogues.

Smoke tests: `.txt` 11 925 tokens -> 31 windows, train CE 2.978, no validation; `.jsonl` with
`--limit 40` and no val file -> 36 train / 4 val dialogues, val CE 4.0945; a 600-token vocabulary
trained from 170 text segments saves v5 metadata, reloads, encodes and decodes; menu startup and
`0` exit work. Six dead constants left over from the optimiser era were removed. The model defaults
are unchanged: hidden 432, 4 layers, `SEQ_LEN` 384, `EVAL_LEN` 128, 1M int16 trigram buckets,
4k last-token counts, row-normalised bag prior with Jeffreys 0.5.
'''
t = t.rstrip('\n') + '\n' + v141
save(p, t)
print('FINDINGS ok')

# HANDOVER
p = r'D:\explore\wav\HANDOVER.md'
t = load(p)
t = rep(t, '**当前最优（2026-10-22 v140）**：`RawHebb+bio+0.18.0.0.py`\n（= `RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384+0.9.258.0.py`）',
        '**当前最优（2026-10-22 v141）**：`RawHebb+bio+0.19.0.0.py`\n（= 0.18 模型配方 + 外壳工程；分支 `RawHebb+shell19+0.9.264.0.py`）')
t = rep(t, '**0.18.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.18.0.0>（commit `234bc49`））',
        '**0.18.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.18.0.0>（commit `234bc49`）、\n**0.19.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.19.0.0>）')
t = rep(t, '**2026-10-22 后续（v140）**',
        '''**2026-10-22 后续（v141）**：**外壳工程版 0.19.0.0，模型数值不变**。恢复 3120012 的
词表生命周期（train/save/load/元数据校验/缺失回退），`.txt` 预训练 / `.jsonl` 对话双管线
（对话按 10% 随机切验证集；超长按 system+最近完整问答对截断），菜单新增在线学习分支
（文件增量 / 对话中学习）。删除 6 个死参数（BATCH_SIZE/LEARNING_RATE/WEIGHT_DECAY/
GRAD_CLIP_NORM/MEM_MOD_DECAY/ADAPT_TAU_MS）。加载 0.18 权重复测 **3.0543/41.86/64.23**，
tokenizer 对 200 条 val 的 eval 切片逐位一致。冒烟：txt 31 窗口 CE 2.978；jsonl 36/4 切分。

**2026-10-22 后续（v140）**''')
save(p, t)
print('HANDOVER ok')

# memory append
p = r'D:\explore\wav\.workbuddy\memory\2026-10-22.md'
t = load(p)
t = t.rstrip('\n') + '''

## v141 / 0.19.0.0（同日追加）

- 用户批评：0.18 只是“核心 + 残缺外壳”，丢了 3120012 的词表逻辑、菜单和 txt/jsonl 语义。
- 0.19 只修外壳，数值不变：加载 0.18 seed43 权重仍为 **3.0543/41.86/64.23**；
  tokenizer 对 200 条 val 的 eval 切片逐位一致。
- 词表：ByteLevel BPE 可训练/保存/加载/校验/回退；重建后 token id 变化，检查点元数据不符会拒绝加载。
- 数据：`.txt` = 预训练（全部 token，默认不验证）；`.jsonl` = 对话（mask，按对话 10% 切验证集）。
  超长对话按 system + 最近完整问答对截断（3120012 `_truncate_pairs`）。
- 菜单：恢复数据集选择、输入容错、词表子菜单；新增 13 在线学习（文件增量 / 对话中学习）。
  在线学习必须先从检查点加载；随机初始化下 Newton 头会爆（‖W_out‖ 52→11813/2 条）。
- 删除死参数：BATCH_SIZE、LEARNING_RATE、WEIGHT_DECAY、GRAD_CLIP_NORM、MEM_MOD_DECAY、ADAPT_TAU_MS。
- 冒烟：txt 11925 tokens/31 窗口 CE 2.978；jsonl limit40 自动切 36/4，val CE 4.0945；
  600-token 词表训练/保存/加载/编解码 OK；菜单启动与退出 OK。
- 文件：`RawHebb+bio+0.19.0.0.py`，分支 `RawHebb+shell19+0.9.264.0.py`，
  生成器 `_v163_shell19.py`、检查 `_v163_check.py`、在线测试 `_v163_online_test.py`。
'''
save(p, t)
print('memory ok')
