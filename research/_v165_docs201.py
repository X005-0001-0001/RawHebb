# -*- coding: utf-8 -*-
import io

ROOT = r'D:\开源系列\RawHebb'


def load(p):
    return io.open(p, encoding='utf-8').read()


def save(p, t):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)


def rep(t, old, new, n=1):
    if old not in t:
        raise SystemExit('MISSING: ' + old[:100])
    return t.replace(old, new, n)


p = ROOT + r'\README.md'
t = load(p)
t = rep(t, 'current release **0.20.0.0**', 'current release **0.20.1.0**')
t = rep(t, 'RawHebb+bio+0.20.0.0.py     single-file application, current',
        'RawHebb+bio+0.20.0.0.py     single-file application (superseded: see 0.20.1.0)\n'
        'RawHebb+bio+0.20.1.0.py     single-file application, current')
t = rep(t, 'RawHebb+shell20+0.9.265.0.py  the exact v142 shell branch that became 0.20.0.0',
        'RawHebb+shell20+0.9.265.0.py  the exact v142 shell branch that became 0.20.0.0\n'
        'RawHebb+shell20+0.9.266.0.py  the exact v143 branch that became 0.20.1.0')
t = rep(t, '| metric | **0.20.0.0** (full window) |',
        '| metric | **0.20.1.0** (full window) |')
t = rep(t, 'protocols and must not be compared: 0.18.0.0 was scored on the last 128 tokens, 0.20.0.0 on the',
        'protocols and must not be compared: 0.18.0.0 was scored on the last 128 tokens, 0.20.1.0 on the')

old_para = '''The model is **better than a window-matched bigram on CE, top-1 and top-5**, with no
back-propagation anywhere. 0.19.0.0 changes **no model math**; it is the 0.18 recipe with a repaired
shell. The vocabulary now has a full lifecycle (train / save / load / metadata validation / fallback)
instead of a hard-wired file, and there are two data pipelines: `.txt` is unsupervised pretraining
(all tokens are targets, no validation by default), `.jsonl` is dialogue (assistant mask, validation
split by dialogue at 10% or an explicit val file). Dialogues longer than the training window keep
the system prompt and the newest complete user/assistant pairs instead of being cut mid-sentence.
A new online-learning menu branch updates the model sample by sample (file streaming or during
chat). Loading the 0.18 checkpoint reproduces CE 3.0543 / top-1 41.86 % / top-5 64.23 % exactly.
'''
new_para = '''The model is **better than a window-matched bigram on CE, top-1 and top-5**, with no
back-propagation anywhere. 0.20.1.0 changes **no model math**; it is the 0.20.0.0 shell after a full
menu pass. Training now saves automatically; plain-text continuation works from very short prompts;
checkpoints no longer carry one-shot `V`/`spike_acc`/`M` state; EOF exits cleanly; and rebuilding a
vocabulary writes `tokenizer.bin/json` instead of overwriting the read-only base. The vocabulary has
a full lifecycle (train / save / load / metadata validation / fallback), and the two data pipelines
remain: `.txt` is unsupervised pretraining (all tokens are targets, no validation by default),
`.jsonl` is dialogue (assistant mask, record-level 10% validation split or an explicit val file).
Dialogues longer than the training window keep the system prompt and the newest complete
user/assistant pairs. Online learning (file streaming or during chat) requires a trained
checkpoint. The 0.18 checkpoint loads cleanly and scores 3.3545 / 38.06 % / 60.60 % under the new
full-window protocol, not the old 3.0543 / 41.86 / 64.23.
'''
t = rep(t, old_para, new_para)

t = rep(t, ' / 0.19.0.0 / 0.20.0.0', ' / 0.19.0.0 / 0.20.0.0 / 0.20.1.0')

old_cn = '''**必须说清楚的现状（0.20.0.0）**：0.20 只改外壳与数据口径，模型配方同 0.18，未重新训练。同一份 0.18 权重在 312 同窗口径（完整训练窗口，166 632 个计分位置）下是 **CE 3.3545 / top-1 38.06% / top-5 60.60%**；旧的 last-128 口径数字（CE 3.0543 / top-1 41.86% /
top-5 64.23%**（完整对话训练 ≤384 token + tri 桶 1M int16 + tri 先验 2 + hidden432 +
行归一化 past-context bag 先验；0.17 是 3.1411/41.59/63.86）。
同窗口、同 mask 的 bigram'''
new_cn = '''**必须说清楚的现状（0.20.1.0）**：0.20 只改外壳与数据口径，模型配方同 0.18，未重新训练。同一份 0.18 权重在 312 同窗口径（完整训练窗口，166 632 个计分位置）下是 **CE 3.3545 / top-1 38.06% / top-5 60.60%**；旧的 last-128 口径是 3.0543 / 41.86% / 64.23%，两套不可比。
同窗口、同 mask 的 bigram'''
t = rep(t, old_cn, new_cn)

old_cn2 = '''0.20.0.0 是把 3120012 的壳子真正移植过来：渲染（system/user/tool/assistant、reasoning、
tool_calls）、`_truncate_pairs`（保留 system + 最近完整问答对，按真实 token 计预算）、
按记录随机抽 10% 验证集、验证与训练同窗。0.19 的菜单 NameError 已修；`.txt` 预训练
（全 token、窗口打乱、默认不验证）与 `.jsonl` 对话（assistant mask）分开。模型配方不变，
因此 0.20 的指标与旧版 last-128 口径不可比：同一份 0.18 权重在 166 632 个位置上是
**CE 3.3545 / top-1 38.06% / top-5 60.60%**，旧口径为 3.0543 / 41.86% / 64.23%。'''
new_cn2 = '''0.20.0.0 是把 3120012 的壳子真正移植过来：渲染（system/user/tool/assistant、reasoning、
tool_calls）、`_truncate_pairs`（保留 system + 最近完整问答对，按真实 token 计预算）、
按记录随机抽 10% 验证集、验证与训练同窗。0.20.1.0 是全菜单实跑后的修复版：训练结束自动保存；
纯文本续写修复短提示的形状错误；检查点不再保存一次性状态；EOF 干净退出；重建词表不再覆盖
只读基座。`.txt` 预训练（全 token、窗口打乱、默认不验证）与 `.jsonl` 对话（assistant mask）
分开。模型配方不变，因此指标与旧版 last-128 口径不可比：同一份 0.18 权重在 166 632 个位置上是
**CE 3.3545 / top-1 38.06% / top-5 60.60%**，旧口径为 3.0543 / 41.86% / 64.23%。'''
t = rep(t, old_cn2, new_cn2)

row20 = t[t.index('| **0.20.0.0** | **usable shell'):]
row20 = row20[:row20.index('\n') + 1]
row201 = ('| **0.20.1.0** | **menu-tested shell release**. 0.20.0.0 ported the 3120012 data chain; '
          '0.20.1.0 fixes what a full menu pass found: automatic checkpoint save after training, '
          'plain-text continuation from short prompts, clean checkpoint loading (transient '
          '`V`/`spike_acc`/`M` no longer stored), EOF exit, and vocabulary rebuilds that do not '
          'overwrite the read-only base. Same model and full-window numbers as 0.20.0.0: CE '
          '**3.3545** / top-1 **38.06 %** / top-5 **60.60 %** (166 632 positions). |\n')
t = t.replace(row20, row201 + row20.replace('| **0.20.0.0** | **usable shell',
                                            '| **0.20.0.0** | **superseded by 0.20.1.0; usable shell'), 1)
save(p, t)
print('README ok')

p = ROOT + r'\CHANGELOG.md'
t = load(p)
i = t.index('## [0.20.0.0]')
entry = '''## [0.20.1.0] - current
* **Full menu test fixes.** Training on menu 1 now saves the checkpoint automatically (no y/n
  prompt). Plain-text continuation no longer crashes on prompts shorter than three tokens: the
  trigram/fourgram key construction now left-pads to the sequence length instead of producing a
  longer tensor; the prompt is also prefixed with `<|bos|>`.
* **Clean checkpoints.** `V`, `spike_acc` and the unused `M` buffer are one-shot run state, not
  model weights; they are no longer stored. Old checkpoints are filtered on load, so the log no
  longer reports "skipped 4 tensors with mismatched shapes". A 1143-token checkpoint drops from
  3.6 GiB to 2.3 GiB.
* **Robust menu.** EOF/Ctrl-D exits cleanly instead of raising a traceback. Rebuilding the
  vocabulary from menu 12 writes `tokenizer.bin/json` and leaves the read-only base
  `tokenizer_v63b4000.*` untouched.
* **Verification:** menu items 1-13 were run end to end (txt and jsonl training, both chat modes,
  save/load/stats/params, rejection sampling, self-consistency, long-context stress, preprocessing,
  evaluation, tokenizer view/round-trip/rebuild, online file/chat). Menu 1 auto-saved; menu 2 txt
  generated; menu 4 loaded 0 skipped / 0 missing / 0 unexpected; menu 12 rebuild preserved the base.
* **Numbers unchanged from 0.20.0.0:** same 0.18 checkpoint under the full-window protocol is CE
  **3.3545** / top-1 **38.06 %** / top-5 **60.60 %** over 166 632 positions. Not comparable to the
  old last-128 numbers.

'''
t = t[:i] + entry + t[i:]
save(p, t)
print('CHANGELOG ok')

p = ROOT + r'\CITATION.cff'
t = load(p)
t = rep(t, 'version: 0.20.0.0', 'version: 0.20.1.0')
save(p, t)
print('CITATION ok')

p = ROOT + r'\docs\FINDINGS.md'
t = load(p)
v143 = '''
### v143 - all menu items exercised (0.20.1.0)

The full menu pass found four real defects, all fixed here.

1. **Menu 1 asked whether to save.** Training now saves to `brain.pt` automatically and sets the
   loaded flag. Verified with a 4-record dialogue file: 3 train / 1 val (record sample), full
   window validation (5 584 scored positions), then `[CKPT] 已保存`.
2. **Menu 2 txt crashed on short prompts.** `你好` tokenises to two ids; the trigram key builder
   used `_ids[:-2]` and the fourgram builder padded three places without truncating, so one tensor
   had length 3 and another length 2 (`RuntimeError: size of tensor a (3) must match ... (2)`).
   Both keys are now built as `cat([pad..., ids])[:T]`; the txt prompt also starts with `<|bos|>`.
   Generation from `你好` and menu exit both succeed.
3. **Loading a fresh checkpoint reported four shape mismatches.** `V` and `spike_acc` were
   registered buffers, so the last forward's `(1,432)` state was saved while a fresh layer had
   `(432,)`. `M` (432³ per layer, 1.3 GiB) was registered but never read. All three are transient
   or dead; they are no longer buffers, and old checkpoints drop those keys on load. Menu 4 now
   reports `skip 0 | missing 0 | unexpected 0`, and a 1143-token checkpoint shrinks from 3.6 GiB
   to 2.3 GiB.
4. **Rebuilding the vocabulary overwrote the base.** From a base-loaded tokenizer the menu passed
   the base paths to the trainer, so a 300-token test overwrote `tokenizer_v63b4000.*`. The base
   was restored from an identical copy in `wav3auto` (2754 tokens, legacy metadata), and the menu
   now always writes `tokenizer.bin/json`; a repeated rebuild leaves the base hash unchanged.

Menu items exercised in a clean directory with only the application and a `.txt`: 1 (txt earlier,
jsonl here), 2 (jsonl and txt), 3, 4, 5, 6, 7 (1 sample, k=1), 8 (k=1), 9 (4 fractions), 10, 11,
12 (view + round-trip + rebuild), 13 (file increment and in-chat learning, 366 target tokens).
All exited 0 with no traceback. `--selftest` still passes with the 2754-token base.

This is a usability release: the model recipe and all performance numbers are unchanged. The same
0.18 checkpoint still scores 3.3545 / 38.06 % / 60.60 % on full training windows, and those numbers
remain incomparable to the old last-128 protocol.
'''
t = t.rstrip('\n') + '\n' + v143
save(p, t)
print('FINDINGS ok')

p = r'D:\explore\wav\HANDOVER.md'
t = load(p)
anchor = '**2026-10-22 后续（v142）**'
new = '''**2026-10-22 后续（v143）**：**0.20.1.0 全菜单测试修复版**。修复：训练结束自动保存；
菜单 2 txt 短提示（<3 token）三/四阶键形状错误；检查点保存一次性 V/spike_acc/M 导致
“跳过形状不符 4 个”（旧档加载时过滤，1143 词表档 3.6→2.3 GiB）；菜单 EOF 干净退出；
菜单 12 重建词表写 tokenizer.bin/json，不覆盖只读基座。菜单 1–13 全部实跑通过，
`--selftest` 通过，基座 2754 已从 wav3auto 恢复。指标不变（0.18 权重全窗 3.3545/38.06/60.60）。

'''
t = rep(t, anchor, new + anchor)
save(p, t)
print('HANDOVER ok')

p = r'D:\explore\wav\.workbuddy\memory\2026-10-22.md'
t = load(p)
t = t.rstrip('\n') + '''

## v143 / 0.20.1.0（全菜单测试）

- 用户要求所有菜单快速实测；已知：训练后应自动保存、菜单 2 txt 加载已有模型报错。
- 修复 4 个真 bug：
  1. 菜单 1 训练后询问保存 → 改为自动保存；
  2. 菜单 2 txt 短提示（2 token）tri/four 键长度不一致 → `cat([pad...,ids])[:T]`，提示加 bos；
  3. V/spike_acc/M 被当 buffer 存进 ckpt（M 1.3GB 从未读取）→ 取消 buffer，旧档加载过滤；
     加载日志 0 skip，1143 档 3.6→2.3 GiB；
  4. 菜单 12 从基座重建会覆盖 tokenizer_v63b4000.* → 固定写 tokenizer.bin/json；
     误覆盖的基座已从 wav3auto 恢复（2754 tokens，hash 保持）。
- 菜单 1–13 全部实跑（干净目录）：txt/jsonl 训练、两种对话、存取/统计/参数、7/8/9/10/11、
  分词器 查看/往返/重建、在线 文件/对话；全 exit 0。
- 0.20.1.0 发布文件由 0.20.0.0 生成；分支 0.9.266.0。
'''
save(p, t)
print('memory ok')
