# -*- coding: utf-8 -*-
import io

p = r'D:\explore\wav\HANDOVER.md'
t = io.open(p, encoding='utf-8').read()
old = '**0.19.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.19.0.0>（commit \\e3d39d5\\））'
new = ('**0.19.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.19.0.0>'
       '（commit `e3d39d5`，已废弃）、\n'
       '**0.20.0.0**：<https://github.com/X005-0001-0001/RawHebb/releases/tag/0.20.0.0>）')
if old not in t:
    raise SystemExit('release line missing')
t = t.replace(old, new, 1)

v142 = '''**2026-10-22 后续（v142）**：**0.20.0.0 可用版**。0.19 的菜单在干净目录点 1 会 `NameError`
（helper 被替换块吃掉）；数据层也只是近似 3120012。0.20 直接移植 312 的渲染（system/tool/
reasoning/tool_calls）、`_truncate_pairs`（保留 system + 最近完整问答对，按真实 token 计预算）、
按记录抽 10% 验证集、验证与训练同窗（`EVAL_LEN=0`）。`.txt` 预训练全 token、窗口打乱、默认不验证。
同一份 0.18 权重在新口径 166 632 位置上 **CE 3.3545/38.06/60.60**，旧 last-128 是
3.0543/41.86/64.23，两套不可比。干净目录菜单训练 2 窗口通过。

'''
anchor = '**2026-10-22 后续（v141）**'
if anchor not in t:
    raise SystemExit('v141 anchor missing')
t = t.replace(anchor, v142 + anchor, 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(t)

mp = r'D:\explore\wav\.workbuddy\memory\2026-10-22.md'
mt = io.open(mp, encoding='utf-8').read()
if 'v142 / 0.20.0.0' not in mt:
    mt = mt.rstrip('\n') + '''

## v142 / 0.20.0.0（同日追加，可用版）

- 0.19 实测在干净目录菜单 1 直接 `NameError`（helpers 被 SECTION 10 替换吃掉）；已修。
- 用户要求“把 312 的壳子拔下来修剪干净适配核心”，不要再拿旧 benchmark 当挡箭牌。
- 0.20 直接移植 312 数据链：
  - render_messages 语义：bos + tools + system/user/tool/assistant + eos；
    assistant 的 reasoning_content/content/tool_calls 参与训练，其余 -100；
  - `_truncate_pairs`：system 保留 + 最近完整问答对，按真实 token 计预算；
  - jsonl 验证按记录随机抽 10%（seed 固定），也可外部 val；
  - 验证与训练同窗（EVAL_LEN=0）；
  - txt 预训练：整文件窗口、全 token、窗口按 seed 打乱、默认不验证。
- 同一 0.18 权重在新口径 166632 位置 **3.3545/38.06/60.60**；旧 last-128 3.0543/41.86/64.23，不可比。
- 测试：合成数据（system/tools/reasoning/截断）、干净目录菜单训练 2 窗口、jsonl 自动切分
  36/4、txt 预训练、在线学习（需已训练权重）。
- 文件：`RawHebb+bio+0.20.0.0.py`，分支 `RawHebb+shell20+0.9.265.0.py`，
  生成 `_v164_build20.py`、测试 `_v164_data_test.py`。
'''
    io.open(mp, 'w', encoding='utf-8', newline='\n').write(mt)
print('ok')
