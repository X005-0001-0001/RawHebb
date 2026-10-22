# -*- coding: utf-8 -*-
"""Create 0.18.0.0 from the full-dialogue training branch (train384, eval 128)."""
import io
import re

SRC = r'D:\explore\wav\RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384+0.9.258.0.py'
DST = r'D:\explore\wav\RawHebb+bio+0.18.0.0.py'
t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384+0.9.258.0',
              'RawHebb+bio+0.18.0.0', 1)
t = t.replace("APP_BRANCH = 'tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384'",
              "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.258.0'", "APP_VERSION = '0.18.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.18.0.0.py')

new_round = '''本轮      : 0.18.0.0：训练窗口从「对话最后 128 token」改为「完整对话」（上限 384），
            评测口径完全不变（仍是每条对话最后 128 个 token）。语料对话平均 472 token、
            中位数 511，所以 128 训练只看到约 1/4 内容；0.18 让计数/状态在线学习时
            看到完整对话，无 BP、无新参数、无魔数。
            500 条筛（训练长度 128→192→256→384→512）：
            3.5864 → 3.5137 → 3.4891 → 3.4529 → 3.4503。
            全量 2850/1ep seed43：CE 3.0543 / top-1 41.86% / top-5 64.23%
            （0.17 为 3.1411/41.59/63.86）。三种子 3.0543/3.0554/3.0561，
            均值 3.0553/41.74/64.23；0.17 均值 3.1434/41.48/63.79。
            训练 384 与 512 全量并列（3.0553 vs 3.0542，差在 1 个种子方差内），
            384 的 top-1 高 0.31、速度快约 30%，故取 384。
            实测 1M 计数表用 int16 与 float32 数值一致（3.1411 vs 3.1411），
            表内存减半；2M int16 筛选 3.5869 未超过 1M 的 3.5864，计数桶轴到顶。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.18.0.0 2026-10-22  训练窗口上限 128→384（评测仍取最后 128 token）；全量 seed43
#                         CE 3.0543 / top-1 41.86% / top-5 64.23%，三种子均值
#                         3.0553/41.74/64.23（0.17 均值 3.1434/41.48/63.79）。
#                         int16 计数表与 float32 数值一致；2M 桶到顶。
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)
io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
