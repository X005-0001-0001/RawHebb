# -*- coding: utf-8 -*-
"""Create 0.15.0.0 from the row-normalised bag branch (four prior 0.5)."""
import io
import re

SRC = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnormprior05+0.9.206.0.py'
DST = r'D:\explore\wav\RawHebb+bio+0.15.0.0.py'
t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri4buck64hidden416four4kbagnormprior05+0.9.206.0',
              'RawHebb+bio+0.15.0.0', 1)
t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4kbagnormprior05'",
              "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.206.0'", "APP_VERSION = '0.15.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.15.0.0.py')

new_round = '''本轮      : 0.15.0.0：在 0.14 的 hidden416 + 4k 计数专家上，把 four 专家的 bag 先验改成
            「过去上下文的归一化条件分布之和」：每个过去 token 的 pair 行先归一化再做因果
            cumsum，罕见上下文与高频上下文各得一票；four 专家的伪计数取 Jeffreys 先验 0.5。
            仍为 SEQ_LEN=128、固定顺序、tri prior 4、64k tri 桶。全量 2850/1ep seed43：
            CE 3.1705 / top-1 40.38% / top-5 63.78%（0.14 为 3.1740/40.43/63.77）。三种子
            3.1705/3.1749/3.1717，均值 3.1724/40.45/63.78；0.14 均值 3.1755/40.45/63.76，
            CE/top-5 改善、top-1 持平。纯 Hebbian 计数回退，无 BP、无 attention、无手填混合系数。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.15.0.0 2026-10-21  行归一化的 past-context bag 先验 + Jeffreys 0.5 four 伪计数；
#                         全量 seed43 CE 3.1705 / top-1 40.38% / top-5 63.78%，
#                         三种子均值 3.1724/40.45/63.78（0.14 均值 3.1755/40.45/63.76）。
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)
io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
