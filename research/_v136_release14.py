# -*- coding: utf-8 -*-
"""Create 0.14.0.0 from hidden416 + 4k expert + bag-of-past prior."""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'RawHebb+tri4buck64hidden416four4kfourbagprior+0.9.186.0.py')
DST = os.path.join(HERE, 'RawHebb+bio+0.14.0.0.py')

t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri4buck64hidden416four4kfourbagprior+0.9.186.0',
              'RawHebb+bio+0.14.0.0', 1)
t = t.replace('分支      : tri4buck64hidden416four4kfourbagprior', '分支      : bio', 1)
t = t.replace('版本      : 0.9.186.0', '版本      : 0.14.0.0', 1)
t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4kfourbagprior'",
              "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.186.0'", "APP_VERSION = '0.14.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.14.0.0.py')

new_round = '''本轮      : 0.14.0.0：在 0.13 的 hidden416 + 4k 计数专家上，把 four 专家的先验从
            三元分布换成「窗口内过去 token 的 bag 分布」：复用精确 pair 表做累计计数，
            没有新计数表、没有新常数。仍为 SEQ_LEN=128、固定顺序、tri prior 4、64k tri 桶。
            全量 2850/1ep seed43：CE 3.1740 / top-1 40.43% / top-5 63.77%（0.13 为
            3.1772/40.43/63.69）。三种子 3.1740/3.1773/3.1753，均值 3.1755/40.45/63.76；
            0.13 均值 3.1793/40.41/63.66，三项均改善。
            纯 Hebbian 计数回退，无 BP、无 attention、无手填混合系数。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.14.0.0 2026-10-20  hidden416 + 4k 计数专家；four 先验改为 past-token bag；
#                         全量 CE 3.1740 / top-1 40.43% / top-5 63.77%，
#                         三种子均值 3.1755/40.45/63.76（0.13 均值 3.1793/40.41/63.66）。
#   0.13.0.0 2026-10-20  hidden416 + 4k 计数专家（tri 先验回退）；
#                         全量 CE 3.1772 / top-1 40.43% / top-5 63.69%。
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)

io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
