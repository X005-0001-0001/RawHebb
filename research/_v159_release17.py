# -*- coding: utf-8 -*-
"""Create 0.17.0.0 from the 1M-bucket triple (tri1M + hidden432 + tri prior 2)."""
import io
import re

SRC = r'D:\explore\wav\RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0.py'
DST = r'D:\explore\wav\RawHebb+bio+0.17.0.0.py'
t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0',
              'RawHebb+bio+0.17.0.0', 1)
t = t.replace("APP_BRANCH = 'tri1Mbuck64hidden432four4kbagnormprior05tri2'",
              "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.251.0'", "APP_VERSION = '0.17.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.17.0.0.py')

new_round = '''本轮      : 0.17.0.0：把 0.16 的 tri 计数桶从 256k 扩到 1M（4 倍，纯计数分辨率）。
            其余不变：tri 先验 2、hidden 432、SEQ_LEN=128、行归一化 bag 先验、Jeffreys 0.5。
            全量 2850/1ep seed43：CE 3.1411 / top-1 41.59% / top-5 63.86%
            （0.16 为 3.1496/41.35/63.84）。三种子 3.1411/3.1458/3.1433，
            均值 3.1434/41.48/63.79；0.16 均值 3.1516/41.20/63.80，CE/top-1 改善、top-5 持平。
            桶轴仍未饱和（64k→256k→1M 全量 3.1584→3.1496→3.1411），但 2M 表约 23GB，
            本机内存到顶。纯 Hebbian 计数回退，无 BP、无 attention、无手填混合系数。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.17.0.0 2026-10-21  tri 桶 256k→1M；全量 seed43 CE 3.1411 / top-1 41.59% / top-5 63.86%，
#                         三种子均值 3.1434/41.48/63.79（0.16 均值 3.1516/41.20/63.80）。
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)
io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
