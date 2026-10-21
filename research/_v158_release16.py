# -*- coding: utf-8 -*-
"""Create 0.16.0.0 from the 256k-bucket triple (tri256k + hidden432 + tri prior 2)."""
import io
import re

SRC = r'D:\explore\wav\RawHebb+tri256kbuck64hidden432four4kbagnormprior05tri2+0.9.244.0.py'
DST = r'D:\explore\wav\RawHebb+bio+0.16.0.0.py'
t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri256kbuck64hidden432four4kbagnormprior05tri2+0.9.244.0',
              'RawHebb+bio+0.16.0.0', 1)
t = t.replace("APP_BRANCH = 'tri256kbuck64hidden432four4kbagnormprior05tri2'",
              "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.244.0'", "APP_VERSION = '0.16.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.16.0.0.py')

new_round = '''本轮      : 0.16.0.0：在 0.15 的行归一化 bag 先验上换三个纯计数刻度：tri 计数桶 64k→256k，
            tri 先验强度 4→2，隐藏 416→432（SEQ_LEN 仍 128，顺序固定，bag 行归一化与
            Jeffreys 0.5 four 伪计数不变）。全量 2850/1ep seed43：
            CE 3.1496 / top-1 41.35% / top-5 63.84%（0.15 为 3.1705/40.38/63.78）。
            三种子 3.1496/3.1546/3.1507，均值 3.1516/41.20/63.80；0.15 均值
            3.1724/40.45/63.78，CE -0.021、top-1 +0.75、top-5 持平。
            纯 Hebbian 计数回退，无 BP、无 attention、无手填混合系数。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.16.0.0 2026-10-21  tri 桶 64k→256k、tri 先验 4→2、hidden 416→432；全量 seed43
#                         CE 3.1496 / top-1 41.35% / top-5 63.84%，三种子均值 3.1516/41.20/63.80
#                         （0.15 均值 3.1724/40.45/63.78）。
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)
io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
