# -*- coding: utf-8 -*-
"""Create 0.12.0.0 from the hidden416 capacity champion."""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'RawHebb+tri4buck64hidden416+0.9.137.0.py')
DST = os.path.join(HERE, 'RawHebb+bio+0.12.0.0.py')

t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri4buck64hidden416+0.9.137.0',
              'RawHebb+bio+0.12.0.0', 1)
t = t.replace('分支      : tri4buck64hidden416', '分支      : bio', 1)
t = t.replace('版本      : 0.9.137.0', '版本      : 0.12.0.0', 1)
t = t.replace("APP_BRANCH = 'tri4buck64hidden416'", "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.137.0'", "APP_VERSION = '0.12.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.12.0.0.py')

new_round = '''本轮      : 0.12.0.0：基座隐藏宽度 128 -> 416，tri 先验 4，tri 桶 65536，
            仍为 SEQ_LEN=128、固定文件顺序、pair+tri 几何混合。
            全量 2850/1ep：CE 3.1946 / top-1 40.34% / top-5 63.52%。
            容量扫描（全量）：hidden 192 3.2248/40.13；224 3.2246/40.21；
            256 3.2182/40.54；288 3.2076/40.20；352 3.2061/40.61；
            384 3.2006/40.53；416 3.1946/40.34。352 的 top-1 最高，
            416 的 CE/top-5 最好；差异在 0.01 CE / 0.3% top-1 量级。
            v123 还测过 layers3/5、hidden64/160/192/224/256/288/320/352/384/
            400/416/448、seq64/96/160/192/256、tri prior 2/3/4/5/8、
            桶 8k/32k/64k/128k、order reverse/length、pair prior 2、
            mix rounds 1/4；见 docs/FINDINGS.md。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.12.0.0 2026-10-20  ��֧ bio���̶�˳�� + hidden 416 + tri prior 4 + 64k tri Ͱ��
#                         ȫ�� CE 3.1946 / top-1 40.34% / top-5 63.52%��
#                         hidden352 �� top-1 40.61%���� CE/top-5 ���Եͣ�
#                         v123 ���ݶ�ɨ�� 192..448��tri prior 2..8��Ͱ 8k..128k��
#                         seq64..256��layers3/5��order/pair/mix ��֧��
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)

io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
