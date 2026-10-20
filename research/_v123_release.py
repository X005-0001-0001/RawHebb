# -*- coding: utf-8 -*-
"""Create 0.12.0.0 from the hidden384 champion."""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'RawHebb+tri4buck64hidden384+0.9.128.0.py')
DST = os.path.join(HERE, 'RawHebb+bio+0.12.0.0.py')

t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri4buck64hidden384+0.9.128.0',
              'RawHebb+bio+0.12.0.0', 1)
t = t.replace('分支      : tri4buck64hidden384', '分支      : bio', 1)
t = t.replace('版本      : 0.9.128.0', '版本      : 0.12.0.0', 1)
t = t.replace("APP_BRANCH = 'tri4buck64hidden384'", "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.128.0'", "APP_VERSION = '0.12.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.12.0.0.py')

new_round = '''本轮      : 0.12.0.0：基座隐藏宽度 128 -> 384，tri 先验 4，tri 桶 65536，
            仍为 SEQ_LEN=128、固定文件顺序、pair+tri 几何混合。
            全量 2850/1ep：CE 3.2006 / top-1 40.53% / top-5 63.33%。
            容量扫描（全量）：hidden 192 3.2248/40.13；224 3.2246/40.21；
            256 3.2182/40.54；288 3.2076/40.20；352 3.2061/40.61；
            384 3.2006/40.53。352 的 top-1 最高，384 的 CE/top-5 最好。
            v123 还测过 layers3/5、hidden64/160/192/224/256/288/320/352/384/
            400/416/448、seq64/96/160/192/256、tri prior 2/3/4/5/8、
            桶 8k/32k/64k/128k、order reverse/length、pair prior 2、
            mix rounds 1/4；见 docs/FINDINGS.md。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.12.0.0 2026-10-20  ��֧ bio���̶�˳�� + hidden 384 + tri prior 4 + 64k tri Ͱ��
#                         ȫ�� CE 3.2006 / top-1 40.53% / top-5 63.33%；
#                         hidden352 �� top-1 40.61%���� CE/top-5 ���Ե͡�
#                         v123 ���ݶ�ɨ�� 192..448��tri prior 2..8��Ͱ 8k..128k��
#                         seq64..256��layers3/5��order/pair/mix ��֧��
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)

io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
