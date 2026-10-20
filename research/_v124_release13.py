# -*- coding: utf-8 -*-
"""Create 0.13.0.0 from hidden416 + 4-gram 4k."""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'RawHebb+tri4buck64hidden416four4k+0.9.164.0.py')
DST = os.path.join(HERE, 'RawHebb+bio+0.13.0.0.py')

t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+tri4buck64hidden416four4k+0.9.164.0',
              'RawHebb+bio+0.13.0.0', 1)
t = t.replace('分支      : tri4buck64hidden416four4k', '分支      : bio', 1)
t = t.replace('版本      : 0.9.164.0', '版本      : 0.13.0.0', 1)
t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4k'", "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.164.0'", "APP_VERSION = '0.13.0.0'", 1)
t = t.replace('RawHebb+bio+0.6.0.0.py', 'RawHebb+bio+0.13.0.0.py')

new_round = '''本轮      : 0.13.0.0：在 0.12 的 hidden416 基座上加入 4-gram 计数专家（4k 哈希桶，
            以 tri 分布为先验回退），仍为 SEQ_LEN=128、固定顺序、tri prior 4、64k tri 桶。
            全量 2850/1ep：CE 3.1772 / top-1 40.43% / top-5 63.69%（seed43）。
            相对 0.12（3.1946/40.34/63.52）：CE -0.0174、top-1 +0.09、top-5 +0.17，
            三项均不差。8k 桶分支的 top-5 更高（63.84%）但 CE/top-1 略低；
            4k 桶是综合冠军，8k 桶作为 top-5 分支保留。
            4-gram 表是 4k×V 的 Hebbian 计数，读回 (C + 4*_tp)/(row+4)，无新魔数。
内核      :'''
t = re.sub(r'本轮      :.*?内核      :', new_round, t, count=1, flags=re.S)

anchor = '#   0.11.0.0 2026-10-20'
lineage = '''#   0.13.0.0 2026-10-20  ��֧ bio��hidden416 + 4-gram 4k ��������ר�ң��� tri �Ȼأ���
#                         ȫ�� CE 3.1772 / top-1 40.43% / top-5 63.69%��
#                         0.12 �� 3.1946/40.34/63.52��4-gram ��������ָ�궼������
#   0.12.0.0 2026-10-20  hidden416 + tri4 + buck64��CE 3.1946 / top-1 40.34%��
#   0.11.0.0 2026-10-20'''
if anchor not in t:
    raise SystemExit('lineage anchor missing')
t = t.replace(anchor, lineage, 1)

io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
