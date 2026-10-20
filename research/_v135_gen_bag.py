# -*- coding: utf-8 -*-
"""Bag-of-past-tokens prior:
   - bagprior: an independent expert P(z | bag of all past window tokens)
   - fourbag: the last-token count expert interpolated with the bag prior
Both reuse the exact pair table; no new counts, no new constants.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'RawHebb+tri4buck64hidden416four4k+0.9.164.0.py')

BAG_CODE = '''                _bagrows = self.pair_count[input_ids[0]]
                _bagcum = torch.cumsum(_bagrows, dim=0)
                _bagprev = torch.cat(
                    [torch.zeros(1, self.vocab_size, dtype=_bagcum.dtype, device=_bagcum.device),
                     _bagcum[:-1]], dim=0)
                _bagsum = _bagprev.sum(dim=-1, keepdim=True)
                _bagp = ((_bagprev + PAIR_PRIOR_STRENGTH * _prior)
                         / (_bagsum + PAIR_PRIOR_STRENGTH))'''

FOUR_APPEND = '                _experts.append(torch.log(_fp.clamp(min=1e-12)).unsqueeze(0))'


def clone(tag, ver):
    t = io.open(BASE, encoding='utf-8').read()
    t = t.replace('RawHebb+tri4buck64hidden416four4k+0.9.164.0',
                  'RawHebb+%s+%s' % (tag, ver), 1)
    t = t.replace('分支      : tri4buck64hidden416four4k', '分支      : %s' % tag, 1)
    t = t.replace('版本      : 0.9.164.0', '版本      : %s' % ver, 1)
    t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4k'",
                  "APP_BRANCH = '%s'" % tag, 1)
    t = t.replace("APP_VERSION = '0.9.164.0'", "APP_VERSION = '%s'" % ver, 1)
    t = t.replace('_ew = torch.zeros(4)', '_ew = torch.zeros(6)', 1)
    return t


# 1) independent bag expert
t = clone('tri4buck64hidden416four4kbagprior', '0.9.185.0')
anchor = 'FOURGRAM_BUCKETS = 4096                 # 哈希桶数（架构量；越大冲突越少、内存越大）'
assert anchor in t
t = t.replace(anchor, anchor + '\nBAG_MEMORY = True                      # 全窗 past-token bag 先验专家', 1)
read = FOUR_APPEND + '''
            if BAG_MEMORY:
''' + BAG_CODE + '''
                _experts.append(torch.log(_bagp.clamp(min=1e-12)).unsqueeze(0))'''
assert FOUR_APPEND in t
t = t.replace(FOUR_APPEND, read, 1)
io.open(os.path.join(HERE, 'RawHebb+tri4buck64hidden416four4kbagprior+0.9.185.0.py'),
        'w', encoding='utf-8', newline='\n').write(t)
print('wrote bagprior')

# 2) four expert with the bag prior instead of the trigram prior
t = clone('tri4buck64hidden416four4kfourbagprior', '0.9.186.0')
anchor = 'FOURGRAM_BUCKETS = 4096                 # 哈希桶数（架构量；越大冲突越少、内存越大）'
assert anchor in t
t = t.replace(anchor, anchor + '\nFOUR_BAG_PRIOR = True', 1)
old = '''                _fp = ((_frow + TRI_PRIOR_STRENGTH * _tp)
                       / (_frs + TRI_PRIOR_STRENGTH))'''
new = '''                _fp = ((_frow + FOUR_PRIOR_STRENGTH * _bagp)
                       / (_frs + FOUR_PRIOR_STRENGTH))'''
assert old in t
# insert bag computation before four block; use the tri block's _prior
bag_anchor = '''            if FOURGRAM_MEMORY:
                _f1 = _prev1'''
assert bag_anchor in t
bag = '''            if FOUR_BAG_PRIOR:
''' + BAG_CODE + '''
            if FOURGRAM_MEMORY:
                _f1 = _prev1'''
t = t.replace(bag_anchor, bag, 1)
t = t.replace('FOURGRAM_BUCKETS = 4096                 # 哈希桶数（架构量；越大冲突越少、内存越大）',
              'FOURGRAM_BUCKETS = 4096                 # 哈希桶数（架构量；越大冲突越少、内存越大）\nFOUR_PRIOR_STRENGTH = 1.0', 1)
t = t.replace(old, new, 1)
io.open(os.path.join(HERE, 'RawHebb+tri4buck64hidden416four4kfourbagprior+0.9.186.0.py'),
        'w', encoding='utf-8', newline='\n').write(t)
print('wrote fourbagprior')
