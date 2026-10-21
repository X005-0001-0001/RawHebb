# -*- coding: utf-8 -*-
"""0.9.196.0: bag of past *conditional* distributions instead of raw counts.

Each past context row is normalised before the causal cumsum, so a rare context
gets the same vote as a frequent one; the final bag is still normalised once.
"""
import io

src = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kfourbagprior+0.9.186.0.py'
dst = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnorm+0.9.196.0.py'
t = io.open(src, encoding='utf-8').read()
t = t.replace('tri4buck64hidden416four4kfourbagprior+0.9.186.0',
              'tri4buck64hidden416four4kbagnorm+0.9.196.0', 1)
t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4kfourbagprior'",
              "APP_BRANCH = 'tri4buck64hidden416four4kbagnorm'", 1)
t = t.replace("APP_VERSION = '0.9.186.0'", "APP_VERSION = '0.9.196.0'", 1)
t = t.replace('FOUR_BAG_PRIOR = True', 'FOUR_BAG_PRIOR = True\nFOUR_BAG_NORM = True', 1)

old = '''                _bagrows = self.pair_count[input_ids[0]]
                _bagcum = torch.cumsum(_bagrows, dim=0)'''
new = '''                _bagrows = self.pair_count[input_ids[0]]
                if FOUR_BAG_NORM:
                    _brs = _bagrows.sum(dim=-1, keepdim=True)
                    _bagrows = ((_bagrows + PAIR_PRIOR_STRENGTH * _prior)
                                / (_brs + PAIR_PRIOR_STRENGTH))
                _bagcum = torch.cumsum(_bagrows, dim=0)'''
if old not in t:
    raise SystemExit('bag block missing')
t = t.replace(old, new, 1)
io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote')
