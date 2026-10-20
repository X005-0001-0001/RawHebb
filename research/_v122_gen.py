# -*- coding: utf-8 -*-
"""v122 branch generator: tri backoff / confidence-adaptive mix / longer windows.

Base is the exact 0.10.0.0 champion source (RawHebb+seq128tri+0.9.33.0.py).
Every branch is written as a standalone file and kept.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'RawHebb+seq128tri+0.9.33.0.py')

TRI_OLD = ('                _tp = (_trow + PAIR_PRIOR_STRENGTH * _prior) / '
           '(_trs + PAIR_PRIOR_STRENGTH)')
TRI_BACKOFF = ('                _tp = (_trow + PAIR_PRIOR_STRENGTH * _p[0]) / '
               '(_trs + PAIR_PRIOR_STRENGTH)')

MIX_OLD = '            logits = sum(_ew[i] * _experts[i] for i in range(_K))'

MIX_NLL_OLD = ('                def _mix_nll(wv):\n'
               '                    return float(F.cross_entropy(sum(wv[i] * _E[i] '
               'for i in range(_K)), _t))')
MIX_NLL_LINEAR = '''                def _mix_nll(wv):
                    _mx = torch.stack(_E).max(dim=0).values
                    _pr = sum(wv[i] * torch.exp(_E[i] - _mx) for i in range(_K))
                    return float(-torch.log(_pr.gather(1, _t.unsqueeze(1)) + 1e-12).mean())'''

MIX_LINEAR = '''            _mx = torch.stack(_experts).max(dim=0).values
            _pr = sum(_ew[i] * torch.exp(_experts[i] - _mx) for i in range(_K))
            logits = torch.log(_pr + 1e-12) + _mx'''

PREFIX_OLD = '            if UNIGRAM_EXPERT:'
PREFIX_NEW = '''            if PREFIX_EXPERT:
                # v122: causal within-window prefix cache. Counts only tokens
                # already seen at or before the current position; no future leak.
                _oh = F.one_hot(input_ids, self.vocab_size).float()
                _pc = _oh.cumsum(dim=1)
                _pp = ((_pc + PAIR_PRIOR_STRENGTH * _prior)
                       / (_pc.sum(dim=-1, keepdim=True) + PAIR_PRIOR_STRENGTH))
                _experts.append(torch.log(_pp.clamp(min=1e-12)))
            if UNIGRAM_EXPERT:'''

PREFIX_DECL_OLD = 'UNIGRAM_EXPERT = False'
PREFIX_DECL_NEW = 'UNIGRAM_EXPERT = False\nPREFIX_EXPERT = False'
MIX_ADAPT = '''            # v122: per-position confidence weights, derived only from counts.
            #   w_k(pos) = w_k_global * n_k(pos) / (n_k(pos) + 1 pseudo-count)
            #   base has no context count, so its weight is w_base * 1.
            _np = _rs
            _nt = _trs.unsqueeze(0)
            _wb = _ew[0] * torch.ones_like(_np)
            _wp = _ew[1] * (_np / (_np + 1.0))
            _wt = _ew[2] * (_nt / (_nt + 1.0))
            _den = (_wb + _wp + _wt).clamp(min=1e-9)
            logits = ((_wb / _den) * _experts[0]
                      + (_wp / _den) * _experts[1]
                      + (_wt / _den) * _experts[2])'''


def make(branch, version, repls):
    with io.open(BASE, encoding='utf-8') as f:
        t = f.read()
    t = t.replace('RawHebb+seq128tri+0.9.33.0', 'RawHebb+%s+%s' % (branch, version), 1)
    t = t.replace('分支      : seq128tri', '分支      : %s' % branch, 1)
    t = t.replace('版本      : 0.9.33.0', '版本      : %s' % version, 1)
    t = t.replace("APP_BRANCH = 'seq128tri'", "APP_BRANCH = '%s'" % branch, 1)
    t = t.replace("APP_VERSION = '0.9.33.0'", "APP_VERSION = '%s'" % version, 1)
    for old, new in repls:
        if old not in t:
            raise SystemExit('MISSING in base: %r' % old[:80])
        t = t.replace(old, new, 1)
    out = os.path.join(HERE, 'RawHebb+%s+%s.py' % (branch, version))
    with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(t)
    print('wrote', os.path.basename(out))


VARIANTS = [
    ('trinoshuf', '0.9.52.0',
     [('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False')]),
    ('tribackoff', '0.9.53.0',
     [(TRI_OLD, TRI_BACKOFF)]),
    ('tribackoffshuf', '0.9.54.0',
     [(TRI_OLD, TRI_BACKOFF), ('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False')]),
    ('triadapt', '0.9.55.0',
     [(MIX_OLD, MIX_ADAPT)]),
    ('tribackoffadapt', '0.9.56.0',
     [(TRI_OLD, TRI_BACKOFF), (MIX_OLD, MIX_ADAPT)]),
    ('seq192tri', '0.9.57.0',
     [('SEQ_LEN = 128', 'SEQ_LEN = 192')]),
    ('seq256tri', '0.9.58.0',
     [('SEQ_LEN = 128', 'SEQ_LEN = 256')]),
    ('tribackoff192', '0.9.59.0',
     [(TRI_OLD, TRI_BACKOFF), ('SEQ_LEN = 128', 'SEQ_LEN = 192')]),
    ('trilin', '0.9.65.0',
     [(MIX_NLL_OLD, MIX_NLL_LINEAR), (MIX_OLD, MIX_LINEAR),
      ('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False')]),
    ('trilinshuf', '0.9.66.0',
     [(MIX_NLL_OLD, MIX_NLL_LINEAR), (MIX_OLD, MIX_LINEAR)]),
    ('trinoshufpref', '0.9.67.0',
     [(PREFIX_DECL_OLD, PREFIX_DECL_NEW), (PREFIX_OLD, PREFIX_NEW),
      ('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False')]),
    ('tripref', '0.9.68.0',
     [(PREFIX_DECL_OLD, PREFIX_DECL_NEW), (PREFIX_OLD, PREFIX_NEW)]),
    ('trinoshufpreflin', '0.9.69.0',
     [(PREFIX_DECL_OLD, PREFIX_DECL_NEW), (PREFIX_OLD, PREFIX_NEW),
      (MIX_NLL_OLD, MIX_NLL_LINEAR), (MIX_OLD, MIX_LINEAR),
      ('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False')]),
]

if __name__ == '__main__':
    for b, v, r in VARIANTS:
        make(b, v, r)
