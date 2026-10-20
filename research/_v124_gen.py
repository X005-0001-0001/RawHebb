# -*- coding: utf-8 -*-
"""v124: new local mechanisms on the hidden384 champion.

Branches: last-successor copy, prompt copy, 4-gram backoff, top-1 mixture.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'RawHebb+tri4buck64hidden384+0.9.128.0.py')

CACHE_DECL_OLD = 'UNIGRAM_EXPERT = False'
CACHE_DECL_NEW = 'UNIGRAM_EXPERT = False\nLASTCOPY_EXPERT = False\nPROMPTCOPY_EXPERT = False'

TRI_APPEND = '                _experts.append(torch.log(_tp).unsqueeze(0))'
LASTCOPY = '''                _experts.append(torch.log(_tp).unsqueeze(0))
            if LASTCOPY_EXPERT:
                # Causal within-window cache: predict the successor most recently
                # observed for the current token in this window.
                _lc_ids = input_ids[0]
                _pred = torch.full((T,), -1, dtype=torch.long, device=dev)
                _last = {}
                for _t in range(T):
                    _cur = int(_lc_ids[_t].item())
                    if _cur in _last:
                        _pred[_t] = _last[_cur]
                    if _t > 0:
                        _last[int(_lc_ids[_t - 1].item())] = _cur
                _cp = _prior.unsqueeze(0).repeat(T, 1).clone()
                _has = _pred >= 0
                if bool(_has.any()):
                    _cp[_has, _pred[_has]] += 1.0
                    _cp[_has] = _cp[_has] / 2.0
                _experts.append(torch.log(_cp.clamp(min=1e-12)).unsqueeze(0))'''

PROMPTCOPY = '''                _experts.append(torch.log(_tp).unsqueeze(0))
            if PROMPTCOPY_EXPERT and labels is not None:
                # Distribution of prompt tokens (labels == -100) for this dialogue.
                _pm = (labels[0] == -100)
                _pid = input_ids[0][_pm]
                _pcount = torch.zeros(self.vocab_size, device=dev)
                if _pid.numel() > 0:
                    _pcount.index_add_(0, _pid,
                                       torch.ones(_pid.numel(), device=dev))
                _pp = ((_pcount + _prior) / (_pcount.sum() + 1.0)).clamp(min=1e-12)
                _experts.append(torch.log(_pp).view(1, 1, -1)
                                .expand(B, T, self.vocab_size))'''

MIX_NLL_OLD = '''                def _mix_nll(wv):
                    return float(F.cross_entropy(sum(wv[i] * _E[i] for i in range(_K)), _t))'''
MIX_NLL_TOP1 = '''                def _mix_nll(wv):
                    _lg = sum(wv[i] * _E[i] for i in range(_K))
                    if MIX_OBJECTIVE == 'top1':
                        return -float((_lg.argmax(-1) == _t).float().mean())
                    return float(F.cross_entropy(_lg, _t))'''

MIX_DECL_OLD = 'MIX_ITERS = 24'
MIX_DECL_NEW = "MIX_ITERS = 24\nMIX_OBJECTIVE = 'nll'"

FOUR_DECL_OLD = 'TRIGRAM_BUCKETS = 16384'  # not used; base has 65536
FOUR_DECL_ANCHOR = 'TRIGRAM_BUCKETS = 65536'
FOUR_DECL_NEW = 'TRIGRAM_BUCKETS = 65536\nFOURGRAM_MEMORY = False\nFOURGRAM_BUCKETS = 65536'

FOUR_INIT_ANCHOR = "            self.register_buffer('tri_count', torch.zeros(TRIGRAM_BUCKETS, vocab_size))"
FOUR_INIT_NEW = '''            self.register_buffer('tri_count', torch.zeros(TRIGRAM_BUCKETS, vocab_size))
            if FOURGRAM_MEMORY:
                self.register_buffer('four_count', torch.zeros(FOURGRAM_BUCKETS, vocab_size))'''

FOUR_READ = '''                _experts.append(torch.log(_tp).unsqueeze(0))
            if FOURGRAM_MEMORY:
                _f1 = _prev1
                _f2 = _prev2
                _p3 = torch.cat([_pad, _pad, _pad, _ids[:-3]])
                _fkeys = (((_p3 * self._tri_base + _f2) * self._tri_base + _f1)
                          % FOURGRAM_BUCKETS)
                _frow = self.four_count[_fkeys]
                _frs = _frow.sum(dim=-1, keepdim=True)
                _fp = ((_frow + TRI_PRIOR_STRENGTH * _tp)
                       / (_frs + TRI_PRIOR_STRENGTH))
                _experts.append(torch.log(_fp.clamp(min=1e-12)).unsqueeze(0))'''

FOUR_UPDATE_ANCHOR = '''                self.tri_count.index_put_(
                    (_xs2, _ys2), torch.ones(_xs2.numel(), dtype=self.tri_count.dtype),
                    accumulate=True)'''
FOUR_UPDATE = '''                self.tri_count.index_put_(
                    (_xs2, _ys2), torch.ones(_xs2.numel(), dtype=self.tri_count.dtype),
                    accumulate=True)
        if FOURGRAM_MEMORY:
            _ids3 = input_ids[0]
            _padv3 = torch.full((1,), self.pad_id, dtype=_ids3.dtype, device=_ids3.device)
            _q1 = torch.cat([_padv3, _ids3[:-1]])
            _q2 = torch.cat([_padv3, _padv3, _ids3[:-2]])
            _q3 = torch.cat([_padv3, _padv3, _padv3, _ids3[:-3]])
            _fkeys = (((_q3 * self._tri_base + _q2) * self._tri_base + _q1)
                      % FOURGRAM_BUCKETS)
            _mk3 = torch.ones_like(mpos[0]) if PAIR_SCOPE == 'all' else mpos[0]
            if bool(_mk3.any()):
                _fx = _fkeys[:T - 1][_mk3]
                _fy = _ids3[1:T][_mk3]
                self.four_count.index_put_(
                    (_fx, _fy),
                    torch.ones(_fx.numel(), dtype=self.four_count.dtype),
                    accumulate=True)'''


def make(branch, version, repls):
    with io.open(BASE, encoding='utf-8') as f:
        t = f.read()
    t = t.replace('RawHebb+tri4buck64hidden384+0.9.128.0',
                  'RawHebb+%s+%s' % (branch, version), 1)
    t = t.replace('分支      : tri4buck64hidden384', '分支      : %s' % branch, 1)
    t = t.replace('版本      : 0.9.128.0', '版本      : %s' % version, 1)
    t = t.replace("APP_BRANCH = 'tri4buck64hidden384'", "APP_BRANCH = '%s'" % branch, 1)
    t = t.replace("APP_VERSION = '0.9.128.0'", "APP_VERSION = '%s'" % version, 1)
    for old, new in repls:
        if old not in t:
            raise SystemExit('MISSING anchor: %r' % old[:80])
        t = t.replace(old, new, 1)
    out = os.path.join(HERE, 'RawHebb+%s+%s.py' % (branch, version))
    with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(t)
    print('wrote', os.path.basename(out))


LASTPROMPT = LASTCOPY + '\n' + PROMPTCOPY.replace(TRI_APPEND, '', 1)

V = [
    ('lastpromptcopy', '0.9.152.0', [
        (CACHE_DECL_OLD, CACHE_DECL_NEW),
        ('LASTCOPY_EXPERT = False', 'LASTCOPY_EXPERT = True'),
        ('PROMPTCOPY_EXPERT = False', 'PROMPTCOPY_EXPERT = True'),
        ('_ew = torch.zeros(4)', '_ew = torch.zeros(6)'),
        (TRI_APPEND, LASTPROMPT),
    ]),
    ('lastcopy', '0.9.148.0', [
        (CACHE_DECL_OLD, CACHE_DECL_NEW),
        ('LASTCOPY_EXPERT = False', 'LASTCOPY_EXPERT = True'),
        (TRI_APPEND, LASTCOPY),
    ]),
    ('promptcopy', '0.9.149.0', [
        (CACHE_DECL_OLD, CACHE_DECL_NEW),
        ('PROMPTCOPY_EXPERT = False', 'PROMPTCOPY_EXPERT = True'),
        (TRI_APPEND, PROMPTCOPY),
    ]),
    ('top1mix', '0.9.151.0', [
        (MIX_DECL_OLD, MIX_DECL_NEW),
        ('MIX_OBJECTIVE = \'nll\'', 'MIX_OBJECTIVE = \'top1\''),
        (MIX_NLL_OLD, MIX_NLL_TOP1),
    ]),
    ('fourbackoff', '0.9.150.0', [
        (FOUR_DECL_ANCHOR, FOUR_DECL_NEW),
        ('FOURGRAM_MEMORY = False', 'FOURGRAM_MEMORY = True'),
        (FOUR_INIT_ANCHOR, FOUR_INIT_NEW),
        (TRI_APPEND, FOUR_READ),
        (FOUR_UPDATE_ANCHOR, FOUR_UPDATE),
    ]),
]

if __name__ == '__main__':
    for v in V:
        make(*v)
