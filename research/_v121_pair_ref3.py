# -*- coding: utf-8 -*-
"""v121 diagnostic: pair-memory ceilings on the SAME 500-row training subset.

Two counting policies:
  mask : only positions whose label != -100 (what cnt currently does)
  all  : every adjacent token pair in the rendered dialogue (prompt + answer)
Evaluation stays strictly on assistant labels, exactly like evaluate().
"""
import os
import random

import torch
import torch.nn.functional as F
import importlib.util as ilu

HERE = os.path.dirname(os.path.abspath(__file__))
spec = ilu.spec_from_file_location('rh', os.path.join(HERE, 'RawHebb+bio+0.6.0.0.py'))
rh = ilu.module_from_spec(spec)
spec.loader.exec_module(rh)

tok = rh.RawHebbTokenizer(os.path.join(HERE, rh.TOK_BIN), os.path.join(HERE, rh.TOK_JSON))
V = tok.vocab_size

rows = list(rh.iter_jsonl(os.path.join(HERE, 'train_dsr1.jsonl')))
random.Random(rh.SEED).shuffle(rows)
rows = rows[:500]

pair_m = torch.zeros(V, V, dtype=torch.float64)
uni_m = torch.zeros(V, dtype=torch.float64)
pair_a = torch.zeros(V, V, dtype=torch.float64)
uni_a = torch.zeros(V, dtype=torch.float64)
ntr = 0
for rec in rows:
    ids, lab = tok.chat_to_ids(rh.get_messages(rec))
    m = lab != -100
    if bool(m.any()):
        xs = ids[:-1][m[1:]]
        ys = lab[1:][m[1:]]
        pair_m.index_put_((xs, ys), torch.ones(xs.numel(), dtype=pair_m.dtype), accumulate=True)
        uni_m.index_add_(0, ys, torch.ones(ys.numel(), dtype=uni_m.dtype))
        ntr += int(m.sum())
    xs = ids[:-1]
    ys = ids[1:]
    pair_a.index_put_((xs, ys), torch.ones(xs.numel(), dtype=pair_a.dtype), accumulate=True)
    uni_a.index_add_(0, ys, torch.ones(ys.numel(), dtype=uni_a.dtype))

print('500-row subset: mask targets', ntr, '| all transitions', int(pair_a.sum()))

tgt, prev = [], []
for rec in rh.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')):
    ids, lab = tok.chat_to_ids(rh.get_messages(rec))
    m = lab != -100
    if bool(m.any()):
        tgt.append(lab[1:][m[1:]])
        prev.append(ids[:-1][m[1:]])
tgt = torch.cat(tgt)
prev = torch.cat(prev)
n = tgt.numel()
print('val positions', n)


def report(name, pair, uni):
    row = pair.sum(1)
    prior = (uni + 1.0) / (uni.sum() + V)
    p = (pair + prior.unsqueeze(0)) / (row + 1.0).unsqueeze(1)
    lg = torch.log(p[prev])
    ce = float(F.cross_entropy(lg, tgt))
    t1 = float((lg.argmax(-1) == tgt).float().mean())
    t5 = float((lg.topk(5, -1).indices == tgt.unsqueeze(-1)).any(-1).float().mean())
    print('%-18s CE=%.4f top1=%.4f top5=%.4f' % (name, ce, t1, t5))
    u = torch.log(prior).unsqueeze(0).expand(n, V)
    best = None
    for i in range(101):
        w = i / 100.0
        mix = (1 - w) * u + w * lg
        ce2 = float(F.cross_entropy(mix, tgt))
        if best is None or ce2 < best[1]:
            best = (w, ce2)
    print('  unigram/pair best w=%.2f CE=%.4f' % best)


report('pair mask-500', pair_m, uni_m)
report('pair all-500', pair_a, uni_a)
