# -*- coding: utf-8 -*-
"""v121 diagnostic: exact mask-matched token-pair memory ceiling.

Counts only positions where labels != -100, exactly like the cnt branch learns:
  pair[x,y] += 1, unigram[y] += 1 for every valid target position.
Then evaluates p(y|x) = (pair[x,y] + prior[y]) / (row[x] + 1), prior = add-one unigram.
"""
import os

import torch
import torch.nn.functional as F
import importlib.util as ilu

HERE = os.path.dirname(os.path.abspath(__file__))
spec = ilu.spec_from_file_location('rh', os.path.join(HERE, 'RawHebb+bio+0.6.0.0.py'))
rh = ilu.module_from_spec(spec)
spec.loader.exec_module(rh)

tok = rh.RawHebbTokenizer(os.path.join(HERE, rh.TOK_BIN), os.path.join(HERE, rh.TOK_JSON))
V = tok.vocab_size

pair = torch.zeros(V, V, dtype=torch.float64)
uni = torch.zeros(V, dtype=torch.float64)
ntr = 0
for rec in rh.iter_jsonl(os.path.join(HERE, 'train_dsr1.jsonl')):
    ids, lab = tok.chat_to_ids(rh.get_messages(rec))
    m = lab != -100
    if not bool(m.any()):
        continue
    xs = ids[:-1][m[1:]]
    ys = lab[1:][m[1:]]
    pair.index_put_((xs, ys), torch.ones(xs.numel(), dtype=pair.dtype), accumulate=True)
    uni.index_add_(0, ys, torch.ones(ys.numel(), dtype=uni.dtype))
    ntr += int(m.sum())
print('train valid targets', ntr, 'pair mass', float(pair.sum()))

tgt, prev = [], []
for rec in rh.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')):
    ids, lab = tok.chat_to_ids(rh.get_messages(rec))
    m = lab != -100
    if not bool(m.any()):
        continue
    tgt.append(lab[1:][m[1:]])
    prev.append(ids[:-1][m[1:]])
tgt = torch.cat(tgt)
prev = torch.cat(prev)
n = tgt.numel()
print('val positions', n)

row = pair.sum(1)
prior = (uni + 1.0) / (uni.sum() + V)
for name, p in [
    ('pair add-1', (pair + 1.0) / (row + 1.0 * V).unsqueeze(1)),
    ('pair unigram-prior x1', (pair + prior.unsqueeze(0)) / (row + 1.0).unsqueeze(1)),
    ('pair unigram-prior x0.5', (pair + 0.5 * prior.unsqueeze(0)) / (row + 0.5).unsqueeze(1)),
]:
    lg = torch.log(p[prev])
    ce = float(F.cross_entropy(lg, tgt))
    t1 = float((lg.argmax(-1) == tgt).float().mean())
    t5 = float((lg.topk(5, -1).indices == tgt.unsqueeze(-1)).any(-1).float().mean())
    print('%-26s CE=%.4f top1=%.4f top5=%.4f' % (name, ce, t1, t5))

u = torch.log(prior).unsqueeze(0).expand(n, V)
lg = torch.log((pair + prior.unsqueeze(0)) / (row + 1.0).unsqueeze(1))[prev]
best = None
for i in range(101):
    w = i / 100.0
    mix = (1 - w) * u + w * lg
    ce = float(F.cross_entropy(mix, tgt))
    if best is None or ce < best[1]:
        best = (w, ce)
print('best unigram/pair-prior log-mixture w=%.2f CE=%.4f' % best)
