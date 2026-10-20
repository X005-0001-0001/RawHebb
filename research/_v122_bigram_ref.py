# -*- coding: utf-8 -*-
"""Corrected, window-matched n-gram reference for 0.11.0.0.

The older v121 reference scripts paired ids[:-1] with labels[1:], which is the
old two-step-ahead bug. Here labels[t] is token t+1, so a standard bigram is
p(labels[t] | ids[t]) at every masked position t.
"""
import importlib.util as ilu
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = ilu.spec_from_file_location('rh', os.path.join(HERE, 'RawHebb+bio+0.11.0.0.py'))
rh = ilu.module_from_spec(spec)
spec.loader.exec_module(rh)
tok = rh.RawHebbTokenizer(os.path.join(HERE, rh.TOK_BIN), os.path.join(HERE, rh.TOK_JSON))
V = tok.vocab_size
T = rh.SEQ_LEN

pair = torch.zeros(V, V, dtype=torch.float64)
uni = torch.zeros(V, dtype=torch.float64)
ntr = 0
for rec in rh.iter_jsonl(os.path.join(HERE, 'train_dsr1.jsonl')):
    ids, lab = tok.chat_to_ids(rh.get_messages(rec), max_len=T)
    m = lab != -100
    if not bool(m.any()):
        continue
    xs = ids[:-1][m[:-1]]
    ys = lab[:-1][m[:-1]]
    pair.index_put_((xs, ys), torch.ones(xs.numel(), dtype=pair.dtype), accumulate=True)
    uni.index_add_(0, ys, torch.ones(ys.numel(), dtype=uni.dtype))
    ntr += int(m.sum())
print('train targets', ntr, 'pair mass', float(pair.sum()), 'window', T)

tgt, prev = [], []
for rec in rh.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')):
    ids, lab = tok.chat_to_ids(rh.get_messages(rec), max_len=T)
    m = lab != -100
    if not bool(m.any()):
        continue
    tgt.append(lab[:-1][m[:-1]])
    prev.append(ids[:-1][m[:-1]])
tgt = torch.cat(tgt)
prev = torch.cat(prev)
n = tgt.numel()
print('val targets', n)

prior = (uni + 1.0) / (uni.sum() + V)
row = pair.sum(1)
for name, p in [
    ('unigram add-1', prior.unsqueeze(0).expand(n, V)),
    ('bigram add-1', (pair + 1.0) / (row + 1.0 * V).unsqueeze(1)),
    ('bigram unigram-prior x1', (pair + prior.unsqueeze(0)) / (row + 1.0).unsqueeze(1)),
]:
    lg = torch.log(p[prev]) if p.shape[0] == V else torch.log(p)
    ce = float(F.cross_entropy(lg, tgt))
    t1 = float((lg.argmax(-1) == tgt).float().mean())
    t5 = float((lg.topk(5, -1).indices == tgt.unsqueeze(-1)).any(-1).float().mean())
    print('%-28s CE=%.4f top1=%.4f top5=%.4f' % (name, ce, t1, t5))

# The model's pair expert uses exactly these counts and prior.
lg = torch.log((pair + prior.unsqueeze(0)) / (row + 1.0).unsqueeze(1))[prev]
u = torch.log(prior).unsqueeze(0).expand(n, V)
best = None
for i in range(101):
    w = i / 100.0
    ce = float(F.cross_entropy((1 - w) * u + w * lg, tgt))
    if best is None or ce < best[1]:
        best = (w, ce)
print('pair/unigram log-mixture best w=%.2f CE=%.4f' % best)
