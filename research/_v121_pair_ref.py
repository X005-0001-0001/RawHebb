# -*- coding: utf-8 -*-
"""v121 diagnostic: token-pair (bigram) reference with the same assistant mask.
Computes CE / top-1 / top-5 for unigram and bigram (various smoothing) on val_dsr1.
Pure counting, no training, CPU only.
"""
import io
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util as ilu

spec = ilu.spec_from_file_location('rh', os.path.join(HERE, 'RawHebb+bio+0.6.0.0.py'))
rh = ilu.module_from_spec(spec)
spec.loader.exec_module(rh)

tok = rh.RawHebbTokenizer(os.path.join(HERE, rh.TOK_BIN), os.path.join(HERE, rh.TOK_JSON))
V = tok.vocab_size
train = list(rh.iter_jsonl(os.path.join(HERE, 'train_dsr1.jsonl')))
val = list(rh.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))

uni = torch.zeros(V, dtype=torch.float64)
bi = torch.zeros(V, V, dtype=torch.float64)
for rec in train:
    for m in rh.get_messages(rec):
        ids = tok.encode(m['content'])
        for i in range(len(ids) - 1):
            bi[ids[i], ids[i + 1]] += 1.0
            uni[ids[i + 1]] += 1.0
        if len(ids) == 1:
            uni[ids[0]] += 1.0

# targets in the same order as evaluate(): assistant content + final <|end|>.
# chat_to_ids labels already encode the exact target sequence and mask.
targets, prevs = [], []
for rec in val:
    ids, lab = tok.chat_to_ids(rh.get_messages(rec))
    for t in range(len(lab)):
        if int(lab[t]) != -100:
            targets.append(int(lab[t]))
            prevs.append(int(ids[t]))
targets = torch.tensor(targets)
prevs = torch.tensor(prevs)
n = targets.numel()
print('positions', n)

def report(name, logits):
    ce = float(torch.nn.functional.cross_entropy(logits, targets))
    t1 = float((logits.argmax(-1) == targets).float().mean())
    k5 = logits.topk(5, -1).indices
    t5 = float((k5 == targets.unsqueeze(-1)).any(-1).float().mean())
    print('%-28s CE=%.4f top1=%.4f top5=%.4f' % (name, ce, t1, t5))

# unigram with add-1 over the *target* token stream (same mask)
u = torch.zeros(V, dtype=torch.float64)
for rec in train:
    ids, lab = tok.chat_to_ids(rh.get_messages(rec))
    for t in range(len(lab)):
        if int(lab[t]) != -100:
            u[int(lab[t])] += 1.0
p_uni = (u + 1.0) / (u.sum() + V)
report('unigram add-1 (mask)', torch.log(p_uni).unsqueeze(0).expand(n, V))

# bigram smoothing variants
row = bi.sum(1)
for name, prior, strength in [
    ('bigram add-0.5', None, 0.5),
    ('bigram add-1', None, 1.0),
    ('bigram unigram-prior x1', p_uni, 1.0),
]:
    if prior is None:
        p = (bi + strength) / (row + strength * V).unsqueeze(1)
    else:
        p = (bi + strength * prior.unsqueeze(0)) / (row + strength).unsqueeze(1)
    report(name, torch.log(p[prevs]))

# oracle best convex mixture in log space, base = unigram, pair = bigram add-0.5
pair = torch.log((bi + 0.5) / (row + 0.5 * V).unsqueeze(1))[prevs]
base = torch.log(p_uni).unsqueeze(0).expand(n, V)
best = None
for i in range(101):
    w = i / 100.0
    lg = (1 - w) * base + w * pair
    ce = float(torch.nn.functional.cross_entropy(lg, targets))
    if best is None or ce < best[1]:
        best = (w, ce)
print('best unigram/bigram log-mixture w=%.2f CE=%.4f' % best)
