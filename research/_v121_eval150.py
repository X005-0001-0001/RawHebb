# -*- coding: utf-8 -*-
"""v121 diagnostic: evaluate 500-sample checkpoints on the full 150-row val set;
sweep the aligncnt mixture weight w."""
import importlib.util as ilu
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(fname):
    spec = ilu.spec_from_file_location('m_' + fname.replace('+', '_').replace('.', '_'),
                                       os.path.join(HERE, fname))
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(m, ckpt):
    torch.set_num_threads(2)
    tok = m.RawHebbTokenizer(os.path.join(HERE, m.TOK_BIN), os.path.join(HERE, m.TOK_JSON))
    model = m.build_model(tok, torch.device('cpu'))
    sd = torch.load(os.path.join(HERE, ckpt), map_location='cpu', weights_only=False)
    cur = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    model.load_state_dict(sd, strict=False)
    rows = list(m.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))
    return m, tok, model, rows


for tag, fname, ckpt in [
    ('0.6.0.0 fresh', 'RawHebb+bio+0.6.0.0.py', '_v121_baseline.pt'),
    ('align', 'RawHebb+align+0.8.0.0.py', '_v121_align.pt'),
    ('aligncnt', 'RawHebb+aligncnt+0.8.0.0.py', '_v121_aligncnt.pt'),
]:
    m, tok, model, rows = build(load_module(fname), ckpt)
    v = m.evaluate(model, tok, rows)
    print('%-16s n=%d CE=%.4f top1=%.4f top5=%.4f' % (tag, v['n'], v['ce'], v['top1'], v['top5']))

print('--- aligncnt w sweep (full val) ---')
m, tok, model, rows = build(load_module('RawHebb+aligncnt+0.8.0.0.py'), '_v121_aligncnt.pt')
for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    model.w_pair.fill_(w)
    v = m.evaluate(model, tok, rows)
    print('w=%.2f CE=%.4f top1=%.4f top5=%.4f' % (w, v['ce'], v['top1'], v['top5']))
