# -*- coding: utf-8 -*-
"""v121 diagnostic: sweep the two readout scalars on saved checkpoints.
  A) cnt branch: geometric-mixture weight w (0 = base+state only, 1 = pair only)
  B) bio/lik branch: memory volume beta
CPU only, no training.
"""
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
    rows = list(m.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))[:40]
    return m, tok, model, rows


print('===== A) cnt: mixture weight w sweep =====')
m, tok, model, rows = build(load_module('RawHebb+cnt+0.8.0.0.py'), '_v121_cnt.pt')
for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    model.w_pair.fill_(w)
    v = m.evaluate(model, tok, rows)
    print('w=%.2f CE=%.4f top1=%.4f top5=%.4f' % (w, v['ce'], v['top1'], v['top5']))

print('===== B) bio: beta sweep =====')
mb, tokb, modelb, rowsb = build(load_module('RawHebb+bio+0.8.0.0.py'), '_v121_bio.pt')
for b in (0.0, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
    modelb.beta.fill_(b)
    modelb.beta_ready.fill_(1)
    v = mb.evaluate(modelb, tokb, rowsb)
    print('beta=%.2f CE=%.4f top1=%.4f top5=%.4f' % (b, v['ce'], v['top1'], v['top5']))

print('===== C) lik checkpoint at beta=1 =====')
ml, tokl, modell, rowsl = build(load_module('RawHebb+lik+0.8.0.0.py'), '_v121_lik.pt')
modell.beta.fill_(1.0)
modell.beta_ready.fill_(1)
v = ml.evaluate(modell, tokl, rowsl)
print('lik@beta=1 CE=%.4f top1=%.4f top5=%.4f' % (v['ce'], v['top1'], v['top5']))
