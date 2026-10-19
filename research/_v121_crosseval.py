# -*- coding: utf-8 -*-
"""v121 diagnostic: same checkpoints under both eval modes (last-window vs chunked)."""
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


CASES = [
    ('tri', 'RawHebb+tri+0.9.25.0.py', '_v121_tri.pt'),
    ('chunk', 'RawHebb+chunk+0.9.48.0.py', '_v121_chunk.pt'),
    ('chunk128', 'RawHebb+chunk128+0.9.49.0.py', '_v121_chunk128.pt'),
    ('chunkpair', 'RawHebb+chunkpair+0.9.50.0.py', '_v121_chunkpair.pt'),
]
for tag, fname, ckpt in CASES:
    m, tok, model, rows = build(load_module(fname), ckpt)
    for mode in (False, True):
        m.CHUNK_EVAL = mode
        v = m.evaluate(model, tok, rows)
        print('%-10s CHUNK_EVAL=%-5s n=%d CE=%.4f top1=%.4f top5=%.4f'
              % (tag, mode, v['n'], v['ce'], v['top1'], v['top5']))
