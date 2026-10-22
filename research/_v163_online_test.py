# -*- coding: utf-8 -*-
import importlib.util as ilu
import os
import sys

import torch

HERE = r'D:\explore\wav'
spec = ilu.spec_from_file_location('m', os.path.join(HERE, 'RawHebb+shell19+0.9.264.0.py'))
m = ilu.module_from_spec(spec)
spec.loader.exec_module(m)

mode = sys.argv[1] if len(sys.argv) > 1 else 'online'


def load_tok(b=None, j=None):
    b = b or os.path.join(HERE, m.TOK_BIN)
    j = j or os.path.join(HERE, m.TOK_JSON)
    t = m.RawHebbTokenizer(b, j)
    assert t.load(), 'tokenizer load failed'
    return t


if mode == 'tokenizer':
    t = load_tok(os.path.join(HERE, '_v163_tok.bin'), os.path.join(HERE, '_v163_tok.json'))
    print('custom tokenizer:', t.describe())
    s = 'RawHebb 词表重建测试：空格 换行 ＡＢＣ。'
    ids = t.encode(s)
    print('vocab', t.vocab_size, 'ids', ids[:12], 'decode_ok', bool(t.decode(ids)))
else:
    torch.set_num_threads(4)
    dev = torch.device('cpu')
    tok = load_tok()
    model = m.build_model(tok, dev)
    if mode == 'loaded':
        assert m.load_model(model, '_v162_full_train384.pt'), 'ckpt load failed'
    before = float(model.W_out.detach().norm())
    n = m.online_learn_file(model, tok, 'val_dsr1.jsonl', limit=2)
    after = float(model.W_out.detach().norm())
    print('online updates', n, 'W_out %.3f -> %.3f' % (before, after),
          'changed', abs(after - before) > 1e-6)
