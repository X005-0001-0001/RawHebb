# -*- coding: utf-8 -*-
import importlib.util as ilu
import io
import os
import sys

import torch

HERE = r'D:\explore\wav'
BRANCH = sys.argv[1]
CKPT = sys.argv[2]
VAL_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 150
spec = ilu.spec_from_file_location('m', os.path.join(HERE, BRANCH))
m = ilu.module_from_spec(spec)
spec.loader.exec_module(m)

dev = m.setup_device()
tok = m.RawHebbTokenizer(os.path.join(HERE, m.TOK_BIN), os.path.join(HERE, m.TOK_JSON))
model = m.build_model(tok, dev)
sd = torch.load(os.path.join(HERE, CKPT), map_location='cpu', mmap=True, weights_only=False)
msd = model.state_dict()
filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
model.load_state_dict(filtered, strict=False)
model.to(dev)
rows = list(m.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))[:VAL_MAX]
res = m.evaluate(model, tok, rows)
print('EVAL %s -> CE=%.4f top1=%.4f top5=%.4f n=%d' % (CKPT, res['ce'], res['top1'], res['top5'], res['n']))
