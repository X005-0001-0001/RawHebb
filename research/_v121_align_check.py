# -*- coding: utf-8 -*-
"""v121 diagnostic: does 0.6.0.0 train/eval use the standard next-token shift?
Loads the fresh baseline checkpoint and scores the SAME logits under three shifts.
"""
import importlib.util as ilu
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = ilu.spec_from_file_location('rh', os.path.join(HERE, 'RawHebb+bio+0.6.0.0.py'))
rh = ilu.module_from_spec(spec)
spec.loader.exec_module(rh)
torch.set_num_threads(2)

tok = rh.RawHebbTokenizer(os.path.join(HERE, rh.TOK_BIN), os.path.join(HERE, rh.TOK_JSON))
m = rh.build_model(tok, torch.device('cpu'))
sd = torch.load(os.path.join(HERE, '_v121_baseline.pt'), map_location='cpu', weights_only=False)
cur = m.state_dict()
sd = {k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
m.load_state_dict(sd, strict=False)
m.eval()

rows = list(rh.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))[:40]
for tag, shift in [('current logits[t] vs labels[t+1]', 1),
                   ('aligned logits[t] vs labels[t]', 0),
                   ('logits[t+1] vs labels[t+1]', -1)]:
    ce = n = t1 = t5 = 0
    with torch.no_grad():
        for rec in rows:
            ids, lab = tok.chat_to_ids(rh.get_messages(rec))
            lg = m(ids.unsqueeze(0))[0]
            L = lab
            if shift == 1:
                a, b = lg[:-1], L[1:]
            elif shift == 0:
                a, b = lg[:-1], L[:-1]
            else:
                a, b = lg[1:], L[1:]
            mm = (b != -100)
            a = a[mm]
            b = b[mm]
            if b.numel() == 0:
                continue
            ce += float(F.cross_entropy(a, b, reduction='sum'))
            t1 += int((a.argmax(-1) == b).sum())
            t5 += int((a.topk(5, -1).indices == b.unsqueeze(-1)).any(-1).sum())
            n += int(b.numel())
    print('%-36s n=%d CE=%.4f top1=%.4f top5=%.4f' % (tag, n, ce / n, t1 / n, t5 / n))
