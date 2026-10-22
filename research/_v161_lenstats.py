# -*- coding: utf-8 -*-
import importlib.util as ilu
import os

HERE = r'D:\explore\wav'
spec = ilu.spec_from_file_location(
    'm', os.path.join(HERE, 'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16train512+0.9.259.0.py'))
m = ilu.module_from_spec(spec)
spec.loader.exec_module(m)
tok = m.RawHebbTokenizer(os.path.join(HERE, m.TOK_BIN), os.path.join(HERE, m.TOK_JSON))
lens = []
for rec in m.iter_jsonl(os.path.join(HERE, 'train_dsr1.jsonl')):
    ids, labels = tok.chat_to_ids(m.get_messages(rec))
    lens.append(len(ids))
lens.sort()
n = len(lens)


def pct(p):
    return lens[min(n - 1, int(p * n))]


print('n=%d min=%d mean=%.1f p50=%d p75=%d p90=%d p95=%d p99=%d max=%d' % (
    n, lens[0], sum(lens) / n, pct(0.5), pct(0.75), pct(0.9), pct(0.95), pct(0.99), lens[-1]))
for t in (128, 192, 256, 384, 512):
    print('<=%d: %.1f%%' % (t, 100.0 * sum(1 for x in lens if x <= t) / n))
