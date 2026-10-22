# -*- coding: utf-8 -*-
import importlib.util as ilu
import os

HERE = r'D:\explore\wav'


def load(name, fn):
    spec = ilu.spec_from_file_location(name, os.path.join(HERE, fn))
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m18 = load('m18', 'RawHebb+bio+0.18.0.0.py')
m19 = load('m19', 'RawHebb+shell19+0.9.264.0.py')
tok18 = m18.RawHebbTokenizer(os.path.join(HERE, m18.TOK_BIN), os.path.join(HERE, m18.TOK_JSON))
tok19 = m19.prepare_tokenizer(train_if_missing=False)
print('vocab', tok18.vocab_size, tok19.vocab_size)
names = {'bos': '<|bos|>', 'eos': '<|eos|>', 'pad': '<|pad|>', 'unk': '<|unk|>',
         'user': '<|user|>', 'assistant': '<|assistant|>', 'end': '<|end|>'}
for n, token in names.items():
    a = tok18._ids.get(token)
    b = tok19._ids.get(token)
    if a != b:
        print('SPECIAL MISMATCH', n, a, b)
samples = ['你好', 'RawHebb：纯 Hebbian，无 BP。', ' 空格\n换行\t制表 ＡＢＣ！', '']
bad = 0
for s in samples:
    if tok18.encode(s) != tok19.encode(s):
        bad += 1
        print('ENC MISMATCH', repr(s), tok18.encode(s)[:20], tok19.encode(s)[:20])
rows = list(m18.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))[:200]
for i, rec in enumerate(rows):
    msgs = m18.get_messages(rec)
    a = tok18.chat_to_ids(msgs)
    b = tok19.chat_to_ids(msgs)
    a = (a[0][-m19.EVAL_LEN:], a[1][-m19.EVAL_LEN:])
    b = (b[0][-m19.EVAL_LEN:], b[1][-m19.EVAL_LEN:])
    if not (a[0].equal(b[0]) and a[1].equal(b[1])):
        bad += 1
        print('CHAT MISMATCH row', i)
        break
print('identity mismatches:', bad)
print('OK' if bad == 0 else 'FAIL')
