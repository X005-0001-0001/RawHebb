# -*- coding: utf-8 -*-
import importlib.util as ilu
import os

HERE = r'D:\explore\wav'
spec = ilu.spec_from_file_location('m', os.path.join(HERE, 'RawHebb+bio+0.20.0.0.py'))
m = ilu.module_from_spec(spec)
spec.loader.exec_module(m)

tok = m.prepare_tokenizer(train_if_missing=False)
short = [
    {'role': 'system', 'content': '你是助手。'},
    {'role': 'user', 'content': '你好'},
    {'role': 'assistant', 'content': '你好！', 'reasoning_content': '想一下'},
]
ids, labels = tok.chat_to_ids(short, max_len=384)
print('short T', len(ids), 'labels', labels.tolist())
print('decoded trained:', repr(tok.decode([int(x) for x in labels[labels != -100]])))
assert labels[0] == -100 and labels[-1] == tok.eos_token_id

long = [{'role': 'system', 'content': '系统提示'}]
for i in range(20):
    long.append({'role': 'user', 'content': '问题 %d' % i})
    long.append({'role': 'assistant', 'content': '回答 %d' % i})
ids2, lab2 = tok.chat_to_ids(long, max_len=120)
txt = tok.decode([int(x) for x in ids2])
print('long T', len(ids2), 'has_system', '系统提示' in txt,
      'has_last', '回答 19' in txt, 'has_first', '回答 0' in txt)
print('tail:', repr(tok.decode([int(x) for x in ids2[-40:]])))
assert '系统提示' in txt and '回答 19' in txt

ids3, lab3 = tok.chat_to_ids(short, max_len=384, tools=[{'name': 'calc'}])
print('tools T', len(ids3), 'has_tool', '工具' in tok.decode([int(x) for x in ids3]))
print('OK')
