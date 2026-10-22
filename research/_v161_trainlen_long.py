# -*- coding: utf-8 -*-
import io

base = io.open(r'D:\explore\wav\RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16+0.9.254.0.py',
               encoding='utf-8').read()
for seq, ver in [(384, '0.9.258.0'), (512, '0.9.259.0')]:
    tag = 'tri1Mbuck64hidden432four4kbagnormprior05tri2int16train%d' % seq
    t = base.replace('SEQ_LEN = 128', 'SEQ_LEN = %d\nEVAL_LEN = 128' % seq, 1)
    t = t.replace('        ids, labels = tok.chat_to_ids(get_messages(rec))\n        m = (labels != -100)',
                  '        ids, labels = tok.chat_to_ids(get_messages(rec))\n'
                  '        if EVAL_LEN and len(ids) > EVAL_LEN:\n'
                  '            ids, labels = ids[-EVAL_LEN:], labels[-EVAL_LEN:]\n'
                  '        m = (labels != -100)', 1)
    t = t.replace('RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2int16+0.9.254.0',
                  'RawHebb+%s+%s' % (tag, ver), 1)
    t = t.replace("APP_BRANCH = 'tri1Mbuck64hidden432four4kbagnormprior05tri2int16'",
                  "APP_BRANCH = '%s'" % tag, 1)
    t = t.replace("APP_VERSION = '0.9.254.0'", "APP_VERSION = '%s'" % ver, 1)
    p = r'D:\explore\wav\RawHebb+%s+%s.py' % (tag, ver)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
    print('wrote', p)
