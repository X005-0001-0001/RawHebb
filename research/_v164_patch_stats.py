# -*- coding: utf-8 -*-
import io

p = r'D:\explore\wav\_v163_shell19.py'
t = io.open(p, encoding='utf-8').read()
old = ("    log_and_print('训练窗口 %d ｜ 评测窗口 %d ｜ 只学 assistant=%s'\n"
       "                  % (SEQ_LEN, EVAL_LEN, QA_MASK))")
new = ("    log_and_print('训练窗口 %d ｜ 评测窗口 %s ｜ 只学 assistant=%s'\n"
       "                  % (SEQ_LEN, '训练同窗' if not EVAL_LEN else EVAL_LEN, QA_MASK))")
if old not in t:
    raise SystemExit('missing')
t = t.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('ok')
