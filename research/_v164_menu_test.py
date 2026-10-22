# -*- coding: utf-8 -*-
"""Menu smoke test for 0.20; each session is a fresh process with piped stdin."""
import os
import subprocess
import sys

DIR = r'C:\Users\X005\AppData\Local\Temp\opencode\rawhebb_clean20'
APP = os.path.join(DIR, 'RawHebb+bio+0.20.0.0.py')
PY = sys.executable


def run(name, lines, timeout=1800):
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['CUDA_VISIBLE_DEVICES'] = ''
    data = '\n'.join(lines) + '\n'
    p = subprocess.run([PY, APP], input=data.encode('utf-8'), cwd=DIR, env=env,
                       capture_output=True, timeout=timeout)
    out = p.stdout.decode('utf-8', 'replace') + p.stderr.decode('utf-8', 'replace')
    path = os.path.join(DIR, '_menu_%s.log' % name)
    open(path, 'w', encoding='utf-8', newline='\n').write(out)
    bad = [ln for ln in out.splitlines()
           if 'Traceback' in ln or ln.startswith(('Error', 'NameError', 'RuntimeError', 'ValueError'))]
    print('=== %s exit=%s chars=%d bad=%s' % (name, p.returncode, len(out), bad[:3]))
    for tag in ('[CKPT]', '[EVAL]', '[STRESS]', '[SFT]', '[CONSIST]', '[ONLINE]', '[DATA]',
                '[TOK]', '[TRAIN]', '助手:', '已自动保存', '纯文本续写'):
        hits = [ln for ln in out.splitlines() if tag in ln]
        for ln in hits[-3:]:
            print('   ' + ln[:200])
    return p.returncode, out


A = ['4', '6', '3',
     '5', '', '', '', '', '8',
     '2', 'jsonl', '你好', 'quit',
     '11', '_t11.jsonl', '2', '',
     '9', '_t9.jsonl',
     '10', '_t10.jsonl', '_split.jsonl', '', '2',
     '12', '1', '3', '0',
     '2', 'txt', '你好']
run('A', A, timeout=1800)

B = ['7', '_t7.jsonl', '_sft.jsonl', '1',
     '8', '_t8.jsonl', '_consist.jsonl', '1',
     '13', '1', '_t13.jsonl', '1', 'n',
     '13', '2', '你好', 'quit',
     '0']
run('B', B, timeout=2400)

C = ['1', '_t1.jsonl', '1', '0', 'auto', 'y', '0']
run('C', C, timeout=1800)
