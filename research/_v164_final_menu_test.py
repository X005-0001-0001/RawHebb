# -*- coding: utf-8 -*-
import os
import subprocess
import sys

DIR = r'C:\Users\X005\AppData\Local\Temp\opencode\rawhebb_clean20'
APP = os.path.join(DIR, 'RawHebb+bio+0.20.0.0.py')
env = dict(os.environ)
env['PYTHONIOENCODING'] = 'utf-8'
env['CUDA_VISIBLE_DEVICES'] = ''


def run(name, lines, timeout=2400):
    p = subprocess.run([sys.executable, APP], input=('\n'.join(lines) + '\n').encode('utf-8'),
                       cwd=DIR, env=env, capture_output=True, timeout=timeout)
    out = p.stdout.decode('utf-8', 'replace') + p.stderr.decode('utf-8', 'replace')
    open(os.path.join(DIR, '_final_%s.log' % name), 'w', encoding='utf-8',
         newline='\n').write(out)
    print('=== %s exit=%s' % (name, p.returncode))
    for ln in out.splitlines():
        if any(k in ln for k in ('[CKPT]', '[TRAIN]', '[EVAL]', '[STRESS]', '[DATA]',
                                 '[TOK]', '[SFT]', '[CONSIST]', '[ONLINE]',
                                 'Traceback', 'Error', '助手:', '纯文本续写')):
            print('   ' + ln[:220])
    return p.returncode


A = ['4', '6', '3',
     '5', '', '', '', '', '8',
     '2', 'jsonl', '你好', 'quit',
     '11', '_t11.jsonl', '2', '',
     '9', '_t9.jsonl',
     '10', '_t10.jsonl', '_split2.jsonl', '', '2',
     '12', '1', '3', '0',
     '2', 'txt', '你好']
run('A', A, timeout=1800)

B = ['7', '_t7.jsonl', '_sft2.jsonl', '1',
     '8', '_t8.jsonl', '_consist2.jsonl', '1',
     '13', '1', '_t13.jsonl', '1', 'n',
     '13', '2', '你好', 'quit',
     '0']
run('B', B, timeout=2400)

C = ['1', '4', '1', '0', 'auto', 'y', '0']
run('C', C, timeout=1800)
