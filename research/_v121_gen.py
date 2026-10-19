# -*- coding: utf-8 -*-
"""v121 branch generator: copy the lab file and patch config lines.
Every output is a standalone version file (kept forever, as requested).
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'RawHebb+lab+0.9.0.0.py')


def make(branch, version, repl):
    with io.open(BASE, encoding='utf-8') as f:
        t = f.read()
    t = t.replace(' RawHebb+lab+0.9.0.0  ——', ' RawHebb+%s+%s  ——' % (branch, version))
    t = re.sub(r'分支      : lab.*\n', '分支      : %-12s （v121 实验分支）\n' % branch, t, count=1)
    t = re.sub(r'版本      : 0\.9\.0\.0.*\n', '版本      : %-12s （v121 实验分支）\n' % version, t, count=1)
    t = t.replace("APP_BRANCH = 'lab'", "APP_BRANCH = '%s'" % branch)
    t = t.replace("APP_VERSION = '0.9.0.0'", "APP_VERSION = '%s'" % version)
    for old, new in repl:
        if old not in t:
            raise SystemExit('MISSING in base: %r' % old)
        t = t.replace(old, new, 1)
    out = os.path.join(HERE, 'RawHebb+%s+%s.py' % (branch, version))
    with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(t)
    print('wrote', os.path.basename(out))


VARIANTS = [
    ('hitmix', '0.9.1.0', [("PAIR_MIX = 'likelihood'", "PAIR_MIX = 'hit'")]),
    ('prior05', '0.9.2.0', [('PAIR_PRIOR_STRENGTH = 1.0', 'PAIR_PRIOR_STRENGTH = 0.5')]),
    ('prior2', '0.9.3.0', [('PAIR_PRIOR_STRENGTH = 1.0', 'PAIR_PRIOR_STRENGTH = 2.0')]),
    ('couple1', '0.9.4.0', [("COUPLING_MODE = 'half'", "COUPLING_MODE = 'one'")]),
    ('coupleinv', '0.9.5.0', [("COUPLING_MODE = 'half'", "COUPLING_MODE = 'inv_layers'")]),
    ('nonoise', '0.9.6.0', [('NOISE_ENABLED = True', 'NOISE_ENABLED = False')]),
    ('nospike', '0.9.7.0', [('SPIKE_ENABLED = True', 'SPIKE_ENABLED = False')]),
    ('bias', '0.9.8.0', [('HEAD_BIAS = False', 'HEAD_BIAS = True')]),
    ('newton2', '0.9.9.0', [('NEWTON_STEPS = 1', 'NEWTON_STEPS = 2')]),
    ('newton4', '0.9.10.0', [('NEWTON_STEPS = 1', 'NEWTON_STEPS = 4')]),
    ('cemix', '0.9.11.0', [("PAIR_MIX = 'likelihood'", "PAIR_MIX = 'ce_top1'")]),
    ('confmix', '0.9.12.0', [("PAIR_MIX = 'likelihood'", "PAIR_MIX = 'confidence'")]),
    ('allpair', '0.9.13.0', [("PAIR_SCOPE = 'mask'", "PAIR_SCOPE = 'all'")]),
    ('layers3', '0.9.14.0', [('NUM_LAYERS = 4', 'NUM_LAYERS = 3')]),
    ('layers5', '0.9.15.0', [('NUM_LAYERS = 4', 'NUM_LAYERS = 5')]),
    ('hidden64', '0.9.16.0', [('HIDDEN = 128', 'HIDDEN = 64')]),
    ('hidden192', '0.9.17.0', [('HIDDEN = 128', 'HIDDEN = 192')]),
    ('tau15', '0.9.18.0', [('TAU_MULT = 2.0', 'TAU_MULT = 1.5')]),
    ('tau3', '0.9.19.0', [('TAU_MULT = 2.0', 'TAU_MULT = 3.0')]),
    ('seq128', '0.9.20.0', [('SEQ_LEN = 200', 'SEQ_LEN = 128')]),
    ('seq384', '0.9.21.0', [('SEQ_LEN = 200', 'SEQ_LEN = 384')]),
    ('noshuffle', '0.9.22.0', [('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False')]),
    ('seed44', '0.9.23.0', [('SEED = 43', 'SEED = 44'), ('SEED_EMBEDDING = 43', 'SEED_EMBEDDING = 44'), ('SEED_FREEZE = 43', 'SEED_FREEZE = 44')]),
    ('seed45', '0.9.24.0', [('SEED = 43', 'SEED = 45'), ('SEED_EMBEDDING = 43', 'SEED_EMBEDDING = 45'), ('SEED_FREEZE = 43', 'SEED_FREEZE = 45')]),
]

if __name__ == '__main__':
    for b, v, r in VARIANTS:
        make(b, v, r)
