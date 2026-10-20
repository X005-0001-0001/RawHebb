# -*- coding: utf-8 -*-
"""v123 branch generator: broad matrix around 0.11.0.0.

Every branch is a standalone version file. Base is RawHebb+bio+0.11.0.0.py.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'RawHebb+bio+0.11.0.0.py')

ORDER_DECL_OLD = 'TRAIN_SHUFFLE = False'
ORDER_DECL_NEW = "TRAIN_SHUFFLE = False\nTRAIN_ORDER = 'file'"

ORDER_BLOCK_OLD = """    rows = list(iter_jsonl(os.path.join(DATA_DIR, path)))
    if TRAIN_SHUFFLE:
        random.Random(SEED).shuffle(rows)
    if limit:
        rows = rows[:limit]
"""
ORDER_BLOCK_NEW = """    rows = list(iter_jsonl(os.path.join(DATA_DIR, path)))
    if TRAIN_SHUFFLE:
        random.Random(SEED).shuffle(rows)
    if limit:
        rows = rows[:limit]
    if TRAIN_ORDER == 'reverse':
        rows = rows[::-1]
    elif TRAIN_ORDER in ('length_asc', 'length_desc'):
        rows.sort(key=lambda r: len(tok.chat_to_ids(get_messages(r))[0]),
                  reverse=(TRAIN_ORDER == 'length_desc'))
"""

TRI_DECL_OLD = 'PAIR_PRIOR_STRENGTH = 1.0'
TRI_DECL_NEW = 'PAIR_PRIOR_STRENGTH = 1.0\nTRI_PRIOR_STRENGTH = 1.0'

TRI_LINE_OLD = ('                _tp = (_trow + PAIR_PRIOR_STRENGTH * _prior) / '
                '(_trs + PAIR_PRIOR_STRENGTH)')
TRI_LINE_NEW = ('                _tp = (_trow + TRI_PRIOR_STRENGTH * _prior) / '
                '(_trs + TRI_PRIOR_STRENGTH)')


def make(branch, version, repls, add_order=False, add_tri=False):
    with io.open(BASE, encoding='utf-8') as f:
        t = f.read()
    t = t.replace('RawHebb+bio+0.11.0.0', 'RawHebb+%s+%s' % (branch, version), 1)
    t = t.replace('分支      : bio', '分支      : %s' % branch, 1)
    t = t.replace('版本      : 0.11.0.0', '版本      : %s' % version, 1)
    t = t.replace("APP_BRANCH = 'bio'", "APP_BRANCH = '%s'" % branch, 1)
    t = t.replace("APP_VERSION = '0.11.0.0'", "APP_VERSION = '%s'" % version, 1)
    if add_order:
        for old, new in [(ORDER_DECL_OLD, ORDER_DECL_NEW), (ORDER_BLOCK_OLD, ORDER_BLOCK_NEW)]:
            if old not in t:
                raise SystemExit('MISSING order anchor: %r' % old[:60])
            t = t.replace(old, new, 1)
    if add_tri:
        for old, new in [(TRI_DECL_OLD, TRI_DECL_NEW), (TRI_LINE_OLD, TRI_LINE_NEW)]:
            if old not in t:
                raise SystemExit('MISSING tri anchor: %r' % old[:60])
            t = t.replace(old, new, 1)
    for old, new in repls:
        if old not in t:
            raise SystemExit('MISSING in base: %r' % old[:80])
        t = t.replace(old, new, 1)
    out = os.path.join(HERE, 'RawHebb+%s+%s.py' % (branch, version))
    with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(t)
    print('wrote', os.path.basename(out))


VARIANTS = [
    # training order
    ('orderrev', '0.9.71.0', [("TRAIN_ORDER = 'file'", "TRAIN_ORDER = 'reverse'")], True, False),
    ('orderlenasc', '0.9.72.0', [("TRAIN_ORDER = 'file'", "TRAIN_ORDER = 'length_asc'")], True, False),
    ('orderlendesc', '0.9.73.0', [("TRAIN_ORDER = 'file'", "TRAIN_ORDER = 'length_desc'")], True, False),
    # pair prior strength (also changes the shared unigram prior)
    ('pairprior05', '0.9.74.0', [('PAIR_PRIOR_STRENGTH = 1.0', 'PAIR_PRIOR_STRENGTH = 0.5')], False, False),
    ('pairprior2', '0.9.75.0', [('PAIR_PRIOR_STRENGTH = 1.0', 'PAIR_PRIOR_STRENGTH = 2.0')], False, False),
    ('pairprior4', '0.9.76.0', [('PAIR_PRIOR_STRENGTH = 1.0', 'PAIR_PRIOR_STRENGTH = 4.0')], False, False),
    # separate tri prior
    ('triprior05', '0.9.77.0', [('TRI_PRIOR_STRENGTH = 1.0', 'TRI_PRIOR_STRENGTH = 0.5')], False, True),
    ('triprior2', '0.9.78.0', [('TRI_PRIOR_STRENGTH = 1.0', 'TRI_PRIOR_STRENGTH = 2.0')], False, True),
    ('triprior4', '0.9.79.0', [('TRI_PRIOR_STRENGTH = 1.0', 'TRI_PRIOR_STRENGTH = 4.0')], False, True),
    # hash width
    ('tribuck8k', '0.9.80.0', [('TRIGRAM_BUCKETS = 16384', 'TRIGRAM_BUCKETS = 8192')], False, False),
    ('tribuck32k', '0.9.81.0', [('TRIGRAM_BUCKETS = 16384', 'TRIGRAM_BUCKETS = 32768')], False, False),
    ('tribuck64k', '0.9.82.0', [('TRIGRAM_BUCKETS = 16384', 'TRIGRAM_BUCKETS = 65536')], False, False),
    # window
    ('seq64tri', '0.9.83.0', [('SEQ_LEN = 128', 'SEQ_LEN = 64')], False, False),
    ('seq96tri', '0.9.84.0', [('SEQ_LEN = 128', 'SEQ_LEN = 96')], False, False),
    ('seq160tri', '0.9.85.0', [('SEQ_LEN = 128', 'SEQ_LEN = 160')], False, False),
    ('seq192tri', '0.9.86.0', [('SEQ_LEN = 128', 'SEQ_LEN = 192')], False, False),
    ('seq256tri', '0.9.87.0', [('SEQ_LEN = 128', 'SEQ_LEN = 256')], False, False),
    # base capacity / dynamics
    ('layers3', '0.9.88.0', [('NUM_LAYERS = 4', 'NUM_LAYERS = 3')], False, False),
    ('layers5', '0.9.89.0', [('NUM_LAYERS = 4', 'NUM_LAYERS = 5')], False, False),
    ('hidden64', '0.9.90.0', [('HIDDEN = 128', 'HIDDEN = 64')], False, False),
    ('hidden192', '0.9.91.0', [('HIDDEN = 128', 'HIDDEN = 192')], False, False),
    ('newton2', '0.9.92.0', [('NEWTON_STEPS = 1', 'NEWTON_STEPS = 2')], False, False),
    ('bias', '0.9.93.0', [('HEAD_BIAS = False', 'HEAD_BIAS = True')], False, False),
    ('nonoise', '0.9.94.0', [('NOISE_ENABLED = True', 'NOISE_ENABLED = False')], False, False),
    ('nospike', '0.9.95.0', [('SPIKE_ENABLED = True', 'SPIKE_ENABLED = False')], False, False),
    ('couple1', '0.9.96.0', [("COUPLING_MODE = 'half'", "COUPLING_MODE = 'one'")], False, False),
    ('coupleinv', '0.9.97.0', [("COUPLING_MODE = 'half'", "COUPLING_MODE = 'inv_layers'")], False, False),
    # mixture budget / scope
    ('mixround1', '0.9.98.0', [('MIX_ROUNDS = 2', 'MIX_ROUNDS = 1')], False, False),
    ('mixround4', '0.9.99.0', [('MIX_ROUNDS = 2', 'MIX_ROUNDS = 4')], False, False),
    ('pairscopeall', '0.9.100.0', [("PAIR_SCOPE = 'mask'", "PAIR_SCOPE = 'all'")], False, False),
]

if __name__ == '__main__':
    for v in VARIANTS:
        make(*v)
