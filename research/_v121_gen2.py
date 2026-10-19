# -*- coding: utf-8 -*-
"""v121 branch generator, round 2: trigram / unigram experts and their combinations."""
from _v121_gen import make

VARIANTS2 = [
    ('tri', '0.9.25.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True')]),
    ('trionly', '0.9.26.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
                             ('PAIR_MEMORY = True', 'PAIR_MEMORY = False')]),
    ('triuni', '0.9.27.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
                            ('UNIGRAM_EXPERT = False', 'UNIGRAM_EXPERT = True')]),
    ('unipair', '0.9.28.0', [('UNIGRAM_EXPERT = False', 'UNIGRAM_EXPERT = True')]),
    ('tri64k', '0.9.29.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
                            ('TRIGRAM_BUCKETS = 16384', 'TRIGRAM_BUCKETS = 65536')]),
    ('tri4096', '0.9.30.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
                             ('TRIGRAM_BUCKETS = 16384', 'TRIGRAM_BUCKETS = 4096')]),
    ('seq96', '0.9.31.0', [('SEQ_LEN = 200', 'SEQ_LEN = 96')]),
    ('seq160', '0.9.32.0', [('SEQ_LEN = 200', 'SEQ_LEN = 160')]),
    ('seq128tri', '0.9.33.0', [('SEQ_LEN = 200', 'SEQ_LEN = 128'),
                               ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True')]),
    ('prior2tri', '0.9.34.0', [('PAIR_PRIOR_STRENGTH = 1.0', 'PAIR_PRIOR_STRENGTH = 2.0'),
                               ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True')]),
    ('layers5tri', '0.9.35.0', [('NUM_LAYERS = 4', 'NUM_LAYERS = 5'),
                                ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True')]),
    ('noshuffletri', '0.9.36.0', [('TRAIN_SHUFFLE = True', 'TRAIN_SHUFFLE = False'),
                                  ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True')]),
    ('mixround1', '0.9.37.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
                               ('MIX_ROUNDS = 2', 'MIX_ROUNDS = 1')]),
    ('mixround4', '0.9.38.0', [('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
                               ('MIX_ROUNDS = 2', 'MIX_ROUNDS = 4')]),
    ('biastri', '0.9.39.0', [('HEAD_BIAS = False', 'HEAD_BIAS = True'),
                             ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True')]),
]

if __name__ == '__main__':
    for b, v, r in VARIANTS2:
        make(b, v, r)
