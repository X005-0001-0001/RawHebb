# -*- coding: utf-8 -*-
from _v121_gen import make

make('kntri', '0.9.45.0', [
    ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
    ("PAIR_SMOOTH = 'prior'", "PAIR_SMOOTH = 'kn'"),
])
make('knonly', '0.9.46.0', [
    ("PAIR_SMOOTH = 'prior'", "PAIR_SMOOTH = 'kn'"),
])
make('kntri64k', '0.9.47.0', [
    ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
    ('TRIGRAM_BUCKETS = 16384', 'TRIGRAM_BUCKETS = 65536'),
    ("PAIR_SMOOTH = 'prior'", "PAIR_SMOOTH = 'kn'"),
])
print('done')
