# -*- coding: utf-8 -*-
from _v121_gen import make

make('chunk', '0.9.48.0', [
    ('CHUNK_MODE = False', 'CHUNK_MODE = True'),
    ('CHUNK_EVAL = False', 'CHUNK_EVAL = True'),
    ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
])
make('chunk128', '0.9.49.0', [
    ('CHUNK_MODE = False', 'CHUNK_MODE = True'),
    ('CHUNK_EVAL = False', 'CHUNK_EVAL = True'),
    ('SEQ_LEN = 200', 'SEQ_LEN = 128'),
    ('TRIGRAM_MEMORY = False', 'TRIGRAM_MEMORY = True'),
])
make('chunkpair', '0.9.50.0', [
    ('CHUNK_MODE = False', 'CHUNK_MODE = True'),
    ('CHUNK_EVAL = False', 'CHUNK_EVAL = True'),
])
print('done')
