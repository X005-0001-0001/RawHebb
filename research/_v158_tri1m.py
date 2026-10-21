# -*- coding: utf-8 -*-
import io

src = r'D:\explore\wav\RawHebb+tri512kbuck64hidden432four4kbagnormprior05tri2+0.9.248.0.py'
t = io.open(src, encoding='utf-8').read()
t = t.replace('TRIGRAM_BUCKETS = 524288', 'TRIGRAM_BUCKETS = 1048576', 1)
t = t.replace('RawHebb+tri512kbuck64hidden432four4kbagnormprior05tri2+0.9.248.0',
              'RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0', 1)
t = t.replace("APP_BRANCH = 'tri512kbuck64hidden432four4kbagnormprior05tri2'",
              "APP_BRANCH = 'tri1Mbuck64hidden432four4kbagnormprior05tri2'", 1)
t = t.replace("APP_VERSION = '0.9.248.0'", "APP_VERSION = '0.9.251.0'", 1)
p = r'D:\explore\wav\RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0.py'
io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', p)
