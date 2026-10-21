# -*- coding: utf-8 -*-
import io

src = r'D:\explore\wav\RawHebb+tri256kbuck64hidden432four4kbagnormprior05tri2+0.9.244.0.py'
for seed, ver in [(44, '0.9.249.0'), (45, '0.9.250.0')]:
    t = io.open(src, encoding='utf-8').read()
    t = t.replace('RawHebb+tri256kbuck64hidden432four4kbagnormprior05tri2+0.9.244.0',
                  'RawHebb+tri256kbuck64hidden432four4kbagnormprior05tri2seed%d+%s' % (seed, ver), 1)
    t = t.replace("APP_BRANCH = 'tri256kbuck64hidden432four4kbagnormprior05tri2'",
                  "APP_BRANCH = 'tri256kbuck64hidden432four4kbagnormprior05tri2seed%d'" % seed, 1)
    t = t.replace("APP_VERSION = '0.9.244.0'", "APP_VERSION = '%s'" % ver, 1)
    t = t.replace('SEED_EMBEDDING = 43', 'SEED_EMBEDDING = %d' % seed, 1)
    t = t.replace('SEED_FREEZE = 43', 'SEED_FREEZE = %d' % seed, 1)
    t = t.replace('SEED = 43', 'SEED = %d' % seed, 1)
    p = r'D:\explore\wav\RawHebb+tri256kbuck64hidden432four4kbagnormprior05tri2seed%d+%s.py' % (seed, ver)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
    print('wrote', p)
