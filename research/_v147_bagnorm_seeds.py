# -*- coding: utf-8 -*-
import io

src = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnorm+0.9.196.0.py'
for seed, ver in [(44, '0.9.203.0'), (45, '0.9.204.0')]:
    t = io.open(src, encoding='utf-8').read()
    t = t.replace('tri4buck64hidden416four4kbagnorm+0.9.196.0',
                  'tri4buck64hidden416four4kbagnormseed%d+%s' % (seed, ver), 1)
    t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4kbagnorm'",
                  "APP_BRANCH = 'tri4buck64hidden416four4kbagnormseed%d'" % seed, 1)
    t = t.replace("APP_VERSION = '0.9.196.0'", "APP_VERSION = '%s'" % ver, 1)
    t = t.replace('SEED_EMBEDDING = 43', 'SEED_EMBEDDING = %d' % seed, 1)
    t = t.replace('SEED_FREEZE = 43', 'SEED_FREEZE = %d' % seed, 1)
    t = t.replace('SEED = 43', 'SEED = %d' % seed, 1)
    dst = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnormseed%d+%s.py' % (seed, ver)
    io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
    print('wrote', dst)
