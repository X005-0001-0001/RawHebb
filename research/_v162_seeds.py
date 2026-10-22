# -*- coding: utf-8 -*-
import io

for tag, src_ver, pairs in [
        ('tri1Mbuck64hidden432four4kbagnormprior05tri2int16train512', '0.9.259.0',
         [(44, '0.9.260.0'), (45, '0.9.261.0')]),
        ('tri1Mbuck64hidden432four4kbagnormprior05tri2int16train384', '0.9.258.0',
         [(44, '0.9.262.0'), (45, '0.9.263.0')])]:
    src = r'D:\explore\wav\RawHebb+%s+%s.py' % (tag, src_ver)
    for seed, ver in pairs:
        t = io.open(src, encoding='utf-8').read()
        t = t.replace('RawHebb+%s+%s' % (tag, src_ver),
                      'RawHebb+%sseed%d+%s' % (tag, seed, ver), 1)
        t = t.replace("APP_BRANCH = '%s'" % tag, "APP_BRANCH = '%sseed%d'" % (tag, seed), 1)
        t = t.replace("APP_VERSION = '%s'" % src_ver, "APP_VERSION = '%s'" % ver, 1)
        t = t.replace('SEED_EMBEDDING = 43', 'SEED_EMBEDDING = %d' % seed, 1)
        t = t.replace('SEED_FREEZE = 43', 'SEED_FREEZE = %d' % seed, 1)
        t = t.replace('SEED = 43', 'SEED = %d' % seed, 1)
        p = r'D:\explore\wav\RawHebb+%sseed%d+%s.py' % (tag, seed, ver)
        io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
        print('wrote', p)
