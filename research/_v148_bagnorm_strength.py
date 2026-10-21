# -*- coding: utf-8 -*-
import io

src = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnorm+0.9.196.0.py'
for val, ver in [('2.0', '0.9.205.0'), ('0.5', '0.9.206.0')]:
    t = io.open(src, encoding='utf-8').read()
    t = t.replace('tri4buck64hidden416four4kbagnorm+0.9.196.0',
                  'tri4buck64hidden416four4kbagnormprior%s+%s' % (val.replace('.', ''), ver), 1)
    t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4kbagnorm'",
                  "APP_BRANCH = 'tri4buck64hidden416four4kbagnormprior%s'"
                  % val.replace('.', ''), 1)
    t = t.replace("APP_VERSION = '0.9.196.0'", "APP_VERSION = '%s'" % ver, 1)
    t = t.replace('FOUR_PRIOR_STRENGTH = 1.0', 'FOUR_PRIOR_STRENGTH = %s' % val, 1)
    dst = r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnormprior%s+%s.py' \
          % (val.replace('.', ''), ver)
    io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
    print('wrote', dst)
