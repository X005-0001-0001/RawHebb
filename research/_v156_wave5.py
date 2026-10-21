# -*- coding: utf-8 -*-
import io

base = io.open(r'D:\explore\wav\RawHebb+tri4buck64hidden416four4kbagnormprior05+0.9.206.0.py',
               encoding='utf-8').read()


def write(tag, ver, t):
    t = t.replace('RawHebb+tri4buck64hidden416four4kbagnormprior05+0.9.206.0',
                  'RawHebb+%s+%s' % (tag, ver), 1)
    t = t.replace("APP_BRANCH = 'tri4buck64hidden416four4kbagnormprior05'",
                  "APP_BRANCH = '%s'" % tag, 1)
    t = t.replace("APP_VERSION = '0.9.206.0'", "APP_VERSION = '%s'" % ver, 1)
    p = r'D:\explore\wav\RawHebb+%s+%s.py' % (tag, ver)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
    print('wrote', p)


def combo(tag, ver, hidden=None, buckets=None, tri=None):
    t = base
    if hidden:
        t = t.replace('HIDDEN = 416', 'HIDDEN = %d' % hidden, 1)
    if buckets:
        t = t.replace('TRIGRAM_BUCKETS = 65536', 'TRIGRAM_BUCKETS = %d' % buckets, 1)
    if tri:
        t = t.replace('TRI_PRIOR_STRENGTH = 4.0', 'TRI_PRIOR_STRENGTH = %s' % tri, 1)
    write(tag, ver, t)


combo('tri128kbuck64hidden432four4kbagnormprior05tri2', '0.9.241.0',
      hidden=432, buckets=131072, tri='2.0')
combo('tri128kbuck64hidden448four4kbagnormprior05tri2', '0.9.242.0',
      hidden=448, buckets=131072, tri='2.0')
combo('tri128kbuck64hidden464four4kbagnormprior05', '0.9.243.0',
      hidden=464, buckets=131072, tri='4.0')
combo('tri256kbuck64hidden432four4kbagnormprior05tri2', '0.9.244.0',
      hidden=432, buckets=262144, tri='2.0')
