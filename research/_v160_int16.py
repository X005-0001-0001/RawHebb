# -*- coding: utf-8 -*-
import io

base = io.open(r'D:\explore\wav\RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0.py',
               encoding='utf-8').read()


def write(tag, ver, buckets):
    t = base
    t = t.replace('TRIGRAM_BUCKETS = 1048576', 'TRIGRAM_BUCKETS = %d' % buckets, 1)
    t = t.replace("self.register_buffer('tri_count', torch.zeros(TRIGRAM_BUCKETS, vocab_size))",
                  "self.register_buffer('tri_count', torch.zeros(TRIGRAM_BUCKETS, vocab_size, dtype=torch.int16))", 1)
    t = t.replace('_trow = self.tri_count[_keys]',
                  '_trow = self.tri_count[_keys].float()', 1)
    t = t.replace('RawHebb+tri1Mbuck64hidden432four4kbagnormprior05tri2+0.9.251.0',
                  'RawHebb+%s+%s' % (tag, ver), 1)
    t = t.replace("APP_BRANCH = 'tri1Mbuck64hidden432four4kbagnormprior05tri2'",
                  "APP_BRANCH = '%s'" % tag, 1)
    t = t.replace("APP_VERSION = '0.9.251.0'", "APP_VERSION = '%s'" % ver, 1)
    p = r'D:\explore\wav\RawHebb+%s+%s.py' % (tag, ver)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
    print('wrote', p)


write('tri1Mbuck64hidden432four4kbagnormprior05tri2int16', '0.9.254.0', 1048576)
write('tri2Mbuck64hidden432four4kbagnormprior05tri2int16', '0.9.255.0', 2097152)
