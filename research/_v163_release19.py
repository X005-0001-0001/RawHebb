# -*- coding: utf-8 -*-
"""Create 0.19.0.0 from the 0.9.264 shell branch (same model recipe as 0.18)."""
import io

SRC = r'D:\explore\wav\RawHebb+shell19+0.9.264.0.py'
DST = r'D:\explore\wav\RawHebb+bio+0.19.0.0.py'
t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+shell19+0.9.264.0', 'RawHebb+bio+0.19.0.0')
t = t.replace("APP_BRANCH = 'shell19'", "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.9.264.0'", "APP_VERSION = '0.19.0.0'", 1)
io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST)
