# -*- coding: utf-8 -*-
import io

SRC = r'D:\explore\wav\RawHebb+bio+0.20.0.0.py'
BRANCH = r'D:\explore\wav\RawHebb+shell20+0.9.266.0.py'
RELEASE = r'D:\explore\wav\RawHebb+bio+0.20.1.0.py'

t = io.open(SRC, encoding='utf-8').read()
t = t.replace('RawHebb+bio+0.19.0.0 —— 纯 Hebbian', 'RawHebb+bio+0.20.1.0 —— 纯 Hebbian', 1)
t = t.replace('RawHebb+bio+0.20.0.0.py', 'RawHebb+bio+0.20.1.0.py')
t = t.replace('0.20.0.0 做了什么（把 3120012 的壳子拔下来，适配 RawHebb 内核）',
              '''0.20.1.0 做了什么（第一轮菜单全量测试修复）
    · 训练结束自动保存，不再询问“是否保存”。
    · 纯文本续写：短提示（<3 token）的三/四阶记忆键补齐修正，不再张量形状报错；提示前加 <|bos|>。
    · 检查点不再保存 V/spike_acc/M 这些一次性运行态；加载旧档时过滤，日志不再“跳过形状不符 4 个”。
    · 菜单 EOF / Ctrl-D 干净退出；菜单 12 重建词表写 tokenizer.bin/json，不覆盖只读基座。
    · 菜单 1–13 全部实跑（见 docs/FINDINGS.md v143）。

0.20.0.0 做了什么（把 3120012 的壳子拔下来，适配 RawHebb 内核）''', 1)
t = t.replace("APP_BRANCH = 'bio'", "APP_BRANCH = 'bio'", 1)
t = t.replace("APP_VERSION = '0.20.0.0'", "APP_VERSION = '0.20.1.0'", 1)
t = t.replace('#   0.20.0.0 2026-10-22  照 3120012 重写数据壳',
              '#   0.20.1.0 2026-10-22  菜单全量测试修复：训练自动保存、txt 短提示、干净加载、\n'
              '#                         EOF 退出、重建词表不覆盖基座。\n'
              '#   0.20.0.0 2026-10-22  照 3120012 重写数据壳', 1)
io.open(RELEASE, 'w', encoding='utf-8', newline='\n').write(t)

b = t.replace('RawHebb+bio+0.20.1.0', 'RawHebb+shell20+0.9.266.0')
b = b.replace("APP_BRANCH = 'bio'", "APP_BRANCH = 'shell20'", 1)
b = b.replace("APP_VERSION = '0.20.1.0'", "APP_VERSION = '0.9.266.0'", 1)
io.open(BRANCH, 'w', encoding='utf-8', newline='\n').write(b)
print('wrote', RELEASE)
print('wrote', BRANCH)
