# -*- coding: utf-8 -*-
"""v114 数据准备：dsr1_v5_0001.jsonl → 训练/验证切分（按对话切，绝不按行随机）。

为什么按对话切：一份语料里同一对话的 user/assistant 是配对的，
若把「同一条对话」拆到两边，验证集就是抄答案。这里同时做三重泄露检查：

  ① 精确重复：对 (user+assistant) 全文做 sha1；重复项**先全局去重**（保留首次出现），再切分；
  ② 跨集重复：切分后再查一次两边是否有相同 sha1（必须为 0）；
  ③ 前缀重复：user 内容前 120 字符是否在另一侧出现过（模板化问答的常见泄露）；
  ④ n-gram 重叠：验证集里 8-gram 有多大比例在训练集出现过（给一个诚实的数字，不假装是 0）。

输出：train_dsr1.jsonl / val_dsr1.jsonl（保持 {"messages":[...]} 原格式）
      _v114_split_stats.txt
"""
import json
import os
import random
import hashlib
import collections
import io

SRC = 'dsr1_v5_0001.jsonl'
OUT_TRAIN = 'train_dsr1.jsonl'
OUT_VAL = 'val_dsr1.jsonl'
VAL_RATIO = 0.05
SEED = 43

rows = []
with io.open(SRC, encoding='utf-8') as f:
    for ln, line in enumerate(f):
        line = line.strip()
        if line:
            d = json.loads(line)
            rows.append((ln, d))
print('[PREP] 原始对话 %d 条' % len(rows))


def text_of(d):
    return '\n'.join(m.get('content', '') for m in d['messages'] if m.get('role') in ('user', 'assistant'))


def user_of(d):
    for m in d['messages']:
        if m.get('role') == 'user':
            return m.get('content', '')
    return ''


# ---- ① 精确去重 ----
seen = {}
dedup = []
dup = 0
for ln, d in rows:
    h = hashlib.sha1(text_of(d).encode('utf-8')).hexdigest()
    if h in seen:
        dup += 1
        continue
    seen[h] = ln
    dedup.append((ln, d, h))
print('[PREP] ① 精确重复剔除 %d 条 ⇒ 剩 %d 条' % (dup, len(dedup)))

# ---- ② 按对话切分 ----
rng = random.Random(SEED)
idx = list(range(len(dedup)))
rng.shuffle(idx)
n_val = max(1, int(round(len(dedup) * VAL_RATIO)))
val_idx = set(idx[:n_val])
train = [dedup[i] for i in range(len(dedup)) if i not in val_idx]
val = [dedup[i] for i in range(len(dedup)) if i in val_idx]
print('[PREP] ② 切分：train %d ｜ val %d（%.1f%%）' % (len(train), len(val), 100.0 * len(val) / len(dedup)))

hs_tr = set(h for _, _, h in train)
hs_va = set(h for _, _, h in val)
print('[PREP] ② 跨集同 sha1 数 = %d（必须 0）' % len(hs_tr & hs_va))

# ---- ③ 前缀泄露 ----
pref_tr = set(user_of(d)[:120] for _, d, _ in train)
pref_va = [user_of(d)[:120] for _, d, _ in val]
hit = sum(1 for p in pref_va if p in pref_tr)
print('[PREP] ③ 验证集 user 前缀（前120字）在训练集出现：%d/%d = %.1f%%'
      % (hit, len(pref_va), 100.0 * hit / max(len(pref_va), 1)))

# ---- ④ 8-gram 重叠 ----
def grams(txt, n=8):
    s = txt.replace(' ', '')
    return set(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


gtr = set()
for _, d, _ in train:
    gtr |= grams(text_of(d))
tot = 0
ov = 0
for _, d, _ in val:
    g = grams(text_of(d))
    tot += len(g)
    ov += len(g & gtr)
print('[PREP] ④ 验证集 8-gram 有多少在训练集出现：%d/%d = %.2f%%（这是 token 级泄露的诚实读数）'
      % (ov, tot, 100.0 * ov / max(tot, 1)))

with io.open(OUT_TRAIN, 'w', encoding='utf-8', newline='\n') as f:
    for _, d, _ in train:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
with io.open(OUT_VAL, 'w', encoding='utf-8', newline='\n') as f:
    for _, d, _ in val:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print('[PREP] 写出 %s（%d 条）/ %s（%d 条）' % (OUT_TRAIN, len(train), OUT_VAL, len(val)))

# ---- 统计 ----
def stats(name, data):
    chars = [len(text_of(d)) for _, d, _ in data]
    chars.sort()
    n = len(chars)
    print('[PREP] %-5s 字符数：mean %.0f ｜ p50 %d ｜ p90 %d ｜ max %d ｜ 总 %d'
          % (name, sum(chars) / n, chars[n // 2], chars[int(n * 0.9)], chars[-1], sum(chars)))
    return sum(chars)


c_tr = stats('train', train)
c_va = stats('val', val)
print('[PREP] 合计 %.2fM 字符（train %.2fM / val %.2fM）' % ((c_tr + c_va) / 1e6, c_tr / 1e6, c_va / 1e6))
