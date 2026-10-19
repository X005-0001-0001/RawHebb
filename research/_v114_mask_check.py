# -*- coding: utf-8 -*-
"""v114 问答 mask 的单元自证（不训练，只看一步更新）。

构造一个「user 段 + assistant 段」的假窗口：
  · user 段的目标 token 全是 7（只在提问里出现的词）
  · assistant 段的目标 token 全是 11
  · labels 在 user 段置 -100，assistant 段保留
然后在同一个初值上分别跑 V114_MASK=1 / 0 的一步 hebbian_update：

  要求 ①：mask=1 时 W_m_vocab[:,7] **一动不动**（提问不学）；
  要求 ②：mask=1 时 W_m_vocab[:,11] 有更新（回答要学）；
  要求 ③：mask=0 时 W_m_vocab[:,7] 也会动（复现 v113 的老行为，证明差异确实来自 mask）；
  要求 ④：mask=1 时报出的 CE == 只在 assistant 段算的 CE（口径一致）。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import torch.nn.functional as F
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.is_initialized = lambda: False
torch.set_num_threads(2)

MASK_ON = os.environ.get('V114_MASK', '1') != '0'
import wavesnn_hebbian_v114 as wh
wh.DEVICE = torch.device('cpu')

V = 2754
T = 48
m = wh.WaveSNN(vocab_size=V, hidden_size=128, num_layers=4, pad_token_id=0,
               use_residual=wh.USE_RESIDUAL, freeze_seed=43, freeze_layers=wh.FREEZE_LAYERS,
               freeze_embedding=True, seed_embedding=43).to('cpu')
m.eval()

torch.manual_seed(7)
# 前 24 个位置：目标=7（提问）；后 23 个位置：目标=11（回答）
ids = torch.randint(3, V, (1, T))
ids[0, 1:25] = 7          # 这些位置「作为目标」出现在 t-1 的 next
ids[0, 25:] = 11
labels = ids.clone()
labels[0, 1:25] = -100    # 提问段不学
# 注意：labels[t] 是位置 t 的标签，模型用 input_ids[:,1:] 当目标 ⇒ 目标序列 = ids[1:]
# 我们让 ids[1:25]=7 表示 t=0..23 的目标是 7；ids[25:]=11 表示 t=24.. 的目标是 11

col7 = m.W_m_vocab[1].data[:, 7].clone()
col11 = m.W_m_vocab[1].data[:, 11].clone()

with torch.no_grad():
    _lg0 = m(ids, labels=labels)          # ★先存更新前的 logits（_v104_ce 就是用它算的）
    _lg0 = (_lg0[0] if isinstance(_lg0, tuple) else _lg0).clone()
    m._step_count = 1
    m.hebbian_update(ids, cached=True)
    ce_masked = getattr(m, '_v104_ce', float('nan'))

d7 = float((m.W_m_vocab[1].data[:, 7] - col7).norm())
d11 = float((m.W_m_vocab[1].data[:, 11] - col11).norm())

# 用**更新前**的同一份 logits 复核口径
if MASK_ON:
    ce_ref = float(F.cross_entropy(_lg0[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1),
                                   ignore_index=-100))
else:
    ce_ref = float(F.cross_entropy(_lg0[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1)))

print('[MASK] V114_MASK=%d' % int(MASK_ON))
print('[MASK] ‖ΔW_m[:,7]‖=%.3e（提问词） ‖ΔW_m[:,11]‖=%.3e（回答词）' % (d7, d11))
print('[MASK] 报出 CE=%.6f ｜ 只算 assistant 段的 CE=%.6f ｜ 差=%.2e'
      % (ce_masked, ce_ref, abs(ce_masked - ce_ref)))

ok = True
if MASK_ON:
    ok &= (d7 < 1e-12)
    ok &= (d11 > 1e-9)
    ok &= (abs(ce_masked - ce_ref) < 1e-5)
else:
    ok &= (d7 > 1e-9)
print('[MASK] ⇒ %s' % ('通过' if ok else '★不通过'))
