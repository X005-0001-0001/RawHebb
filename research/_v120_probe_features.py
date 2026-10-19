# -*- coding: utf-8 -*-
"""特征探针（诊断用，**不是部署的学习规则**）：把每层状态与词向量抽出来缓存，
再用**闭式岭回归**（离线解）预测下一个 token。

目的：一刀切开两种可能——
  (a) 闭式解出来的 top-1 也只有 5%  ⇒ **特征不行**（状态里没有 token 身份），改规则没用；
  (b) 闭式解能到 15%+             ⇒ **特征够，是在线局部规则收敛太慢**，那就该换成
      递归最小二乘（仍然是纯在线、无反传、局部）而不是继续调步长。

只跑一次前向并把特征落盘，之后所有探针都是秒级。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['V112_L0_READ'] = '1'
os.environ['V116_BASE_SRC'] = 'last'

import json
import io
import importlib
import time

import math

import torch
import torch.nn.functional as F
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.is_initialized = lambda: False
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('rawhebb_app', 'RawHebb+bio+0.3.0.0.py')
wh = _ilu.module_from_spec(_spec); _spec.loader.exec_module(wh)
wh.DEVICE = torch.device('cpu')



tok = wh.RawHebbTokenizer(os.path.join(wh.DATA_DIR, wh.TOK_BIN),
                          os.path.join(wh.DATA_DIR, wh.TOK_JSON))
V = tok.vocab_size
m = wh.build_model(tok, torch.device('cpu'))
m.eval()
print('[FEAT] 模块就绪 vocab=%d' % V)

N_TRAIN, N_VAL = int(os.environ.get('FEAT_TRAIN', '250')), int(os.environ.get('FEAT_VAL', '40'))
MASKW = -(10 ** 9)   # 单文件内核用 -100，这里保持一致


def collect(ds, path, n):
    feats, tgts = [], []
    rows = [json.loads(l) for l in io.open(path, encoding='utf-8') if l.strip()][:n]
    t0 = time.time()
    with torch.no_grad():
        for i, rec in enumerate(rows):
            ids, lab = tok.chat_to_ids(wh.get_messages(rec), max_len=200)
            if ids.numel() > 200:
                ids, lab = ids[-200:], lab[-200:]
            ids, lab = ids.unsqueeze(0), lab.unsqueeze(0)     # (1,T)
            keep = (lab[0, 1:] != -100)
            if int(keep.sum()) < 16:
                continue
            m(ids, labels=lab)
            Vtr = m._buf['V']
            layers = [Vtr[li][0, :-1, :] for li in range(4)]
            emb = m.embedding(ids)[0, :-1, :]
            x = torch.cat([emb] + layers, dim=-1)[keep]      # (n, 5*128)
            feats.append(x.half())
            tgts.append(lab[0, 1:][keep])
            if (i + 1) % 50 == 0:
                print('  %s %d/%d  %.0fs' % (os.path.basename(path), i + 1, len(rows), time.time() - t0), flush=True)
    return torch.cat(feats), torch.cat(tgts)


Xtr, Ytr = collect(None, 'train_dsr1.jsonl', N_TRAIN)
Xva, Yva = collect(None, 'val_dsr1.jsonl', N_VAL)
print('[FEAT] 训练 %s ／ 验证 %s' % (tuple(Xtr.shape), tuple(Xva.shape)))
torch.save({'Xtr': Xtr, 'Ytr': Ytr, 'Xva': Xva, 'Yva': Yva}, '_v120_feats.pt')

# ---------------- 闭式岭回归探针（仅诊断）----------------
def probe(Xt, Yt, Xv, Yv, cols, tag):
    A = Xt[:, cols].float()
    B = Xv[:, cols].float()
    mu, sd = A.mean(0, keepdim=True), A.std(0, keepdim=True).clamp(min=1e-6)
    A = (A - mu) / sd
    B = (B - mu) / sd
    G = A.T @ A
    best = None
    for lam in (1e-3, 1e-1, 1e1, 1e3):
        W = torch.linalg.solve(G + lam * torch.eye(G.shape[0]), A.T @ F.one_hot(Yt, V).float())
        lg = B @ W
        ce = float(F.cross_entropy(lg, Yv))
        t1 = float((lg.argmax(-1) == Yv).float().mean())
        if best is None or ce < best[1]:
            best = (lam, ce, t1)
    print('[PROBE] %-34s lam=%.0e  验证 CE=%.4f  top1=%.4f' % (tag, best[0], best[1], best[2]))
    return best


emb = list(range(0, 128))
L0, L1, L2, L3 = (list(range(128 * (i + 1), 128 * (i + 2))) for i in range(4))
probe(Xtr, Ytr, Xva, Yva, emb, '仅词向量（输入身份）')
probe(Xtr, Ytr, Xva, Yva, L0, '仅 L0 状态 τ=10')
probe(Xtr, Ytr, Xva, Yva, L3, '仅 L3 状态 τ=80')
probe(Xtr, Ytr, Xva, Yva, emb + L0 + L1 + L2 + L3, '词向量 + 全部四层状态')
print('[PROBE] unigram top1 参考 = %.4f ｜ bigram CE 参考 = 3.9613'
      % float(torch.bincount(Ytr, minlength=V).max() / len(Ytr)))