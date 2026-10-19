# -*- coding: utf-8 -*-
"""v113 探针 0：L1/L2/L3 读出的共线度（不训练，只诊断）。

回答三个问题：
  Q1 三层读出两两有多像？（全局 cos + 逐位置 cos）
  Q2 用另外两层线性重构某一层，能解释多少？（R²）——这是「共线」的严格定义
  Q3 **去掉被另外两层解释掉的部分之后，剩下的残差还有没有预测力？**
     这一条决定下一步该做「去共线」还是「赢者通吃」：
       · 残差 top1 明显高于基线 ⇒ 每层有**私有信息**，正交化/侧抑制能把它捞出来；
       · 残差 top1 ≈ 基线      ⇒ 三层是同一句话的噪声副本，求和必然放大噪声，
                                 正确做法是竞争（只让最好的那层说话），不是加权求和。
  Q4 现在的合并有几成是「重复计数」：C = ‖Σw·lm‖² / Σw²‖lm‖²（=1 表示完全同向）

口径：只在 35 个训练窗上跑前向（labels 不为 None ⇒ forward 会缓存每层读出），
     每层读出已按 RMS=1 归一（模块里做的），所以 cos 与 w 可直接比。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# ★必须在 import 模块之前设：模块在导入时就读环境变量定 τ
SD = os.environ.get('V113_PROBE_SD', 'v112_sd.pt')
MULT = os.environ.get('V113_PROBE_MULT', '2')
CAP = os.environ.get('V113_PROBE_CAP', '100')
os.environ['V112_TAU_MULT'] = MULT
os.environ['V112_TAU_CAP'] = CAP
import numpy as np
import torch
import torch.nn.functional as F
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.is_initialized = lambda: False
torch.set_num_threads(4)
import wavesnn_hebbian_v112 as wh
wh.DEVICE = torch.device('cpu')



class Tok(wh.FastBPE):
    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        from tokenizers import Tokenizer
        self.initial_vocab_size = data.get('initial_vocab_size', wh.VOCAB_SIZE)
        self.special_tokens_list = data.get('special_tokens_list', wh.SPECIAL_TOKENS_LIST.copy())
        jp = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '_tokenizer.json')
        self._tokenizer = Tokenizer.from_file(jp); self._update_special_ids(); self._loaded_json = jp
        return True


tok = Tok(wh.VOCAB_SIZE); tok.load('tokenizer_v63b4000.bin')
ids = tok.encode(open('story_v63.txt', encoding='utf-8').read())
V = tok.vocab_size
m = wh.WaveSNN(vocab_size=V, hidden_size=128, num_layers=4, pad_token_id=tok.pad_token_id,
               use_residual=wh.USE_RESIDUAL, freeze_seed=43, freeze_layers=wh.FREEZE_LAYERS,
               freeze_embedding=True, seed_embedding=43).to('cpu')
sd = torch.load(SD, map_location='cpu', weights_only=False)
mm, uu = m.load_state_dict(sd, strict=False)
m.eval()
W = [float(x) for x in m._v111_layer_weights.data]
IDX = [1, 2, 3]
print('[PROBE] sd=%s ｜ MULT=%s CAP=%s ｜ γ=%.4f ｜ w=%s ｜ 读出层=%s'
      % (SD, MULT, CAP, float(m.v111_gamma.data), ['%.4f' % x for x in W], IDX))

WINS = [torch.tensor([ids[i * 200:(i + 1) * 200]], dtype=torch.long) for i in range(35)]

cos_glob = np.zeros((3, 3)); cos_pos = np.zeros((3, 3)); R2 = np.zeros((3, 3))
top1_raw = np.zeros(3); top5_raw = np.zeros(3)
top1_res = np.zeros(3); top5_res = np.zeros(3)
res_frac = np.zeros(3)
C_idx = []; npos = 0

with torch.no_grad():
    for x in WINS:
        m(x, labels=x)
        lms = m._v112_lm_list_buf                       # 每层 (1,T,V)，RMS=1
        idx = list(m._v112_idx_buf)
        L = [lms[k][0, :-1, :] for k in range(len(idx))]   # (T-1,V)
        y = x[0, 1:]
        T1 = L[0].shape[0]; npos += T1
        Lf = [t.reshape(-1) for t in L]
        for a in range(3):
            for b in range(3):
                cos_glob[a, b] += float(Lf[a] @ Lf[b] / (Lf[a].norm() * Lf[b].norm() + 1e-12))
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                cos_pos[a, b] += float(F.cosine_similarity(L[a], L[b], dim=-1).mean())
        # 逐位置：用其他两层线性重构本层
        for a in range(3):
            others = [L[b] for b in range(3) if b != a]           # 各 (T-1,V)
            A = torch.stack([torch.stack([(others[p] * others[q]).sum(-1) for q in range(2)], -1)
                             for p in range(2)], -1)               # (T-1,2,2)
            rhs = torch.stack([(others[p] * L[a]).sum(-1) for p in range(2)], -1)  # (T-1,2)
            A = A + 1e-6 * torch.eye(2).expand_as(A)
            coef = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)            # (T-1,2)
            pred = coef[:, 0:1] * others[0] + coef[:, 1:2] * others[1]
            res = L[a] - pred
            ss_res = res.pow(2).sum(-1); ss_tot = L[a].pow(2).sum(-1).clamp(min=1e-12)
            R2[a, 0] += float((1 - ss_res / ss_tot).mean())
            res_frac[a] += float((ss_res.sqrt() / ss_tot.sqrt()).mean())
            top1_raw[a] += float((L[a].argmax(-1) == y).float().mean())
            top5_raw[a] += float((torch.topk(L[a], 5, -1).indices == y.unsqueeze(-1)).any(-1).float().mean())
            top1_res[a] += float((res.argmax(-1) == y).float().mean())
            top5_res[a] += float((torch.topk(res, 5, -1).indices == y.unsqueeze(-1)).any(-1).float().mean())
        # Q4 合并的共线度
        w = torch.tensor([W[i] for i in idx])
        S = sum(w[k] * L[k] for k in range(3))
        num = float(S.pow(2).sum())
        den = float(sum((w[k] ** 2) * L[k].pow(2).sum() for k in range(3)))
        C_idx.append(num / max(den, 1e-12))

n = len(WINS)
cos_glob /= n; cos_pos /= n; R2 /= n; res_frac /= n
top1_raw /= n; top5_raw /= n; top1_res /= n; top5_res /= n
base = 1.0 / V
C = float(np.mean(C_idx))

print('\n[Q1] 全局 cos（三层读出展平后）            [逐位置 cos 均值]')
for a in range(3):
    print('   L%d: %s   |   %s'
          % (IDX[a], ' '.join('%+.4f' % cos_glob[a, b] for b in range(3)),
             ' '.join('%+.4f' % cos_pos[a, b] for b in range(3))))
print('\n[Q2] 用另外两层线性重构本层的 R² / 残差相对范数（1−R² 的开方）：')
for a in range(3):
    print('   L%d: R²=%.4f  残差‖·‖/‖·‖=%.4f' % (IDX[a], R2[a, 0], res_frac[a]))
print('\n[Q3] 原读出 vs 去共线残差 的 top-1（随机基线 %.2e）：' % base)
for a in range(3):
    print('   L%d: 原 top1=%.6f（%.1fx） top5=%.6f ｜ 残差 top1=%.6f（%.1fx） top5=%.6f ｜ w=%+.4f'
          % (IDX[a], top1_raw[a], top1_raw[a] / base, top5_raw[a],
             top1_res[a], top1_res[a] / base, top5_res[a], W[IDX[a]]))
print('\n[Q4] 合并共线度 C = ‖Σw·lm‖² / Σw²‖lm‖² = %.4f（1.0=完全同向、3.0=三层正交）' % C)
print('     含义：C≈1 ⇒ 加权和几乎等于「把同一句话重复三遍再放大」，'
      '有效音量是 Σw² 而不是 (Σw)²。')
