# -*- coding: utf-8 -*-
"""v113 探针 4：信息在不在状态里？——用 kNN 给「表征」和「线性读出」分责。

问题：v113 把合并修好了一点，但推理 CE 仍是 7.3469（v112 7.3477），门③ 仍差 0.016。
      到底卡在哪？
        (a) 表征里没有信息   ⇒ 换读出/合并都没用，必须改状态动力学
        (b) 表征里有，线性读出学不出来 ⇒ 是「样本 / 容量」问题（128×2754 的矩阵用 7k 位置拟合）

判据：对每层的 V(t)（128 维）做 1-NN 检索，看下一个 token 的命中率：
  · kNN ≫ 线性读出（0.43%） ⇒ 信息在状态里，是线性映射没学会 ⇒ (b)
  · kNN ≈ 线性读出           ⇒ 状态里就没多少可用的 ⇒ (a)
两个口径：
  inter-win：只允许匹配**别的窗口**的位置（更接近「学了规律」）
  intra-win：允许同窗口其它位置（更接近「查表/近邻记忆」的上限）
另给 bigram 基线（用当前 token 查下一个 token 的频次表），标定「L0 那种查表」大概在什么量级。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['V112_TAU_MULT'] = '2'
os.environ['V112_TAU_CAP'] = '100'
import numpy as np
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.is_initialized = lambda: False
torch.set_num_threads(4)
import wavesnn_hebbian_v113 as wh
wh.DEVICE = torch.device('cpu')

SD = os.environ.get('V113_PROBE_SD', 'v113_sd.pt')


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
m.load_state_dict(torch.load(SD, map_location='cpu', weights_only=False), strict=False)
m.eval()
print('[KNN] sd=%s ｜ γ=%.4f ｜ w=%s' % (SD, float(m.v111_gamma.data),
                                        [round(float(x), 4) for x in m._v111_layer_weights.data]))

WINS = [torch.tensor([ids[i * 200:(i + 1) * 200]], dtype=torch.long) for i in range(35)]
Xs = {1: [], 2: [], 3: []}
ys, wids = [], []
with torch.no_grad():
    for wi, x in enumerate(WINS):
        m(x, labels=x)
        for li in (1, 2, 3):
            Xs[li].append(m._cached_layer_outputs[li][0, :-1, :])      # (T-1,H)
        ys.append(x[0, 1:]); wids += [wi] * (x.shape[1] - 1)

y = torch.cat(ys); wid = torch.tensor(wids)
N = y.numel()
base = 1.0 / V
print('[KNN] 位置数 N=%d ｜ 窗口 %d ｜ vocab %d ｜ 随机基线 top1=%.2e top5=%.2e'
      % (N, len(WINS), V, base, 5 * base))

# ---- bigram 基线（留一法：查 (当前 token, 下一 token) 频次，排除自身位置）----
prev = torch.cat([WINS[wi][0, :-1] for wi in range(len(WINS))])
big = torch.zeros(V, V)
for i in range(N):
    big[prev[i], y[i]] += 1
pred = []
for i in range(N):
    row = big[prev[i]].clone(); row[y[i]] -= 1
    pred.append(int(row.argmax()))
pred = torch.tensor(pred)
print('[KNN] bigram 基线（留一）：top1=%.6f（%.1fx）' % (float((pred == y).float().mean()),
                                                     float((pred == y).float().mean()) / base))

for li in (1, 2, 3):
    X = torch.cat(Xs[li], 0)
    X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    S = X @ X.T                                    # 余弦相似度 (N,N)
    S.fill_diagonal_(-2.0)
    same_win = (wid[:, None] == wid[None, :])
    for tag, mask in (('inter-win', ~same_win), ('intra-win', same_win)):
        Sm = S.masked_fill(mask, -2.0)
        k1 = Sm.argmax(1)
        acc1 = float((y[k1] == y).float().mean())
        top5 = torch.topk(Sm, 5, 1).indices
        acc5 = float((top5 == y.unsqueeze(1)).any(1).float().mean())
        print('[KNN] L%d %-9s top1=%.6f（%.1fx） top5=%.6f'
              % (li, tag, acc1, acc1 / base, acc5))
print('[KNN] 判读：若 inter-win 的 kNN 明显高于线性读出的 0.0043(11.9x baseline) ⇒ '
      '信息在状态里、是线性映射没学会（容量/样本问题）；若接近 ⇒ 状态本身没多少可用信息。')
