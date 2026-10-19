# -*- coding: utf-8 -*-
"""v115 评测：在一个**完全没参与训练**的验证集上，给出可比的 CE/top1/top5 与 n-gram 参考线。

为什么单独写：训练脚本里的 evaluate 只报 CE/top1，且 bigram 参考线是别处算的。
这里把口径固定死，任何模型权重都能直接跑：
  · 只统计 assistant 的 token（labels ≠ -100）
  · CE / top1 / top5 三个数一起报，避免「CE 好看但 top1 崩」
  · bigram / trigram 用**训练集**统计、加 0.5 平滑，在同一批位置上报 CE
  · 分位置段报 CE（前 25% / 中 50% / 后 25%），看「越往后是否越准」（上下文是否真的被用上）
"""
import argparse
import io
import json
import math
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
import torch.nn.functional as F
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.is_initialized = lambda: False
torch.set_num_threads(4)
import importlib

ap = argparse.ArgumentParser()
ap.add_argument('--mod', default='wavesnn_hebbian_v115')
ap.add_argument('--sd', default='v115_sd.pt')
ap.add_argument('--val', default='val_dsr1.jsonl')
ap.add_argument('--train', default='train_dsr1.jsonl')
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--n', type=int, default=0, help='验证集最多用多少条（0=全量 150 条）')
ap.add_argument('--l0read', type=int, default=1)
args = ap.parse_args()

os.environ['V112_L0_READ'] = str(args.l0read)
wh = importlib.import_module(args.mod)
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
V = tok.vocab_size
m = wh.WaveSNN(vocab_size=V, hidden_size=128, num_layers=4, pad_token_id=tok.pad_token_id,
               use_residual=wh.USE_RESIDUAL, freeze_seed=43, freeze_layers=wh.FREEZE_LAYERS,
               freeze_embedding=True, seed_embedding=43).to('cpu')
sd = torch.load(args.sd, map_location='cpu', weights_only=False)
mm, uu = m.load_state_dict(sd, strict=False)
m.eval()
print('[EVAL] mod=%s sd=%s ｜ missing=%s unexpected=%s ｜ γ=%.4f ｜ w=%s'
      % (args.mod, args.sd, list(mm), list(uu), float(m.v111_gamma.data),
         [round(float(x), 4) for x in m._v111_layer_weights.data]))


def rows(path, n=0):
    out = []
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
                if n and len(out) >= n:
                    break
    return out


train_rows = rows(args.train)
val_rows = rows(args.val, args.n)
print('[EVAL] train %d 条 ｜ val %d 条' % (len(train_rows), len(val_rows)))

# ---- n-gram 参考线（在 assistant 的连续 token 上统计）----
def ng_counts(k):
    c = {}
    for d in train_rows:
        for msg in d['messages']:
            ids = tok.encode(msg.get('content', ''))
            for i in range(len(ids) - k + 1):
                key = tuple(ids[i:i + k - 1]) if k > 1 else ()
                c[(key, ids[i + k - 1])] = c.get((key, ids[i + k - 1]), 0) + 1
    return c


def ng_den(k, c):
    if k == 1:
        return None, sum(c.values())
    den = {}
    for (key, _), v in c.items():
        den[key] = den.get(key, 0) + v
    return den, None


def ng_ce(k, c):
    den, total = ng_den(k, c)
    tot, nll = 0, 0.0
    for d in val_rows:
        for msg in d['messages']:
            if msg.get('role') != 'assistant':
                continue
            ids = tok.encode(msg.get('content', ''))
            for i in range(len(ids) - k + 1):
                key = tuple(ids[i:i + k - 1]) if k > 1 else ()
                dn = den.get(key, 0) if k > 1 else total
                num = c.get((key, ids[i + k - 1]), 0)
                p = (num + 0.5) / (dn + 0.5 * V)
                nll += -math.log(p); tot += 1
    return nll / max(tot, 1), tot


for k in (1, 2):
    c = ng_counts(k)
    ce_k, n_k = ng_ce(k, c)
    print('[EVAL] %d-gram 参考线：CE=%.4f（%d 位置）' % (k + 1, ce_k, n_k))

# ---- 模型 ----
_ds = wh.StreamingJSONLDataset(args.val, tok, max_seq_len=args.T)
ce_sum, n_tok, top1, top5 = 0.0, 0, 0, 0
pos_ce = [[0.0, 0] for _ in range(3)]
per_sample = []
with torch.no_grad():
    for d in val_rows:
        ids, labels = _ds._tokenize_conversation(d.get('messages') or d.get('conversations') or [])
        if ids.numel() > args.T:
            ids, labels = ids[-args.T:], labels[-args.T:]
        ids, labels = ids.unsqueeze(0), labels.unsqueeze(0)
        if (labels != -100).sum() == 0:
            continue
        out = m(ids, labels=labels)
        lg = out[0] if isinstance(out, tuple) else out
        lg = lg[:, :-1, :].reshape(-1, V)
        tg = labels[:, 1:].reshape(-1)
        msk = (tg != -100)
        ce = F.cross_entropy(lg[msk], tg[msk], reduction='sum')
        ce_sum += float(ce); n_tok += int(msk.sum())
        pred = lg[msk].argmax(-1)
        top1 += int((pred == tg[msk]).sum())
        k5 = torch.topk(lg[msk], 5, -1).indices
        top5 += int((k5 == tg[msk].unsqueeze(-1)).any(-1).sum())
        # 位置分段（在被 mask 的位置序列上按前/中/后切）
        idx = torch.nonzero(msk, as_tuple=False).flatten()
        if idx.numel() >= 9:
            a, b = idx.numel() // 4, idx.numel() * 3 // 4
            for j, sl in enumerate((slice(0, a), slice(a, b), slice(b, None))):
                ii = idx[sl]
                pos_ce[j][0] += float(F.cross_entropy(lg[ii], tg[ii], reduction='sum'))
                pos_ce[j][1] += int(ii.numel())
        per_sample.append(float(ce) / max(int(msk.sum()), 1))

print('[EVAL] ★ 模型：CE=%.4f ｜ top1=%.4f ｜ top5=%.4f ｜ 计分位置=%d'
      % (ce_sum / max(n_tok, 1), top1 / max(n_tok, 1), top5 / max(n_tok, 1), n_tok))
print('[EVAL] 分段 CE（前25%%/中50%%/后25%% 的回答位置）：%.4f / %.4f / %.4f'
      % tuple(pos_ce[j][0] / max(pos_ce[j][1], 1) for j in range(3)))
per_sample.sort()
print('[EVAL] 逐条 CE：p10=%.3f p50=%.3f p90=%.3f（越集中越好）'
      % (per_sample[len(per_sample) // 10], per_sample[len(per_sample) // 2],
         per_sample[int(len(per_sample) * 0.9)]))
