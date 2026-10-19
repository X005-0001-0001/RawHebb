# -*- coding: utf-8 -*-
"""v116 训练/验证：v114 训练脚本的 GPU 版 + 可调 hidden + 模块可切换。

与 v114 的差异（仅此三项，其余一字未改）：
  ① --device cpu|cuda（cuda 时不再把 torch.cuda 钉死为 False；启动后会自报 PID）
  ② --hidden 可调（攻 128 维瓶颈）
  ③ 模块由 V116_MOD 指定（默认 wavesnn_hebbian_v116）
原 v114 脚本保留不动，见 train_v114.py。

数据流（单一真源）：直接用模块里的 `StreamingJSONLDataset._tokenize_conversation`，
它已经实现了「user 段 labels=-100、assistant 段保留、<|end|> 计入回答」的问答 mask。
本脚本只做三件事：切到 T、逐样本更新、每 epoch 在**独立验证集**上算 masked CE。

口径：
  · 训练 CE / 验证 CE 一律 ignore_index=-100（只统计 assistant 的 token）
  · top-1 也只在 assistant 的 token 上统计
  · 参考线：bigram（用训练集统计的 next-token 频次表，在验证集上算 CE）
  · 验证集与训练集按对话切分（`_v114_prep_data.py`），互不重叠

用法：
  python train_v114.py --epochs 1 --limit 400 --T 200
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import importlib
DEV = 'cpu'
MODNAME = 'wavesnn_hebbian_v116'


ap = argparse.ArgumentParser()
ap.add_argument('--train', default='train_dsr1.jsonl')
ap.add_argument('--val', default='val_dsr1.jsonl')
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--limit', type=int, default=0, help='每 epoch 只用前 N 条（0=全量）')
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--seed', type=int, default=43)
ap.add_argument('--save', default='v114_sd.pt')
ap.add_argument('--init', default='', help='从某个 sd 热启动（默认随机初始化）')
ap.add_argument('--log', default='v114.log')
ap.add_argument('--tsv', default='results_v114.tsv')
ap.add_argument('--val-max', type=int, default=40, help='验证集最多用多少条（0=全量）')
ap.add_argument('--bigram-cache', default='_v114_bigram.json')
ap.add_argument('--eval-every', type=int, default=0, help='每 N 个样本做一次验证（0=只在 epoch 末）')
ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--mod', default='wavesnn_hebbian_v116')
ap.add_argument('--batch', type=int, default=1, help='>1 时把等长样本堆成批，摊薄 GPU 核启动开销')
args = ap.parse_args()


if args.device == 'cuda':
    if not torch.cuda.is_available():
        raise SystemExit('★--device cuda 但 torch.cuda.is_available()=False')
    DEV = 'cuda'
    torch.set_num_threads(2)
    print('[GPU] 启用 CUDA：%s ｜ 本进程 PID=%d' % (torch.cuda.get_device_name(0), os.getpid()), flush=True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.is_initialized = lambda: False
    torch.set_num_threads(int(os.environ.get('V116_THREADS', '4')))
wh = importlib.import_module(args.mod)
wh.DEVICE = torch.device(DEV)
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
logf = io.open(args.log, 'a', encoding='utf-8')


def log(s):
    print(s, flush=True)
    logf.write(s + '\n'); logf.flush()


log('=' * 70)
log('[V114TRAIN] train=%s val=%s epochs=%d limit=%s T=%d seed=%d init=%s'
    % (args.train, args.val, args.epochs, args.limit or 'all', args.T, args.seed, args.init or 'scratch'))

model = wh.WaveSNN(vocab_size=V, hidden_size=args.hidden, num_layers=4, pad_token_id=tok.pad_token_id,
                   use_residual=wh.USE_RESIDUAL, freeze_seed=args.seed, freeze_layers=wh.FREEZE_LAYERS,
                   freeze_embedding=True, seed_embedding=args.seed).to(DEV)
if args.init:
    sd = torch.load(args.init, map_location='cpu', weights_only=False)
    mm, uu = model.load_state_dict(sd, strict=False)
    log('[V114TRAIN] 热启动 %s ｜ missing=%s unexpected=%s' % (args.init, list(mm), list(uu)))
log('[V114TRAIN] 参数量 %.2fM ｜ vocab %d ｜ T=%d' % (sum(p.numel() for p in model.parameters()) / 1e6, V, args.T))

# ---- 取数：用模块自己的 tokenizer 逻辑（单一真源）----
_ds = wh.StreamingJSONLDataset(args.train, tok, max_seq_len=args.T)


def load_rows(path, limit=0):
    out = []
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


_rows = load_rows(args.train)

# bigram 参考线（只用训练集统计）
def bigram_ref(train_rows, val_rows, maxn=400):
    cnt = {}
    tot = {}
    for d in train_rows:
        for m in d['messages']:
            ids = tok.encode(m.get('content', ''))
            for a, b in zip(ids[:-1], ids[1:]):
                cnt[(a, b)] = cnt.get((a, b), 0) + 1
                tot[a] = tot.get(a, 0) + 1
    n = 0
    nll = 0.0
    for d in val_rows[:maxn]:
        for m in d['messages']:
            if m.get('role') != 'assistant':
                continue
            ids = tok.encode(m.get('content', ''))
            for a, b in zip(ids[:-1], ids[1:]):
                p = (cnt.get((a, b), 0) + 0.5) / (tot.get(a, 0) + 0.5 * V)
                nll += -math.log(p)
                n += 1
    return nll / max(n, 1), n


_val_rows = load_rows(args.val)
if os.path.exists(args.bigram_cache):
    _bc = json.load(open(args.bigram_cache))
    _ce_bg, _n_bg = _bc['ce'], _bc['n']
    log('[V114TRAIN] bigram 参考线（缓存）：CE=%.4f（%d 位置）' % (_ce_bg, _n_bg))
else:
    _ce_bg, _n_bg = bigram_ref(_rows, _val_rows)
    json.dump({'ce': _ce_bg, 'n': _n_bg}, open(args.bigram_cache, 'w'))
    log('[V114TRAIN] bigram 参考线（训练集统计 → 验证集 assistant 段）：CE=%.4f（%d 个位置）' % (_ce_bg, _n_bg))
log('[V114TRAIN] bigram 参考线（训练集统计 → 验证集 assistant 段）：CE=%.4f（%d 个位置）' % (_ce_bg, _n_bg))


def prep(row):
    convs = row.get('messages') or row.get('conversations') or []
    ids, labels = _ds._tokenize_conversation(convs)
    if ids.numel() > args.T:
        ids, labels = ids[-args.T:], labels[-args.T:]
    return ids.unsqueeze(0).to(DEV), labels.unsqueeze(0).to(DEV)


def collate(items, pad_id):
    """把一批 (ids, labels) 补齐成同长并堆叠。padding 放在尾部、labels=-100，
    因果结构 ⇒ 真实位置的状态不受影响；记忆写入也被 mask 挡住。"""
    L = max(int(x.shape[0]) for x, _ in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    for i, (x, y) in enumerate(items):
        ids[i, :x.shape[0]] = x
        lab[i, :y.shape[0]] = y
    return ids.to(DEV), lab.to(DEV)


def evaluate(rows, maxn=0):
    model.eval()
    ce_sum, ntok, top1, nval = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, d in enumerate(rows):
            if maxn and i >= maxn:
                break
            ids, labels = prep(d)
            if (labels != -100).sum() == 0:
                continue
            out = model(ids, labels=labels)
            lg = out[0] if isinstance(out, tuple) else out
            ce = F.cross_entropy(lg[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1),
                                 ignore_index=-100, reduction='sum')
            m = (labels[:, 1:] != -100)
            pred = lg[:, :-1, :].argmax(-1)
            top1 += float((pred[m] == labels[:, 1:][m]).float().sum())
            ce_sum += float(ce); ntok += int(m.sum())
    return ce_sum / max(ntok, 1), top1 / max(ntok, 1), ntok


# ---- 训练 ----
t0 = time.time()
history = []
for ep in range(1, args.epochs + 1):
    rng = random.Random(args.seed + ep)
    order = list(range(len(_rows)))
    rng.shuffle(order)
    if args.limit:
        order = order[:args.limit]
    model.train()
    ep_ce, ep_n = 0.0, 0
    t_ep = time.time()
    # 按长度分桶后再组批，避免 padding 浪费（长度大多已截到 T）
    prepared = []
    for idx in order:
        ids, labels = prep(_rows[idx])
        if (labels != -100).sum() == 0:
            continue
        prepared.append((int(ids.shape[1]), idx, ids[0], labels[0]))   # 去掉 prep 的前导 batch 维
    prepared.sort(key=lambda z: -z[0])
    batches = [prepared[i:i + args.batch] for i in range(0, len(prepared), args.batch)]
    order_n = len(prepared)
    for k, bt in enumerate(batches):
        ids, labels = collate([(z[2], z[3]) for z in bt], tok.pad_token_id)
        with torch.no_grad():
            model(ids, labels=labels)
            model._step_count = getattr(model, '_step_count', 0) + 1
            model.hebbian_update(ids, cached=True)
            # ★_v104_ce 是 hebbian_update 里算的 ⇒ 必须先更新再读（否则第一步读到 nan，整轮都被污染）
            _n_eff = int((labels[:, 1:] != -100).sum())
            ep_ce += float(getattr(model, '_v104_ce', float('nan'))) * _n_eff
            ep_n += _n_eff
        _done = min((k + 1) * args.batch, order_n)
        if args.eval_every and _done % args.eval_every < args.batch:
            v_ce, v_top1, v_n = evaluate(_val_rows, args.val_max)
            log('[V114MID] ep%d %d/%d ｜ 训练 CE(累计)=%.4f ｜ **验证 CE=%.4f** top1=%.4f ｜ γ=%.3f ｜ %.0fs'
                % (ep, k + 1, len(order), ep_ce / max(ep_n, 1), v_ce, v_top1,
                   float(model.v111_gamma.data), time.time() - t0))
        if _done % 100 < args.batch or k == 0:
            log('[V114TRAIN] ep%d %d/%d ｜ 训练 CE(累计)=%.4f ｜ %.1fs (%.3fs/样本)'
                % (ep, _done, order_n, ep_ce / max(ep_n, 1), time.time() - t_ep,
                   (time.time() - t_ep) / max(_done, 1)))
    tr_ce = ep_ce / max(ep_n, 1)
    v_ce, v_top1, v_n = evaluate(_val_rows, args.val_max)
    log('[V114EVAL] ep%d ｜ 训练 CE=%.4f ｜ **验证 CE=%.4f**（%d 位置）｜ 验证 top1=%.4f ｜ '
        'γ=%.4f ｜ w=%s ｜ wall=%.0fs'
        % (ep, tr_ce, v_ce, v_n, v_top1, float(model.v111_gamma.data),
           [round(float(x), 4) for x in model._v111_layer_weights.data], time.time() - t0))
    history.append((ep, tr_ce, v_ce, v_top1))

# ---- 落盘 ----
torch.save(model.state_dict(), args.save)
log('[V114TRAIN] 末态权重已存 %s（%.1f MiB）' % (args.save, os.path.getsize(args.save) / 2 ** 20))

# TSV 追加
new = not os.path.exists(args.tsv)
with io.open(args.tsv, 'a', encoding='utf-8', newline='\n') as f:
    if new:
        f.write('run\tepochs\tlimit\tT\tCE_train_last\tCE_val_last\ttop1_val\tCE_bigram\t'
                'gamma\tw_L0\tw_L1\tw_L2\tw_L3\twall_s\n')
    ep, tr_ce, v_ce, v_top1 = history[-1]
    f.write('v114\t%d\t%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s\t%.1f\n'
            % (args.epochs, args.limit or 'all', args.T, tr_ce, v_ce, v_top1, _ce_bg,
               float(model.v111_gamma.data),
               '\t'.join('%.6f' % float(x) for x in model._v111_layer_weights.data),
               time.time() - t0))
log('[V114TRAIN] 完成：%d epoch ｜ 末轮验证 CE=%.4f ｜ bigram 参考 %.4f ｜ 总耗时 %.0fs'
    % (args.epochs, history[-1][2], _ce_bg, time.time() - t0))
logf.close()
