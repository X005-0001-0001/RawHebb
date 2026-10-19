# -*- coding: utf-8 -*-
"""
================================================================================================
 RawHebb+bio+0.1.0.0  —— 纯 Hebbian（无反向传播）语言模型 · 单文件应用
================================================================================================
分支      : bio          （仿生分支：膜电位/阈值放电/回流管/局部突触可塑性）
版本      : 0.1.0.0        （每次改动请复制本文件、改名升版、保留旧版，见文件末尾「版本谱系」）
内核      : 本文件 SECTION 3 的 RawHebbKernel —— 替换 3120012elite.py 的「复值 SSM + AdamW +
            LinearAttentionSummarizer」内核。**只换内核**：菜单结构与交互逻辑沿用原版。
数据层    : JSONL messages/conversations 双读 + 问答 mask + 按对话切分防泄露（沿用原版数据哲学）
禁止项    : 反向传播 / 全局梯度 / AdamW / attention / SSM / 手填常数 / 外挂重置监控器

与原菜单的对应（保留原结构，逐项映射到新内核）：
   1 训练模型              → SECTION 6  train_hebbian      （纯局部规则，masked loss）
   2 对话模式              → SECTION 7  chat_loop          （jsonl 对话 / txt 续写）
   3 保存模型 / 4 加载模型 → SECTION 6  save_model/load_model
   5 调整生成参数          → SECTION 0  TEMPERATURE / TOP_P / TOP_K / REPEAT_PENALTY
   6 查看模型统计信息      → SECTION 6  print_stats        （本内核的冻结/能量/发放率口径）
   7 规则验证拒绝采样      → SECTION 8  rejection_sampling （无梯度自蒸馏，产出 SFT jsonl）
   8 基础规则奖励 RL       → SECTION 9  self_consistency  （★RL 需要策略梯度=BP，与纲领冲突，
                              故替换为「多次采样一致性投票筛选」，同样不需要梯度）
   9 GRPO 规则强化学习     → SECTION 9  longctx_stress    （★同上；替换为长上下文分段接力评测）
  10 退出
  11 数据预处理：切分      → SECTION 4  preprocess_dialogs
  12 独立验证集评测        → SECTION 6  evaluate           （本项目新增：防泄露口径）

用法：
   python "RawHebb+bio+0.1.0.0.py"              # 进菜单
   python "RawHebb+bio+0.1.0.0.py" --selftest   # 内核自检（不读数据，30 秒内结束）
   python "RawHebb+bio+0.1.0.0.py" --train --train-file train_dsr1.jsonl --epochs 1 --limit 200
================================================================================================
"""

# =================================================================================================
# SECTION 0 · 全局配置（**所有超参与开关都集中在这里**，其余代码不允许出现裸常数）
# =================================================================================================
APP_NAME = 'RawHebb'
APP_BRANCH = 'bio'
APP_VERSION = '0.1.0.0'
APP_ID = '%s+%s+%s' % (APP_NAME, APP_BRANCH, APP_VERSION)

# ---- 路径（一律相对本文件所在目录，不越界）----
HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
DATA_DIR = HERE
TOK_BIN = 'tokenizer_v63b4000.bin'
TOK_JSON = 'tokenizer_v63b4000_tokenizer.json'
CKPT = 'brain.pt'                       # 原版同名，保持习惯
LOG_FILE = 'rawhebb.log'

# ---- 生物常数（全部由「节律周期 : 时间常数」的物理关系推出，不允许手填）----
CORTEX_HZ = 40.0                        # γ 节律 40Hz（基座）
THETA_HZ = 8.0                          # θ 节律 8Hz（基座）
TAU_M_MS = 10.0                         # 膜时间常数 10ms（文献 10-20ms 取低端）
DT_MS = 1.0                             # 时间步 1ms（40Hz 采样下的整数细分）
LEAK = float(__import__('math').exp(-DT_MS / TAU_M_MS))      # ≈0.9048
MEM_MOD_DECAY = 0.995                   # 回流管记忆的慢衰减（= exp(-1/τ_echo) 形式，可改）
ADAPT_TAU_MS = 200.0                    # 适应（ homeostasis 目标的时间窗）

# ---- 架构 ----
HIDDEN = 128
NUM_LAYERS = 4
SEQ_LEN = 200
FREEZE_LAYERS = 0                       # 隐藏层 W 冻结（δ≡0，纯前向）
FREEZE_EMBEDDING = True
SEED_EMBEDDING = 43
SEED_FREEZE = 43

# ---- 多时间常数（v112/v115 结论：深层 τ 更长；倍率与上限可调，0 表示用基座 LEAK）----
TAU_MULT = 2.0                          # τ_li = TAU_M_MS * TAU_MULT**li
TAU_CAP_STEPS = 100.0                   # τ 上限（步）；0 = 不限
L0_TAU = 0.0                            # 0 = L0 沿用 LEAK；>0 时 L0 用自己的 τ（v115 开关）

# ---- 记忆与读出 ----
READ_LAYERS = 'all'                     # 'all' = L0..L3-1；'deep' = 只 L1..L3-1
READ_NORM_L2 = True                     # 读出按 ‖w‖₂ 归一（能量保持，v113）
LR_MEM = 0.02                           # 记忆列原型的局部步长 V111_LR_M
MEM_DECAY_EVERY = 10                    # 每 N 步给记忆做一次慢衰减
LR_OUT = 0.01                           # 输出头 LMS 的**相对**步长（每步走 ‖W_out‖ 的这个比例）
LR_GAMMA = 0.001                        # γ 的局部步长
LR_W = 0.001                            # 层权重的局部步长
W_CLAMP = 3.0                           # 层权重护栏 ±3
GAMMA_CLAMP = (0.1, 10.0)
QA_MASK = True                          # 问答 mask：只学 assistant（labels=-100 处不学）

# ---- 生成 ----
TEMPERATURE = 0.8
TOP_P = 0.92
TOP_K = 0
REPEAT_PENALTY = 1.08
MAX_NEW_TOKENS = 160

# ---- 训练 ----
EPOCHS = 1
LEARNING_RATE = 1.0                     # 保留原接口；本内核无 optimizer，此值仅作日志口径
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 0.0                    # 保留原接口；本内核无梯度，必须为 0
LOG_INTERVAL = 25
BATCH_SIZE = 1
VAL_RATIO = 0.05
SEED = 43

# ---- 运行时 ----
DEVICE = 'cpu'                          # 'cpu' / 'cuda'（用 GPU 前请确认没有别人的任务）
THREADS = 4
LIMIT_SAMPLES = 0                       # 0 = 全量；调试时设小值

# =================================================================================================
# SECTION 1 · 日志与工具
# =================================================================================================
import argparse
import fnmatch
import hashlib
import io
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_logf = None


def log_and_print(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    global _logf
    if _logf is None:
        _logf = io.open(os.path.join(HERE, LOG_FILE), 'a', encoding='utf-8')
    _logf.write(s + '\n')
    _logf.flush()


def close_log_file():
    global _logf
    if _logf is not None:
        _logf.close()
        _logf = None


def setup_device():
    global DEVICE
    d = torch.device(DEVICE)
    if DEVICE == 'cuda':
        assert torch.cuda.is_available(), 'DEVICE=cuda 但 torch.cuda.is_available() 为 False'
        log_and_print('[DEVICE] 使用 GPU：%s（%s）' % (torch.cuda.get_device_name(0), d))
    else:
        d = torch.device('cpu')
        log_and_print('[DEVICE] 使用 CPU（threads=%d）' % torch.get_num_threads())
    return d


def deterministic(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =================================================================================================
# SECTION 2 · 分词器（ByteLevel BPE，特殊 token 与基座一致）
# =================================================================================================
class RawHebbTokenizer(object):
    """薄封装：加载基座 BPE 词表 + 特殊 token；不重建词表。"""

    def __init__(self, bin_path, json_path):
        from tokenizers import Tokenizer
        data = torch.load(bin_path, map_location='cpu', weights_only=False)
        self.initial_vocab_size = int(data.get('initial_vocab_size', 2754))
        self.special_tokens_list = list(data.get('special_tokens_list', []))
        self._tok = Tokenizer.from_file(json_path)
        self._ids = {t: self._tok.token_to_id(t) for t in self.special_tokens_list}
        for t, i in self._ids.items():
            if i is not None:
                self._tok.add_special_tokens([t])
        self._cache = {}

    def _id(self, name):
        return self._ids.get(name)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        out = self._tok.encode(text, add_special_tokens=False)
        return out.ids

    def decode(self, ids):
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=True)

    @property
    def bos_token_id(self):
        return self._id('<|bos|>')

    @property
    def eos_token_id(self):
        return self._id('<|eos|>')

    @property
    def user_token_id(self):
        return self._id('<|user|>')

    @property
    def assistant_token_id(self):
        return self._id('<|assistant|>')

    @property
    def end_token_id(self):
        return self._id('<|end|>')

    @property
    def pad_token_id(self):
        return self._id('<|pad|>')

    def chat_to_ids(self, msgs, max_len=SEQ_LEN, mask_assistant=True):
        """把一轮对话渲染成 (input_ids, labels)。**问答 mask 在这里落地**：
        user 段与格式 token 的 labels 一律 -100，只有 assistant 的内容与收尾 <|end|> 计入损失。"""
        tokens, m = [], []
        for t in msgs:
            role = t.get('role')
            cid = self.encode(t.get('content', ''))
            if role == 'user':
                tokens += [self.user_token_id]; m += [False]
                tokens += cid; m += [False] * len(cid)
                tokens += [self.end_token_id]; m += [False]
            elif role == 'assistant':
                tokens += [self.assistant_token_id]; m += [False]
                tokens += cid; m += ([True] * len(cid) if mask_assistant else [False] * len(cid))
                tokens += [self.end_token_id]; m += [True]
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        m = [False] + m + [True]
        if len(tokens) > max_len:                      # 超长：保留最近的上下文
            tokens, m = tokens[-max_len:], m[-max_len:]
        ids = tokens[:-1]
        labels = [t if m[i + 1] else -100 for i, t in enumerate(tokens[1:])]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# =================================================================================================
# SECTION 3 · 内核：RawHebbKernel（纯前向 + 局部可塑性，无 BP）
# =================================================================================================
class RawHebbLayer(nn.Module):
    """单个波神经元层：膜电位积分 → 阈值放电 → 回流管（写 t+1 / 倒流读）。"""

    def __init__(self, idx, hidden, leak, seed):
        super().__init__()
        self.layer_idx = idx
        self.hidden = hidden
        self.leak = nn.Parameter(torch.tensor(leak))
        self.W = nn.Parameter(torch.empty(hidden, hidden))
        gen = torch.Generator().manual_seed(seed * 1000 + idx)
        with torch.no_grad():
            self.W.copy_(torch.randn(hidden, hidden, generator=gen) * (1.0 / math.sqrt(hidden)))
        self.register_buffer('theta', torch.full((hidden,), 0.5))
        self.register_buffer('V', torch.zeros(hidden))
        self.register_buffer('spike_acc', torch.zeros(1, hidden))
        self.register_buffer('M', torch.zeros(hidden, hidden, hidden))   # 回流管：每通道一个三阶记忆
        self.avg_firing = 0.0

    def reset_state(self, device):
        self.V = torch.zeros(self.hidden, device=device)
        self.spike_acc = torch.zeros(1, self.hidden, device=device)

    def step(self, incoming, noise):
        """一个时间步：V = 漏电积分 + 输入（中心化），超过阈值则放电。返回 (V, spike)。"""
        inc = incoming - incoming.mean()
        V = inc + self.leak * self.V
        spike = (V > self.theta).float()
        self.V = V * (1.0 - spike)
        self.spike_acc += spike
        return V, spike


class RawHebbKernel(nn.Module):
    """RawHebb 内核。

    结构（与 wavesnn_hebbian_v115 同源，已蒸馏为单文件、无死代码）：
      · 多层 leaky integrator，τ_li = TAU_M_MS·TAU_MULT**li（L0 可单独指定）
      · 读出：base = 最后一层 V 经 W_out（LMS 学） + γ·Σ w_li·lm_li / ‖w‖₂
      · lm_li = V_li @ W_m[li]，**每层先 RMS 归一到 1**，避免层间音量互吞
      · 局部规则（全部只看本层的读出与误差，无跨层反传）：
          W_out ← W_out + LR_OUT·mean_t(e_t ⊗ V_t)      e_t = onehot(y) − softmax(logits)
          W_m[li][:, y] ← +LR_MEM·(V_li(t) − W_m[li][:, y])
          γ ← γ + LR_GAMMA·mean_t(e_t·读出)
          w_li ← w_li + LR_W·mean_t(e_t·lm_li)
    """

    def __init__(self, vocab_size, hidden=HIDDEN, num_layers=NUM_LAYERS,
                 pad_token_id=0, freeze_layers=FREEZE_LAYERS,
                 freeze_embedding=FREEZE_EMBEDDING, seed=SEED_FREEZE,
                 seed_embedding=SEED_EMBEDDING, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.freeze_layers = freeze_layers
        self.freeze_embedding = freeze_embedding
        self.seed_embedding = seed_embedding
        self.device = device
        self.step_count = 0

        self.embedding = nn.Embedding(vocab_size, hidden)
        gen = torch.Generator().manual_seed(seed_embedding)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(vocab_size, hidden, generator=gen)
                                       * (1.0 / math.sqrt(hidden)))
        self.embedding.weight.requires_grad = not freeze_embedding

        # 各层 τ：L0 默认 LEAK；li≥1 按 TAU_MULT 增长并受 TAU_CAP 限制
        leaks = []
        for li in range(num_layers):
            if li == 0:
                tau = L0_TAU if L0_TAU > 0 else DT_MS / -math.log(max(LEAK, 1e-12))
            else:
                tau = TAU_M_MS * (TAU_MULT ** li)
                if TAU_CAP_STEPS > 0:
                    tau = min(tau, TAU_CAP_STEPS)
            leaks.append(float(math.exp(-DT_MS / tau)))
        self.layers = nn.ModuleList([RawHebbLayer(i, hidden, leaks[i], SEED_FREEZE)
                                     for i in range(num_layers)])
        for i, l in enumerate(self.layers):
            if i >= 1 + freeze_layers:
                l.W.requires_grad_(False)          # 隐藏层 W 冻结 ⇒ δ≡0，无 BP

        self.W_out = nn.Parameter(torch.randn(vocab_size, hidden, generator=gen)
                                  / math.sqrt(hidden))
        self.W_m = nn.ParameterList([nn.Parameter(torch.zeros(hidden, vocab_size))
                                     for _ in range(num_layers)])
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.layer_w = nn.Parameter(torch.ones(num_layers))

        # 读出层（READ_LAYERS='deep' 时把 L0 排除）
        self.read_idx = list(range(num_layers)) if READ_LAYERS == 'all' \
            else list(range(1, num_layers))

        self._buf = {}
        self.to(device)

    # ---------- 前向（**全程 no_grad**：本内核没有任何 autograd 路径）----------
    @torch.no_grad()
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        dev = input_ids.device
        for l in self.layers:
            l.reset_state(dev)
        Vs = [[] for _ in range(self.num_layers)]
        spikes = [[] for _ in range(self.num_layers)]
        noise = torch.randn(self.num_layers, T, self.hidden, device=dev)
        for t in range(T):
            h = self.embedding(input_ids[:, t])
            for li, layer in enumerate(self.layers):
                inc = h
                if li > 0:
                    inc = h + 0.5 * Vs[li - 1][-1]
                V, sp = layer.step(inc, noise[li, t])
                Vs[li].append(V)
                spikes[li].append(sp)
                if li < self.num_layers - 1:
                    h = V
        Vlist = [torch.stack(Vs[li], dim=1) for li in range(self.num_layers)]      # (B,T,H)
        self._buf['V'] = Vlist
        self._buf['spikes'] = [torch.stack(spikes[li], dim=1) for li in range(self.num_layers)]

        last = Vlist[-1]
        # 底板读出：V 先归一到 RMS=1、W_out 行归一 ⇒ logits 量级 O(1)，CE 在 ln(vocab) 附近起步
        last_n = last / last.pow(2).mean().sqrt().clamp(min=1e-6)
        base = last_n @ F.normalize(self.W_out, dim=-1).T
        lms = []
        for li in self.read_idx:
            lm = Vlist[li] @ self.W_m[li]
            lm = lm / lm.pow(2).mean().sqrt().clamp(min=1e-6)     # 每层 RMS=1
            lms.append(lm)
        w = self.layer_w[self.read_idx]
        merged = sum(w[k] * lms[k] for k in range(len(lms)))
        if READ_NORM_L2:
            merged = merged / w.pow(2).sum().sqrt().clamp(min=1e-6)
        logits = base + self.gamma * merged
        # 缓存读出中间量：局部规则必须用**与前向同一份**张量，否则口径会漂
        self._buf['merged'] = merged
        self._buf['logits'] = logits
        self._buf['lm_flat'] = [lm[:, :T - 1, :].reshape(-1, self.vocab_size) for lm in lms]
        self._buf['scale'] = self._v105_scale if hasattr(self, '_v105_scale') else \
            (last.pow(2).mean().sqrt() / 1.0).clamp(min=1e-6) / last.pow(2).mean().sqrt().clamp(min=1e-6)

        if labels is not None:
            self._buf['labels'] = labels
            m = (labels[:, 1:] != -100)
            if m.any():
                self._buf['ce'] = float(F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, self.vocab_size)[m.reshape(-1)],
                    labels[:, 1:].reshape(-1)[m.reshape(-1)]))
            else:
                self._buf['ce'] = float('nan')
        return logits

    # ---------- 局部可塑性（无梯度、无反传）----------
    @torch.no_grad()
    def hebbian_update(self, input_ids, lr=LEARNING_RATE):
        labels = self._buf.get('labels')
        B, T = input_ids.shape
        Vlist = self._buf['V']
        # 误差一律用前向缓存的 logits/merged（不重算、不额外前向）
        merged = self._buf.get('merged')
        if merged is None or labels is None:
            return
        tgt = labels[:, 1:T]
        mpos = (tgt != -100) if QA_MASK else torch.ones_like(tgt, dtype=torch.bool)
        n_valid = float(max(int(mpos.sum()), 1))
        safe = tgt.clamp(min=0)
        p = F.softmax(self._buf['logits'][:, :T - 1, :], dim=-1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(2, safe.unsqueeze(-1), 1.0)
        err = onehot - p
        if QA_MASK:
            err = err * mpos.unsqueeze(-1).float()

        # ① 输出头 LMS（CE 在概率空间的精确梯度，对隐藏层零反传）
        last_n = Vlist[-1] / Vlist[-1].pow(2).mean().sqrt().clamp(min=1e-6)
        h = last_n[:, :T - 1, :].reshape(-1, self.hidden)
        dW = torch.einsum('no,nh->oh', err.reshape(-1, self.vocab_size), h) / n_valid
        # 相对步长：把更新方向归一后走 ‖W_out‖ 的固定比例 ⇒ 与梯度尺度无关、天然不会发散，
        # 不需要「超阈值就重置」那种外挂监控器（纲领第 4 条）
        _dn = float(dW.norm())
        if _dn > 0:
            dW = dW / _dn * (LR_OUT * float(self.W_out.norm()))
        self.W_out.data += dW

        # ② 记忆列原型（只在有效位置写；V 各层已 RMS 归一，避免层间尺度差）
        if self.step_count % MEM_DECAY_EVERY == 0:
            for li in self.read_idx:
                self.W_m[li].data *= 0.99
        for li in self.read_idx:
            Vn = Vlist[li] / Vlist[li].pow(2).mean().sqrt().clamp(min=1e-6)
            for t in range(T - 1):
                if QA_MASK and not bool(mpos[0, t]):
                    continue
                y = int(input_ids[0, t + 1].item())
                if 0 <= y < self.vocab_size:
                    self.W_m[li].data[:, y] += LR_MEM * (Vn[0, t] - self.W_m[li].data[:, y])

        # ③ γ 与层权重：只看「误差 × 读出」，不跨层
        #   γ 用**合并后**的读出（与前向一致）；层权重用**各层自己的**读出
        err_flat = err.reshape(-1, self.vocab_size)
        merged_flat = merged[:, :T - 1, :].reshape(-1, self.vocab_size)
        lm_flat = self._buf['lm_flat']
        g_gamma = float((err_flat * merged_flat).sum()) / n_valid
        self.gamma.data.fill_(min(max(float(self.gamma.data) + LR_GAMMA * g_gamma,
                                       GAMMA_CLAMP[0]), GAMMA_CLAMP[1]))
        for k, li in enumerate(self.read_idx):
            gl = float((err_flat * lm_flat[k]).sum()) / n_valid
            nw = float(self.layer_w.data[li]) + LR_W * gl
            self.layer_w.data[li] = min(max(nw, -W_CLAMP), W_CLAMP)

        self.step_count += 1

    # ---------- 读数 ----------
    def stats(self):
        fr = []
        for l in self.layers:
            fr.append(round(float(l.spike_acc.mean()), 5))
        return {
            'layers': self.num_layers, 'hidden': self.hidden, 'vocab': self.vocab_size,
            'params': sum(p.numel() for p in self.parameters()),
            'frozen_embedding': self.freeze_embedding,
            'frozen_layers': self.freeze_layers,
            'leak': [round(float(l.leak), 6) for l in self.layers],
            'W_out_norm': round(float(self.W_out.norm()), 4),
            'W_m_norm': [round(float(p.norm()), 4) for p in self.W_m],
            'gamma': round(float(self.gamma), 4),
            'layer_w': [round(float(x), 4) for x in self.layer_w],
            'spike_acc_mean': fr,
            'read_idx': self.read_idx,
            'read_norm_l2': READ_NORM_L2,
            'qa_mask': QA_MASK,
        }


# =================================================================================================
# SECTION 4 · 数据层：JSONL 双读 + 问答 mask + 防泄露切分 + 超长切分
# =================================================================================================
def iter_jsonl(path):
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_messages(rec):
    """v5 schema 优先（messages），旧 conversations 也读（与原版一致的双读）。"""
    for key in ('messages', 'conversations', 'conversation', 'dialog'):
        v = rec.get(key)
        if isinstance(v, list) and v:
            out = []
            for t in v:
                if isinstance(t, dict):
                    role = t.get('role') or t.get('from') or 'user'
                    content = t.get('content')
                    if content is None:
                        content = t.get('value') or ''
                    out.append({'role': role, 'content': content})
                elif isinstance(t, str):
                    out.append({'role': 'user', 'content': t})
            return out
    return []


def split_by_dialogue(src, train_out, val_out, ratio=VAL_RATIO, seed=SEED, verbose=True):
    """**按对话切分**（绝不按行随机），并做三重泄露检查：精确重复 / 跨集同 sha1 / 前缀重复。
       返回 (n_train, n_val, 报告字符串)。"""
    rows = list(iter_jsonl(src))
    seen, dedup = {}, []
    dup = 0
    for r in rows:
        txt = '\n'.join(t['content'] for t in get_messages(r))
        h = hashlib.sha1(txt.encode('utf-8')).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen[h] = 1
        dedup.append((r, h))
    rng = random.Random(seed)
    idx = list(range(len(dedup)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(dedup) * ratio)))
    val_i = set(idx[:n_val])
    tr = [r for i, (r, _) in enumerate(dedup) if i not in val_i]
    va = [r for i, (r, _) in enumerate(dedup) if i in val_i]
    cross = len(set(h for _, h in [(0, 0)]) )  # placeholder
    hs_tr = set(h for i, (r, h) in enumerate(dedup) if i not in val_i)
    hs_va = set(h for i, (r, h) in enumerate(dedup) if i in val_i)
    pre_tr = set(t['content'][:120] for r in tr for t in get_messages(r) if t['role'] == 'user')
    pre_hit = sum(1 for r in va for t in get_messages(r)
                  if t['role'] == 'user' and t['content'][:120] in pre_tr)
    rep = ('切分 %d/%d（val %.1f%%）｜ 精确重复剔除 %d ｜ 跨集同 sha1 %d ｜ '
           '验证集 user 前缀命中 %d/%d'
           % (len(tr), len(va), 100.0 * len(va) / max(len(dedup), 1), dup,
              len(hs_tr & hs_va), pre_hit, len(va)))
    with io.open(train_out, 'w', encoding='utf-8', newline='\n') as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with io.open(val_out, 'w', encoding='utf-8', newline='\n') as f:
        for r in va:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if verbose:
        log_and_print('[DATA] ' + rep)
    return len(tr), len(va), rep


def preprocess_dialogs_menu():
    """菜单 11：超长对话切分（多轮切单轮 / 单轮超长丢弃），沿用原版意图。"""
    src = input('输入 jsonl [dsr1_v5_0001.jsonl]: ').strip() or 'dsr1_v5_0001.jsonl'
    dst = input('输出 jsonl [dsr1_split_train.jsonl]: ').strip() or 'dsr1_split_train.jsonl'
    maxlen = int(input('单条上限（token，0=用 SEQ_LEN）[%d]: ' % SEQ_LEN).strip() or SEQ_LEN)
    mode = input('多轮怎么处理？1=只留最后一轮 2=每轮拆成单条 [2]: ').strip() or '2'
    n_in = n_out = n_drop = 0
    with io.open(dst, 'w', encoding='utf-8', newline='\n') as f:
        for rec in iter_jsonl(os.path.join(DATA_DIR, src)):
            n_in += 1
            msgs = get_messages(rec)
            pairs = [(msgs[i], msgs[i + 1]) for i in range(len(msgs) - 1)
                     if msgs[i]['role'] == 'user' and msgs[i + 1]['role'] == 'assistant']
            if not pairs:
                n_drop += 1
                continue
            if mode == '1':
                pairs = pairs[-1:]
            for u, a in pairs:
                approx = len(u['content']) + len(a['content'])
                if maxlen > 0 and approx > maxlen * 8:      # 粗筛：1 token ≈ 4-8 字符
                    n_drop += 1
                    continue
                f.write(json.dumps({'messages': [u, a]}, ensure_ascii=False) + '\n')
                n_out += 1
    log_and_print('[DATA] 切分完成：输入 %d 条 → 输出 %d 条，丢弃 %d 条 → %s' % (n_in, n_out, n_drop, dst))


# =================================================================================================
# SECTION 5 · 采样与生成（无梯度）
# =================================================================================================
@torch.no_grad()
def generate(model, tok, prompt_ids, max_new=MAX_NEW_TOKENS, temp=TEMPERATURE, top_p=TOP_P,
             top_k=TOP_K, rep=REPEAT_PENALTY, stop_ids=()):
    model.eval()
    ids = list(prompt_ids)
    out = []
    seen = {}
    for _ in range(max_new):
        x = torch.tensor([ids[-SEQ_LEN:]], dtype=torch.long, device=model.device)
        logits = model(x)[0, -1, :].clone()
        if rep != 1.0:
            for t in set(ids):
                if logits[t] < 0:
                    logits[t] = logits[t] * rep
        if temp > 0:
            logits = logits / temp
        if top_k and top_k > 0:
            v, i = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf')).scatter_(0, i, v)
        probs = F.softmax(logits, -1)
        if top_p and 0 < top_p < 1:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            cut = int((cum - sp > top_p).sum()) + 1
            keep = si[:cut]
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[keep] = True
            sp = sp[:cut] / sp[:cut].sum()
            nid = int(keep[torch.multinomial(sp, 1)].item())
        else:
            nid = int(torch.multinomial(probs, 1).item())
        if nid in stop_ids:
            break
        ids.append(nid)
        out.append(nid)
    return out


# =================================================================================================
# SECTION 6 · 训练 / 评测 / 存取
# =================================================================================================
def build_model(tok, device, hidden=HIDDEN, layers=NUM_LAYERS):
    return RawHebbKernel(tok.vocab_size, hidden=hidden, num_layers=layers,
                         pad_token_id=tok.pad_token_id, device=device)


def save_model(model, path=None):
    path = os.path.join(DATA_DIR, path or CKPT)
    torch.save(model.state_dict(), path)
    log_and_print('[CKPT] 已保存 %s（%.1f MiB）' % (path, os.path.getsize(path) / 2 ** 20))


def load_model(model, path=None):
    path = os.path.join(DATA_DIR, path or CKPT)
    if not os.path.exists(path):
        return False
    sd = torch.load(path, map_location=model.device, weights_only=False)
    mm, uu = model.load_state_dict(sd, strict=False)
    log_and_print('[CKPT] 已加载 %s ｜ missing=%s unexpected=%s' % (path, list(mm), list(uu)))
    return True


def print_stats(model, tok):
    st = model.stats()
    log_and_print('架构：%d 层 × %d 维 ｜ 词表 %d ｜ 参数量 %d' % (st['layers'], st['hidden'], st['vocab'], st['params']))
    log_and_print('leak（τ 派生）: %s' % st['leak'])
    log_and_print('读出层 %s ｜ ‖w‖₂ 归一 %s ｜ 问答 mask %s' % (st['read_idx'], st['read_norm_l2'], st['qa_mask']))
    log_and_print('γ=%.4f ｜ 层权重 w=%s' % (st['gamma'], st['layer_w']))
    log_and_print('‖W_out‖=%.2f ｜ ‖W_m‖=%s' % (st['W_out_norm'], st['W_m_norm']))
    log_and_print('发放率（累计）: %s' % st['spike_acc_mean'])
    log_and_print('嵌入冻结 %s ｜ 隐藏层冻结 %d 层' % (st['frozen_embedding'], st['frozen_layers']))


def train_hebbian(model, tok, path, epochs=EPOCHS, limit=LIMIT_SAMPLES, val_file=None,
                  val_max=40, eval_every=0, logfile=None):
    """纯 Hebbian 训练：forward → 局部规则 → forward。无 BP、无 optimizer。
    损失只在 assistant 的 token 上计算（QA_MASK）。"""
    dev = model.device
    rows = list(iter_jsonl(os.path.join(DATA_DIR, path)))
    random.Random(SEED).shuffle(rows)
    if limit:
        rows = rows[:limit]
    log_and_print('[TRAIN] 数据 %s：%d 条 ｜ epoch=%d ｜ T=%d ｜ 只学 assistant=%s'
                  % (path, len(rows), epochs, SEQ_LEN, QA_MASK))
    val_rows = list(iter_jsonl(os.path.join(DATA_DIR, val_file)))[:val_max] if val_file else []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ce_sum, n_ce, t_ep = 0.0, 0, time.time()
        for k, rec in enumerate(rows):
            ids, labels = tok.chat_to_ids(get_messages(rec))
            if (labels != -100).sum() == 0:
                continue
            ids = ids.unsqueeze(0).to(dev)
            labels = labels.unsqueeze(0).to(dev)
            model(ids, labels=labels)
            model.hebbian_update(ids)
            n = int((labels[:, 1:] != -100).sum())
            ce_sum += float(model._buf.get('ce', float('nan'))) * n
            n_ce += n
            if eval_every and (k + 1) % eval_every == 0 and val_rows:
                v = evaluate(model, tok, val_rows)
                log_and_print('[MID] ep%d %d/%d ｜ 训练CE=%.4f ｜ **验证CE=%.4f** top1=%.4f ｜ %.0fs'
                              % (ep, k + 1, len(rows), ce_sum / max(n_ce, 1), v['ce'], v['top1'],
                                 time.time() - t0))
            if (k + 1) % LOG_INTERVAL == 0:
                log_and_print('[TRAIN] ep%d %d/%d ｜ CE(累计)=%.4f ｜ %.1fs'
                              % (ep, k + 1, len(rows), ce_sum / max(n_ce, 1), time.time() - t_ep))
        msg = '[TRAIN] ep%d 完成 ｜ 训练 CE=%.4f ｜ %.0fs' % (ep, ce_sum / max(n_ce, 1), time.time() - t_ep)
        if val_rows:
            v = evaluate(model, tok, val_rows)
            msg += ' ｜ **验证 CE=%.4f top1=%.4f top5=%.4f**' % (v['ce'], v['top1'], v['top5'])
        log_and_print(msg)
    if logfile:
        log_and_print('[TRAIN] 日志已写 %s' % logfile)
    return ce_sum / max(n_ce, 1)


@torch.no_grad()
def evaluate(model, tok, rows, ngram_ref=False):
    """独立验证集评测：**只统计 assistant token**（labels≠-100），与训练口径严格一致。"""
    model.eval()
    dev = model.device
    ce, n, t1, t5 = 0.0, 0, 0, 0
    for rec in rows:
        ids, labels = tok.chat_to_ids(get_messages(rec))
        m = (labels != -100)
        if m.sum() == 0:
            continue
        x = ids.unsqueeze(0).to(dev)
        lg = model(x)[0]
        lg = lg[:-1]
        tg = labels[1:]
        mm = m[1:]
        if mm.sum() == 0:
            continue
        ce += float(F.cross_entropy(lg[mm], tg[mm], reduction='sum'))
        pred = lg[mm].argmax(-1)
        t1 += int((pred == tg[mm]).sum())
        k5 = torch.topk(lg[mm], 5, -1).indices
        t5 += int((k5 == tg[mm].unsqueeze(-1)).any(-1).sum())
        n += int(mm.sum())
    return {'ce': ce / max(n, 1), 'top1': t1 / max(n, 1), 'top5': t5 / max(n, 1), 'n': n}


def ngram_reference(tok, train_path, val_rows, k=2):
    """n-gram 参考线（只用训练集统计）。k=1 → unigram，k=2 → bigram。"""
    cnt, den = {}, {}
    for rec in iter_jsonl(os.path.join(DATA_DIR, train_path)):
        for t in get_messages(rec):
            ids = tok.encode(t['content'])
            for i in range(len(ids) - k + 1):
                key = tuple(ids[i:i + k - 1]) if k > 1 else ()
                cnt[(key, ids[i + k - 1])] = cnt.get((key, ids[i + k - 1]), 0) + 1
                den[key] = den.get(key, 0) + 1
    nll, n = 0.0, 0
    for rec in val_rows:
        for t in get_messages(rec):
            if t['role'] != 'assistant':
                continue
            ids = tok.encode(t['content'])
            for i in range(len(ids) - k + 1):
                key = tuple(ids[i:i + k - 1]) if k > 1 else ()
                p = (cnt.get((key, ids[i + k - 1]), 0) + 0.5) / (den.get(key, 0) + 0.5 * tok.vocab_size)
                nll += -math.log(p)
                n += 1
    return nll / max(n, 1), n


# =================================================================================================
# SECTION 7 · 对话
# =================================================================================================
def chat_loop(model, tok, mode='jsonl'):
    model.eval()
    stop = (tok.eos_token_id, tok.end_token_id)
    if mode == 'txt':
        log_and_print('纯文本续写模式：输入任意文本，Ctrl-C 结束。')
        while True:
            try:
                s = input('文本: ')
            except (EOFError, KeyboardInterrupt):
                return
            ids = tok.encode(s)
            out = generate(model, tok, ids, stop_ids=stop)
            print(tok.decode(out))
    else:
        log_and_print('对话模式（jsonl）：输入 空格=重开一问，quit 退出。')
        while True:
            try:
                q = input('用户: ').strip()
            except (EOFError, KeyboardInterrupt):
                return
            if q.lower() in ('quit', 'exit'):
                return
            if not q:
                continue
            prompt = [tok.bos_token_id, tok.user_token_id] + tok.encode(q) + \
                     [tok.end_token_id, tok.assistant_token_id]
            out = generate(model, tok, prompt, stop_ids=stop)
            log_and_print('助手: ' + tok.decode(out))


# =================================================================================================
# SECTION 8 · 规则验证拒绝采样（菜单 7）：无梯度自蒸馏，产出 SFT jsonl
# =================================================================================================
def rejection_sampling(model, tok, src, dst, k=4, rule='nonempty'):
    """采样 k 条，只保留通过规则（默认：非空且长度达标）的回答，写成 SFT jsonl。
       这是**纯采样 + 过滤**，没有梯度、没有奖励反传。"""
    recs = list(iter_jsonl(os.path.join(DATA_DIR, src)))
    n_in = n_out = 0
    with io.open(os.path.join(DATA_DIR, dst), 'w', encoding='utf-8', newline='\n') as f:
        for rec in recs:
            msgs = get_messages(rec)
            if len(msgs) < 2:
                continue
            n_in += 1
            u = msgs[0]
            prompt = [tok.bos_token_id, tok.user_token_id] + tok.encode(u['content']) + \
                     [tok.end_token_id, tok.assistant_token_id]
            cands = [tok.decode(generate(model, tok, prompt, max_new=120))
                     for _ in range(k)]
            good = [c for c in cands if len(c.strip()) >= 8]
            if not good:
                continue
            best = max(good, key=len)          # 规则示例：取最完整的一条
            f.write(json.dumps({'messages': [{'role': 'user', 'content': u['content']},
                                             {'role': 'assistant', 'content': best}]},
                               ensure_ascii=False) + '\n')
            n_out += 1
    log_and_print('[SFT] 拒绝采样：输入 %d → 输出 %d → %s' % (n_in, n_out, dst))


# =================================================================================================
# SECTION 9 · 原菜单 8/9 的无梯度替代
# =================================================================================================
def self_consistency(model, tok, src, dst, k=5):
    """原「基础规则奖励 RL」的替代：**多次采样 + 一致性投票**，没有梯度、没有策略更新。"""
    recs = list(iter_jsonl(os.path.join(DATA_DIR, src)))[:200]
    agree = []
    outp = os.path.join(DATA_DIR, dst)
    with io.open(outp, 'w', encoding='utf-8', newline='\n') as f:
        for rec in recs:
            msgs = get_messages(rec)
            if len(msgs) < 2:
                continue
            u = msgs[0]
            prompt = [tok.bos_token_id, tok.user_token_id] + tok.encode(u['content']) + \
                     [tok.end_token_id, tok.assistant_token_id]
            cands = [tok.decode(generate(model, tok, prompt, max_new=100)) for _ in range(k)]
            cands = [c for c in cands if c.strip()]
            if not cands:
                continue
            counts = {}
            for c in cands:
                counts[c] = counts.get(c, 0) + 1
            best, n = max(counts.items(), key=lambda kv: kv[1])
            agree.append(n / float(k))
            f.write(json.dumps({'messages': [{'role': 'user', 'content': u['content']},
                                             {'role': 'assistant', 'content': best}],
                                'consistency': n / float(k)}, ensure_ascii=False) + '\n')
    log_and_print('[CONSIST] 平均一致率 %.3f（k=%d）→ %s' % (sum(agree) / max(len(agree), 1), k, dst))


def longctx_stress(model, tok, val_file, seg=4):
    """原「GRPO」的替代：长上下文分段接力评测——把验证对话按段切开，
       看「读了多少上文」对回答质量的影响（纯评测，无梯度）。"""
    rows = list(iter_jsonl(os.path.join(DATA_DIR, val_file)))[:40]
    res = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        sub = []
        for rec in rows:
            msgs = get_messages(rec)
            aid = [t for t in msgs if t['role'] == 'assistant']
            if not aid:
                continue
            txt = aid[0]['content']
            cut = max(8, int(len(txt) * frac))
            keep = {'messages': [{'role': 'user', 'content': msgs[0]['content']},
                                 {'role': 'assistant', 'content': txt[:cut]}]}
            sub.append(keep)
        v = evaluate(model, tok, sub)
        res.append((frac, v['ce'], v['top1'], v['n']))
        log_and_print('[STRESS] 回答保留 %.0f%% ｜ CE=%.4f top1=%.4f（%d 位置）'
                      % (frac * 100, v['ce'], v['top1'], v['n']))
    return res


# =================================================================================================
# SECTION 10 · 菜单（沿用 3120012elite.py 的结构与交互，逐项映射到新内核）
# =================================================================================================
def main_menu():
    global TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY
    dev = setup_device()
    torch.set_num_threads(THREADS)
    deterministic(SEED)
    tok_path = os.path.join(DATA_DIR, TOK_BIN)
    tok_json = os.path.join(DATA_DIR, TOK_JSON)
    if not (os.path.exists(tok_path) and os.path.exists(tok_json)):
        log_and_print('★找不到词表 %s / %s —— 本内核必须与基座词表对齐。' % (TOK_BIN, TOK_JSON))
        return
    tok = RawHebbTokenizer(tok_path, tok_json)
    log_and_print('%s 已启动 ｜ 内核=RawHebbKernel（无 BP）｜ 词表 %d' % (APP_ID, tok.vocab_size))
    model = build_model(tok, dev)
    model_loaded = False

    def ensure_loaded():
        nonlocal model_loaded
        if model_loaded:
            return True
        if load_model(model):
            model_loaded = True
            return True
        log_and_print('未找到 %s，使用随机初始化（从头训练）。' % CKPT)
        return False

    while True:
        print('\n=== RawHebb 训练与对话系统菜单 ===')
        print('1. 训练模型（继续训练）')
        print('2. 对话模式')
        print('3. 保存模型')
        print('4. 加载模型')
        print('5. 调整生成参数')
        print('6. 查看模型统计信息')
        print('7. 规则验证拒绝采样：生成正确回答 SFT 数据')
        print('8. 自一致性投票筛选（原「规则奖励 RL」的**无梯度替代**）')
        print('9. 长上下文分段接力评测（原「GRPO」的**无梯度替代**）')
        print('10. 退出')
        print('11. 数据预处理：超长对话切分')
        print('12. 独立验证集评测（防泄露口径）')
        choice = input('请选择: ').strip()

        if choice == '1':
            path = input('训练 jsonl [train_dsr1.jsonl]: ').strip() or 'train_dsr1.jsonl'
            ep = int(input('epoch 数 [%d]: ' % EPOCHS).strip() or EPOCHS)
            lim = int(input('样本上限（0=全量）[%d]: ' % LIMIT_SAMPLES).strip() or 0)
            valf = input('验证集（可空）[val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            ensure_loaded()
            if input('开始训练？(y/n): ').lower() == 'y':
                train_hebbian(model, tok, path, epochs=ep, limit=lim, val_file=valf)
        elif choice == '2':
            ensure_loaded()
            mode = input('对话格式 (jsonl) 还是纯文本续写 (txt)？[jsonl]: ').strip().lower()
            chat_loop(model, tok, mode=('txt' if mode == 'txt' else 'jsonl'))
        elif choice == '3':
            save_model(model)
            model_loaded = True
        elif choice == '4':
            if load_model(model):
                model_loaded = True
        elif choice == '5':
            try:
                TEMPERATURE = float(input('temperature [%.2f]: ' % TEMPERATURE).strip() or TEMPERATURE)
                TOP_P = float(input('top_p [%.2f]: ' % TOP_P).strip() or TOP_P)
                TOP_K = int(input('top_k [%d]: ' % TOP_K).strip() or TOP_K)
                REPEAT_PENALTY = float(input('repeat_penalty [%.2f]: ' % REPEAT_PENALTY).strip() or REPEAT_PENALTY)
            except ValueError:
                log_and_print('输入无效，保持原值')
        elif choice == '6':
            print_stats(model, tok)
        elif choice == '7':
            ensure_loaded()
            src = input('输入 jsonl [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            dst = input('输出 SFT jsonl [sft_raw.jsonl]: ').strip() or 'sft_raw.jsonl'
            k = int(input('每题采样数 [4]: ').strip() or 4)
            rejection_sampling(model, tok, src, dst, k=k)
        elif choice == '8':
            ensure_loaded()
            src = input('输入 jsonl [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            dst = input('输出一致性数据 [consistency.jsonl]: ').strip() or 'consistency.jsonl'
            k = int(input('采样次数 [5]: ').strip() or 5)
            self_consistency(model, tok, src, dst, k=k)
        elif choice == '9':
            ensure_loaded()
            vf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            longctx_stress(model, tok, vf)
        elif choice == '11':
            preprocess_dialogs_menu()
        elif choice == '12':
            ensure_loaded()
            vf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            n = int(input('用多少条 [40]: ').strip() or 40)
            rows = list(iter_jsonl(os.path.join(DATA_DIR, vf)))[:n]
            v = evaluate(model, tok, rows)
            log_and_print('[EVAL] ★CE=%.4f ｜ top1=%.4f ｜ top5=%.4f ｜ 计分位置 %d'
                          % (v['ce'], v['top1'], v['top5'], v['n']))
            tf = input('同时算 n-gram 参考线？需要训练集路径 [train_dsr1.jsonl]: ').strip()
            if tf and os.path.exists(os.path.join(DATA_DIR, tf)):
                for k in (1, 2):
                    ce, npos = ngram_reference(tok, tf, rows, k=k)
                    log_and_print('[EVAL] %d-gram 参考线：CE=%.4f（%d 位置）' % (k, ce, npos))
        elif choice == '10':
            break
        else:
            log_and_print('无效选择，请重新输入')
    close_log_file()


# =================================================================================================
# 自检：不读数据，验证内核的局部规则与 mask 是否按预期工作
# =================================================================================================
def selftest():
    dev = torch.device('cpu')
    torch.set_num_threads(2)
    tok = RawHebbTokenizer(os.path.join(DATA_DIR, TOK_BIN), os.path.join(DATA_DIR, TOK_JSON))
    model = RawHebbKernel(tok.vocab_size, device=dev)
    ids, labels = tok.chat_to_ids([{'role': 'user', 'content': '你好'},
                                   {'role': 'assistant', 'content': '世界'}])
    print('[SELFTEST] ids=%s' % ids.tolist()[:16])
    print('[SELFTEST] labels=%s（-100 = 提问段，不学）' % labels.tolist()[:16])
    x = ids.unsqueeze(0)
    y = labels.unsqueeze(0)
    out = model(x, labels=y)
    print('[SELFTEST] logits=%s ｜ masked CE=%.4f（只算 assistant 段）'
          % (tuple(out.shape), model._buf['ce']))
    print('[SELFTEST] V=%.3f ｜ 发散=%s' % (float(out.abs().mean()),
                                          bool(torch.isfinite(out).all())))
    before = float(model.W_out.norm())
    model.hebbian_update(x)
    after = float(model.W_out.norm())
    print('[SELFTEST] 一步更新：‖W_out‖ %.3f → %.3f（变化 %s）'
          % (before, after, '有' if abs(after - before) > 1e-9 else '★无'))
    print('[SELFTEST] OK —— 内核可运行、mask 生效、无 NaN。')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--train-file', default='train_dsr1.jsonl')
    ap.add_argument('--val-file', default='val_dsr1.jsonl')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--limit', type=int, default=LIMIT_SAMPLES)
    ap.add_argument('--val-max', type=int, default=40)
    ap.add_argument('--eval-every', type=int, default=0)
    ap.add_argument('--save', default='')
    ap.add_argument('--split', action='store_true', help='按对话切分训练/验证集')
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.split:
        deterministic()
        tok = RawHebbTokenizer(os.path.join(DATA_DIR, TOK_BIN), os.path.join(DATA_DIR, TOK_JSON))
        split_by_dialogue('dsr1_v5_0001.jsonl', 'train_dsr1.jsonl', 'val_dsr1.jsonl')
    elif args.train:
        dev = setup_device()
        torch.set_num_threads(THREADS)
        deterministic(SEED)
        tok = RawHebbTokenizer(os.path.join(DATA_DIR, TOK_BIN), os.path.join(DATA_DIR, TOK_JSON))
        model = build_model(tok, dev)
        if os.path.exists(CKPT):
            load_model(model)
        train_hebbian(model, tok, args.train_file, epochs=args.epochs, limit=args.limit,
                      val_file=args.val_file, val_max=args.val_max, eval_every=args.eval_every)
        save_model(model, args.save or CKPT)
    else:
        main_menu()

# =================================================================================================
# 版本谱系（每次改动：复制本文件 → 改 → 改名升版，旧版保留在同目录）
#   0.1.0.0 2026-10-19  首版（四段版本号约定：a.b.c.d，不带 v 前缀）：内核替换为 RawHebbKernel（纯 Hebbian）；菜单沿用 3120012elite
#                         的 11 项结构，8/9 项因与「拒绝 BP」冲突替换为无梯度等价物；
#                         新增 12 项独立验证集评测（防泄露口径）。
# =================================================================================================
