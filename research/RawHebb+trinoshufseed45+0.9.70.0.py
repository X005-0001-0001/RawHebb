# -*- coding: utf-8 -*-
"""
================================================================================================
 RawHebb+trinoshufseed45+0.9.70.0  —— 纯 Hebbian（无反向传播）语言模型 · 单文件实验台
================================================================================================
分支      : trinoshufseed45    （v121 实验分支）
版本      : 0.9.70.0     （v121 实验分支）
本轮      : 只加开关，不改默认行为：HEAD_BIAS / NEWTON_STEPS / COUPLING_MODE /
            NOISE_ENABLED / SPIKE_ENABLED / PAIR_SCOPE / PAIR_MIX / TRAIN_SHUFFLE。
            默认值与 aligncnt 逐位一致，便于生成对照分支。
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
   python "RawHebb+bio+0.6.0.0.py"              # 进菜单
   python "RawHebb+bio+0.6.0.0.py" --selftest   # 内核自检（不读数据，30 秒内结束）
   python "RawHebb+bio+0.6.0.0.py" --train --train-file train_dsr1.jsonl --epochs 1 --limit 200
================================================================================================
"""

# =================================================================================================
# SECTION 0 · 全局配置（**所有超参与开关都集中在这里**，其余代码不允许出现裸常数）
# =================================================================================================
APP_NAME = 'RawHebb'
APP_BRANCH = 'trinoshufseed45'
APP_VERSION = '0.9.70.0'
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
SEQ_LEN = 128
FREEZE_LAYERS = 0                       # 隐藏层 W 冻结（δ≡0，纯前向）
FREEZE_EMBEDDING = True
SEED_EMBEDDING = 45
SEED_FREEZE = 45

# ---- 多时间常数（v112/v115 结论：深层 τ 更长；倍率与上限可调，0 表示用基座 LEAK）----
TAU_MULT = 2.0                          # τ_li = TAU_M_MS * TAU_MULT**li
TAU_CAP_STEPS = 100.0                   # τ 上限（步）；0 = 不限
L0_TAU = 0.0                            # 0 = L0 沿用 LEAK；>0 时 L0 用自己的 τ（v115 开关）

# ---- 记忆与读出（0.2.0.0：**没有任何可调常数**）----
# 学习率、混合系数、记忆衰减全部由各自的运行统计或生物时间常数决定：
#   · 输出头 LMS : 方向归一 + 1/(t+1) 调度      ⇒ 与梯度尺度无关、总位移有界
#   · 记忆写入   : 指数平均，衰减 = 该层的膜泄漏   ⇒ 遗忘曲线 = 膜时间常数（同一套生物常数）
#   · 读出混合   : 各层的「超随机命中率」归一     ⇒ 尺度由证据决定，天然 ∈[0,1]、Σw=1
#   · γ / 护栏   : 删除（尺度由混合权重承担，不再需要学一个标量或夹一个区间）
# 剩下的自由参数只有**架构与物理量**（层数/宽度/窗口/τ），没有「调出来的」学习率。
READ_LAYERS = 'all'                     # 'all' = L0..L3-1；'deep' = 只 L1..L3-1
QA_MASK = True                          # 问答 mask：只学 assistant（labels=-100 处不学）
# ---- 0.8.0.0：精确 token 对记忆（bigram 式联想）----
# 0.6.0.0 的记忆列是 128 维软原型；bigram 表是 V×V 精确计数。两者信息不同，故并联。
# 更新是纯 Hebbian 累加（W[x,y] += 1），没有梯度、没有学习率。
# 先验 = 全局 unigram 分布，强度 1 = **一个伪观测**（最小非零证据单位），不是调出来的平滑系数。
PAIR_MEMORY = True
PAIR_PRIOR_STRENGTH = 1.0
PAIR_MIX = 'likelihood'                 # likelihood / hit / ce_top1 / confidence / product
PAIR_SCOPE = 'mask'                     # mask = 只统计 assistant 目标；all = 所有相邻 token
# ---- 实验台开关（默认 = aligncnt 行为）----
HEAD_BIAS = False                       # 输出头加常数列 1（截距）；Gram 自动求偏置
NEWTON_STEPS = 1                        # 输出头每窗的准 Newton 步数
COUPLING_MODE = 'half'                  # half / one / inv_layers（层间耦合系数）
NOISE_ENABLED = True                    # 旧代码生成但未使用；False 省掉 RNG 开销
SPIKE_ENABLED = True                    # 阈值放电后的复位；False = 纯漏电积分
TRAIN_SHUFFLE = False                    # 训练集是否按 SEED 打乱
# ---- 多专家（0.9.x 实验台）：base / pair / trigram / unigram ----
TRIGRAM_MEMORY = True                  # 二阶上下文（前两个 token）哈希计数专家
TRIGRAM_BUCKETS = 16384                 # 哈希桶数（架构量；越大冲突越少、内存越大）
UNIGRAM_EXPERT = False                  # 显式 unigram 专家（运行计数，非手填先验）
MIX_ROUNDS = 2                          # 专家权重的坐标下降轮数（每轮对每个权做一次黄金分割）
MIX_ITERS = 24                          # 每次黄金分割的迭代数（计算预算，非常数调参）

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
SEED = 45

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


def golden_min(f, lo, hi, iters=32):
    """一维黄金分割搜索（只依赖 (√5−1)/2，不含任何可调步长）。
    用于「混合权重/温度」这种 1 维标量：目标函数由当前窗的似然给出。"""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = float(f(c)), float(f(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = float(f(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = float(f(d))
    return 0.5 * (a + b)


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
        # ★0.3.0.0：输入增益取**本层的 (1−leak)**。稳态 V ≈ inc/(1−leak)，
        #   不补偿时每层把方差放大 1/(1−leak) 倍：L0 10.5 × L1 20 × L2 40 × L3 80
        #   ⇒ 级联 67 万倍，实测 L3 的特征已经是 inf，而底板读出的正是这堆垃圾。
        #   乘 (1−leak) 后每层稳态方差 = 输入方差，级联不再放大。
        #   这个增益不是新常数 —— 它就是该层自己的膜泄漏。
        inc = (incoming - incoming.mean()) * (1.0 - float(self.leak))
        V = inc + self.leak * self.V
        if SPIKE_ENABLED:
            spike = (V > self.theta).float()
            self.V = V * (1.0 - spike)
        else:
            spike = torch.zeros_like(V)
            self.V = V
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
        self.pad_id = int(pad_token_id)
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

        # 层间耦合系数：0.5 不是生物常数，而是一个旧的手填数；实验台给它三种结构口径。
        if COUPLING_MODE == 'one':
            self.coupling = 1.0
        elif COUPLING_MODE == 'inv_layers':
            self.coupling = 1.0 / max(num_layers, 1)
        else:
            self.coupling = 0.5
        # ★0.4.0.0：读出接的是「词向量 ⊕ 四层状态」共 (L+1)×H 维，所以输出头按这个宽度建
        # 实验台：HEAD_BIAS=True 时再接一列常数 1（线性读出带截距）。
        self._fdim = hidden * (num_layers + 1) + (1 if HEAD_BIAS else 0)
        self.W_out = nn.Parameter(torch.randn(vocab_size, self._fdim, generator=gen)
                                  / math.sqrt(self._fdim))
        self.W_m = nn.ParameterList([nn.Parameter(torch.zeros(hidden, vocab_size))
                                     for _ in range(num_layers)])
        # 每个记忆通道被写入的次数（浮点 buffer，不参与梯度）
        self.register_buffer('m_count', torch.zeros(num_layers, vocab_size))
        # ★0.8.0.0：精确 token 对联想记忆。pair_count[x,y] = 训练中见过的 x→y 次数；
        #   uni_count[y] = 目标侧 unigram 计数（只统计 mask 内的目标，与学习口径一致）。
        self.register_buffer('uni_count', torch.zeros(vocab_size))
        if PAIR_MEMORY:
            self.register_buffer('pair_count', torch.zeros(vocab_size, vocab_size))
            self.register_buffer('w_pair', torch.zeros(()))     # 混合权重（由似然/命中率自定）
            self.register_buffer('hit_base', torch.zeros(()))
            self.register_buffer('hit_pair', torch.zeros(()))
            self.register_buffer('seen_pair', torch.zeros(()))
        # ★0.9.x 实验台：多专家权重（base/pair/tri/uni 的顺序），初值 = 全给 base。
        _ew = torch.zeros(4)
        _ew[0] = 1.0
        self.register_buffer('expert_w', _ew)
        if TRIGRAM_MEMORY:
            # 二阶上下文哈希计数：key = (t-2, t-1) 的 2^12 进制拼接再取模。
            # 2^12 = 大于词表的最小 2 的幂 ⇒ 拼接无碰撞丢失（架构量，不是调出来的）。
            self.register_buffer('tri_count', torch.zeros(TRIGRAM_BUCKETS, vocab_size))
            self._tri_base = int(1 << (vocab_size - 1).bit_length())
        # γ 与可调层权重在 0.2.0.0 中**删除**：混合系数改由运行命中率决定（见 forward）
        self.layer_w = nn.Parameter(torch.ones(num_layers), requires_grad=False)
        self.register_buffer('hit', torch.zeros(num_layers))
        self.register_buffer('seen', torch.zeros(num_layers))
        # ★0.3.0.0：读出改用**运行充分统计量** A=Σxx^T、B=Σxy，输出头取 W=A⁻¹B。
        #   它是「到目前为止看过的全部数据的最优线性读出」，逐窗在线、纯局部、不反传，
        #   **没有学习率、没有遗忘因子、没有正则系数**（抖动只取 trace(A)/H 的机器级量级）。
        self.register_buffer('gram_A', torch.zeros(self._fdim, self._fdim))
        self.register_buffer('gram_B', torch.zeros(self._fdim, vocab_size))
        self.register_buffer('gram_n', torch.zeros(()))          # 已累计的有效位置数
        self.register_buffer('gram_n_at_solve', torch.zeros(())) # 上次求解时的位置数
        # 0.5.0.0：全局温度已删除 —— Gauss-Newton 拟合自带正确尺度，不再需要外挂标定

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
        noise = torch.randn(self.num_layers, T, self.hidden, device=dev) if NOISE_ENABLED else None
        for t in range(T):
            h = self.embedding(input_ids[:, t])
            for li, layer in enumerate(self.layers):
                inc = h
                if li > 0:
                    inc = h + self.coupling * Vs[li - 1][-1]
                V, sp = layer.step(inc, None if noise is None else noise[li, t])
                Vs[li].append(V)
                spikes[li].append(sp)
                if li < self.num_layers - 1:
                    h = V
        Vlist = [torch.stack(Vs[li], dim=1) for li in range(self.num_layers)]      # (B,T,H)
        self._buf['V'] = Vlist
        self._buf['spikes'] = [torch.stack(spikes[li], dim=1) for li in range(self.num_layers)]

        last = Vlist[-1]
        # 底板读出：V 先归一到 RMS=1、W_out 行归一 ⇒ logits 量级 O(1)，CE 在 ln(vocab) 附近起步
        # ★0.4.0.0：读出特征 = 词向量 ⊕ 四层状态（每个都归一到 RMS=1 ⇒ 尺度可比，
        #   最小二乘自动决定各尺度该占多少权重）。
        #   依据（闭式探针，500 条训练 / 40 条验证）：
        #     仅词向量 top-1 13.47% ｜ 仅 L0 11.45% ｜ 四层状态 13.11% ｜ 全部拼一起 **16.39%**
        #   —— 各种时间尺度里确实有互补信息，读哪一层不该是我手挑的，那就全给它。
        self._feat = torch.cat(
            [self.embedding(input_ids)] + [Vlist[li] for li in range(self.num_layers)], dim=-1)
        feat_n = self._feat / self._feat.pow(2).mean(dim=(0, 1), keepdim=True).sqrt().clamp(min=1e-6)
        if HEAD_BIAS:
            feat_n = torch.cat([feat_n, torch.ones(B, T, 1, device=dev, dtype=feat_n.dtype)], dim=-1)
        self._feat_n = feat_n
        base = feat_n @ F.normalize(self.W_out, dim=-1).T
        lms = []
        for li in self.read_idx:
            lm = Vlist[li] @ self.W_m[li]                        # (B,T,V)
            # ★只在「写过记忆的通道」上做除性归一。
            #   若按全词表 RMS 归一，训练初期只有几百个通道非零 ⇒ 分母被大量 0 拉小 ⇒
            #   非零通道被放大 √(V/非零数) 倍（实测到 merged 峰值 44，CE 从 8 直接炸到 25）。
            act = self.m_count[li] > 0
            if bool(act.any()):
                sub = lm[..., act]
                mu = sub.mean(dim=-1, keepdim=True)
                sd = sub.std(dim=-1, keepdim=True).clamp(min=1e-6)
                lm = torch.where(act, (lm - mu) / sd, torch.zeros_like(lm))
            else:
                lm = lm - lm.mean(dim=-1, keepdim=True)
            lms.append(lm)
        # 混合系数：每层「说对过多少次 / 说过多少次」，**扣掉随机基线 1/V**（词表给出的结构量）。
        # 归一化后 Σw=1 ⇒ 读出幅度天然有界；没有证据就闭嘴（w=0），不会像裸求和那样炸。
        _hit = self.hit[self.read_idx]
        _seen = self.seen[self.read_idx].clamp(min=1.0)
        w = (_hit / _seen - 1.0 / self.vocab_size).clamp(min=0.0)
        _ws = float(w.sum())
        w = w / _ws if _ws > 1e-9 else torch.full_like(w, 1.0 / w.numel())
        self._buf['w_mix'] = w.detach()
        merged = sum(w[k] * lms[k] for k in range(len(lms)))
        # 两个记忆专家共用的先验：运行 unigram 分布，强度 = 一个伪观测。
        _uni = self.uni_count + PAIR_PRIOR_STRENGTH
        _prior = _uni / _uni.sum().clamp(min=1e-12)
        if TRIGRAM_MEMORY or UNIGRAM_EXPERT:
            # ★0.9.x：通用多专家混合（base / pair / tri / uni）。
            #   权重在概率单纯形上，用坐标下降 + 黄金分割逐权优化：
            #   每一轮把某个权在 [0, 1−其余权和] 上搜到 NLL 最小，再轮下一个。
            #   没有学习率、没有手填权重，轮数/迭代数是计算预算。
            _experts = [base + merged]
            if PAIR_MEMORY:
                _row = self.pair_count[input_ids]
                _rs = _row.sum(dim=-1, keepdim=True)
                _p = (_row + PAIR_PRIOR_STRENGTH * _prior) / (_rs + PAIR_PRIOR_STRENGTH)
                _experts.append(torch.log(_p))
            if TRIGRAM_MEMORY:
                _ids = input_ids[0]
                _pad = torch.full((1,), int(getattr(self, 'pad_id', 0)),
                                  dtype=_ids.dtype, device=dev)
                _prev1 = torch.cat([_pad, _ids[:-1]])
                _prev2 = torch.cat([_pad, _pad, _ids[:-2]])
                _keys = ((_prev2 * self._tri_base + _prev1) % TRIGRAM_BUCKETS)
                _trow = self.tri_count[_keys]
                _trs = _trow.sum(dim=-1, keepdim=True)
                _tp = (_trow + PAIR_PRIOR_STRENGTH * _prior) / (_trs + PAIR_PRIOR_STRENGTH)
                _experts.append(torch.log(_tp).unsqueeze(0))
            if UNIGRAM_EXPERT:
                _experts.append(torch.log(_prior).view(1, 1, self.vocab_size)
                                .expand(B, T, self.vocab_size))
            _K = len(_experts)
            _ew = self.expert_w[:_K].clone()
            if labels is not None and bool((labels[:, :-1] != -100).any()):
                _mm = (labels[:, :-1] != -100)
                _E = [e[:, :-1, :][_mm] for e in _experts]
                _t = labels[:, :-1][_mm]

                def _mix_nll(wv):
                    return float(F.cross_entropy(sum(wv[i] * _E[i] for i in range(_K)), _t))

                for _round in range(max(1, MIX_ROUNDS)):
                    for _i in range(_K):
                        _rem = 1.0 - float(sum(_ew[j] for j in range(_K) if j != _i))
                        if _rem <= 0:
                            _ew[_i] = 0.0
                            continue

                        def _f(x, _i=_i):
                            _ew[_i] = x
                            return _mix_nll(_ew)

                        _ew[_i] = golden_min(_f, 0.0, _rem, iters=MIX_ITERS)
                self.expert_w[:_K] = _ew
            logits = sum(_ew[i] * _experts[i] for i in range(_K))
            self._buf['pair_w'] = float(_ew[1]) if _K > 1 else 0.0
            self._buf['expert_w'] = [round(float(x), 6) for x in _ew]
        elif PAIR_MEMORY:
            # ★0.8.0.0：精确 token 对联想记忆的读出。
            #   先验 = 运行 unigram 分布；强度 = PAIR_PRIOR_STRENGTH（一个伪观测）。
            #   p(y|x) = (count[x,y] + λ·p_uni(y)) / (row_sum[x] + λ)
            #   这是一个真正的条件分布（行和 1），不是软原型的内积。
            _row = self.pair_count[input_ids]                       # (B,T,V)
            _rs = _row.sum(dim=-1, keepdim=True)
            _p = (_row + PAIR_PRIOR_STRENGTH * _prior) / (_rs + PAIR_PRIOR_STRENGTH)
            pair_logp = torch.log(_p)
            _w = float(self.w_pair)
            _base_ctx = base + merged
            if PAIR_MIX == 'hit':
                # 用「谁把 argmax 说对」的证据决定混合权重（离散信号，优化 top-1）。
                _rb = max(float(self.hit_base) / max(float(self.seen_pair), 1.0)
                          - 1.0 / self.vocab_size, 0.0)
                _rp = max(float(self.hit_pair) / max(float(self.seen_pair), 1.0)
                          - 1.0 / self.vocab_size, 0.0)
                _w = _rp / (_rb + _rp) if (_rb + _rp) > 1e-12 else 0.0
            if labels is not None and bool((labels[:, :-1] != -100).any()):
                _mm = (labels[:, :-1] != -100)
                _b = _base_ctx[:, :-1, :][_mm]
                _g = pair_logp[:, :-1, :][_mm]
                _t = labels[:, :-1][_mm]

                def _nll(wv):
                    return float(F.cross_entropy((1.0 - wv) * _b + wv * _g, _t))

                if PAIR_MIX in ('likelihood', 'confidence'):
                    # 混合权重 w 是 1 维标量：argmin_w CE((1−w)·base_ctx + w·pair_logp)
                    # 用黄金分割搜索，无学习率、无步长、无边界手填（w 的物理范围就是 [0,1]）。
                    self.w_pair.fill_(golden_min(_nll, 0.0, 1.0))
                    _w = float(self.w_pair)
                elif PAIR_MIX == 'ce_top1':
                    # 先找 CE 最优 w；若它的 top-1 低于底板，就在 [0, w_ce] 里二分找
                    # 「top-1 不低于底板」的边界。两个判据都来自当前窗，不用验证集。
                    _w_ce = golden_min(_nll, 0.0, 1.0)
                    _base_top1 = float((_b.argmax(-1) == _t).float().mean())

                    def _top1(wv):
                        return float((((1.0 - wv) * _b + wv * _g).argmax(-1) == _t)
                                     .float().mean())

                    if _top1(_w_ce) >= _base_top1:
                        _w = _w_ce
                    else:
                        _lo, _hi = 0.0, _w_ce
                        for _ in range(16):
                            _mid = 0.5 * (_lo + _hi)
                            if _top1(_mid) >= _base_top1:
                                _lo = _mid
                            else:
                                _hi = _mid
                        _w = _lo
                    self.w_pair.fill_(_w)
                # 诊断/证据：两支各自 argmax 的命中率（不参与 likelihood 模式的 w）
                _hb = (_b.argmax(-1) == _t).sum()
                _hp = (_g.argmax(-1) == _t).sum()
                self.hit_base += float(_hb)
                self.hit_pair += float(_hp)
                self.seen_pair += float(_mm.sum())
            if PAIR_MIX == 'confidence':
                # 每个位置按该源 token 的写入次数给权重：n/(n+λ)。证据少的行走底板。
                _conf = _rs / (_rs + PAIR_PRIOR_STRENGTH)
                _wt = _w * _conf
                logits = (1.0 - _wt) * _base_ctx + _wt * pair_logp
            else:
                logits = (1.0 - _w) * _base_ctx + _w * pair_logp
            self._buf['pair_w'] = _w
        else:
            logits = base + merged
        # 缓存读出中间量：局部规则必须用**与前向同一份**张量，否则口径会漂
        self._buf['merged'] = merged
        self._buf['logits'] = logits
        self._buf['lm_flat'] = [lm[:, :T - 1, :].reshape(-1, self.vocab_size) for lm in lms]
        self._buf['scale'] = self._v105_scale if hasattr(self, '_v105_scale') else \
            (last.pow(2).mean().sqrt() / 1.0).clamp(min=1e-6) / last.pow(2).mean().sqrt().clamp(min=1e-6)

        if labels is not None:
            self._buf['labels'] = labels
            # ★aligncnt：logits[t] 对 labels[t]（标准下一 token）。
            m = (labels[:, :-1] != -100)
            if m.any():
                self._buf['ce'] = float(F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, self.vocab_size)[m.reshape(-1)],
                    labels[:, :-1].reshape(-1)[m.reshape(-1)]))
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
        # ★aligncnt：源位置 t 的目标是 labels[t]（标准下一 token）。
        tgt = labels[:, :T - 1]
        mpos = (tgt != -100) if QA_MASK else torch.ones_like(tgt, dtype=torch.bool)
        n_valid = float(max(int(mpos.sum()), 1))
        safe = tgt.clamp(min=0)
        p = F.softmax(self._buf['logits'][:, :T - 1, :], dim=-1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(2, safe.unsqueeze(-1), 1.0)
        err = onehot - p
        if QA_MASK:
            err = err * mpos.unsqueeze(-1).float()

        # ① 输出头：**运行 Gram 的最小二乘解**（不再是梯度步长）
        #   依据：闭式探针实测「同一批特征，闭式最小二乘 top-1 13.47%，
        #   而在线 LMS 只有 4.33%」⇒ 差在收敛速度，不在信息量。
        #   维护充分统计量即可：逐窗在线、纯局部、无学习率/遗忘因子/正则。
        h = self._feat_n[:, :T - 1, :].reshape(-1, self._fdim)
        _mk = mpos.reshape(-1)
        _hv = h[_mk]
        _yv = safe.reshape(-1)[_mk]
        self.gram_A += _hv.T @ _hv
        self.gram_B.index_add_(1, _yv, _hv.T)
        self.gram_n += float(_yv.numel())
        # ★0.5.0.0：在线 Gauss-Newton 精修（0.4.0.0 的最小二乘只解决了排序，没解决概率）
        #   0.4.0.0 实测：闭式解 top-1 有 16.4% 但 CE 只有 7.88 —— 最小二乘拟合 one-hot
        #   天然欠自信，直接当 logits 用会把概率做平。
        #   这里改用 CE 意义下的 Newton 步：对每个输出单元 o，
        #       w_o ← w_o + A⁻¹ x (y_o − p_o)，  A = Σ x xᵀ（累积协方差）
        #   步长完全由数据决定（A⁻¹），**没有学习率**；每个突触只读自己的误差信号和
        #   自己的输入协方差，不回传到任何隐藏层。
        #   A⁻¹ 每窗重算（640×640，成本可忽略）⇒ 连「倍增调度」这个参数也去掉了。
        _jit = float(self.gram_A.diagonal().mean()) * 1e-6        # 数值兜底，非可调超参
        _A = self.gram_A + _jit * torch.eye(self._fdim, device=self.gram_A.device)
        # A⁻¹M 直接用 solve(A, M)（数值更稳，语义相同）
        # 用当前 W 算概率残差（p − y），再乘 A⁻¹ 做一步 Newton
        # 随机 Newton 步（SNGF）：**每窗全量覆写**（0.5.0.0 的 bug 就是把上一步的 W 累加进来，
        # 于是 −A⁻¹B 每窗都被重复减一次 ⇒ 漂移 ⇒ top-1 掉到 1%）。
        # 现在：以最小二乘解为起点，用**当前窗**的残差做一次 Newton 修正，
        # Hessian 用全量累积的 A。覆写 ⇒ 无漂移、无学习率。
        _W_ls = torch.linalg.solve(_A, self.gram_B).T            # (V,F) 最小二乘解
        _W_new = _W_ls
        for _ in range(max(1, NEWTON_STEPS)):
            _p = F.softmax(_hv @ _W_new.T, dim=-1)                # 用当前 W 算概率
            _g = _p.T @ _hv - F.one_hot(_yv, self.vocab_size).float().T @ _hv
            _W_new = _W_new - torch.linalg.solve(_A, _g.T).T      # 准 Newton（A 预条件）
        if torch.isfinite(_W_new).all():
            self.W_out.data.copy_(_W_new)

        # ② 记忆列原型：指数平均，**衰减 = 该层的膜泄漏**。
        #   遗忘曲线直接等于膜时间常数 —— 不引入任何新的可调常数。
        #   （0.1.1.0 用固定 lr=0.02 + 每 10 步 ×0.99；全程 1/(n+1) 平均更不行：
        #    recency 被抹平 ⇒ 列原型共线 ⇒ 命中率≈随机。实测见 docs/FINDINGS.md。）
        for li in self.read_idx:
            _a = float(1.0 - float(self.layers[li].leak))
            Vn = Vlist[li] / Vlist[li].pow(2).mean().sqrt().clamp(min=1e-6)
            for t in range(T - 1):
                if QA_MASK and not bool(mpos[0, t]):
                    continue
                y = int(input_ids[0, t + 1].item())
                if 0 <= y < self.vocab_size:
                    _cur = self.W_m[li].data[:, y]
                    self.W_m[li].data[:, y] = (1.0 - _a) * _cur + _a * Vn[0, t]
                    self.m_count[li, y] += 1.0

        # ②b 精确 token 对记忆：纯 Hebbian 累加。只统计 mask 内的目标位置，
        #    与模型的学习口径一致；这是 bigram 表式的局部联想，不是全局统计外挂。
        if PAIR_MEMORY or TRIGRAM_MEMORY:
            _mk = torch.ones_like(mpos[0]) if PAIR_SCOPE == 'all' else mpos[0]
            if bool(_mk.any()):
                _xs = input_ids[0, :T - 1][_mk]
                _ys = input_ids[0, 1:T][_mk]
                if PAIR_MEMORY:
                    self.pair_count.index_put_(
                        (_xs, _ys), torch.ones(_xs.numel(), dtype=self.pair_count.dtype),
                        accumulate=True)
                self.uni_count.index_add_(
                    0, _ys, torch.ones(_ys.numel(), dtype=self.uni_count.dtype))
        if TRIGRAM_MEMORY:
            _ids2 = input_ids[0]
            _padv = torch.full((1,), self.pad_id, dtype=_ids2.dtype, device=_ids2.device)
            _p1 = torch.cat([_padv, _ids2[:-1]])
            _p2 = torch.cat([_padv, _padv, _ids2[:-2]])
            _keys = ((_p2 * self._tri_base + _p1) % TRIGRAM_BUCKETS)
            _mk2 = torch.ones_like(mpos[0]) if PAIR_SCOPE == 'all' else mpos[0]
            if bool(_mk2.any()):
                _xs2 = _keys[:T - 1][_mk2]
                _ys2 = _ids2[1:T][_mk2]
                self.tri_count.index_put_(
                    (_xs2, _ys2), torch.ones(_xs2.numel(), dtype=self.tri_count.dtype),
                    accumulate=True)

        # ③ 层权重 = 各自读出的**运行命中率**（离散信号：argmax 有没有说对）。
        #   为什么换掉连续相关量：v117 实测，连续量奖励的是「把分布做尖」，
        #   CE 变好而 top-1 变差；top-1 需要的是「把 argmax 做对」。
        #   命中率天然 ∈[0,1] ⇒ ±3 护栏不再需要。
        err_flat = err.reshape(-1, self.vocab_size)
        lm_flat = self._buf['lm_flat']
        tgt = safe.reshape(-1)
        msk = mpos.reshape(-1)
        for k, li in enumerate(self.read_idx):
            hit_n = float((lm_flat[k][msk].argmax(-1) == tgt[msk]).sum())
            self.hit[li] += hit_n
            self.seen[li] += float(msk.sum())
            self.layer_w.data[li] = float(self.hit[li]) / max(float(self.seen[li]), 1.0)

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
            'mix_w': [round(float(x), 6) for x in self._buf.get('w_mix', torch.zeros(0))],
            'layer_w': [round(float(x), 4) for x in self.layer_w],
            'spike_acc_mean': fr,
            'read_idx': self.read_idx,
            'mix_rule': 'hit rate above chance (1/V), normalised to sum 1',
            'qa_mask': QA_MASK,
            'pair_w': round(float(self.w_pair), 6) if PAIR_MEMORY else None,
            'pair_seen': float(self.seen_pair) if PAIR_MEMORY else 0.0,
            'pair_hit_base': float(self.hit_base) if PAIR_MEMORY else 0.0,
            'pair_hit_pair': float(self.hit_pair) if PAIR_MEMORY else 0.0,
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
    # 跨版本兼容：形状对不上的键直接跳过并报出来（多数是运行态的 state buffer，
    # 例如 layers.*.V —— 它每次 forward 开头都会被重置，本来就不该从检查点恢复）
    cur = model.state_dict()
    skip = [k for k in list(sd.keys())
            if k in cur and tuple(cur[k].shape) != tuple(sd[k].shape)]
    for k in skip:
        sd.pop(k)
    mm, uu = model.load_state_dict(sd, strict=False)
    log_and_print('[CKPT] 已加载 %s ｜ 形状不匹配已跳过 %d 个：%s ｜ missing=%d unexpected=%d'
                  % (path, len(skip), skip[:6], len(mm), len(uu)))
    return True


def print_stats(model, tok):
    st = model.stats()
    log_and_print('架构：%d 层 × %d 维 ｜ 词表 %d ｜ 参数量 %d' % (st['layers'], st['hidden'], st['vocab'], st['params']))
    log_and_print('leak（τ 派生）: %s' % st['leak'])
    log_and_print('读出层 %s ｜ 问答 mask %s' % (st['read_idx'], st['qa_mask']))
    log_and_print('混合权重 w（超随机命中率）: %s' % (st['mix_w'],))
    log_and_print('token 对混合权重 w_pair=%.4f ｜ 证据位置 %d ｜ 命中 base=%d pair=%d'
                  % (st['pair_w'] or 0.0, st['pair_seen'], st['pair_hit_base'], st['pair_hit_pair']))
    log_and_print('‖W_out‖=%.2f ｜ ‖W_m‖=%s' % (st['W_out_norm'], st['W_m_norm']))
    log_and_print('发放率（累计）: %s' % st['spike_acc_mean'])
    log_and_print('嵌入冻结 %s ｜ 隐藏层冻结 %d 层' % (st['frozen_embedding'], st['frozen_layers']))


def train_hebbian(model, tok, path, epochs=EPOCHS, limit=LIMIT_SAMPLES, val_file=None,
                  val_max=40, eval_every=0, logfile=None):
    """纯 Hebbian 训练：forward → 局部规则 → forward。无 BP、无 optimizer。
    损失只在 assistant 的 token 上计算（QA_MASK）。"""
    dev = model.device
    rows = list(iter_jsonl(os.path.join(DATA_DIR, path)))
    if TRAIN_SHUFFLE:
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
            n = int((labels[:, :-1] != -100).sum())
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
        tg = labels[:-1]          # ★aligncnt：标准下一 token
        mm = m[:-1]
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
    ap.add_argument('--fresh', action='store_true', help='不加载 brain.pt（并行实验必须用）')
    ap.add_argument('--threads', type=int, default=THREADS)
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.split:
        deterministic()
        tok = RawHebbTokenizer(os.path.join(DATA_DIR, TOK_BIN), os.path.join(DATA_DIR, TOK_JSON))
        split_by_dialogue('dsr1_v5_0001.jsonl', 'train_dsr1.jsonl', 'val_dsr1.jsonl')
    elif args.train:
        dev = setup_device()
        torch.set_num_threads(args.threads)
        deterministic(SEED)
        tok = RawHebbTokenizer(os.path.join(DATA_DIR, TOK_BIN), os.path.join(DATA_DIR, TOK_JSON))
        model = build_model(tok, dev)
        if not args.fresh and os.path.exists(os.path.join(DATA_DIR, CKPT)):
            load_model(model)
        train_hebbian(model, tok, args.train_file, epochs=args.epochs, limit=args.limit,
                      val_file=args.val_file, val_max=args.val_max, eval_every=args.eval_every)
        save_model(model, args.save or CKPT)
    else:
        main_menu()

# =================================================================================================
# 版本谱系（每次改动：复制本文件 → 改 → 改名升版，旧版保留在同目录）
#   0.8.0.0 2026-10-19  分支 aligncnt：**先修目标位移 off-by-one，再并联精确 token 对记忆**。
#                         旧版 logits[t] 对 labels[t+1]（隔一个 token）；本轮改为 logits[t] 对
#                         labels[t]。token 对计数在正确对齐后写入：源 token t → 目标 token t+1，
#                         mask 也改用 labels[t]（与损失同一个目标）。
#   0.8.0.0 2026-10-19  分支 cnt：**精确 token 对联想记忆**与慢状态记忆并联。
#                         0.6.0.0 的记忆列是 128 维软原型；bigram 表是 V×V 精确计数。
#                         本轮把后者也做成 Hebbian 表（W[x,y]+=1），读出用
#                         p(y|x)=(count+λ·p_uni)/(row+λ)，λ=一个伪观测；
#                         两支在 log 概率空间做几何混合，w 由当前窗似然一维搜索自定。
#                         不替代回流管/每层自指：W_m 慢状态支路原样保留。
#   0.1.0.0 2026-10-19  首版（四段版本号约定：a.b.c.d，不带 v 前缀）：内核替换为 RawHebbKernel（纯 Hebbian）；菜单沿用 3120012elite
#                         的 11 项结构，8/9 项因与「拒绝 BP」冲突替换为无梯度等价物；
#                         新增 12 项独立验证集评测（防泄露口径）。
#                         ★保留此版作为历史（tag = 0.1.0.0）（含已修的两个缺陷，见 0.1.1）。
#   0.6.0.0 2026-10-19  修 0.5.0.0 的漂移 bug：Newton 步必须**每窗全量覆写**，
#                         否则 −A⁻¹B 被反复累减（0.5.0.0 实测 top-1 掉到 1%）。
#   0.5.0.0 2026-10-19  ★读出改为「在线 Gauss-Newton」，全局温度删除
#                         0.4.0.0 的教训：闭式最小二乘把 top-1 拉到 16%，但 CE 仍是 7.88 ——
#                         最小二乘拟合 one-hot 天然欠自信，直接当 logits 用会把概率做平。
#                         改成 CE 意义下的 Newton 步：w_o ← w_o + A⁻¹x(y_o − p_o)，
#                         步长完全由累积协方差 A⁻¹ 决定（没有学习率），每个突触只读
#                         自己的误差信号与输入协方差，不回传任何隐藏层。
#                         A⁻¹ 每窗重算 ⇒ 连 0.4.0.0 的「倍增调度」也一并去掉。
#   0.4.0.0 2026-10-19  ★读出改为「多时间尺度 + 在线闭式最小二乘 + 自标定温度」
#                         闭式探针（500 训练 / 40 验证）实测上限：
#                           仅词向量 top-1 13.47% ｜ 仅 L0 11.45% ｜ 四层状态 13.11%
#                           词向量 ⊕ 四层状态 **16.39%**（我们在训的模型当时只有 5%）
#                         ① 读出特征 = 词向量 ⊕ 四层状态（各自 RMS=1），读哪一层不再是手挑；
#                         ② W = A⁻¹B，A/B 是运行充分统计量；证据翻倍才重解（倍增调度，零参数）；
#                         ③ 全局温度用有限差分 Newton 对齐似然（标的是数据，不是手）。
#                         三条都不含学习率/遗忘因子/正则系数。
#   0.3.0.0 2026-10-19  ★修掉「深层状态溢出」+ 读出改为在线闭式最小二乘
#                         ① 层间增益 = 本层 (1−leak)。稳态 V ≈ inc/(1−leak)，不补偿时
#                            每层放大 1/(1−leak) 倍：L0 10.5 × L1 20 × L2 40 × L3 80 ⇒ 级联
#                            67 万倍，实测 L3 特征已是 inf —— 底板读出的正是这堆垃圾，
#                            这就是 CE 卡在 unigram 水平的直接原因。
#                         ② 输出头：运行充分统计量 A=Σxx^T、B=Σxy，取 W=A⁻¹B。闭式探针证明
#                            同一批特征闭式解 top-1 13.47%、在线 LMS 只有 4.33% ⇒ 差在收敛。
#                            该规则逐窗在线、纯局部、无学习率/遗忘因子/正则系数。
#   0.2.0.0 2026-10-19  ★零魔数：删掉全部学习率常数与两个护栏
#                         ① 输出头 LMS：方向归一 + 1/(t+1) 调度（替代固定 LR_OUT）
#                         ② 记忆写入：指数平均，衰减 = 该层膜泄漏（替代 LR_MEM + 手动衰减）
#                         ③ 读出混合：各层「超随机命中率」归一（替代 γ、LR_W、±3 护栏）
#                         动机：v117 实测 CE 与 top-1 零和，旧 γ 规则奖励的是「把分布做尖」；
#                         连续相关量改成离散命中信号后，500 条 CPU 上验证 top-1 由 2.85% 提到 4.33%。
#                         剩下的自由参数只有架构与生物物理量，没有「调出来的」学习率。
#   0.1.1.0 2026-10-19  修 0.1.0.0 的两个实测缺陷：
#                         ① 记忆读出按全词表 RMS 归一 ⇒ 训练初期非零通道被放大 √N 倍
#                            （实测 merged 峰值 44、CE 8.4→24.9）。改为**只在写过记忆的通道上**
#                            做减均值/除标准差，其余通道置零（除性归一的正确口径）。
#                         ② W_out 的 LMS 改为**相对步长**（走 ‖W_out‖ 的固定比例），
#                            与梯度尺度解耦，不需要「超阈值重置」的外挂监控器。
# =================================================================================================
