# BioWaveLayer + Hebbian 外积学习（无 BP、无反向追踪、无信用分配）
# ============================================================
import json
import math
import unicodedata
import os
import random
import time
import contextlib
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, IterableDataset

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    raise ImportError("请安装 tokenizers 库: pip install tokenizers")

# ------------------------------------------------------------
# 配置参数
# ------------------------------------------------------------
VOCAB_SIZE = 800
HIDDEN_SIZE = 128
NUM_LAYERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 5.0
BATCH_SIZE = 1
EPOCHS = 20
LOG_INTERVAL = 1
TEMPERATURE = 0.7
TOP_P = 0.9
GRADIENT_ACCUMULATION_STEPS = 1
USE_AMP = False
WINDOW_OVERLAP = 0.0
LOSS_CHUNK_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'

MAX_SEQ_LEN = 200
EARLY_STOP_PATIENCE = 3
NUM_WORKERS = 0

# Ultra 参数（保持外壳兼容，不实际使用）
ULTRA_RS_SAMPLES_PER_QUESTION = 8
ULTRA_RS_MAX_NEW_TOKENS = 512
ULTRA_RL_EPOCHS = 1
ULTRA_RL_MAX_NEW_TOKENS = 512
ULTRA_RL_LEARNING_RATE = 1e-6
ULTRA_RL_BASELINE_MOMENTUM = 0.9
ULTRA_GRPO_EPOCHS = 1
ULTRA_GRPO_GROUP_SIZE = 8
ULTRA_GRPO_MAX_NEW_TOKENS = 512
ULTRA_GRPO_LEARNING_RATE = 1e-6
ULTRA_GRPO_INNER_STEPS = 1
ULTRA_GRPO_CLIP_EPSILON = 0.2
ULTRA_GRPO_OLD_POLICY_PENALTY = 0.01

FREEZE_LAYERS = 0
SEED_FREEZE = 42
FREEZE_EMBEDDING = True   # 固定为 True（词向量不学）
SEED_EMBEDDING = 42
USE_RESIDUAL = True

# 全局随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# BioWaveLayer 超参数
# ------------------------------------------------------------
BIO_THRESHOLD_MIN = 0.5
BIO_THRESHOLD_MAX = 2.0
BIO_REFRACTORY_CHOICES = [2, 3, 4, 5]
# v88：A_global/f_global/f_local 三个手填常量从这里删除，改为由下方「v88 节律常数推导块」
#       从生物学常数推出（theta 8Hz / gamma 40Hz / TAU_M_MS 与节律周期之比）。
#       BIO_A_LOCAL_RANGE 保留：逐神经元相对分布不变，整体幅度由 BIO_A_LOCAL_SCALE 承担。
BIO_A_LOCAL_RANGE = (0.05, 0.3)
BIO_SIGMA_NOISE_RANGE = (0.01, 0.05)
BIO_NEURON_ATP_CAPACITY_RANGE = (5.0, 20.0)
BIO_NEURON_ATP_RECOVERY_RANGE = (0.1, 0.5)
BIO_ATP_PER_SPIKE = 1.0
BIO_RELEASE_PROB_RANGE = (0.3, 0.7)
BIO_VESICLES_CHOICES = [1, 2, 3, 4]
BIO_SYN_ATP_CAPACITY_RANGE = (2.0, 8.0)
BIO_SYN_ATP_RECOVERY_RANGE = (0.05, 0.3)
BIO_ATP_PER_VESICLE = 0.5
BIO_LEARNING_RATE = 0.01       # Hebbian 更新率

# ---- v51.5 P 家族：生物学时间常数块（模块级）----
TAU_M_MS = 10.0             # 膜时间常数（文献 10-20ms，取低端）
TAU_STP_MS = 500.0          # 短期突触可塑性时间常数（文献百 ms ~ 秒）
DT_MS = 1.0
LEAK = math.exp(-DT_MS / TAU_M_MS)         # ≈0.9048374180359595
M_DECAY = math.exp(-DT_MS / TAU_STP_MS)    # ≈0.9980019986673331

# ---- v88：节律常数推导块（零手填系数；频率抄文献频段，幅度=膜时间常数与节律周期之比）----
# 频段：海马 theta 4-8Hz（取上端 8Hz）；皮层 gamma 30-80Hz（取下端 40Hz）。
# 换算：f_cycles_per_step = F_HZ * DT_MS / 1000（DT_MS=1ms 时间步）。
F_THETA_HZ = 8.0
F_GAMMA_HZ = 40.0
TAU_THETA_MS = 1000.0 / F_THETA_HZ      # 125ms = theta 周期
TAU_GAMMA_MS = 1000.0 / F_GAMMA_HZ      # 25ms = gamma 周期
BIO_THRESHOLD_MID = (BIO_THRESHOLD_MIN + BIO_THRESHOLD_MAX) / 2      # = 1.25
BIO_F_GLOBAL = F_THETA_HZ * DT_MS / 1000.0        # = 0.008 周期/步（8Hz）
BIO_F_LOCAL_CYC = F_GAMMA_HZ * DT_MS / 1000.0     # = 0.04 周期/步（40Hz）
# 退化 range：f_local 由「逐神经元随机频率」改为「全体同一 gamma 频率」。
#   uniform_(v, v) 与 uniform_(lo, hi) 的 RNG 消耗逐位相同（已实测 n=8/128 两种情况）
#   ⇒ 后续 psi/sigma_noise/release_prob/n_vesicles/syn_*/weight 的初值一位不动，
#      state_dict 键集合不变（[INIT-EQ] 逐位比对口径保持）。
BIO_F_LOCAL_RANGE = (BIO_F_LOCAL_CYC, BIO_F_LOCAL_CYC)
# 幅度推导：膜电位对周期 T_r 的正弦驱动只能保留 TAU_M_MS/T_r 的比例（膜积分低通），
#   故驱动幅度 = 阈值中点 × (TAU_M_MS / T_rhythm_MS)：动力学可积的比值，不是手填系数。
BIO_A_GLOBAL = BIO_THRESHOLD_MID * (TAU_M_MS / TAU_THETA_MS)      # 1.25 × 10/125 = 0.1
BIO_A_LOCAL_BASE = BIO_THRESHOLD_MID * (TAU_M_MS / TAU_GAMMA_MS)  # 1.25 × 10/25 = 0.5
# 逐神经元缩放：把保留的 uniform(0.05,0.3) 相对分布同比例放大到 A_local_base 量级
BIO_A_LOCAL_SCALE = BIO_A_LOCAL_BASE / BIO_A_LOCAL_RANGE[1]       # = 0.5/0.3 = 5/3
BIO_ATP_PER_WEIGHT_UPDATE = 0.1

BIO_PRUNE_THRESHOLD = 200
BIO_PRUNE_FACTOR = 0.95
BIO_W_MAX = 1.0
BIO_W_MIN = 0.0

BIO_AUTO_SCALE_ENABLED = True
BIO_AUTO_SCALE_RATIO = 0.5
BIO_AUTO_SCALE_MOMENTUM = 0.99

SCHEME = 'hebb'   # 保留变量，不再使用

# ------------------------------------------------------------
# 日志
# ------------------------------------------------------------
_LOG_FILE = None

def init_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open("training_log.txt", "a", encoding='utf-8')
        _LOG_FILE.write(f"\n--- 新会话开始于 {datetime.now()} ---\n")
        _LOG_FILE.flush()

_LOG_FLUSH_COUNTER = 0
_LOG_FLUSH_INTERVAL = 50

def log_and_print(*args, **kwargs):
    import sys
    global _LOG_FLUSH_COUNTER
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(str(arg) for arg in args) + end
    sys.stdout.write(msg)
    sys.stdout.flush()
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg)
        _LOG_FLUSH_COUNTER += 1
        if _LOG_FLUSH_COUNTER >= _LOG_FLUSH_INTERVAL:
            _LOG_FILE.flush()
            _LOG_FLUSH_COUNTER = 0

def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None

# ------------------------------------------------------------
# 工具
# ------------------------------------------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFKC', text).lower()

SPECIAL_TOKENS_LIST = ['<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|user|>', '<|assistant|>', '<|end|>']
_v64_note = "v64_F: 稳态缩放信号源改为发放率 EMA (layer_outputs->layer_firings) + [ACC]/[FR] 诊断插桩；基座 v60_S3"

_v70_note = "v70_L: 基座 v64_F；结构性改动=解绑输出头(W_out 独立 (vocab,H)，logits=V@W_out_norm.T，W_out 用与 embedding 对齐同规则 lr*0.1 更新)；回流管/读取段/t+1/稳态缩放不动"

_v74_note = "v74_T: 基座 v70_L；单点改动=_fire 膜电位更新里 incoming 先减群内均值(对比度)；leak/theta/V_rest/回流管/解码器全部不动"

_v81_note = "v81_GG: 基座 v74_T；新增海马体稀疏索引支路（W_dg 固定 top-k=20 稀疏化 → dg_sparse 入 logits + W_index_out 对齐），回流管 M/V 动力学一字不动；唯一偏离=__init__ 块尾补 manual_seed(42) 以保证 layers/M 初值与 v74_T 逐位一致"
_v82_note = "v82_II: 基座 v81_GG（海马体稀疏索引在位）；单点删除 hebbian_update 的 embedding 对齐段 ⇒ 码本真冻结 ⇒ dg_sparse 与训练状态无关；W_out 对齐(目标 V)、W_index_out 对齐(目标 dg_sparse)、回流管 M/t+1/倒流/V 动力学一字不动"
_v83_note = "v83_KK: 基座 v82_II（embedding 冻结、索引支路在位）；单点改动=dg_sparse 加时间累积 dg_context[t]=0.9*dg_context[t-1]+dg_sparse[t] 后按步归一化，logits 与 W_index_out 对齐目标同时换成 dg_context；回流管 M/t+1/倒流/V 动力学/W_out 对齐/dg_sparse 逐 token 计算一字不动；注：_dg_decay=0.9 是指令允许的简化近似，理论值 exp(-1/10)=0.904837；register_buffer(_dg_context_ema) 按指令注册但 forward 用局部 ema，未消费该 buffer"
_v87_note = "v87_UUa: 基座 v83_KK；本轮唯一改动=hebbian_update 的 W 更新行把 DA 换成 abs(DA)（保留强度、去掉方向符号）；回流管 M 结构/t+1 写入/倒流读取/embedding 与 W_dg 冻结/W_out 与 W_index_out 对齐/软边界/稳态缩放/修剪/动力学 a 一字未动；DA 仍计算供 [DIAG]/[ACC]"
_v88_note = "v88_VV: 基座 v87_UUa（abs(DA) 不回归）；本轮唯一改动=compute_V_rest 喂给的四个数的**来源**（公式一字不动）：f_global 0.1→8Hz×DT_MS/1000=0.008、f_local uniform(0.5,3.0)→40Hz×DT_MS/1000=0.04（退化 range，RNG 位等、键集合不变）、A_global 0.3→1.25×(TAU_M_MS/TAU_THETA_MS)=0.1、A_local uniform(0.05,0.3)×(5/3)⇒有效 uniform(0.0833,0.5)；compute_V_rest 公式/_fire/leak/theta/回流管 M 结构/t+1 写入/倒流读取/W_out 与 W_index_out 对齐/软边界/稳态缩放/修剪/embedding 与 W_dg 冻结/动力学 a 一字未动；零新增手填系数（频率=文献 theta/gamma 频段，幅度=膜时间常数与节律周期之比）"
V104_LR_OUT = 0.100   # v104：输出头 LMS 的学习率（绝对步长）

V111_LR_M = 0.020        # 每层关联矩阵的 Hebbian 步长（与 v110 同值，便于跨轮对照）
V111_LR_G = 0.050        # γ 的局部步长（与 v110 同值）
V111_LR_W = 0.010        # 层权重的局部步长（指令给定）
V111_W_CLAMP = 3.0       # 层权重的**数值护栏**（±3）：不设 0 地板 ⇒ 塌缩必须被看见，而不是被夹住
V111_DECAY_RATE = 0.990  # 指令给定：每 V111_DECAY_EVERY 窗乘一次
V111_DECAY_EVERY = 10
# slow(默认，指令) | old(v110 原样 self._mod_decay/窗=0.998002) | none(不衰减，极值对照)
V111_DECAY = os.environ.get('V111_DECAY', 'slow')
# ===== v112 =====
V112_TAU_MULT = float(os.environ.get('V112_TAU_MULT', '2'))    # 每深一层膜时间常数 ×MULT
V112_TAU_CAP = float(os.environ.get('V112_TAU_CAP', '100'))    # τ 上限（步）；0=不设上限
V112_GAIN = os.environ.get('V112_GAIN', 'var')                 # var|dc|raw（见文件头 A/B）
V112_NO_L0_READ = os.environ.get('V112_L0_READ', '0') != '1'   # True=读出只走 L1-L3
# ===== v113 =====
# 读出按 ‖w‖₂ 归一（能量保持）。1=开（默认）；0=复现 v112 的裸求和（A/B 用）
V113_NORM = os.environ.get('V113_NORM', '1') != '0'
# ===== v114 =====
# 问答 mask：1=只在该位置 labels ≠ -100（即 assistant 的 token）上学习与计数；0=回到 v113 行为
V114_MASK = os.environ.get('V114_MASK', '1') != '0'
# ===== v115 =====
# L0（快通道）独立的膜时间常数，单位=步；0=沿用基座 LEAK（τ=10 步）
V115_L0_TAU = float(os.environ.get('V115_L0_TAU', '0'))

_v112_note = "v112: 基座 v111。三处改动（老通道/配平/W 冻结/W_out LMS 不动）：①写入只给 L1-L3，W_m_vocab[0] 恒零冻结；②L1-L3 膜时间常数按层增长 τ_li=TAU_M_MS·MULT^li（默认 10/20/40/80 步，上限 T/2=100），L0 不动；③读出只走 L1-L3（L0 读出切断，w0 成死参数保留 1.0 不参与任何计算）。★技术判断 A：τ 必须 ≤ T/2——指令的 3^li 会让 L3 的 τ=270>T=200，状态退化成整窗均值、失去位置分辨率；★技术判断 B：λ 变大方差放大约 4.4 倍，用 sqrt((1−λ²)/(1−λ0²)) 做方差保持的输入补偿（零手填），否则深层直接饱和放电；V112_GAIN=raw 即字面指令（不补偿），dc 为保直流增益。"

_v113_note = "v113: 基座 v112（三处物理改动全保留：L0 记忆冻结/τ 按层增长/读出只走 L1-L3）。本轮唯一改动=读出合并的能量保持归一：Σw·lm 之后除以 ‖w‖₂=sqrt(Σw²)（零手填，只用现有 w）。依据：三层读出逐位置 cos=0.93~0.95、互为线性重构 R²=0.91~0.94；七种子集合并里 L2+L3(7.3225)/L1+L3(7.3236) 都赢过三层全加(7.3477)，而三层全加的音量扫描最优 s=0.75(7.3121)、两层最优 s=1.00 ⇒ 差额主要是音量过大而非纯冗余。V113_NORM=0 复现 v112 裸求和（位等自证）。"

_v114_note = "v114: 基座 v113（三处物理改动 + 读出按 ‖w‖₂ 归一，全部保留）。本轮唯一改动=问答 mask：hebbian_update 的目标从 input_ids 改为 labels（-100 处不学），误差乘 mask、分母改有效位置数、记忆写入逐位置跳过、CE 用 ignore_index=-100。动机：原实现无视 dataset 的 -100，会把 user 的提问也当目标学，与 forward 显示的 masked CE 口径不一致。V114_MASK=0 回到 v113。"

_v115_note = "v115: 基座 v114（问答 mask 全保留）。本轮唯一改动=L0 快通道的膜时间常数可独立设置：V115_L0_TAU（步，0=沿用 LEAK 即 τ=10）。依据：真实语料上 L0 记忆开比关好 0.52 CE，而 τ=10 的 V_L0 是最近 10 步 embedding 的滑窗均值，用它做逐 token 模板匹配天生对不准；τ→1 才对应「当前是哪个词」。会改变第 0 层 leak 的初值（INIT-EQ 只在这一项不同，属预期）。"

# ------------------------------------------------------------
# 分词器（外壳不变）
# ------------------------------------------------------------
class FastBPE:
    def __init__(self, initial_vocab_size: int = VOCAB_SIZE):
        self.initial_vocab_size = initial_vocab_size
        self.special_tokens_list = SPECIAL_TOKENS_LIST.copy()
        self._tokenizer = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.user_token_id = None
        self.assistant_token_id = None
        self.end_token_id = None
        self._vocab_size_cache = None

    def train(self, texts, verbose=False):
        normalized = [normalize_text(t) for t in texts]
        tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.initial_vocab_size,
            special_tokens=self.special_tokens_list,
            min_frequency=2, show_progress=verbose)
        tokenizer.train_from_iterator(normalized, trainer=trainer)
        self._tokenizer = tokenizer
        self._update_special_ids()

    def _update_special_ids(self):
        if self._tokenizer is None:
            return
        vocab = self._tokenizer.get_vocab()
        self._vocab_size_cache = self._tokenizer.get_vocab_size()
        self.bos_token_id = vocab.get('<|bos|>', 0)
        self.eos_token_id = vocab.get('<|eos|>', 1)
        self.pad_token_id = vocab.get('<|pad|>', 2)
        self.unk_token_id = vocab.get('<|unk|>', 3)
        self.user_token_id = vocab.get('<|user|>', 4)
        self.assistant_token_id = vocab.get('<|assistant|>', 5)
        self.end_token_id = vocab.get('<|end|>', 6)

    def encode(self, text):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未训练或未加载")
        norm_text = normalize_text(text)
        ids = self._tokenizer.encode(norm_text).ids
        return [tid if tid < self._vocab_size_cache else self.unk_token_id for tid in ids]

    def decode(self, ids):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未训练或未加载")
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未初始化")
        dir_name, file_name = os.path.split(path)
        file_base = os.path.splitext(file_name)[0]
        tokenizer_path = os.path.join(dir_name, f"{file_base}_tokenizer.json")
        self._tokenizer.save(tokenizer_path)
        state = {'initial_vocab_size': self.initial_vocab_size,
                 'special_tokens_list': self.special_tokens_list}
        torch.save(state, path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            self.initial_vocab_size = data.get('initial_vocab_size', VOCAB_SIZE)
            self.special_tokens_list = data.get('special_tokens_list', SPECIAL_TOKENS_LIST.copy())
        except Exception as e:
            log_and_print(f"加载 tokenizer 元数据失败 ({path}): {e}")
            return False
        dir_name, file_name = os.path.split(path)
        file_base = os.path.splitext(file_name)[0]
        tokenizer_path = os.path.join(dir_name, f"{file_base}_tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.pre_tokenizer = Whitespace()
                self._tokenizer.decoder = decoders.BPEDecoder()
                self._update_special_ids()
                log_and_print(f"成功从 {tokenizer_path} 加载 tokenizer")
                return True
            except Exception as e:
                log_and_print(f"加载 tokenizer.json 失败: {e}")
                return False
        return False

    @property
    def vocab_size(self):
        if self._tokenizer is None:
            return len(self.special_tokens_list)
        if self._vocab_size_cache is None:
            self._vocab_size_cache = self._tokenizer.get_vocab_size()
        return self._vocab_size_cache

# ------------------------------------------------------------
# 学习率调度器（保留但不用）
# ------------------------------------------------------------
class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_steps=None, eta_min=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / (self.warmup_steps + 1e-8)
        elif self.decay_steps is not None:
            t = step - self.warmup_steps
            progress = min(t / self.decay_steps, 1.0)
            factor = self.eta_min + (1 - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            t = step - self.warmup_steps
            factor = self.eta_min + (1 - self.eta_min) * (self.warmup_steps / (self.warmup_steps + t)) ** 0.5
        return [base_lr * factor for base_lr in self.base_lrs]

# ------------------------------------------------------------
# RMSNorm（保留）
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if x.numel() == 0:
            return x
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# ------------------------------------------------------------
# 数据集（外壳不变）
# ------------------------------------------------------------
class StreamingJSONLDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                 epoch=0, is_val=False, assigned_indices=None):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.is_val = is_val
        self.assigned_indices = assigned_indices
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.user_id = tokenizer.user_token_id
        self.assistant_id = tokenizer.assistant_token_id
        self.end_id = tokenizer.end_token_id
        self.pad_id = tokenizer.pad_token_id

    def _encode(self, text):
        return self.tokenizer.encode(text)

    def _tokenize_conversation(self, convs):
        kept_pairs = []
        token_count = 0
        id_cache = {}
        i = len(convs) - 1
        while i >= 1:
            assistant_turn = convs[i]
            user_turn = convs[i - 1]
            if assistant_turn.get('role') == 'assistant' and user_turn.get('role') == 'user':
                pair = [user_turn, assistant_turn]
                pair_tokens = 0
                for turn in pair:
                    content = turn.get('content', '')
                    ids = id_cache.get(content)
                    if ids is None:
                        ids = self._encode(content)
                        id_cache[content] = ids
                    pair_tokens += len(ids) + 2
                kept_pairs.insert(0, pair)
                token_count += pair_tokens
                if token_count + 2 >= self.max_seq_len:
                    break
                i -= 2
            else:
                i -= 1
        convs = [turn for pair in kept_pairs for turn in pair]
        tokens = []
        assistant_mask = []
        for turn in convs:
            role = turn['role']
            content = turn['content']
            content_ids = id_cache.get(content)
            if content_ids is None:
                content_ids = self._encode(content)
                id_cache[content] = content_ids
            if role == 'user':
                tokens.append(self.user_id)
                assistant_mask.append(False)
                tokens.extend(content_ids)
                assistant_mask.extend([False] * len(content_ids))
                tokens.append(self.end_id)
                assistant_mask.append(False)
            elif role == 'assistant':
                tokens.append(self.assistant_id)
                assistant_mask.append(False)
                tokens.extend(content_ids)
                assistant_mask.extend([True] * len(content_ids))
                tokens.append(self.end_id)
                assistant_mask.append(True)
        tokens = [self.bos_id] + tokens
        assistant_mask = [False] + assistant_mask
        tokens.append(self.eos_id)
        assistant_mask.append(True)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        labels = [lbl if mask else -100 for lbl, mask in zip(labels, assistant_mask[1:])]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(42 + self.epoch * 1000 + worker_id)
        buffer = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx % num_workers != worker_id:
                    continue
                if self.assigned_indices is not None and line_idx not in self.assigned_indices:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    convs = data.get('conversations', [])
                    if not convs:
                        continue
                    sample = self._tokenize_conversation(convs)
                except (json.JSONDecodeError, KeyError):
                    continue
                if self.is_val:
                    yield sample
                else:
                    buffer.append(sample)
                    if len(buffer) >= 1000:
                        rng.shuffle(buffer)
                        while buffer:
                            yield buffer.pop()
        if not self.is_val and buffer:
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

class TextFileDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                 epoch=0, is_val=False, window_overlap=0.0):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.is_val = is_val
        self.window_overlap = window_overlap
        self._token_ids = None
        self._num_windows = 0

    def _ensure_tokenized(self):
        if self._token_ids is not None:
            return
        cache_file = self.file_path + f".len{self.max_seq_len}.norm.ids.bin"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(self.file_path):
            total_ids = os.path.getsize(cache_file) // 4
            self._token_ids = np.memmap(cache_file, dtype=np.int32, mode='r', shape=(total_ids,))
            step = max(1, int(self.max_seq_len * (1 - self.window_overlap)))
            window_size = self.max_seq_len + 1
            self._num_windows = max(0, (total_ids - window_size) // step + 1)
            return
        log_and_print(f"首次 tokenize 缓存生成中...")
        t0 = time.time()
        vocab_size = self.tokenizer.vocab_size
        unk_id = self.tokenizer.unk_token_id
        chunk_size = 4096
        total_ids = 0
        buf = []
        with open(cache_file, 'wb') as out, open(self.file_path, 'r', encoding='utf-8') as f:
            def flush():
                nonlocal total_ids
                if not buf:
                    return
                normalized = [normalize_text(t) for t in buf]
                encodings = self.tokenizer._tokenizer.encode_batch(normalized)
                chunk_ids = []
                for enc in encodings:
                    chunk_ids.extend(tid if tid < vocab_size else unk_id for tid in enc.ids)
                ids = np.array(chunk_ids, dtype=np.int32)
                ids.tofile(out)
                total_ids += ids.size
                buf.clear()
            for line in f:
                if not line.strip():
                    continue
                buf.append(line.rstrip('\n\r') + '\n')
                if len(buf) >= chunk_size:
                    flush()
            flush()
        elapsed = time.time() - t0
        if total_ids == 0:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            self._token_ids = None
            self._num_windows = 0
            return
        log_and_print(f"tokenize 完成：{total_ids:,} tokens，耗时 {elapsed:.1f}s")
        self._token_ids = np.memmap(cache_file, dtype=np.int32, mode='r', shape=(total_ids,))
        step = max(1, int(self.max_seq_len * (1 - self.window_overlap)))
        window_size = self.max_seq_len + 1
        self._num_windows = max(0, (total_ids - window_size) // step + 1)

    def __iter__(self):
        self._ensure_tokenized()
        if self._token_ids is None or self._num_windows == 0:
            return
        rng = random.Random(42 + self.epoch * 1000)
        step = max(1, int(self.max_seq_len * (1 - self.window_overlap)))
        indices = list(range(self._num_windows))
        if not self.is_val:
            rng.shuffle(indices)
        for idx in indices:
            start = idx * step
            end = start + self.max_seq_len + 1
            window = self._token_ids[start:end]
            input_ids = torch.tensor(np.array(window[:self.max_seq_len]), dtype=torch.long)
            labels = torch.tensor(np.array(window[1:self.max_seq_len + 1]), dtype=torch.long)
            yield input_ids, labels

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        pad_id = self.pad_id
        valid_batch = []
        for inp, lbl in batch:
            try:
                if inp.numel() > 0 and lbl.numel() > 0:
                    valid_batch.append((inp, lbl))
            except:
                continue
        if not valid_batch:
            return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)
        input_ids_list, labels_list = zip(*valid_batch)
        max_len = max(len(ids) for ids in input_ids_list)
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        for inp, lbl in zip(input_ids_list, labels_list):
            pad_len = max_len - len(inp)
            if pad_len > 0:
                inp = F.pad(inp, (0, pad_len), value=pad_id)
                lbl = F.pad(lbl, (0, pad_len), value=-100)
            mask = (inp != pad_id).long()
            padded_inputs.append(inp)
            padded_labels.append(lbl)
            attention_masks.append(mask)
        return torch.stack(padded_inputs), torch.stack(padded_labels), torch.stack(attention_masks)

def make_collate_fn(pad_id):
    return Collator(pad_id)

def _seed_worker(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

# ------------------------------------------------------------
# BioWaveLayer
# ------------------------------------------------------------
_LEARNING_DISABLED_DURING_GENERATE = False

def _make_generator(seed_offset):
    g = torch.Generator()
    g.manual_seed(42 + int(seed_offset))
    return g

def _draw_choices(choices, size, generator):
    k = len(choices)
    idx = torch.randint(0, k, size, generator=generator)
    tbl = torch.tensor(choices, dtype=torch.long)
    return tbl[idx]

class BioWaveLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer_idx=-1,
                 threshold_range=(BIO_THRESHOLD_MIN, BIO_THRESHOLD_MAX),
                 refractory_choices=BIO_REFRACTORY_CHOICES,
                 release_prob_range=BIO_RELEASE_PROB_RANGE,
                 vesicles_choices=BIO_VESICLES_CHOICES,
                 neuron_atp_capacity_range=BIO_NEURON_ATP_CAPACITY_RANGE,
                 neuron_atp_recovery_range=BIO_NEURON_ATP_RECOVERY_RANGE,
                 syn_atp_capacity_range=BIO_SYN_ATP_CAPACITY_RANGE,
                 syn_atp_recovery_range=BIO_SYN_ATP_RECOVERY_RANGE,
                 atp_per_spike=BIO_ATP_PER_SPIKE,
                 atp_per_vesicle=BIO_ATP_PER_VESICLE,
                 atp_per_weight_update=BIO_ATP_PER_WEIGHT_UPDATE,
                 learning_rate=BIO_LEARNING_RATE):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_idx = layer_idx
        g = _make_generator(layer_idx if layer_idx >= 0 else 0)

        self.register_buffer('theta', torch.empty(output_dim).uniform_(*threshold_range, generator=g))
        self.register_buffer('refractory_len', _draw_choices(refractory_choices, (output_dim,), g))
        self.A_global = BIO_A_GLOBAL
        self.f_global = BIO_F_GLOBAL
        self.register_buffer('phi', torch.empty(output_dim).uniform_(0, 2 * math.pi, generator=g))
        self.register_buffer('A_local', torch.empty(output_dim).uniform_(*BIO_A_LOCAL_RANGE, generator=g) * BIO_A_LOCAL_SCALE)
        self.register_buffer('f_local', torch.empty(output_dim).uniform_(*BIO_F_LOCAL_RANGE, generator=g))
        self.register_buffer('psi', torch.empty(output_dim).uniform_(0, 2 * math.pi, generator=g))
        self.register_buffer('sigma_noise', torch.empty(output_dim).uniform_(*BIO_SIGMA_NOISE_RANGE, generator=g))
        self.register_buffer('neuron_atp_capacity', torch.empty(output_dim).uniform_(*neuron_atp_capacity_range, generator=g))
        self.register_buffer('neuron_atp_recovery', torch.empty(output_dim).uniform_(*neuron_atp_recovery_range, generator=g))

        self.register_buffer('release_prob', torch.empty(output_dim, input_dim).uniform_(*release_prob_range, generator=g))
        self.register_buffer('n_vesicles', _draw_choices(vesicles_choices, (output_dim, input_dim), g))
        self.register_buffer('syn_atp_capacity', torch.empty(output_dim, input_dim).uniform_(*syn_atp_capacity_range, generator=g))
        self.register_buffer('syn_atp_recovery', torch.empty(output_dim, input_dim).uniform_(*syn_atp_recovery_range, generator=g))
        self.register_buffer('eligibility', torch.zeros(output_dim, input_dim))

        self.weight = nn.Parameter(torch.empty(output_dim, input_dim).uniform_(-0.4, 0.4, generator=g))

        self.w_max = BIO_W_MAX
        self.w_min = -self.w_max   # 允许 [-w_max, +w_max]
        self.register_buffer('inactive_counter', torch.zeros(output_dim, input_dim, dtype=torch.long))

        self.register_buffer('running_incoming_std', torch.ones(1))
        self.register_buffer('incoming_scale', torch.ones(1))

        self.atp_per_spike = atp_per_spike
        self.atp_per_vesicle = atp_per_vesicle
        self.atp_per_weight_update = atp_per_weight_update
        self.lr = learning_rate
        self.leak = nn.Parameter(torch.tensor(LEAK))  # v51.5 P1: exp(-DT_MS/TAU_M_MS)
        # ★v115：只给 L0（快通道）一个更短的时间常数 —— τ→1 才能抓住「当前是哪个词」
        if layer_idx == 0 and V115_L0_TAU > 0:
            self.leak = nn.Parameter(torch.tensor(float(math.exp(-DT_MS / V115_L0_TAU))))
            self.leak_v115 = float(math.exp(-DT_MS / V115_L0_TAU))
        # v112：深层膜时间常数按层增长（L0 保持 τ=TAU_M_MS=10 步）。
        #   τ_li = TAU_M_MS·MULT^li，并设上限 V112_TAU_CAP 步（默认 T/2=100）：
        #   τ 超过窗长时 λ^T 仍有 ~0.5 权重 ⇒ V(t) 变成整窗均值、逐位置读出失去分辨率。
        _tau112 = TAU_M_MS * (V112_TAU_MULT ** max(layer_idx, 0))
        if V112_TAU_CAP > 0:
            _tau112 = min(_tau112, V112_TAU_CAP)
        self.leak_v112 = float(math.exp(-DT_MS / _tau112))
        # 方差保持的输入缩放：λ 变大时 V 的方差按 1/sqrt(1−λ²) 放大，这里按同一因子缩小输入，
        #   零手填——分子分母都由 λ 与 LEAK 算出。
        self.in_scale_v112 = float(math.sqrt((1.0 - self.leak_v112 ** 2) / (1.0 - LEAK ** 2)))

        # 稳态突触缩放（Turrigiano 1998）
        self.register_buffer('avg_firing_rate', torch.tensor(0.1))
        self.target_firing_rate = 0.1     # 目标发放率
        self.scaling_rate = 0.001         # 缩放学习率（慢速）

    def init_state(self, device, batch_size=1):
        return {
            'V': torch.zeros(batch_size, self.output_dim, device=device),
            'refractory': torch.zeros(batch_size, self.output_dim, dtype=torch.long, device=device),
            'neuron_atp': self.neuron_atp_capacity.clone().unsqueeze(0).expand(batch_size, -1).contiguous().to(device),
            'syn_atp': self.syn_atp_capacity.clone().unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device),
            'fired': torch.zeros(batch_size, self.output_dim, dtype=torch.bool, device=device),
            'overshoot': torch.zeros(batch_size, self.output_dim, device=device),
            'incoming': torch.zeros(batch_size, self.output_dim, device=device),
            'firing_history': torch.zeros(batch_size, self.output_dim, device=device),
        }

    def compute_V_rest(self, t, B, device):
        t_tensor = torch.tensor(float(t), device=device)
        global_wave = self.A_global * torch.sin(2 * math.pi * self.f_global * t_tensor + self.phi)
        local_wave = self.A_local * torch.sin(2 * math.pi * self.f_local * t_tensor + self.psi)
        noise = torch.randn(self.output_dim, device=device) * self.sigma_noise
        V_rest = (global_wave + local_wave + noise).unsqueeze(0)
        return V_rest.expand(B, -1)

    def _fire(self, state, t, B, device, external_current=None):
        if external_current is not None:
            state['incoming'] = external_current
        V_rest = self.compute_V_rest(t, B, device)
        # v74-T：对比度——incoming 减去本层神经元群内均值（逐样本、逐时间步），
        # 使所有神经元共享的那部分电流不再进入 V 的累积；V_rest 与 leak 一字不动
        incoming = state['incoming']
        incoming_centered = incoming - incoming.mean(dim=-1, keepdim=True)
        # v112：L1-L3 用更长的膜时间常数（L0 一字不动；MULT=1 时 leak_v112≡LEAK）
        _lk112 = self.leak if self.layer_idx <= 0 else self.leak_v112
        state['V'] = V_rest + incoming_centered + _lk112 * state['V']

        can_fire = (state['refractory'] == 0)
        enough_atp = (state['neuron_atp'] >= self.atp_per_spike)
        fired = (state['V'] >= self.theta.unsqueeze(0)) & can_fire & enough_atp
        state['fired'] = fired
        state['overshoot'] = torch.where(
            fired, state['V'] / self.theta.unsqueeze(0), torch.zeros_like(state['V']))
        state['V'] = torch.where(fired, torch.zeros_like(state['V']), state['V'])

        state['refractory'] = torch.clamp(state['refractory'] - 1, min=0)
        state['refractory'][fired] = self.refractory_len.unsqueeze(0).expand(B, -1)[fired]
        state['neuron_atp'][fired] -= self.atp_per_spike
        state['firing_history'] = 0.99 * state['firing_history'] + 0.01 * fired.float()
        return fired

    def step(self, x_t, state, t=0):
        B = x_t.shape[0]
        return self._fire(state, t, B, x_t.device, external_current=x_t)

    def forward(self, x, time_mask=None, return_state=False, t_start=0):
        B, T, _ = x.shape
        device = x.device
        state = self.init_state(device, B)
        outputs = torch.zeros(B, T, self.output_dim, device=device)
        for tl in range(T):
            fired = self._fire(state, t_start + tl, B, device, external_current=x[:, tl, :])
            outputs[:, tl, :] = fired.float()
            state['incoming'] = torch.zeros_like(state['incoming'])
            state['neuron_atp'] = torch.clamp(state['neuron_atp'] + self.neuron_atp_recovery.unsqueeze(0),
                                              max=self.neuron_atp_capacity.unsqueeze(0))
            state['syn_atp'] = torch.clamp(state['syn_atp'] + self.syn_atp_recovery.unsqueeze(0),
                                           max=self.syn_atp_capacity.unsqueeze(0))
        if time_mask is not None:
            outputs = outputs * time_mask.unsqueeze(-1).float()
        if return_state:
            return outputs, state
        return outputs

# ------------------------------------------------------------
# 层间电流传递（保留超射放大、源端归一化、囊泡释放）
# ------------------------------------------------------------
def propagate_current(all_fired, all_layers, all_overshoot, update_inactive=False, rng_seed=None):
    B = all_fired[0].shape[0]
    device = all_fired[0].device
    incoming = [torch.zeros_like(f) for f in all_fired]

    for li in range(len(all_layers) - 1):
        src_layer = all_layers[li]
        dst_layer = all_layers[li + 1]
        src_fired = all_fired[li].float()
        src_over = all_overshoot[li]
        w = dst_layer.weight
        rp = dst_layer.release_prob
        n_ves = dst_layer.n_vesicles
        D_dst = dst_layer.output_dim
        D_src = src_layer.output_dim
        max_ves = int(n_ves.max().item())

        if rng_seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(rng_seed + li)
            rand = torch.rand(B, D_dst, D_src, max_ves, device=device, generator=g)
        else:
            rand = torch.rand(B, D_dst, D_src, max_ves, device=device)
        ves_idx = torch.arange(max_ves, device=device).view(1, 1, -1)
        ves_mask = (ves_idx < n_ves.unsqueeze(-1))
        release_mask = (rand < rp.unsqueeze(-1).unsqueeze(0)) & ves_mask.unsqueeze(0)

        ves_current = w.unsqueeze(-1).unsqueeze(0) / n_ves.unsqueeze(-1).clamp(min=1).unsqueeze(0)
        ves_current = ves_current.expand(B, -1, -1, max_ves)
        actual_ves = ves_current * release_mask.float()
        actual_per_syn = actual_ves.sum(dim=-1)

        src_fired_exp = src_fired.unsqueeze(1)
        actual_per_syn = actual_per_syn * src_fired_exp

        total_from_src = actual_per_syn.sum(dim=1, keepdim=True)
        # 真正的超射守恒：总量 = 超射量，与权重无关
        # 权重只决定"分配比例"，不决定"总量"
        desired_release = src_over * src_fired   # 严格 = 超射量
        src_output_budget = desired_release.unsqueeze(1)
        safe_total = torch.where(total_from_src > 1e-8, total_from_src, torch.ones_like(total_from_src))
        normalized = actual_per_syn / safe_total * src_output_budget

        dst_incoming = normalized.sum(dim=-1)
        incoming[li + 1] = dst_incoming

        if update_inactive:
            received = (actual_per_syn > 1e-6).long()
            active_per_syn = received.max(dim=0).values
            dst_layer.inactive_counter = torch.where(
                active_per_syn > 0,
                torch.zeros_like(dst_layer.inactive_counter),
                dst_layer.inactive_counter + 1)
    return incoming

# ------------------------------------------------------------
# WaveSNN：Hebbian 版
# ------------------------------------------------------------
class WaveSNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers,
                 pad_token_id=0, use_residual=True,
                 freeze_seed=42, freeze_layers=0,
                 freeze_embedding=False, seed_embedding=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.use_residual = use_residual
        self.freeze_seed = freeze_seed
        self.freeze_layers = freeze_layers
        self.freeze_embedding = freeze_embedding
        self.seed_embedding = seed_embedding
        self.eos_token_id = None
        self.bos_token_id = None
        self.unk_token_id = None
        self.user_token_id = None
        self.assistant_token_id = None
        self.end_token_id = None

        self._gen_t = 0

        from collections import deque
        self._acc_history = deque(maxlen=20)   # 最近 20 次正确率

        # 词向量层：冻结，随机初始化
        torch.manual_seed(seed_embedding)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embedding.weight, std=1.0)
        self.embedding.weight.requires_grad = False
        # v70-L：独立输出头（不冻结，靠 Hebbian 更新）。取数走 seed_embedding 流，
        # 保证 embedding/layers/M 的初值与 v64_F 逐位一致
        self.W_out = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.W_out, std=1.0)
        # v110：token-id 关联记忆（H,V）。列 = 该 token 的「原型状态」，Hebbian 写入、无 BP。
        #   ★放在 W_out 初始化之后，避免多消耗 RNG 改变既有初值（INIT-EQ 会让位等失败）。
        # v111：M5「每层自指」——每层一个 vocab 关联矩阵（H,V），列 = 该 token 在该层的原型状态。
        #   zeros 初始化不消耗 RNG ⇒ 与基座 v110 的公共张量仍逐位相同（INIT-EQ 可查）。
        self.W_m_vocab = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, vocab_size)) for _ in range(num_layers)])
        self.v111_gamma = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # 各层读出的加权系数：初始等权 1.0，由 CE 概率误差局部学（无 BP）
        self._v111_layer_weights = nn.Parameter(torch.ones(num_layers), requires_grad=False)
        torch.manual_seed(42)
        # ========== v81-GG：海马体稀疏索引（CLS：快、稀疏、pattern-separated，不经 V 积分） ==========
        # 输入=当前 token 的冻结随机 embedding；输出=H_index 维里只有 k_index 个非零
        # W_dg 固定不学（dentate gyrus 的连线在发育期定好），W_index_out 不冻结、靠 Hebbian 更新
        self.H_index = hidden_size           # 索引维数与 hidden 相同
        self.k_index = 20                    # 稀疏度：每步只有 20 个活跃神经元
        torch.manual_seed(seed_embedding + 1000)   # 独立种子，与 embedding 不共享
        self.W_dg = nn.Parameter(torch.empty(self.H_index, hidden_size))
        nn.init.normal_(self.W_dg, std=1.0)
        self.W_dg.requires_grad = False      # pattern separator 不学习
        torch.manual_seed(42)
        self.W_index_out = nn.Parameter(torch.empty(vocab_size, self.H_index))
        nn.init.normal_(self.W_index_out, std=1.0)
        # ========== v83-KK：dg_context 的时间累积状态 ==========
        # decay 从短期突触可塑性常数派生：TAU_STP_MS=500 步，为在 T=200 窗内留足上下文深度取 /50=10 步
        # ⇒ 理论值 exp(-1/10)=0.904837；指令允许用 0.9 作简化近似（本行按指令写 0.9，runner 里并列打印两者）
        self._dg_decay = 0.9
        self.register_buffer('_dg_context_ema', torch.zeros(self.H_index))
        torch.manual_seed(42)                # ★ 块尾复位（唯一偏离）：保证 layers/M 初值与 v74_T 逐位一致

        # BioWaveLayer 们
        frozen_count = min(freeze_layers, num_layers)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = BioWaveLayer(hidden_size, hidden_size, layer_idx=i)
            # 保留可训练属性（虽然不用 BP，但 Hebbian 要访问 .weight.data）
            self.layers.append(layer)
        for i in range(frozen_count):
            for p in self.layers[i].parameters():
                p.requires_grad = False

        # 冻结层内权重（不再被 BP 更新，只被 Hebbian 更新）
        for layer in self.layers:
            layer.weight.requires_grad = False
            layer.leak.requires_grad = False

        self.forbidden_ids = {pad_token_id}

        # ---- v58 柱化：每层一套回指张量，内含 N 个宽 D=64 的柱 ----
        # M_li (N, D, D, D)：柱内 k/i/j 三轴同索引切片；cross_li (N, 2, D, D)：环形左/右邻
        # 参数量 O(H^3/D^2)：H=512 时 4 层共 33.5MB（P2 全局稠密需 2.15GB）
        self._a_pool = [torch.ones(1) for _ in range(num_layers)]
        self._b0 = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_layers)])
        self._W_a = nn.ParameterList([nn.Parameter(torch.eye(hidden_size) * 0.01) for _ in range(num_layers)])
        self.D = 64
        if hidden_size % self.D != 0:
            raise ValueError("v58 要求 hidden_size 是 D=64 的整数倍，收到 %d" % hidden_size)
        self.N = hidden_size // self.D
        # v60-S3：k 轴由 D 扩到 3D，分段 [左邻柱 0:D | 本柱 D:2D | 右邻柱 2D:3D]
        # cross 不再是独立通道（无自己的学习/稳定机制），全部共用 M 的 sat/tanh/max_ema
        self.K = 3 * self.D
        self.modulation_tensors = nn.ParameterList([
            nn.Parameter(torch.zeros(self.N, self.K, self.D, self.D))
            for _ in range(num_layers)
        ])
        self.cross_tensors = nn.ParameterList()
        self._theta_cross = []
        # 初始化：小的随机值
        for M in self.modulation_tensors:
            nn.init.normal_(M, std=0.01)

        # 神经调质状态
        self._step_count = 0
        self._serotonin_beta = 0.001
        self._mod_decay = M_DECAY  # v51.5 P1: exp(-DT_MS/TAU_STP_MS)
        self._mod_lr = 0.01        # 调制张量的写入学习率

        # 自监督熵状态（调质用）
        self._last_entropy_norm = 1.0
        self._entropy_ema = 1.0
        self._ht_ema = 0.5
        # M3：能量守恒读出状态
        self._readout_energy = None
        # C2：max EMA 状态（逐层）
        self._max_ema = [1.0] * num_layers

    _v515_note = "P1=v50 E2 逐字 + 膜 leak 0.9→exp(-1/10)=0.9048374、M 衰减 0.9→exp(-1/500)=0.9980020（写入段两处 0.9 字面量改读 _mod_decay）；tanh 界仍手填 sat=2.0"

    _v515_note = "P2=P1 + tanh 饱和点 sat=本层 V_post 的 RMS（clamp min 1e-3），替代手填 2.0"

    def _make_states(self, device, batch_size):
        return [layer.init_state(device, batch_size) for layer in self.layers]

    def _core_step(self, token_emb_t, states, t, rng_seed=None):
        """单步推进。states 是 BioWaveLayer 状态列表。"""
        B = token_emb_t.shape[0]
        device = token_emb_t.device

        # 第 0 层注入外部电流
        states[0]['incoming'] = token_emb_t

        # 各层发放
        all_fired = []
        all_overshoot = []
        for li, layer in enumerate(self.layers):
            fired = layer._fire(states[li], t, B, device)
            all_fired.append(fired)
            all_overshoot.append(states[li]['overshoot'])

        # 层间电流传递
        if len(self.layers) > 1:
            incoming = propagate_current(all_fired, self.layers, all_overshoot,
                                         update_inactive=False, rng_seed=rng_seed)
            for li in range(1, len(self.layers)):
                layer = self.layers[li]
                raw = incoming[li]
                if BIO_AUTO_SCALE_ENABLED:
                    with torch.no_grad():
                        target = layer.theta.mean() * BIO_AUTO_SCALE_RATIO
                        # v112：λ 变大时按「方差保持」（默认）/「直流增益保持」缩放输入；
                        #   raw = 不补偿（= 指令字面，深层会饱和，作对照）。
                        if V112_GAIN == 'var':
                            target = target * layer.in_scale_v112
                        elif V112_GAIN == 'dc':
                            target = target * ((1.0 - layer.leak_v112) / (1.0 - LEAK))
                        raw_std = raw.std().clamp(min=1e-6)
                        if layer.running_incoming_std.item() == 1.0:
                            layer.running_incoming_std.fill_(raw_std)
                        layer.running_incoming_std.mul_(BIO_AUTO_SCALE_MOMENTUM).add_(
                            (1 - BIO_AUTO_SCALE_MOMENTUM) * raw_std)
                        # s 保持 tensor，全程 GPU：raw * s 自动 broadcast
                        s = target / layer.running_incoming_std.clamp(min=1e-6)
                        layer.incoming_scale.copy_(s)
                        raw = raw * s
                states[li]['incoming'] = raw

        # ATP 恢复
        for li, layer in enumerate(self.layers):
            states[li]['neuron_atp'] = torch.clamp(
                states[li]['neuron_atp'] + layer.neuron_atp_recovery.unsqueeze(0),
                max=layer.neuron_atp_capacity.unsqueeze(0))
            states[li]['syn_atp'] = torch.clamp(
                states[li]['syn_atp'] + layer.syn_atp_recovery.unsqueeze(0),
                max=layer.syn_atp_capacity.unsqueeze(0))

        # ===== v100-B：M 读出当预测电流（纯正预测编码：预测来自模型自己的记忆）=====
        # 读出与 hebbian_update 的倒流读取同构：query = 顶层当前活动（本步的 V），
        #   blocks[b,c,i,j] = Σ_k h_k[b,c,k]·M_li[c,k,i,j] ⇒ M 预测的共激活 [out_i, in_j]
        # 投影到 V 空间（用户允许的投影）：pred[b,c,i] = Σ_j blocks[b,c,i,j]·pre[b,c,j]
        #   pre = 本层当前输入（propagate 已填好，即 t+1 步的输入，与 M 的 t+1 写入语义对齐）；
        #   i=out 索引 ⇒ pred 与 states[li]['V'] 同形 (B,H)。
        # 电流形式与 v99-B 同构：I = (1−LEAK)·(pred − V)，一次向预测的弛豫（不是覆盖）。
        #   LEAK = exp(−DT_MS/TAU_M_MS)，(1−LEAK)=膜一步内对新输入的开放度，零手填。
        # ★与 v99-B 的本质区别：pred 来自 M（训练/推理同一计算）而非 t+1 真实 embedding
        #   ⇒ 不存在「拿着答案做题」；推理时 M 即冻结的记忆，预测照常发生。
        # ★刻度：pred 范数归一到当前 V（per-sample、detach）。M 读出的绝对尺度从未校准到
        #   V（它在基座里只当过 δ 的 direction），归一是让弛豫幅度受控（≤2(1−LEAK)‖V‖），
        #   预测的方向/pattern 完整保留。与刀 A 的 mod_scale 同一思想。
        # ★no_grad：M 是 nn.Parameter，包住防止 200 步前向把整张图留在显存里。
        if not getattr(self, '_v100_inj_off', False):
            with torch.no_grad():
                _B100 = token_emb_t.shape[0]
                if t == 0:
                    self._v100_inj_win = [0.0] * len(self.layers)
                    self._v100_inj_cnt = 0
                    self._v103_cur = []              # v103：本窗最后一层预测电流（按 t 追加，各 (B,H)）
                for _li100 in range(1, len(self.layers)):
                    _M100 = self.modulation_tensors[_li100]                 # (N,3D,D,D)
                    _q100 = states[-1]['V'].view(_B100, self.N, self.D)     # 顶层当前活动
                    _hk100 = torch.cat([torch.roll(_q100, 1, dims=1), _q100,
                                        torch.roll(_q100, -1, dims=1)], dim=2)   # (B,N,3D)
                    _blk100 = torch.einsum('bck,ckij->bcij', _hk100, _M100)      # (B,N,D,D)
                    _pre100 = states[_li100]['incoming'].view(_B100, self.N, self.D)
                    _pred100 = torch.einsum('bcij,bcj->bci', _blk100, _pre100).reshape(_B100, -1)
                    _vn100 = states[_li100]['V'].norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    _pred100 = _pred100 * (_vn100 / _pred100.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                    _I100 = (1.0 - LEAK) * (_pred100 - states[_li100]['V'])
                    states[_li100]['V'] = states[_li100]['V'] + _I100
                    self._v100_inj_win[_li100] += float(_I100.norm())
                    # v103：把最后一层的电流按 t 记下来（供 hebbian_update 直接当更新方向）
                    #   _core_step 拿到的 x_t 已是 (B,H)，T 维度不在这里 ⇒ 用列表逐 t 追加，
                    #   hebbian_update 里再 stack 成 (B,T,H)。
                    if _li100 == len(self.layers) - 1:
                        self._v103_cur.append(_I100.detach())
                self._v100_inj_cnt += 1
        hidden_last = states[-1]['V']
        return hidden_last, all_fired

    def forward(self, input_ids, labels=None, return_per_sample_loss=False,
                position_offset=None, attention_mask=None, rng_seed=None,
                return_layer_states=False, return_atp=False):
        if input_ids.numel() == 0:
            return torch.empty(0, 0, self.vocab_size, device=input_ids.device)

        B, T = input_ids.shape
        device = input_ids.device
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)

        # 冻结词向量
        x = self.embedding(input_ids)  # (B, T, H)

        states = self._make_states(device, B)
        hidden_outputs = torch.zeros(B, T, self.hidden_size, device=device)

        # 记录（用于 Hebbian）
        layer_inputs = [torch.zeros(B, T, self.hidden_size, device=device) for _ in self.layers]
        layer_outputs = [torch.zeros(B, T, self.hidden_size, device=device) for _ in self.layers]
        layer_firings = [torch.zeros(B, T, self.hidden_size, device=device) for _ in self.layers]
        need_atp = return_atp or (labels is not None)
        if need_atp:
            layer_atp = [torch.zeros(B, T, self.hidden_size, device=device)
                         for _ in self.layers]

        for t in range(T):
            # 记录本时间步各层的输入
            layer_inputs[0][:, t, :] = x[:, t, :]
            for li in range(1, len(self.layers)):
                layer_inputs[li][:, t, :] = states[li]['incoming']

            hidden_last, _ = self._core_step(x[:, t, :], states, t, rng_seed=rng_seed)
            hidden_outputs[:, t, :] = hidden_last

            # 记录本时间步各层的膜电位
            for li in range(len(self.layers)):
                layer_outputs[li][:, t, :] = states[li]['V']
                layer_firings[li][:, t, :] = states[li]['firing_history']
            if need_atp:
                for li in range(len(self.layers)):
                    atp_ratio = (states[li]['neuron_atp']
                                 / self.layers[li].neuron_atp_capacity.unsqueeze(0).clamp(min=1e-6))
                    layer_atp[li][:, t, :] = atp_ratio

        # 解码（v70-L 解绑输出头）：V @ 独立 W_out 的归一化行^T；不再复用 embedding，
        # 也去掉 h_dir 归一化与 energy_now 能量缩放（按指令）
        # ---- v81-GG：海马体索引（token → 固定投影 → top-k 稀疏 → 单位范数；x 与 hebbian 的 target_emb 同源） ----
        dg_raw = x @ self.W_dg.T                        # (B, T, H_index) 固定随机投影
        topk_vals, topk_idx = dg_raw.abs().topk(self.k_index, dim=-1)
        dg_sparse = torch.zeros_like(dg_raw)
        dg_sparse.scatter_(-1, topk_idx, dg_raw.gather(-1, topk_idx))   # 保留原符号
        dg_sparse = dg_sparse / dg_sparse.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # ---- v83-KK：时间累积（衰减 EMA，逐 batch 独立；每个 forward 内 ema 从零开始）----
        dg_context = torch.zeros_like(dg_sparse)                    # (B, T, H_index)
        ema = torch.zeros(dg_sparse.shape[0], dg_sparse.shape[2],
                          device=dg_sparse.device, dtype=dg_sparse.dtype)
        for _t in range(dg_sparse.shape[1]):
            ema = self._dg_decay * ema + dg_sparse[:, _t, :]
            dg_context[:, _t, :] = ema
        # 归一化：让每个时间步的 dg_context 与 dg_sparse 同量级（单位范数）
        dg_context = dg_context / dg_context.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        W_out_norm = F.normalize(self.W_out, dim=1)
        W_index_norm = F.normalize(self.W_index_out, dim=1)
        # v83-KK：索引项换成携带上下文累积的 dg_context（dg_sparse 仍在位、仍被计算，只是不再直接入 logits）
        # ===== v105：支路尺度配平（把 V 支路的「音量」压到与索引支路同量级）=====
        # v104 地板诊断：完整 9.415 / 去掉索引支路 9.704 / **去掉 W_out 支路 7.627**。
        #   机制：V 范数≈28、W_out_norm 行是单位向量 ⇒ V 支 logits ~O(10)；dg_context 是单位范数
        #   ⇒ 索引支路 ~O(1)。10 倍噪声盖住 1 倍有效信号。
        # 配平：scale = mean‖dg_context‖ / ‖V‖（逐 (b,t)，只由实时范数算出、零手填常数），
        #   clamp(max=1.0) 只允许压、不允许放大（放大等于把噪声再放大）。
        _v105_norm_v = hidden_outputs.norm(dim=-1, keepdim=True).clamp(min=1e-6)     # (B,T,1)
        _v105_tgt = dg_context.norm(dim=-1, keepdim=True).mean(dim=(0, 1), keepdim=True)  # (1,1,1)
        _v105_scale = (_v105_tgt / _v105_norm_v).clamp(max=1.0)                      # (B,T,1)
        logits = (hidden_outputs * _v105_scale) @ W_out_norm.T + dg_context @ W_index_norm.T
        # ===== v111：M5「每层自指」——每层一个 vocab 读出头，逐位置读出后按层权重叠加 =====
        #   ★对草案的修正：草案用 layer_outputs[li].mean(dim=1)（整窗均值）⇒ 每个位置得到同一组
        #     logits，只能学词频先验（v108/v110 两次踩过）。这里用逐位置状态 V_li(t)。
        #   每层读出各自 RMS 归一到 1（层内音量与状态范数解耦）⇒ 层权重只表达「信不信这一层」。
        # v112：**读出只走 L1-L3**（L0 的记忆冻结在零，读它没有意义；V112_L0_READ=1 可保留作对照）
        _v112_idx = list(range(1, len(self.layers))) if V112_NO_L0_READ else list(range(len(self.layers)))
        _v112_lm_list = []
        for _li112 in _v112_idx:
            _lm112 = layer_outputs[_li112] @ self.W_m_vocab[_li112]                # (B,T,V)
            _lm112 = _lm112 / _lm112.pow(2).mean().sqrt().clamp(min=1e-6)          # 每层 RMS=1
            _v112_lm_list.append(_lm112)
        _v111_lm = _v112_lm_list[0] * self._v111_layer_weights[_v112_idx[0]]
        for _k112 in range(1, len(_v112_lm_list)):
            _v111_lm = _v111_lm + _v112_lm_list[_k112] * self._v111_layer_weights[_v112_idx[_k112]]
        # ★v113：按权重向量的 L2 范数归一 —— 三层近共线（逐位置 cos 0.93~0.95）时，
        #   裸求和会把音量放大 Σw 倍，而有效信息只有一份；除以 ‖w‖₂ 让「加一层」不再白送音量。
        if V113_NORM:
            _w113 = self._v111_layer_weights[_v112_idx]
            _v111_lm = _v111_lm / _w113.pow(2).sum().sqrt().clamp(min=1e-6)
        if not getattr(self, '_v112_off', False):
            logits = logits + self.v111_gamma * _v111_lm
        if labels is not None:
            self._v111_lm_buf = _v111_lm.detach()
            self._v112_lm_list_buf = [x.detach() for x in _v112_lm_list]
            self._v112_idx_buf = list(_v112_idx)
            self._v111_rms_v = float(_v111_lm.detach().pow(2).mean().sqrt())
        self._v105_scale_buf = _v105_scale.detach()      # 供 hebbian_update 用同一特征做 LMS
        self._v105_scale_mean = float(_v105_scale.mean())
        # v104：训练前向（有 labels）把 logits 存下来，供 hebbian_update 算输出头误差。
        #   ★必须存：train_from_scratch 走的是 hebbian_update(cached=True)，不会自己再跑 forward。
        if labels is not None:
            self._v104_logits = logits.detach()
            # ★v114：把带 -100 的 labels 存下来，hebbian_update 用它做问答 mask
            self._v114_labels = labels.detach() if torch.is_tensor(labels) else None

        # ---- 自监督熵：无标签，调质用 ----
        with torch.no_grad():
            _logits = logits.reshape(-1, self.vocab_size)
            _p = F.softmax(_logits, dim=-1)
            _H = -(_p * torch.log(_p + 1e-8)).sum(dim=-1).mean()
            H_norm = (_H / math.log(self.vocab_size)).clamp(0.0, 1.0)
            # v50 E1: 仅存张量引用（capture 期写入 graph 池，replay 每步刷新），EMA 移出 forward
            self._last_entropy_norm = H_norm

        # ---- 计算多巴胺信号（奖励预测误差的简化版）----
        if labels is not None:
            with torch.no_grad():
                # 取预测的 argmax 和真实 label 比较
                pred = logits[:, :-1, :].argmax(dim=-1)   # (B, T-1)
                true = labels[:, 1:]                       # (B, T-1)
                mask = (true != -100)
                if mask.any():
                    match = (pred == true) & mask
                    # DA = 正确率 - 随机基线
                    acc = match.float().sum() / mask.float().sum()
                    acc_val = acc.item()
                    if len(self._acc_history) >= 5:
                        baseline = sum(self._acc_history) / len(self._acc_history)
                    else:
                        baseline = 1.0 / self.vocab_size
                    # ===== v99-A：DA 从「绝对差值」改为「相对改进率」=====
                    # 旧口径：DA = acc − baseline，均值只有 4.7e-4（v88_VV 参照 3.9e-3），
                    #   乘上 lr(0.01) 与 coact(0.12) 后单窗 δ 只有 5.6e-7 ⇒ W 相对变化 3.5e-4/3epoch。
                    # 新口径：DA = (acc − b) / b，b = max(baseline, 1/vocab_size)。
                    #   除以 1/vocab_size（随机瞎猜的均匀基线）⇒ 量纲变成「相对随机基线的改进倍数」。
                    #   ★诚实标注：模型 acc 几乎恒为 0 ⇒ DA 会几乎恒为 −1，等价于把 |DA| 从 4.7e-4
                    #   提到 1.0（约 2100 倍）。它此时不再是「奖励误差」，而是一把恒定强度的门。
                    #   这正是本刀要测的：**在这个量级下，随机方向会不会被放大成爆炸。**
                    _b99 = max(baseline, 1.0 / self.vocab_size)
                    self._last_da = (acc_val - _b99) / _b99
                    self._v99_baseline = float(_b99)
                    self._acc_history.append(acc_val)
                else:
                    self._last_da = 0.0
        else:
            self._last_da = 0.0

        # v47 F2：带 labels 的 forward 把 states 缓存下来，供 hebbian_update 复用
        if labels is not None:
            self._cached_layer_inputs = layer_inputs
            self._cached_layer_outputs = layer_outputs
            self._cached_layer_firings = layer_firings
            self._cached_layer_atp = layer_atp

        if return_layer_states and return_atp:
            return logits, hidden_outputs, layer_inputs, layer_outputs, layer_firings, layer_atp
        if return_layer_states:
            return logits, hidden_outputs, layer_inputs, layer_outputs, layer_firings
        return logits

    # ---- v51.5 诊断通道：按 epoch 汇总的标量轨迹（sat / p99thr / knorm_dev / sm_max ...）----
    def _v515_add(self, key, val):
        st = self.__dict__.setdefault('_v515_stat', {})
        st.setdefault(key, []).append(float(val))

    def _v515_report(self):
        st = self.__dict__.get('_v515_stat', {})
        out = []
        for k in sorted(st):
            vs = st[k]
            if not vs:
                out.append(f"{k}=-")
                continue
            if k == 'trig':
                out.append(f"{k}=sum({sum(vs):.0f}/{len(vs)})")
            else:
                out.append(f"{k}=mean({sum(vs)/len(vs):.5f})/max({max(vs):.5f})/last({vs[-1]:.5f})")
            st[k] = []
        return " ".join(out) if out else "-"

    def hebbian_update(self, input_ids, lr=None, cached=False):
        """16 根回指管子：写入（t+1 打标记）+ 读取（最后一层激活调制）。无 BP。"""
        if lr is None:
            lr = BIO_LEARNING_RATE

        with torch.no_grad():
            # 多巴胺由训练循环里带 labels 的那次 forward 算好；
            # 本方法内部的 forward（无 labels）会把 _last_da 重置为 0，必须先缓存。
            _da = getattr(self, '_last_da', 0.0)

            if cached:
                layer_inputs = self._cached_layer_inputs
                layer_outputs = self._cached_layer_outputs
                layer_firings = self._cached_layer_firings
                layer_atp = self._cached_layer_atp
            else:
                _, _, layer_inputs, layer_outputs, layer_firings, layer_atp = self.forward(
                    input_ids, return_layer_states=True, return_atp=True)

            B, T = input_ids.shape
            target_emb = self.embedding(input_ids)   # (B, T, H)
            DA = _da
            h_last_all = layer_outputs[-1]           # (B, T, H) 最后一层激活

            # ========== 写入阶段（V2 向量化：batched einsum + 几何衰减闭式累积） ==========
            # 逐步 tanh/leak 无法并入闭式，只在整体写入后各应用一次（语义差异，报告注明）
            for li in range(1, len(self.layers)):
                M_li = self.modulation_tensors[li]           # (N, D, D, D)
                h_in = layer_inputs[li][:, :T - 1, :]        # (B, T-1, H)
                h_out = layer_outputs[li][:, :T - 1, :]
                # v58：k/i/j 三轴按同一神经元索引切成 N 段，柱内稠密、柱间块对角
                Nc, Dc = self.N, self.D
                activity_batch = torch.einsum('btpi,btpj->btpij',
                                              h_out.view(B, T - 1, Nc, Dc),
                                              h_in.view(B, T - 1, Nc, Dc)).mean(dim=0)  # (T-1,N,D,D)
                if li == 1:
                    targets = target_emb[:, 1:, :].mean(dim=0)                # (T-1,H)
                else:
                    targets = layer_outputs[li - 1][:, 1:, :].mean(dim=0)
                targets = targets.view(T - 1, Nc, Dc)
                # v60-S3：t+1 标记按同样的 3 段拼接（环形 roll），与读取端 k 轴一一对应
                targets_k = torch.cat([torch.roll(targets, 1, dims=1), targets,
                                       torch.roll(targets, -1, dims=1)], dim=2)  # (T-1,N,3D)
                denom = targets_k.pow(2).sum(dim=-1) + 1.0                     # (T-1,N) 整条 k 的范数
                targets_n = targets_k / denom.unsqueeze(-1)
                n_w = T - 1
                # M_final = decay^(T-1)*M_init + Σ_t decay^(T-2-t)*write_t（闭式，与 P2 同形）
                exponents = torch.arange(n_w - 1, -1, -1, device=M_li.device, dtype=torch.float32)
                decay_powers = torch.exp2(exponents * math.log(self._mod_decay, 2))  # v51.5 P1: 读 τ 派生 decay
                w_t = targets_n * decay_powers.view(-1, 1, 1)                  # (T-1,N,D)
                delta_M = torch.einsum('tck,tcij->ckij', w_t, activity_batch)  # (N,D,D,D)
                assert delta_M.shape == M_li.data.shape, (delta_M.shape, M_li.data.shape)
                M_li.data = (self._mod_decay ** n_w) * M_li.data + self._mod_lr * delta_M  # v51.5 P1
                V_post_rms = layer_outputs[li].pow(2).mean().sqrt()   # 当前层 V 的 RMS
                sat = V_post_rms.clamp(min=1e-3)                      # 动态饱和点，不手填
                M_li.data = torch.tanh(M_li.data / sat) * sat
                self._v515_add('sat', sat)
                if not hasattr(self, '_M_norm_ema'):
                    self._M_norm_ema = [1.0] * len(self.layers)
                norm_now = M_li.data.norm().item()
                self._M_norm_ema[li] = (0.999 ** n_w) * self._M_norm_ema[li] + (1 - 0.999 ** n_w) * norm_now
                self._max_ema[li] = (0.99 ** n_w) * self._max_ema[li] + (1 - 0.99 ** n_w) * M_li.data.abs().max().item()

            # ========== 读取阶段：最后一层激活经过回指管子 ==========
            for li, layer in enumerate(self.layers):
                if li == 0:
                    continue

                M_li = self.modulation_tensors[li]   # (H, H, H)

                # 资格迹更新（V1 向量化：T 维 batched einsum）
                h_pre = layer_inputs[li]
                h_post = layer_outputs[li]
                act_batch = torch.einsum('bti,btj->btij', h_pre, h_post).mean(dim=0)  # (T,H,H)
                scale_batch = (h_pre.norm(dim=-1).mean(dim=0)
                               * h_post.norm(dim=-1).mean(dim=0)).clamp(min=1e-6)     # (T,)
                act_batch = act_batch / scale_batch.view(-1, 1, 1)
                # v97：删除资格迹累积（跨时间信用分配 ≈ 模糊 BP）；act_batch 保留为瞬时共激活

                # 读取回指管子：h_last × M_li 求和（v58：柱内块对角 + 环形邻柱 cross）
                h_last_mean = h_last_all.mean(dim=(0, 1))    # (H,)
                h_seg = h_last_mean.view(self.N, self.D)     # (N, D)

                # 调制 = Σ_k h_seg[c,k] × M_li[c,k,i,j] → (N,D,D)，再装回块对角 (H,H)
                # ORGaNICs 动态 a
                a_old = self._a_pool[li].to(M_li.device)
                y_activity = h_last_mean.pow(2).mean()
                a_new = a_old + 0.01 * (-a_old + 1.0 * self._b0[li].pow(2) + torch.einsum('ij,j->i', self._W_a[li], y_activity * a_old)).mean()
                a_new = F.softplus(a_new)
                self._a_pool[li] = a_new.detach().cpu()
                if self._step_count % 100 == 0:
                    print(f"[DIAG] a_pool L{li}={a_new.item():.4f}")
                # v60-S3：k 向量 = [左邻柱段 | 本柱段 | 右邻柱段]，逐元素点积，不再是邻柱标量均值
                h_k = torch.cat([torch.roll(h_seg, 1, dims=0), h_seg,
                                 torch.roll(h_seg, -1, dims=0)], dim=1)         # (N, 3D)
                blocks = torch.einsum('ck,ckij->cij', h_k, M_li)                # (N,D,D)
                Hh = h_last_mean.numel()
                modulation_full = torch.zeros(Hh, Hh, device=M_li.device, dtype=M_li.dtype)
                cidx = torch.arange(self.N, device=M_li.device)
                modulation_full.view(self.N, self.D, self.N, self.D)[cidx, :, cidx, :] = blocks
                modulation = modulation_full / torch.sqrt(a_new)

                # v97：删除 ach_mask（均值恒 0.5 的纯噪声）与旧标量 ATP 门；改用「共激活 × 突触后 ATP」
                # 更新
                # ===== v87-UUa：DA 绝对值化（保留「该不该学」的强度，去掉「往哪学」的符号）=====
                # 修正后诊断：DA 是全局标量、符号在 38% 的相邻窗对里翻转 ⇒ δ 长程方向=掷硬币。
                # 本档令 δ ∝ |DA|：方向完全由 modulation（回指管 M 读出）决定，DA 只当强度门。
                # DA 仍照旧计算（[DIAG]/[ACC] 口径不变），只是不再向 δ 提供符号。
                # ===== v97-P1：δ 的方向 = 真实共激活（pre ⊗ post 时间平均）× 突触后 ATP =====
                coact_mean = act_batch.mean(dim=0)                        # (H,H)
                post_atp = layer_atp[li].mean(dim=(0, 1))                 # (H,)
                atp_gate_matrix = post_atp.unsqueeze(0).clamp(0.1, 1.0)   # (1,H)→(H,H) 广播
                # ===== v100-A：预测编码的惊讶度（surprise minimization）=====
                # 回流管 M 根据之前的记忆，预测「当前共激活应该是 modulation」；
                # 预测误差（惊讶度）= 真实共激活 − M 的预测。Error=0 ⇒ 见怪不怪 ⇒ 不更新。
                # ★索引方向核对（实现细节）：M 写入用 h_out⊗h_in ⇒ [out,in]；读取端
                #   act_batch 用 h_pre⊗h_post ⇒ coact_mean 是 [in,out]。相差一次转置，
                #   直接相减是「拿向量和它的转置比」⇒ 先对 modulation 取 .t() 归为同向。
                # mod_scale = ‖coact‖/‖modulation‖：把预测归一到真实共激活的量级（尺度自由），
                #   .detach() 表示这是刻度不是学习信号。‖modulation‖=0（M 未写过）时
                #   mod_scaled≡0 ⇒ surprise≡coact_mean，自动退化为 v99_A 口径。
                mod_scale = (coact_mean.norm() / modulation.norm().clamp(min=1e-6)).detach()
                mod_scaled = modulation.t() * mod_scale
                surprise = coact_mean - mod_scaled
                # ===== v102：纯正预测编码 —— 误差回到「预测下一状态」的空间 =====
                # ★先纠一个前提（实现前逐行核对过）：M 的读出 modulation **不是词向量**，它是 M 在
                #   神经元空间的共激活预测（blocks=einsum('ck,ckij->cij', h_k, M_li)，M 由 h_out⊗h_in
                #   写入），与 coact_mean 同形 [H,H]，v100 已用 .t() 对齐 ⇒ 不存在「跨维度乱减」。
                #   v101 门② 未过的真实原因：**共激活误差不是预测误差** —— 谁跟谁一起放电，
                #   与「下一个词该是什么」没有必然关系。
                # v102：把误差换成真正的「预测下一状态」误差（局部预测编码 / LMS，无 BP、无全局损失）：
                #   target_t = 该层在 t+1 的目标（L1=下一词 embedding；L2/L3=上一层在 t+1 的激活）
                #   pred_t   = W · h_in_t                       # 当前突触把本步输入映射到输出空间
                #   e_t      = target_t − pred_t
                #   ΔW       = lr·|DA|·mean_t(e_t ⊗ h_in_t)     # 时间逐点配对
                # ★为什么不是 outer(mean_t e, mean_t h)：那是 E[e]⊗E[h]，丢掉 t 的配对且恒为
                #   rank-1（T-1 个独立方向被压成 1 个）。Σ_t e_t⊗h_t 才是 ½Σ‖target−W·h_in‖²
                #   对 W 的负梯度 ⇒ 真正的预测编码学习规则。
                # ★该规则只用本层输入、本层突触、本层目标 ⇒ 纯局部；M 的预测电流注入（拉 V）不动。
                # ===== v104：隐藏层 W 全冻结（δ≡0）=====
                # v100~v103 的证据：无逐层反传时，W 的任何局部规则都只会把它推胖或乱走
                #   （v103 用电流驱动 ⇒ W_L3 10 epoch +70%、CE 10.41→11.13）。
                # 分工：W 只负责「把输入变成电流」，不负责预测词；学习全部交给输出头 W_out。
                # 保留稳态缩放/软边界/修剪（都是无量纲的自稳项，不含方向学习）。
                delta = torch.zeros_like(layer.weight.data)
                assert delta.shape == layer.weight.data.shape, (delta.shape, layer.weight.data.shape)
                self._v100_surprise_norm = float(surprise.norm())
                self._v100_coact_norm = float(coact_mean.norm())
                self._v100_mod_norm = float(modulation.norm())
                self._v100_mod_scale = float(mod_scale)
                self._v100_surprise_frac = float(surprise.norm() / coact_mean.norm().clamp(min=1e-12))
                self._v100_cos_pred = float(torch.dot(mod_scaled.flatten(), coact_mean.flatten())
                                            / (mod_scaled.norm() * coact_mean.norm() + 1e-30))
                self._v97_coact_norm = float(coact_mean.norm())
                # v103：v102 的预测误差读数已随规则一起删除（本轮无误差标量）
                # 与共激活朝向的对照（[out,in] vs coact.t()），跨轮可比
                self._v97_delta_cos = float(torch.dot(delta.flatten(), coact_mean.t().flatten())
                                            / (delta.norm() * coact_mean.norm() + 1e-30))
                # CV 调制
                if not hasattr(self, '_mod_history'):
                    self._mod_history = [[] for _ in range(self.num_layers)]
                self._mod_history[li].append(modulation.norm().item())
                if len(self._mod_history[li]) > 10:
                    self._mod_history[li].pop(0)
                if len(self._mod_history[li]) >= 10:
                    mu = sum(self._mod_history[li]) / 10
                    sigma = (sum((x - mu)**2 for x in self._mod_history[li]) / 10) ** 0.5
                    cv = sigma / max(mu, 1e-6)
                    cv_factor = 1.0 + 0.5 * torch.tanh(torch.tensor(cv / 2.0))
                else:
                    cv_factor = 1.0
                # v97：CV 缩放不再作用于 δ（cv 仍记录，供诊断对照）

                layer.weight.data += delta
                # 软边界：|w| 超过 0.5 就开始非线性衰减（v13.2 基准 0.5 版，CE 7.13）
                abs_w = layer.weight.data.abs()
                # 平滑的"超界惩罚"：越界越多，衰减越快
                penalty = torch.relu(abs_w - 0.5 * layer.w_max) ** 2 * 0.1
                layer.weight.data -= layer.weight.data.sign() * penalty

                # ---- 稳态突触缩放：按平均发放率慢速调节 ----
                actual_fr = layer_firings[li].mean().item()   # v64-A：稳态缩放信号源=发放率 EMA（非 V 均值）
                layer.avg_firing_rate = 0.99 * layer.avg_firing_rate + 0.01 * actual_fr
                scale = 1.0 + layer.scaling_rate * (layer.target_firing_rate - layer.avg_firing_rate)
                layer.weight.data *= scale

            # ========== v82-II：embedding 冻结（单点删除，去掉唯一的 V→embedding 写入路径） ==========
            # 理由：embedding 是「感觉映射」，发育期基本固定，不应被顶层反馈（V 派生的 h_t）直接驱动。
            # v81 实测：任何连到 V 的学习路径都会把 V 的公共成分传播下去，使下游 dg_sparse 的
            # pattern separation 从一个 epoch 内塌缩（窗口间余弦 0.18 → 0.96）。
            # __init__ 的 embedding.weight.requires_grad=False 只禁用反向传播，真正的写入就在被删的这段；
            # 删掉后训练期内 embedding.weight.data 不再被任何代码触碰（由 runner 的 [EMB-FROZEN] 逐窗核验）。

            # ========== v90-BBa：W_out 对齐 t+1 + 量级匹配（目标范数对齐到 V 的历史水平） ==========
            # 物理依据：V 行范数 ≈ 28（v74_T 起稳定），embedding 行范数 ≈ √H ≈ 11.3（std=1.0 冻结）；
            # 直接对齐 embedding 会把 W_out 目标范数拉低 60%（v89 AAa 实测 741→492）。
            # scale = ‖V‖/‖emb‖ 全部实时算出、.detach() 只作量级校准，不参与学习、无手填常数。
            # v94-Q1：基准用「首 epoch 的 V 范数中位数」——既不用首窗瞬时值（v93 P3 锁高 2.1x），
            # 也不用逐窗实时值（v92 D 正反馈）。首 epoch 内 scale=1.0，让 V 走完初始瞬态。
            if self._step_count <= 70:
                if not hasattr(self, '_v_norm_buffer'):
                    self._v_norm_buffer = []
                self._v_norm_buffer.append(h_last_all.norm(dim=-1).mean().item())
                scale = 1.0
            else:
                if not hasattr(self, '_v_norm_ref'):
                    import statistics
                    self._v_norm_ref = torch.tensor(statistics.median(self._v_norm_buffer),
                                                    device=h_last_all.device, dtype=torch.float32)
                    del self._v_norm_buffer
                emb_norm = target_emb[:, 1:, :].norm(dim=-1).mean()
                scale = (self._v_norm_ref / emb_norm.clamp(min=1e-6)).detach()
            self._v90_scale = float(scale)                          # 逐窗诊断读数（不参与学习）
            # ========== v104：输出头学「真答案 − 预测概率」（readout LMS，对隐藏层零反传）==========
            # 外部经验的正确落法：W_out 是 [vocab,H]，它的「猜测」是词表分布 softmax(logits)，
            #   所以误差只能在概率空间里算（embedding 空间的 target_emb 与它维度不同，不能相减）。
            #   e_t = onehot(y_{t+1}) − softmax(logits_t)   ∈ R^V
            #   ΔW_out += lr_out · mean_t( e_t ⊗ V_t )      V_t = 最后一层 V
            # 这是 CE 对 W_out 的精确梯度（前向用 F.normalize(W_out) 的行 ⇒ 径向分量会被
            #   归一化吸收，对 raw W_out 直接更新等价于切向投影）。
            _logits = getattr(self, '_v104_logits', None)
            if _logits is not None and _logits.shape[0] == B and _logits.shape[1] >= T:
                _lg = _logits[:, :T - 1, :].reshape(-1, self.vocab_size)
                # ★v114：目标优先取 labels（含 -100），并据此构造 mask；没有 mask 时退回 input_ids
                _lab114 = getattr(self, '_v114_labels', None)
                _use_mask = (V114_MASK and _lab114 is not None
                             and _lab114.shape[0] == B and _lab114.shape[1] >= T)
                if _use_mask:
                    _yt = _lab114[:, 1:T]                       # (B,T-1)，被 mask 处 = -100
                    _mpos = (_yt != -100)                       # 布尔
                    _y_target = _yt.clamp(min=0)
                else:
                    _yt = input_ids[:, 1:]
                    _mpos = torch.ones_like(_yt, dtype=torch.bool)
                    _y_target = _yt
                _y = _y_target.reshape(-1)
                _N_valid = float(_mpos.sum().item()) if _use_mask else float(_yt.numel())
                _N_valid = max(_N_valid, 1.0)
                _p = F.softmax(_lg, dim=-1)
                _onehot = torch.zeros_like(_p)
                _onehot.scatter_(1, _y.unsqueeze(1), 1.0)
                _err = _onehot - _p                                  # (N,V) 真答案 − 预测概率
                if _use_mask:
                    # ★v114：被 mask 的位置误差置 0 —— 不学提问、不学格式，只学回答
                    _err = _err * _mpos.reshape(-1).unsqueeze(-1).to(_err.dtype)
                # v105：与 forward 的 V 支同尺度（否则 LMS 优化的是另一个特征尺度，与 forward 脱节）
                _h_raw = layer_outputs[-1][:, :T - 1, :]
                _sc105 = getattr(self, '_v105_scale_buf', None)
                if _sc105 is not None and _sc105.shape[1] >= T:
                    _h_raw = _h_raw * _sc105[:, :T - 1, :]
                _h = _h_raw.reshape(-1, _h_raw.shape[-1])
                _dW = torch.einsum('no,nh->oh', _err, _h) / _N_valid
                self.W_out.data += V104_LR_OUT * _dW

            # ===== v112：只让 L1-L3 写入（L0 的记忆矩阵冻结在零，不再学习）=====
            #   对每个位置 t、每一层 li≥1：把该层状态 V_li(t) 拉向「下一个 token 的列原型」。
            if not getattr(self, '_v112_off', False):
                _idx112 = list(range(1, len(self.layers))) if V112_NO_L0_READ else list(range(len(self.layers)))
                _wn111 = []
                for _li112 in _idx112:
                    _hT112 = layer_outputs[_li112]                        # (B,T,H)
                    for _t112 in range(T - 1):
                        if _use_mask and not bool(_mpos[0, _t112]):
                            continue                                   # ★v114：不在这写提问
                        _tid = int(input_ids[0, _t112 + 1].item())
                        if 0 <= _tid < self.vocab_size:
                            self.W_m_vocab[_li112].data[:, _tid] += V111_LR_M * (
                                _hT112[0, _t112] - self.W_m_vocab[_li112].data[:, _tid])
                for _li112 in range(len(self.layers)):
                    _wn111.append(float(self.W_m_vocab[_li112].norm()))
                # ---- 衰减（v111 已证与 0.998002/窗 等价，不动）----
                if V111_DECAY == 'none':
                    pass
                elif V111_DECAY == 'old':
                    for _li112 in _idx112:
                        self.W_m_vocab[_li112].data *= self._mod_decay
                elif self._step_count % V111_DECAY_EVERY == 0:
                    for _li112 in _idx112:
                        self.W_m_vocab[_li112].data *= V111_DECAY_RATE
                self._v111_wm_norm_v = _wn111
                # ---- γ 的局部更新（同 v110/v111：CE 概率误差 × 加权总读出）----
                _lmb = getattr(self, '_v111_lm_buf', None)
                if _lmb is not None and _lmb.shape[1] >= T:
                    _lmv = _lmb[:, :T - 1, :].reshape(-1, self.vocab_size)
                    _N111 = _N_valid          # ★v114：按有效（未 mask）位置数平均
                    _g111 = float((_err * _lmv).sum() / _N111)
                    _newg = float(self.v111_gamma.data) + V111_LR_G * _g111
                    self.v111_gamma.data.fill_(min(max(_newg, 0.1), 10.0))
                    self._v111_gamma_v = float(self.v111_gamma.data)
                    # ---- 层权重局部学（只更新参与读出的层）----
                    _lms = getattr(self, '_v112_lm_list_buf', None)
                    _idxb = getattr(self, '_v112_idx_buf', None)
                    if _lms is not None and _idxb is not None:
                        for _k112 in range(len(_idxb)):
                            _li112 = _idxb[_k112]
                            _lmli = _lms[_k112][:, :T - 1, :].reshape(-1, self.vocab_size)
                            _gl = float((_err * _lmli).sum() / _N111)
                            _nw = float(self._v111_layer_weights.data[_li112]) + V111_LR_W * _gl
                            self._v111_layer_weights.data[_li112] = min(max(_nw, -V111_W_CLAMP), V111_W_CLAMP)
                        self._v111_w_v = [float(x) for x in self._v111_layer_weights.data]
            self._v104_err_norm = float(_err.norm())
            self._v104_dW_norm = float(_dW.norm())
            self._v104_ce = float(F.cross_entropy(_lg, _yt.reshape(-1), ignore_index=-100))

            # ========== v81-GG：索引读出对齐（与 W_out 对齐同规则、同速 lr*0.1；目标=dg_sparse，不是 V） ==========
            # dg_sparse 在此重算：与 forward 同一份 embedding 输出、同一串算子 ⇒ 逐位相同，零额外前向
            dg_raw = target_emb @ self.W_dg.T
            topk_vals, topk_idx = dg_raw.abs().topk(self.k_index, dim=-1)
            dg_sparse = torch.zeros_like(dg_raw)
            dg_sparse.scatter_(-1, topk_idx, dg_raw.gather(-1, topk_idx))
            dg_sparse = dg_sparse / dg_sparse.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            # ---- v83-KK：时间累积（衰减 EMA，逐 batch 独立；每个 forward 内 ema 从零开始）----
            dg_context = torch.zeros_like(dg_sparse)                    # (B, T, H_index)
            ema = torch.zeros(dg_sparse.shape[0], dg_sparse.shape[2],
                              device=dg_sparse.device, dtype=dg_sparse.dtype)
            for _t in range(dg_sparse.shape[1]):
                ema = self._dg_decay * ema + dg_sparse[:, _t, :]
                dg_context[:, _t, :] = ema
            # 归一化：让每个时间步的 dg_context 与 dg_sparse 同量级（单位范数）
            dg_context = dg_context / dg_context.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            for t in range(T - 1):
                next_ids = input_ids[:, t + 1]
                for b in range(B):
                    tid = next_ids[b].item()
                    if 0 <= tid < self.vocab_size:
                        target_vec = self.W_index_out.data[tid]
                        self.W_index_out.data[tid] = target_vec + lr * 0.1 * (dg_context[b, t] - target_vec)

            # ---- 突触修剪（刹车 3）：层范数超限则全局 ×0.99 ----
            for layer in self.layers:
                wn = layer.weight.data.norm()
                if wn > 50.0:
                    layer.weight.data *= 0.99

            # ---- ATP 池比值监控 ----
            self._last_atp_ratio = [layer_atp[li].mean().item()
                                    for li in range(len(self.layers))]
            # ---- v22 H0a 诊断日志 ----
            M_total = sum(m.norm().item() for m in self.modulation_tensors)
            W_total = max(layer.weight.norm().item() for layer in self.layers)
            log_and_print(f"[DIAG] M_norm={M_total:.4f} W_norm={W_total:.4f} DA={getattr(self,'_last_da',0.0):.4f}")

    @torch.no_grad()
    def prefill(self, input_ids):
        device = input_ids.device
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        states = self._make_states(device, B)
        self._gen_t = 0
        last_logits = None
        for t in range(T):
            hidden_last, _ = self._core_step(x[:, t, :], states, self._gen_t)
            emb_norm = F.normalize(self.embedding.weight, dim=1)
            logits = hidden_last @ emb_norm.T
            last_logits = logits
            self._gen_t += 1
        return last_logits, states

    def step(self, token_id, states):
        device = states[0]['V'].device
        token_emb = self.embedding(
            torch.tensor([token_id], dtype=torch.long, device=device))
        hidden_last, _ = self._core_step(token_emb, states, self._gen_t)
        emb_norm = F.normalize(self.embedding.weight, dim=1)
        logits = hidden_last @ emb_norm.T
        self._gen_t += 1
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=MAX_SEQ_LEN,
                 temperature=0.7, top_p=0.9, stream=None):
        global _LEARNING_DISABLED_DURING_GENERATE
        was_training = self.training
        self.eval()
        _LEARNING_DISABLED_DURING_GENERATE = True
        try:
            device = input_ids.device
            batch_size = input_ids.shape[0]
            if batch_size != 1:
                raise ValueError("生成时仅支持 batch_size = 1")

            stop_tokens = set()
            if self.eos_token_id is not None:
                stop_tokens.add(self.eos_token_id)
            if self.end_token_id is not None:
                stop_tokens.add(self.end_token_id)

            forbidden = self.forbidden_ids.copy()
            if self.pad_token_id is not None:
                forbidden.add(self.pad_token_id)

            prompt_ids = input_ids[0].tolist()

            if len(prompt_ids) == 0:
                _bos = getattr(self, 'bos_token_id', None)
                prompt_ids = [_bos if _bos is not None else 0]
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            logits, states = self.prefill(input_ids)

            generated = []
            for _ in range(max_new_tokens):
                last_logits = logits[0].float()
                probs = F.softmax(last_logits / temperature, dim=-1)
                for fid in forbidden:
                    if fid < self.vocab_size:
                        probs[fid] = 0.0
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[0] = False
                sorted_probs[mask] = 0.0
                if sorted_probs.sum() == 0:
                    uniform = torch.ones(self.vocab_size, device=device) / self.vocab_size
                    for fid in forbidden:
                        if fid < self.vocab_size:
                            uniform[fid] = 0.0
                    if uniform.sum() == 0:
                        uniform = torch.ones(self.vocab_size, device=device) / self.vocab_size
                    else:
                        uniform = uniform / uniform.sum()
                    sorted_probs, sorted_indices = torch.sort(uniform, descending=True)
                else:
                    sorted_probs = sorted_probs / sorted_probs.sum()
                next_token = sorted_indices[torch.multinomial(sorted_probs, 1).item()].item()

                generated.append(next_token)
                if stream is not None:
                    stream(next_token)
                if next_token in stop_tokens:
                    break
                logits = self.step(next_token, states)
            return generated
        finally:
            _LEARNING_DISABLED_DURING_GENERATE = False
            if was_training:
                self.train()

# ------------------------------------------------------------
# 保存 / 加载
# ------------------------------------------------------------
def get_next_checkpoint_id():
    timestamp = int(time.time() * 1000)
    rand = random.randint(0, 10000)
    return f"{timestamp}_{rand}"

def save_checkpoint(model, epoch, custom_filename=None):
    if custom_filename is None:
        filename = f"checkpoint_{get_next_checkpoint_id()}.pt"
    else:
        filename = custom_filename
    save_dict = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    torch.save(save_dict, filename)
    log_and_print(f"保存 checkpoint 至 {filename} (epoch {epoch})")
    return filename

def create_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers=0,
                      pin_memory=PIN_MEMORY, drop_last=False):
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g, persistent_workers=(num_workers > 0))

def setup_model_token_ids(model, tokenizer):
    model.eos_token_id = tokenizer.eos_token_id
    model.bos_token_id = tokenizer.bos_token_id
    model.unk_token_id = tokenizer.unk_token_id
    model.user_token_id = tokenizer.user_token_id
    model.assistant_token_id = tokenizer.assistant_token_id
    model.pad_token_id = tokenizer.pad_token_id
    model.end_token_id = tokenizer.end_token_id
    model.forbidden_ids = {tokenizer.pad_token_id} if tokenizer.pad_token_id is not None else set()

# ------------------------------------------------------------
# 文件选择
# ------------------------------------------------------------
def select_training_file():
    files = []
    for f in sorted(os.listdir('.')):
        if f.startswith('.'):
            continue
        if f.endswith('.txt') or f.endswith('.jsonl'):
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0
            files.append((f, size))
    if not files:
        log_and_print("当前目录下未找到 .txt 或 .jsonl 文件。")
        return None, None
    print("\n可用数据集文件：")
    for i, (name, size) in enumerate(files, 1):
        if size >= 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {i}. {name} ({size_str})")
    while True:
        raw = input("请输入文件序号: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue
        if idx < 1 or idx > len(files):
            print(f"序号超出范围，请输入 1-{len(files)}。")
            continue
        selected, _ = files[idx - 1]
        ext = os.path.splitext(selected)[1].lower()
        log_and_print(f"已选择文件: {selected}")
        return selected, ext

# ------------------------------------------------------------
# 训练函数：Hebbian 版，无 BP
# ------------------------------------------------------------

def eval_quick(model, dataset, tokenizer):
    """v22 统一：epoch 末公平评估（4 窗口）。保存/恢复 RNG 状态，不污染训练轨迹。"""
    model.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_states = {}
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            cuda_states[d] = torch.cuda.get_rng_state(d)
    tot_ce = 0.0; n = 0; c1 = 0; c5 = 0; tp = 0; nan_flag = 0
    try:
        with torch.no_grad():
            for input_ids, labels in dataset:
                input_ids = input_ids.unsqueeze(0).to(DEVICE)
                labels = labels.unsqueeze(0).to(DEVICE)
                logits = model(input_ids)
                V = logits.shape[-1]
                lg = logits[:, :-1, :].reshape(-1, V)
                tg = labels[:, 1:].reshape(-1)
                mask = (tg != -100) & (tg != tokenizer.pad_token_id)
                if mask.sum() == 0:
                    continue
                if not torch.isfinite(lg).all():
                    nan_flag = 1
                ce = F.cross_entropy(lg[mask], tg[mask]).item()
                tot_ce += ce; n += 1
                preds = lg[mask].argmax(dim=-1)
                c1 += (preds == tg[mask]).sum().item()
                top5 = lg[mask].topk(5, dim=-1).indices
                c5 += (top5 == tg[mask].unsqueeze(1)).any(dim=-1).sum().item()
                tp += mask.sum().item()
    finally:
        torch.random.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            for d, s in cuda_states.items():
                torch.cuda.set_rng_state(s, d)
        model.train()
    if n == 0 or tp == 0:
        return float('nan'), 0.0, 0.0, nan_flag
    return tot_ce / n, c1 / tp, c5 / tp, nan_flag

def train_from_scratch(model, tokenizer, data_path, batch_size, epochs, lr,
                       weight_decay, grad_clip, log_interval,
                       accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                       file_type='jsonl'):
    device = DEVICE
    model.to(device)
    log_and_print(f"Hebbian 模式：开始训练。数据集: {data_path}")

    try:
        import psutil
        process = psutil.Process(os.getpid())
        _has_psutil = True
    except ImportError:
        process = None
        _has_psutil = False

    with open("training_log.txt", 'a', encoding='utf-8') as f:
        f.write(f"--- Hebbian 训练开始 ({data_path}) ---\n")

    setup_model_token_ids(model, tokenizer)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    if file_type == '.txt':
        train_dataset = TextFileDataset(data_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                                        epoch=0, is_val=False, window_overlap=WINDOW_OVERLAP)
        train_dataset._ensure_tokenized()
        total_samples = train_dataset._num_windows
        est_batches_per_epoch = max(1, total_samples // batch_size) if total_samples > 0 else 1
        log_and_print(f"txt 流式模式：{total_samples} 个训练窗口。")
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for l in f if l.strip())
        est_batches_per_epoch = max(1, total_samples // batch_size) if total_samples > 0 else 1000
        train_dataset = StreamingJSONLDataset(
            file_path=data_path, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN,
            epoch=0, is_val=False, assigned_indices=None)

    num_workers = NUM_WORKERS
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=PIN_MEMORY, drop_last=True)

    log_and_print(f"Hebbian 更新率 = {BIO_LEARNING_RATE}，无 optimizer，无 BP")

    interrupted = False
    try:
        for epoch in range(epochs):
            if hasattr(train_dataset, 'epoch'):
                train_dataset.epoch = epoch
            model.train()
            step_loss_sum = 0.0
            step_count = 0
            consumed_samples = 0

            for step, batch in enumerate(train_loader):
                input_ids, labels, attn_mask = batch
                if input_ids.numel() == 0:
                    continue
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # 前向 + 显示 loss + Hebbian 更新（全在 no_grad 里）
                with torch.no_grad():
                    logits = model(input_ids, labels=labels)  # (B, T, V)；顺带计算 _last_da

                    # 显示 loss（只是显示，不 backward）
                    V = logits.shape[-1]
                    valid_logits = logits[:, :-1, :].reshape(-1, V)
                    valid_labels = labels[:, 1:].reshape(-1)
                    loss_display = F.cross_entropy(
                        valid_logits, valid_labels, ignore_index=-100)

                    # 血清素：动态 lr，早期大、后期小
                    model._step_count += 1
                    # lr：线性预热 100 步到 1.0 + 余弦退火到 0.1
                    model._step_count += 1
                    current_lr = 1.0   # Q2：固定 lr=1.0

                    # v50 E1: 熵 EMA 在训练循环更新（eager 里 .item() 无害）
                    _Ht = getattr(model, '_last_entropy_norm', 1.0)
                    _Hn = float(_Ht.item()) if torch.is_tensor(_Ht) else float(_Ht)
                    model._entropy_ema = 0.99 * getattr(model, '_entropy_ema', 1.0) + 0.01 * _Hn
                    # 5-HT 自监督：低熵=高置信=稳定 -> 5-HT 高
                    H = model._entropy_ema
                    ht_raw = 0.5 * (1.0 + math.tanh((0.5 - H) * 4.0))
                    model._ht_ema = 0.99 * getattr(model, '_ht_ema', 0.5) + 0.01 * ht_raw
                    ht_signal = model._ht_ema
                    # 温和乘子，中心 1.0，不再砍半
                    lr_mult = 0.95 + 0.1 * ht_signal
                    current_lr = current_lr * lr_mult
                    model._last_ht_signal = ht_signal
                    model._last_lr_mult = lr_mult
                    model._last_ht_H = H

                    model.hebbian_update(input_ids, lr=current_lr, cached=True)
                    # v64-B：诊断。此刻 _cached_layer_* 尚未被 delattr，可直接读
                    if model._step_count % 10 == 0:
                        _pred = logits[:, :-1, :].argmax(dim=-1)
                        _true = labels[:, 1:]
                        _mask = (_true != -100)
                        if _mask.any():
                            _acc = (((_pred == _true) & _mask).float().sum()
                                    / _mask.float().sum()).item()
                            _hist = list(model._acc_history)
                            _base = sum(_hist) / max(len(_hist), 1)
                            log_and_print(f"[ACC] step={model._step_count} acc={_acc:.4f} "
                                          f"baseline={_base:.4f} "
                                          f"DA={getattr(model, '_last_da', 0.0):.6f} "
                                          f"hist_n={len(_hist)}")
                        _lf = getattr(model, '_cached_layer_firings', None)
                        _lo = getattr(model, '_cached_layer_outputs', None)
                        for _li in range(model.num_layers):
                            _real = _lf[_li].mean().item() if _lf is not None else float('nan')
                            _vm = _lo[_li].mean().item() if _lo is not None else float('nan')
                            _ho = model.layers[_li].avg_firing_rate.item()
                            log_and_print(f"[FR] L{_li} real_fr={_real:.4f} V_mean={_vm:.4f} "
                                          f"homeo_avg={_ho:.4f}")
                    # v48 FIX(D1): 释放 F2 缓存引用，避免分配器路径膨胀
                    for _attr in ('_cached_layer_inputs', '_cached_layer_outputs',
                                  '_cached_layer_firings', '_cached_layer_atp'):
                        if hasattr(model, _attr):
                            delattr(model, _attr)

                step_loss_sum += loss_display.item()
                step_count += 1
                consumed_samples += input_ids.shape[0]

                if (step + 1) % log_interval == 0:
                    avg_loss = step_loss_sum / step_count if step_count > 0 else 0.0
                    avg_ppl = math.exp(min(avg_loss, 20))
                    mem_str = ""
                    if _has_psutil and process is not None:
                        try:
                            mem_str = f" Memory: {process.memory_info().rss / 1024**2:.1f} MB"
                        except:
                            mem_str = ""
                    pct_str = ""
                    if total_samples > 0:
                        pct_str = f" ({consumed_samples * 100 / total_samples:.1f}%)"
                    log_and_print(
                        f"Epoch {epoch+1}/{epochs} Step {step+1} "
                        f"CE(display)={avg_loss:.4f} PPL={avg_ppl:.2f} "
                        f"Samples: {consumed_samples}{pct_str}{mem_str}")
                    for li, layer in enumerate(model.layers):
                        wn = layer.weight.norm().item()
                        rp = layer.release_prob.mean().item()
                        log_and_print(f"  [Bio L{li}] Wn={wn:.4f} release_prob={rp:.3f}")
                    for li, M in enumerate(model.modulation_tensors):
                        log_and_print(f"  [Mod L{li}] Mn={M.norm().item():.4f}")
                    for li, layer in enumerate(model.layers):
                        fr = layer.avg_firing_rate.item()
                        log_and_print(f"  [Homeo L{li}] avg_fr={fr:.4f}")
                    ht = getattr(model, '_last_ht_signal', 0.0)
                    lr_mult = getattr(model, '_last_lr_mult', 1.0)
                    H = getattr(model, '_last_ht_H', 1.0)
                    log_and_print(f"  [5-HT] ht_signal={ht:.4f} H_ema={H:.4f} lr_mult={lr_mult:.4f} lr={current_lr:.4f}")
                    atp_vals = getattr(model, '_last_atp_ratio', [1.0] * len(model.layers))
                    atp_log = " ".join([f"L{li}={v:.3f}" for li, v in enumerate(atp_vals)])
                    log_and_print(f"  [ATP] {atp_log}")
                    step_loss_sum = 0.0
                    step_count = 0

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()

            log_and_print(f"Epoch {epoch+1} 无验证集，跳过验证与早停。")
            # ---- v22 统一日志：每 epoch 末一行 ----
            # v47 F3：每 5 个 epoch 评估一次
            if (epoch + 1) % 5 == 0:
                try:
                    e_ce, e_top1, e_top5, e_nan = eval_quick(model, train_dataset, tokenizer)
                    M_n = 0.0
                    for li in range(len(model.layers)):
                        if hasattr(model, 'modulation_tensors') and li < len(model.modulation_tensors):
                            M_n += model.modulation_tensors[li].norm().item()
                        elif hasattr(model, 'M_a') and li < len(model.M_a):
                            M_n += torch.einsum('kr,rij->kij', model.M_a[li], model.M_b[li]).norm().item()
                        elif hasattr(model, 'M_cp_a') and li < len(model.M_cp_a):
                            M_n += torch.einsum('rk,ri,rj->kij', model.M_cp_a[li], model.M_cp_b[li], model.M_cp_c[li]).norm().item()
                    W_n = max(layer.weight.norm().item() for layer in model.layers)
                    DA = getattr(model, '_last_da', 0.0)
                    log_and_print(f"EPOCH={epoch+1} CE={e_ce:.4f} top1={e_top1*100:.3f} top5={e_top5*100:.3f} M_norm={M_n:.4f} W_norm={W_n:.4f} DA={DA:.4f} NaN={int(e_nan)}")
                    log_and_print("[V515] ep=" + str(epoch + 1) + " " + model._v515_report())
                except Exception as ex:
                    log_and_print(f"EPOCH={epoch+1} eval 失败: {ex}")
            else:
                log_and_print(f"Epoch {epoch+1} 跳过 eval（每 5 epoch 一次）。")
            ckpt_name = f"checkpoint_hebb_epoch{epoch+1}.pt"
            save_checkpoint(model, epoch+1, custom_filename=ckpt_name)

            step_loss_sum = 0.0
            step_count = 0
    except KeyboardInterrupt:
        interrupted = True
        log_and_print("\n训练被用户中断 (Ctrl+C)，正在保存当前模型...")
        save_checkpoint(model, epoch + 1, custom_filename="interrupted_hebb.pt")
        return

    if not interrupted:
        log_and_print("Hebbian 训练完成")

# ------------------------------------------------------------
# 对话循环（外壳不变）
# ------------------------------------------------------------
def chat_loop(model, tokenizer, mode="jsonl"):
    device = DEVICE
    model.to(device)
    model.eval()
    setup_model_token_ids(model, tokenizer)

    MAX_HISTORY_TOKENS = MAX_SEQ_LEN

    if mode == "txt":
        log_and_print("进入纯文本续写模式（txt），输入 exit 退出，reset 重置上下文。")
        context_ids = []
        while True:
            user_input = input("\n输入: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                context_ids = []
                log_and_print("上下文已重置。")
                continue
            new_ids = tokenizer.encode(user_input)
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_HISTORY_TOKENS:
                context_ids = context_ids[-MAX_HISTORY_TOKENS:]
            input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
            generated_ids = []
            def stream_token(token_id):
                generated_ids.append(token_id)
                text = tokenizer.decode([token_id])
                if text:
                    print(text, end="", flush=True)
            model.generate(input_tensor, max_new_tokens=MAX_SEQ_LEN,
                           temperature=TEMPERATURE, top_p=TOP_P, stream=stream_token)
            context_ids.extend(generated_ids)
            if len(context_ids) > MAX_HISTORY_TOKENS:
                context_ids = context_ids[-MAX_HISTORY_TOKENS:]
            print()
        print()
    else:
        log_and_print("进入对话模式（jsonl），输入 exit 退出，reset 重置上下文。")
        history_turns = []
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                history_turns = []
                log_and_print("上下文已重置。")
                continue
            user_ids = [tokenizer.user_token_id] + tokenizer.encode(user_input) + [tokenizer.end_token_id]
            flat_history = [tok for turn in history_turns for tok in turn]
            full_input = [tokenizer.bos_token_id] + flat_history + user_ids + [tokenizer.assistant_token_id]
            input_tensor = torch.tensor([full_input], dtype=torch.long, device=device)
            generated_ids = []
            print("助手: ", end="", flush=True)
            def stream_token_jsonl(token_id):
                generated_ids.append(token_id)
                text = tokenizer.decode([token_id])
                if text:
                    print(text, end="", flush=True)
            model.generate(input_tensor, max_new_tokens=MAX_SEQ_LEN,
                           temperature=TEMPERATURE, top_p=TOP_P, stream=stream_token_jsonl)
            print()
            history_turns.append(user_ids)
            history_turns.append([tokenizer.assistant_token_id] + generated_ids)
            total = sum(len(t) for t in history_turns)
            while total > MAX_HISTORY_TOKENS and len(history_turns) > 1:
                total -= len(history_turns.pop(0))

# ------------------------------------------------------------
# Ultra 系列（保留外壳，兼容调用）
# ------------------------------------------------------------
def ultra_read_int(prompt, default, minimum=1):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"请输入不小于 {minimum} 的整数。")

def ultra_read_float(prompt, default, minimum=0.0):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"请输入不小于 {minimum} 的数字。")

def ultra_select_problem_file():
    files = []
    for name in sorted(os.listdir('.')):
        if name.startswith('.') or not name.endswith('.jsonl'):
            continue
        try:
            size = os.path.getsize(name)
        except OSError:
            size = 0
        files.append((name, size))
    if not files:
        log_and_print("当前目录没有可用的 .jsonl 题库。")
        return None
    print("\n可用 JSONL 题库：")
    for i, (name, size) in enumerate(files, 1):
        print(f"  {i}. {name} ({size / 1024:.1f} KB)")
    while True:
        raw = input("请输入题库序号: ").strip()
        try:
            index = int(raw)
            if 1 <= index <= len(files):
                return files[index - 1][0]
        except ValueError:
            pass
        print(f"请输入 1-{len(files)}。")

def ultra_iter_problems(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                log_and_print(f"跳过题库第 {line_no} 行：JSON 解析失败: {e}")
                continue
            question = item.get('question', item.get('prompt'))
            answers = item.get('answers', item.get('answer'))
            if isinstance(question, list):
                question = '\n'.join(str(turn.get('content', turn)) if isinstance(turn, dict) else str(turn)
                                     for turn in question)
            if not isinstance(question, str) or answers is None:
                continue
            if not isinstance(answers, list):
                answers = [answers]
            answers = [str(a) for a in answers if str(a).strip()]
            if not answers:
                continue
            yield {'question': question, 'answers': answers,
                   'type': str(item.get('type', 'auto')).lower(),
                   'line_no': line_no}

def ultra_count_problems(path):
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def ultra_iter_problems_shuffled(path, buffer_size=10000, seed=0):
    rng = random.Random(seed)
    buf = []
    for p in ultra_iter_problems(path):
        buf.append(p)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            while buf:
                yield buf.pop()
    if buf:
        rng.shuffle(buf)
        while buf:
            yield buf.pop()

def ultra_infer_answer_type(problem):
    import re
    answer_type = problem.get('type', 'auto')
    if answer_type != 'auto':
        return answer_type
    answers = [a.strip() for a in problem['answers']]
    if answers and all(re.fullmatch(r'[A-Za-z]', a) for a in answers):
        return 'choice'
    if answers and all(re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', a.replace(',', ''))
                       for a in answers):
        return 'number'
    return 'text'

def ultra_normalize_answer(value, answer_type):
    import re
    import unicodedata
    from decimal import Decimal, InvalidOperation
    from fractions import Fraction
    text = unicodedata.normalize('NFKC', str(value)).strip()
    if answer_type == 'choice':
        choices = re.findall(r'(?<![A-Za-z])[A-Za-z](?![A-Za-z])', text)
        return choices[-1].upper() if choices else text.upper()
    if answer_type == 'number':
        matches = re.findall(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', text.replace(',', ''))
        if matches:
            raw = matches[-1]
            try:
                if '/' in raw:
                    left, right = raw.split('/', 1)
                    vf = Fraction(Decimal(left)) / Fraction(Decimal(right))
                else:
                    vf = Fraction(Decimal(raw))
                return f"{vf.numerator}/{vf.denominator}"
            except (InvalidOperation, ZeroDivisionError):
                pass
    text = re.sub(r'\s+', '', text.lower())
    return text.strip('。.!！?？,，;；:：')

def ultra_extract_final_answer(response):
    import re
    boxed = re.findall(r'\\boxed\s*\{([^{}]+)\}', response)
    if boxed:
        return boxed[-1].strip()
    marked = re.findall(r'(?:最终答案|答案|final\s+answer)\s*[:：]\s*([^\r\n]+)', response,
                        flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip()
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines[-1] if lines else ''

def ultra_rule_reward(response, problem):
    predicted = ultra_extract_final_answer(response)
    answer_type = ultra_infer_answer_type(problem)
    pred_norm = ultra_normalize_answer(predicted, answer_type)
    accepted = {ultra_normalize_answer(a, answer_type) for a in problem['answers']}
    return (1.0 if pred_norm in accepted else 0.0), predicted

def ultra_make_prompt(problem):
    return f"{problem['question']}\n\n请自由推理。最后必须单独写一行：最终答案：你的答案"

def ultra_prompt_ids(tokenizer, prompt):
    return ([tokenizer.bos_token_id, tokenizer.user_token_id]
            + tokenizer.encode(prompt)
            + [tokenizer.end_token_id, tokenizer.assistant_token_id])

def ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens):
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated_ids = model.generate(input_tensor, max_new_tokens=max_new_tokens,
                                   temperature=TEMPERATURE, top_p=TOP_P)
    return generated_ids, tokenizer.decode(generated_ids)

def ultra_write_sft_line(f, prompt, response):
    record = {'conversations': [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': response}]}
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    f.flush()

def ultra_rejection_sampling(model, tokenizer, problem_path, output_path,
                             samples_per_question, max_new_tokens):
    if os.path.abspath(problem_path) == os.path.abspath(output_path):
        raise ValueError("输出文件不能覆盖输入题库。")
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    was_training = model.training
    model.to(DEVICE)
    model.eval()
    kept = 0
    tried = 0
    seen = set()
    with open(output_path, 'w', encoding='utf-8') as out:
        with torch.no_grad():
            for index, problem in enumerate(ultra_iter_problems(problem_path), 1):
                prompt = ultra_make_prompt(problem)
                prompt_ids = ultra_prompt_ids(tokenizer, prompt)
                local_kept = 0
                for _ in range(samples_per_question):
                    generated_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                    tried += 1
                    reward, predicted = ultra_rule_reward(response, problem)
                    key = (problem['line_no'], response)
                    if reward > 0 and key not in seen:
                        ultra_write_sft_line(out, prompt, response)
                        seen.add(key)
                        kept += 1
                        local_kept += 1
                log_and_print(f"[RS {index}/{total_problems}] 正确保留 {local_kept}/{samples_per_question}")
    if was_training:
        model.train()
    log_and_print(f"拒绝采样完成：尝试 {tried} 个回答，保留 {kept} 个正确回答")

def ultra_completion_log_probs(model, prompt_ids, completion_ids):
    if not completion_ids:
        return torch.empty(0, device=DEVICE)
    full_ids = prompt_ids + completion_ids
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
    logits = model(input_ids)
    start = len(prompt_ids) - 1
    selected_logits = logits[:, start:start + len(completion_ids), :]
    targets = torch.tensor(completion_ids, dtype=torch.long, device=DEVICE).view(1, -1, 1)
    return F.log_softmax(selected_logits, dim=-1).gather(-1, targets).squeeze(0).squeeze(-1)

def ultra_save_model_only(model, filename, epoch):
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, filename)
    log_and_print(f"仅保存模型至 {filename}")

def ultra_rejection_sampling_menu(model, tokenizer):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    output_path = input("正确回答 SFT 输出文件 [ultra_rejection_sft.jsonl]: ").strip() or 'ultra_rejection_sft.jsonl'
    samples = ultra_read_int("每题采样回答数", ULTRA_RS_SAMPLES_PER_QUESTION)
    max_new_tokens = ultra_read_int("每个回答最多生成 token 数", ULTRA_RS_MAX_NEW_TOKENS)
    ultra_rejection_sampling(model, tokenizer, problem_path, output_path, samples, max_new_tokens)

# ------------------------------------------------------------
# 加载 brain.pt
# ------------------------------------------------------------
def load_brain_pt_into(model):
    if not os.path.exists("brain.pt"):
        return False
    checkpoint = torch.load("brain.pt", map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return True

# ------------------------------------------------------------
# 主菜单
# ------------------------------------------------------------
def main():
    global TEMPERATURE, TOP_P
    init_log_file()
    tokenizer = FastBPE(VOCAB_SIZE)
    tokenizer_loaded = False
    try:
        if os.path.exists("tokenizer_incremental.bin") and os.path.exists("tokenizer_incremental_tokenizer.json"):
            tokenizer_loaded = tokenizer.load("tokenizer_incremental.bin")
            if tokenizer_loaded:
                log_and_print("成功加载已有分词器。")
    except Exception as e:
        log_and_print(f"加载分词器异常: {e}")

    if not tokenizer_loaded:
        log_and_print("准备训练新分词器。")
        try:
            with open("dialogues.jsonl", 'r', encoding='utf-8') as f:
                all_text = []
                char_set = set()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    for turn in data.get('conversations', []):
                        content = turn.get('content', '')
                        normalized = normalize_text(content)
                        all_text.append(normalized)
                        for ch in normalized:
                            char_set.add(ch)
                unique_chars = len(char_set)
                if VOCAB_SIZE < unique_chars + len(SPECIAL_TOKENS_LIST):
                    log_and_print(f"警告：VOCAB_SIZE={VOCAB_SIZE} < 字符数+特殊token={unique_chars + len(SPECIAL_TOKENS_LIST)}")
                tokenizer.train(all_text, verbose=True)
                tokenizer.save("tokenizer_incremental.bin")
                tokenizer._tokenizer.save("tokenizer_incremental_tokenizer.json")
        except FileNotFoundError:
            log_and_print("未找到 dialogues.jsonl，扫描 .txt 文件...")
            all_text = []
            for f in sorted(os.listdir('.')):
                if f.startswith('.') or not f.endswith('.txt'):
                    continue
                try:
                    with open(f, 'r', encoding='utf-8') as tf:
                        for line in tf:
                            line = line.strip()
                            if line:
                                all_text.append(normalize_text(line))
                    log_and_print(f"  已读取 {f}")
                except Exception as e:
                    log_and_print(f"  跳过 {f}: {e}")
            if all_text:
                tokenizer.train(all_text, verbose=True)
            else:
                tokenizer.train([""], verbose=False)
            tokenizer.save("tokenizer_incremental.bin")
            if tokenizer._tokenizer is not None:
                tokenizer._tokenizer.save("tokenizer_incremental_tokenizer.json")

    model = WaveSNN(
        vocab_size=tokenizer.vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pad_token_id=tokenizer.pad_token_id,
        use_residual=USE_RESIDUAL,
        freeze_seed=SEED_FREEZE,
        freeze_layers=FREEZE_LAYERS,
        freeze_embedding=True,
        seed_embedding=SEED_EMBEDDING)

    setup_model_token_ids(model, tokenizer)

    model_loaded = False
    if os.path.exists("brain.pt"):
        log_and_print("检测到 brain.pt，按需加载。")
    else:
        log_and_print("未找到 brain.pt，使用随机初始化模型。")

    def ensure_model_loaded():
        nonlocal model_loaded
        if model_loaded:
            return True
        if not os.path.exists("brain.pt"):
            return False
        try:
            load_brain_pt_into(model)
            setup_model_token_ids(model, tokenizer)
        except Exception as e:
            log_and_print(f"加载 brain.pt 失败: {e}。程序退出。")
            close_log_file()
            raise SystemExit(1)
        model_loaded = True
        log_and_print("已加载 brain.pt 模型权重。")
        return True

    while True:
        print("\n=== Wave-SNN (Hebbian) 训练与对话系统 ===")
        print("1. 训练模型")
        print("2. 对话模式")
        print("3. 保存模型")
        print("4. 加载模型")
        print("5. 调整生成参数")
        print("6. 查看模型统计信息")
        print("7. 规则验证拒绝采样")
        print("8. (保留)")
        print("9. (保留)")
        print("10. 退出")
        choice = input("请选择: ").strip()

        if choice == '1':
            data_path, file_ext = select_training_file()
            if data_path is None:
                continue
            ensure_model_loaded()
            if input("开始训练？(y/n): ").lower() == 'y':
                train_from_scratch(model, tokenizer, data_path,
                                   BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                                   GRAD_CLIP_NORM, LOG_INTERVAL,
                                   accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                                   file_type=file_ext)
        elif choice == '2':
            ensure_model_loaded()
            chat_mode = input("对话格式 (jsonl) 还是纯文本续写 (txt)？[jsonl]: ").strip().lower()
            if chat_mode not in ("jsonl", "txt"):
                chat_mode = "jsonl"
            chat_loop(model, tokenizer, mode=chat_mode)
        elif choice == '3':
            try:
                torch.save({'model_state_dict': model.state_dict(), 'epoch': 0}, "brain.pt")
                log_and_print("Model saved to brain.pt.")
                model_loaded = True
            except Exception as e:
                log_and_print(f"保存失败: {e}")
        elif choice == '4':
            try:
                if load_brain_pt_into(model):
                    setup_model_token_ids(model, tokenizer)
                    log_and_print("模型加载成功。")
                    model_loaded = True
                else:
                    log_and_print("未找到 brain.pt。")
            except Exception as e:
                log_and_print(f"加载失败: {e}")
                return
        elif choice == '5':
            try:
                TEMPERATURE = float(input(f"temperature (0.1-1.5) [当前 {TEMPERATURE:.2f}]: ") or TEMPERATURE)
                TOP_P = float(input(f"top_p (0.0-1.0) [当前 {TOP_P:.2f}]: ") or TOP_P)
            except ValueError:
                log_and_print("输入无效，保持原值")
        elif choice == '6':
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log_and_print(f"可训练参数: {trainable_params} / 总参数: {total_params}")
            log_and_print(f"词表大小: {model.vocab_size}")
            log_and_print(f"架构：BioWaveLayer × {model.num_layers} + Hebbian 外积学习")
            log_and_print(f"词向量层：冻结（不学习）")
            log_and_print(f"Hebbian 更新率: {BIO_LEARNING_RATE}")
        elif choice == '7':
            ensure_model_loaded()
            ultra_rejection_sampling_menu(model, tokenizer)
        elif choice == '8' or choice == '9':
            log_and_print("此模式下不可用。")
        elif choice == '10':
            break
        else:
            log_and_print("无效选择，请重新输入")
    close_log_file()

if __name__ == "__main__":
    main()

_v58_note = "C = P2 eager 柱化：M (N,D,D,D) 每层、cross (N,2,D,D) 每层、D=64、N=H/64；写入 k/i/j 同索引切片、读取块对角+环形邻柱；门控/资格迹/ATP/soft-boundary/sat/tanh 全部不动"
_v60_note = "v60-S3 = v58_C 的 k 轴 3D 扩宽：M_li (N, 3D, D, D)，k=[左邻|本柱|右邻]（环形 roll）；写入端 t+1 标记同段拼接、读取端 h_k 逐元素点积；cross_tensors/_theta_cross 独立通道取消；sat、tanh 饱和、max_ema、_M_norm_ema 沿用全局（非分柱）口径。"
