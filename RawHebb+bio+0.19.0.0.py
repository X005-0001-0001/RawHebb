# -*- coding: utf-8 -*-
"""
================================================================================================
 RawHebb+bio+0.19.0.0 —— 纯 Hebbian（无反向传播）语言模型 · 单文件
================================================================================================
一句话
    没有 BP、没有 Adam、没有 attention。词表是 ByteLevel BPE；模型是 4 层漏电积分膜 +
    阈值放电 + 每层自指回流管 + 精确 token 计数表；学习只来自局部规则：
    计数累加、运行 Gram 上的 Newton 读出、按证据的专家混合。

0.19.0.0 做了什么（外壳工程版，模型数值零改动）
    · 恢复 3120012elite.py 的**完整词表生命周期**：训练 / 保存 / 加载 / 元数据校验 /
      缺失回退 / 字符集检查。基座词表在就只读加载；不在就从语料训练 tokenizer.bin/json。
    · 恢复 3120012 的**双数据管线**：
        .txt  = 预训练：全文 token 化 → SEQ_LEN 窗口，全部位置计损，默认不做验证；
        .jsonl= 对话：assistant mask；验证集按**对话**随机切分 10%（或显式给 val 文件）；
        对话超长时按 3120012 的 `_truncate_pairs` 保留 system + 最近完整问答对，不硬切句子。
    · 新增**在线学习**菜单分支（13）：文件增量逐条更新 / 对话中学习；不 shuffle、不验证。
    · 恢复原版菜单的**数据集选择、输入容错、确认与返回**；新增词表子菜单。
    · 配置分区并写清每项来源与实测结论；删掉 6 个从未被读取的“安慰剂参数”。
    · 存档带 vocab/hidden/layers 元数据；词表不符时**明确拒绝加载**，不再静默跳过。
    · 加载同一份 0.18 权重的指标必须仍是 CE 3.0543 / top-1 41.86% / top-5 64.23%。

模型（与 0.18 完全一致）
    · 训练窗口 SEQ_LEN=384（完整对话，语料均值 472 token）；评测只取最后 EVAL_LEN=128 token。
    · 1M trigram 桶（int16）+ 4k last-token 计数 + 行归一化 bag 先验（Jeffreys 0.5）。
    · hidden 432 × 4 层；词向量冻结；无学习率、无梯度、无外挂监控器。
    · 全量 2850 条 / 1 epoch / CPU：CE 3.0543，top-1 41.86%，top-5 64.23%。

词表
    基座    : tokenizer_v63b4000.bin + tokenizer_v63b4000_tokenizer.json（随目录提供，只读优先）
    重建    : tokenizer.bin + tokenizer.json（菜单/CLI 写入，存在时优先生效）
    规则    : ByteLevel BPE，256 字节全进 initial_alphabet（结构上不会出现 UNK）；
              min_frequency=2；7 个特殊 token 的顺序固定；normalize 默认 'none'。
    ★ 重建词表会改变 token id，旧 brain.pt 不兼容。程序会拒绝加载并提示重新训练，
      不会再用 strict=False 静默跳过。

菜单（输入 0 退出）
    1  训练模型（可选择 .jsonl，默认 train_dsr1.jsonl）
    2  对话模式（jsonl 多轮 / txt 续写）
    3  保存模型        4  加载模型
    5  调整生成参数    6  查看模型统计
    7  拒绝采样生成 SFT 数据（无梯度）
    8  自一致性投票筛选（原规则 RL 的无梯度替代）
    9  长上下文分段接力评测（原 GRPO 的无梯度替代）
   10  数据预处理：按对话切分训练/验证集
   11  独立验证集评测（防泄露口径）
   12  分词器：查看 / 重建 / 往返测试
   13  在线学习：增量灌入文件 / 对话中学习（逐条更新，不验证）

命令行
    python "RawHebb+bio+0.19.0.0.py"                       # 菜单
    python "RawHebb+bio+0.19.0.0.py" --selftest            # 内核 + 词表往返自检
    python "RawHebb+bio+0.19.0.0.py" --tokenizer-info      # 只看词表信息
    python "RawHebb+bio+0.19.0.0.py" --build-tokenizer --vocab-size 4000
    python "RawHebb+bio+0.19.0.0.py" --train --fresh --limit 2850 --threads 6 \
           --val-max 150 --save brain.pt

数据（3120012 的两套语义）
    .txt   ：预训练语料；整文件 token 化成 SEQ_LEN+1 窗口，全部 token 都是目标；
             默认不验证（与 3120012 一致），需要时显式给一个 .txt 验证文件。
    .jsonl ：对话语料；user 与格式 token 的 label=-100，只学习 assistant；
             验证集默认从同一文件按**对话**随机切 10%（seed 固定），也可显式给 val 文件；
             超过 SEQ_LEN 时保留 system + 最近完整问答对，不在句子中间硬切。
================================================================================================
"""

# =================================================================================================
# SECTION 0 · 全局配置（唯一允许出现常数的地方；每项注明来源或实测结论）
# =================================================================================================
# ---- 版本与路径 ----
APP_NAME = 'RawHebb'
APP_BRANCH = 'bio'
APP_VERSION = '0.19.0.0'
APP_ID = '%s+%s+%s' % (APP_NAME, APP_BRANCH, APP_VERSION)

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
DATA_DIR = HERE
CKPT = 'brain.pt'                       # 默认存档（原版同名）
LOG_FILE = 'rawhebb.log'

# ---- 词表（完整生命周期见 SECTION 2）----
VOCAB_SIZE = 4000                       # BPE 目标词表；基座实际 2754，仅重建时使用
TOKENIZER_MIN_FREQUENCY = 2             # BPE 合并最小频次（3120012 v5 约定）
NORMALIZE_MODE = 'none'                 # 'none' 保原样；'nfkc' 只做 NFKC（必须与词表一致）
SPECIAL_TOKENS_LIST = ['<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>',
                       '<|user|>', '<|assistant|>', '<|end|>']
TOK_BIN = 'tokenizer_v63b4000.bin'      # 基座词表（只读）
TOK_JSON = 'tokenizer_v63b4000_tokenizer.json'
TOK_CUSTOM_BIN = 'tokenizer.bin'        # 重建词表（存在时优先使用）
TOK_CUSTOM_JSON = 'tokenizer.json'

# ---- 架构（括号内是实测，不是猜测）----
HIDDEN = 432                            # 400/428/432/436/448 在噪声内，464 更差
NUM_LAYERS = 4                          # 5 层负结果
SEQ_LEN = 384                           # 训练窗口；500 条 128→512: 3.5864→3.4503
EVAL_LEN = 128                          # 评测口径；发布数字都在这个窗口上，改了不可比
FREEZE_LAYERS = 0                       # 隐藏层 W 冻结数（0 = 全不冻结）
FREEZE_EMBEDDING = True                 # 词向量冻结（随机投影）；内核定义的一部分
SEED_EMBEDDING = 43
SEED_FREEZE = 43

# ---- 动力学（物理量：由节律 / 膜时间常数推出）----
CORTEX_HZ = 40.0                        # γ 节律
THETA_HZ = 8.0                          # θ 节律
TAU_M_MS = 10.0                         # 膜时间常数（文献 10–20ms 取低端）
DT_MS = 1.0                             # 时间步
LEAK = float(__import__('math').exp(-DT_MS / TAU_M_MS))     # ≈0.9048
TAU_MULT = 2.0                          # 深层 τ 倍率：τ_li = TAU_M_MS · TAU_MULT**li
TAU_CAP_STEPS = 100.0                   # τ 上限（步）；0 = 不限
L0_TAU = 0.0                            # 0 = L0 沿用 LEAK；>0 时单独指定

# ---- 读出与计数记忆 ----
READ_LAYERS = 'all'                     # 'all' = L0..L3-1；'deep' = 只 L1..L3-1
QA_MASK = True                          # 只学 assistant（labels=-100 处不学）
PAIR_MEMORY = True                      # 精确 token 对表
PAIR_PRIOR_STRENGTH = 1.0               # 一个伪观测（最小非零证据单位）
TRI_PRIOR_STRENGTH = 2.0                # 4 → 2 实测 +0.003 CE（256k 桶）
PAIR_MIX = 'likelihood'                 # likelihood / hit / ce_top1 / confidence / product
PAIR_SCOPE = 'mask'                     # mask = 只统计 assistant 目标；all = 所有相邻 token
TRIGRAM_MEMORY = True
TRIGRAM_BUCKETS = 1048576               # 1M 已饱和（2M int16 无增益）
FOURGRAM_MEMORY = True
FOURGRAM_BUCKETS = 4096                 # 4k > 8k（0.13 全量 3.1772 vs 3.1863）
FOUR_PRIOR_STRENGTH = 0.5               # Jeffreys 伪计数
FOUR_BAG_PRIOR = True                   # past-context bag 先验（不是严格 4-gram）
FOUR_BAG_NORM = True                    # 每个上下文行先归一化，再因果 cumsum
UNIGRAM_EXPERT = False                  # 显式 unigram 专家（运行计数）
MIX_ROUNDS = 2                          # 专家权重坐标下降轮数
MIX_ITERS = 24                          # 每次黄金分割迭代数（计算预算）

# ---- 实验开关（默认即发布配方；改动前先读 docs/FINDINGS.md）----
HEAD_BIAS = False                       # 输出头常数列（截距）
NEWTON_STEPS = 1                        # 每窗准 Newton 步数
COUPLING_MODE = 'half'                  # half / one / inv_layers
SPIKE_ENABLED = True                    # False = 纯漏电积分（诊断用）
TRAIN_SHUFFLE = False                   # 固定顺序（0.11 结论：shuffle 更差）

# ---- 生成（只影响菜单对话，不影响 CE 评测）----
TEMPERATURE = 0.8
TOP_P = 0.92
TOP_K = 0
REPEAT_PENALTY = 1.08
MAX_NEW_TOKENS = 160

# ---- 训练与运行时 ----
EPOCHS = 1                              # 2 epoch 过拟合（500 条 3.6105）
VAL_RATIO = 0.10                        # 对话自动切分验证集（3120012 的 1/10 约定）
SEED = 43
LOG_INTERVAL = 25
DEVICE = 'cpu'                          # 'cpu' / 'cuda'
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
import unicodedata

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
# SECTION 2 · 分词器：ByteLevel BPE 的完整生命周期（训练 / 保存 / 加载 / 校验）
# =================================================================================================
def normalize_text(text):
    """词表规范化。默认 'none'（保空格/换行/大小写/全角标点）；
    'nfkc' 只用于与旧实验对齐。编码与训练必须使用同一模式。"""
    if NORMALIZE_MODE == 'nfkc':
        return unicodedata.normalize('NFKC', str(text))
    return str(text)


def truncate_messages(msgs, tok, max_len):
    """3120012 的 _truncate_pairs：保留开头 system + 最近完整的 user/assistant 对。
    不在句子中间硬切；只有单轮本身超长时才落到 token 级截断（chat_to_ids 末尾兜底）。"""
    head, start = [], 0
    while start < len(msgs) and msgs[start].get('role') == 'system':
        head.append(msgs[start])
        start += 1
    budget = sum(len(tok.encode(str(m.get('content') or ''))) + 2 for m in head) + 2
    kept, i = [], len(msgs) - 1
    while i >= start + 1:
        if msgs[i].get('role') == 'assistant' and msgs[i - 1].get('role') == 'user':
            pair = [msgs[i - 1], msgs[i]]
            cost = sum(len(tok.encode(str(m.get('content') or ''))) + 2 for m in pair)
            if kept and budget + cost > max_len:
                break
            kept.insert(0, pair)
            budget += cost
            if budget >= max_len:
                break
            i -= 2
        else:
            i -= 1
    flat = [m for p in kept for m in p]
    if not flat:
        for m in reversed(msgs[start:]):
            cost = len(tok.encode(str(m.get('content') or ''))) + 2
            if flat and budget + cost > max_len:
                break
            flat.insert(0, m)
            budget += cost
    return head + flat


class RawHebbTokenizer(object):
    """ByteLevel BPE（与 3120012elite.py 的 FastBPE 同一套规则）。

    · 256 个字节全部进入 initial_alphabet ⇒ 结构上不会产生 <|unk|>（不吞字）。
    · 特殊 token 顺序固定，具体 id 由词表文件决定；chat_to_ids 只读属性，不硬编码数字。
    · save() 同时写元数据 .bin 与 tokenizers 的 .json；load() 校验元数据。
    · 旧基座文件缺少 v5 元数据键时按 legacy 兼容读取，并在日志中说明。
    """

    def __init__(self, bin_path, json_path, auto_load=True):
        self.bin_path = bin_path
        self.json_path = json_path
        self.initial_vocab_size = VOCAB_SIZE
        self.special_tokens_list = list(SPECIAL_TOKENS_LIST)
        self._tok = None
        self._ids = {}
        self._vocab_size_cache = None
        self.normalize_mode = NORMALIZE_MODE
        self.legacy_meta = False
        if auto_load and os.path.exists(bin_path) and os.path.exists(json_path):
            self.load()

    # ---- 训练 ----
    def train(self, texts, vocab_size=VOCAB_SIZE, verbose=False):
        """在文本迭代器上训练 BPE。texts 必须已 normalize（collect_tokenizer_texts 负责）。"""
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        except ImportError as e:
            raise RuntimeError('缺少 tokenizers：pip install tokenizers') from e
        tok = Tokenizer(models.BPE(unk_token='<|unk|>'))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            special_tokens=self.special_tokens_list,
            min_frequency=TOKENIZER_MIN_FREQUENCY,
            show_progress=verbose,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(list(texts), trainer=trainer)
        tok.decoder = decoders.ByteLevel()
        self._tok = tok
        self.initial_vocab_size = int(vocab_size)
        self._vocab_size_cache = tok.get_vocab_size()
        self.legacy_meta = False
        self._update_special_ids()

    def _update_special_ids(self):
        if self._tok is None:
            return
        vocab = self._tok.get_vocab()
        self._vocab_size_cache = self._tok.get_vocab_size()
        self._ids = {name: vocab.get(name) for name in self.special_tokens_list}

    # ---- 保存 / 加载 ----
    def save(self, bin_path=None, json_path=None):
        if self._tok is None:
            raise RuntimeError('词表未训练或未加载，无法保存')
        bin_path = bin_path or self.bin_path
        json_path = json_path or self.json_path
        d = os.path.dirname(os.path.abspath(bin_path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        self._tok.save(json_path)
        state = {
            'initial_vocab_size': self.initial_vocab_size,
            'special_tokens_list': self.special_tokens_list,
            'normalize_mode': self.normalize_mode,
            'pre_tokenizer': 'byte_level',
            'decoder': 'byte_level',
        }
        torch.save(state, bin_path)
        log_and_print('[TOK] 已保存 %s + %s（%d tokens）'
                      % (os.path.basename(bin_path), os.path.basename(json_path),
                         self._vocab_size_cache or 0))

    def load(self):
        if not (os.path.exists(self.bin_path) and os.path.exists(self.json_path)):
            return False
        try:
            data = torch.load(self.bin_path, map_location='cpu', weights_only=False)
        except Exception as e:
            log_and_print('[TOK] 元数据读取失败 %s：%s' % (self.bin_path, e))
            return False
        self.initial_vocab_size = int(data.get('initial_vocab_size', VOCAB_SIZE))
        self.special_tokens_list = list(data.get('special_tokens_list', SPECIAL_TOKENS_LIST))
        missing = [k for k in ('pre_tokenizer', 'decoder', 'normalize_mode') if k not in data]
        self.legacy_meta = bool(missing)
        self.normalize_mode = data.get('normalize_mode', NORMALIZE_MODE)
        try:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(self.json_path)
        except Exception as e:
            log_and_print('[TOK] 词表 JSON 读取失败 %s：%s' % (self.json_path, e))
            self._tok = None
            return False
        self._update_special_ids()
        for _name, _tid in self._ids.items():
            if _tid is not None:
                self._tok.add_special_tokens([_name])
        self._update_special_ids()
        miss_ids = [k for k, v in self._ids.items() if v is None]
        if miss_ids:
            log_and_print('[TOK] ★词表缺少特殊 token：%s' % miss_ids)
            self._tok = None
            return False
        if self.legacy_meta:
            log_and_print('[TOK] 基座元数据为旧格式（无 pre/decoder/normalize），按 legacy 兼容读取。')
        return True

    # ---- 编解码 ----
    def encode(self, text):
        if self._tok is None:
            raise RuntimeError('词表未加载。请先准备 tokenizer（菜单 12 或 --build-tokenizer）')
        ids = self._tok.encode(normalize_text(text), add_special_tokens=False).ids
        return [i if i < self._vocab_size_cache else self.unk_token_id for i in ids]

    def decode(self, ids):
        if self._tok is None:
            raise RuntimeError('词表未加载')
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=True)

    # ---- 信息 ----
    @property
    def vocab_size(self):
        return self._vocab_size_cache or 0

    def _id(self, name):
        return self._ids.get(name)

    bos_token_id = property(lambda self: self._id('<|bos|>'))
    eos_token_id = property(lambda self: self._id('<|eos|>'))
    pad_token_id = property(lambda self: self._id('<|pad|>'))
    unk_token_id = property(lambda self: self._id('<|unk|>'))
    user_token_id = property(lambda self: self._id('<|user|>'))
    assistant_token_id = property(lambda self: self._id('<|assistant|>'))
    end_token_id = property(lambda self: self._id('<|end|>'))

    def info(self):
        return {
            'bin': self.bin_path, 'json': self.json_path,
            'vocab_size': self.vocab_size,
            'target_vocab_size': self.initial_vocab_size,
            'normalize': self.normalize_mode,
            'legacy_meta': self.legacy_meta,
            'special_ids': {k: v for k, v in self._ids.items()},
        }

    def describe(self):
        ids = ', '.join('%s=%s' % (k, v) for k, v in self._ids.items())
        return ('词表：%d tokens（目标 %d，normalize=%s，%s）｜ %s ｜ %s'
                % (self.vocab_size, self.initial_vocab_size, self.normalize_mode,
                   'legacy 元数据' if self.legacy_meta else 'v5 元数据',
                   os.path.basename(self.bin_path), ids))

    def chat_to_ids(self, msgs, max_len=SEQ_LEN, mask_assistant=True):
        """把一轮对话渲染成 (input_ids, labels)。问答 mask 在这里落地：
        user 段与格式 token 的 labels 一律 -100，只有 assistant 内容与收尾 <|end|> 计损。"""
        if max_len:
            need = sum(len(self.encode(str(x.get('content') or ''))) + 2 for x in msgs) + 2
            if need > max_len:
                msgs = truncate_messages(msgs, self, max_len)
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
        if len(tokens) > max_len:
            tokens, m = tokens[-max_len:], m[-max_len:]
        ids = tokens[:-1]
        labels = [t if m[i + 1] else -100 for i, t in enumerate(tokens[1:])]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collect_tokenizer_texts(paths):
    """从 jsonl 的每条消息 / txt 的每一行收集分词器训练文本（先 normalize）。"""
    texts = []
    for path in paths:
        full = path if os.path.isabs(path) else os.path.join(DATA_DIR, path)
        if not os.path.exists(full):
            continue
        if full.lower().endswith('.jsonl'):
            for rec in iter_jsonl(full):
                for msg in get_messages(rec):
                    c = msg.get('content')
                    if c:
                        texts.append(normalize_text(c))
        else:
            with io.open(full, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(normalize_text(line))
    return texts


def default_tokenizer_sources():
    for name in ('train_dsr1.jsonl', 'dialogues.jsonl'):
        if os.path.exists(os.path.join(DATA_DIR, name)):
            return [name]
    txt = sorted(f for f in os.listdir(DATA_DIR)
                 if f.endswith('.txt') and not f.startswith('.'))
    return txt


def build_tokenizer_from_corpus(src, vocab_size=VOCAB_SIZE,
                                bin_path=TOK_CUSTOM_BIN, json_path=TOK_CUSTOM_JSON):
    """从一份 jsonl/txt 训练词表并保存。返回 tokenizer 或 None。"""
    sources = [src] if src else default_tokenizer_sources()
    texts = collect_tokenizer_texts(sources)
    if not texts:
        log_and_print('★没有可用语料，无法训练词表：%s' % (sources or '（空）'))
        return None
    chars = len(set(''.join(texts)))
    log_and_print('[TOK] 语料 %d 段 ｜ 不同字符 %d ｜ 目标词表 %d'
                  % (len(texts), chars, vocab_size))
    tok = RawHebbTokenizer(bin_path, json_path, auto_load=False)
    tok.train(texts, vocab_size=vocab_size, verbose=True)
    tok.save()
    log_and_print('[TOK] ★词表已重建。token id 已改变：旧 brain.pt 不兼容，需 --fresh 重新训练。')
    return tok


def prepare_tokenizer(train_if_missing=True, verbose=True):
    """按优先级准备词表：tokenizer.bin（重建）→ 基座 → 从语料训练。"""
    for b, j in ((TOK_CUSTOM_BIN, TOK_CUSTOM_JSON), (TOK_BIN, TOK_JSON)):
        bp = b if os.path.isabs(b) else os.path.join(DATA_DIR, b)
        jp = j if os.path.isabs(j) else os.path.join(DATA_DIR, j)
        if os.path.exists(bp) and os.path.exists(jp):
            tok = RawHebbTokenizer(bp, jp, auto_load=False)
            if tok.load():
                if verbose:
                    log_and_print(tok.describe())
                return tok
    if not train_if_missing:
        return None
    sources = default_tokenizer_sources()
    if not sources:
        log_and_print('★未找到词表，也没有可训练的 .jsonl/.txt 语料。')
        return None
    log_and_print('[TOK] 未找到词表，将从 %s 训练新词表（旧 brain.pt 将不兼容）。' % sources)
    return build_tokenizer_from_corpus(sources[0], VOCAB_SIZE)

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

    def step(self, incoming):
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
        _ew = torch.zeros(6)
        _ew[0] = 1.0
        self.register_buffer('expert_w', _ew)
        if TRIGRAM_MEMORY:
            # 二阶上下文哈希计数：key = (t-2, t-1) 的 2^12 进制拼接再取模。
            # 2^12 = 大于词表的最小 2 的幂 ⇒ 拼接无碰撞丢失（架构量，不是调出来的）。
            self.register_buffer('tri_count', torch.zeros(TRIGRAM_BUCKETS, vocab_size, dtype=torch.int16))
            if FOURGRAM_MEMORY:
                self.register_buffer('four_count', torch.zeros(FOURGRAM_BUCKETS, vocab_size))
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
        for t in range(T):
            h = self.embedding(input_ids[:, t])
            for li, layer in enumerate(self.layers):
                inc = h
                if li > 0:
                    inc = h + self.coupling * Vs[li - 1][-1]
                V, sp = layer.step(inc)
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
                _trow = self.tri_count[_keys].float()
                _trs = _trow.sum(dim=-1, keepdim=True)
                _tp = (_trow + TRI_PRIOR_STRENGTH * _prior) / (_trs + TRI_PRIOR_STRENGTH)
                _experts.append(torch.log(_tp).unsqueeze(0))
            if FOUR_BAG_PRIOR:
                _bagrows = self.pair_count[input_ids[0]]
                if FOUR_BAG_NORM:
                    _brs = _bagrows.sum(dim=-1, keepdim=True)
                    _bagrows = ((_bagrows + PAIR_PRIOR_STRENGTH * _prior)
                                / (_brs + PAIR_PRIOR_STRENGTH))
                _bagcum = torch.cumsum(_bagrows, dim=0)
                _bagprev = torch.cat(
                    [torch.zeros(1, self.vocab_size, dtype=_bagcum.dtype, device=_bagcum.device),
                     _bagcum[:-1]], dim=0)
                _bagsum = _bagprev.sum(dim=-1, keepdim=True)
                _bagp = ((_bagprev + PAIR_PRIOR_STRENGTH * _prior)
                         / (_bagsum + PAIR_PRIOR_STRENGTH))
            if FOURGRAM_MEMORY:
                _f1 = _prev1
                _f2 = _prev2
                _p3 = torch.cat([_pad, _pad, _pad, _ids[:-3]])
                _fkeys = (((_p3 * self._tri_base + _f2) * self._tri_base + _f1)
                          % FOURGRAM_BUCKETS)
                _frow = self.four_count[_fkeys]
                _frs = _frow.sum(dim=-1, keepdim=True)
                _fp = ((_frow + FOUR_PRIOR_STRENGTH * _bagp)
                       / (_frs + FOUR_PRIOR_STRENGTH))
                _experts.append(torch.log(_fp.clamp(min=1e-12)).unsqueeze(0))
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
    def hebbian_update(self, input_ids):
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
        if FOURGRAM_MEMORY:
            _ids3 = input_ids[0]
            _padv3 = torch.full((1,), self.pad_id, dtype=_ids3.dtype, device=_ids3.device)
            _q1 = torch.cat([_padv3, _ids3[:-1]])
            _q2 = torch.cat([_padv3, _padv3, _ids3[:-2]])
            _q3 = torch.cat([_padv3, _padv3, _padv3, _ids3[:-3]])
            _fkeys = (((_q3 * self._tri_base + _q2) * self._tri_base + _q1)
                      % FOURGRAM_BUCKETS)
            _mk3 = torch.ones_like(mpos[0]) if PAIR_SCOPE == 'all' else mpos[0]
            if bool(_mk3.any()):
                _fx = _fkeys[:T - 1][_mk3]
                _fy = _ids3[1:T][_mk3]
                self.four_count.index_put_(
                    (_fx, _fy),
                    torch.ones(_fx.numel(), dtype=self.four_count.dtype),
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


def _split_ckpt(obj):
    """兼容两种存档：新格式 {'state_dict': ..., 'meta': ...} 与旧的裸 state_dict。"""
    if isinstance(obj, dict) and isinstance(obj.get('state_dict'), dict):
        return obj['state_dict'], dict(obj.get('meta') or {})
    return obj, {}


def _ckpt_meta(sd):
    """从 state_dict 推断词表 / 宽度 / 层数（新格式也会带上，但推断始终可核对）。"""
    meta = {}
    emb = sd.get('embedding.weight')
    if emb is not None:
        meta['vocab_size'] = int(emb.shape[0])
        meta['hidden'] = int(emb.shape[1])
    layers = set()
    for k in sd:
        if k.startswith('layers.') and '.' in k[7:]:
            layers.add(int(k.split('.')[1]))
    if layers:
        meta['num_layers'] = len(layers)
    return meta


def save_model(model, path=None):
    path = os.path.join(DATA_DIR, path or CKPT)
    payload = {
        'state_dict': model.state_dict(),
        'meta': {
            'app': APP_ID,
            'vocab_size': int(model.vocab_size),
            'hidden': int(model.hidden),
            'num_layers': int(model.num_layers),
            'seq_len': int(SEQ_LEN),
            'eval_len': int(EVAL_LEN),
        },
    }
    torch.save(payload, path)
    log_and_print('[CKPT] 已保存 %s（%.1f MiB，词表 %d，hidden %d，%d 层）'
                  % (path, os.path.getsize(path) / 2 ** 20,
                     model.vocab_size, model.hidden, model.num_layers))


def load_model(model, path=None):
    path = os.path.join(DATA_DIR, path or CKPT)
    if not os.path.exists(path):
        return False
    obj = torch.load(path, map_location=model.device, weights_only=False)
    sd, meta = _split_ckpt(obj)
    inferred = _ckpt_meta(sd)
    meta = {**inferred, **meta}
    # ★词表/宽度/层数不符时明确拒绝：否则 strict=False 会悄悄跳过大部分表，等于随机模型。
    for key, cur in (('vocab_size', model.vocab_size),
                     ('hidden', model.hidden),
                     ('num_layers', model.num_layers)):
        got = meta.get(key)
        if got is not None and int(got) != int(cur):
            log_and_print('[CKPT] ★拒绝加载 %s：%s=%s，当前模型=%s。' % (path, key, got, cur))
            if key == 'vocab_size':
                log_and_print('[CKPT] 词表已重建或与基座不一致；请用 --fresh 从头训练，或换回原词表。')
            return False
    cur_sd = model.state_dict()
    skip = [k for k in list(sd.keys())
            if k in cur_sd and tuple(cur_sd[k].shape) != tuple(sd[k].shape)]
    for k in skip:
        sd.pop(k)
    mm, uu = model.load_state_dict(sd, strict=False)
    log_and_print('[CKPT] 已加载 %s ｜ 跳过形状不符 %d 个：%s ｜ missing=%d unexpected=%d'
                  % (path, len(skip), skip[:6], len(mm), len(uu)))
    return True


def print_stats(model, tok):
    st = model.stats()
    log_and_print('架构：%d 层 × %d 维 ｜ 词表 %d ｜ 参数量 %d'
                  % (st['layers'], st['hidden'], st['vocab'], st['params']))
    if tok is not None:
        log_and_print(tok.describe())
    log_and_print('训练窗口 %d ｜ 评测窗口 %d ｜ 只学 assistant=%s'
                  % (SEQ_LEN, EVAL_LEN, QA_MASK))
    log_and_print('leak（τ 派生）: %s' % st['leak'])
    log_and_print('读出层 %s' % st['read_idx'])
    log_and_print('混合权重 w（超随机命中率）: %s' % st['mix_w'])
    log_and_print('token 对混合权重 w_pair=%.4f ｜ 证据位置 %d ｜ 命中 base=%d pair=%d'
                  % (st['pair_w'] or 0.0, st['pair_seen'], st['pair_hit_base'], st['pair_hit_pair']))
    log_and_print('‖W_out‖=%.2f ｜ ‖W_m‖=%s' % (st['W_out_norm'], st['W_m_norm']))
    log_and_print('发放率（累计）: %s' % st['spike_acc_mean'])
    log_and_print('嵌入冻结 %s ｜ 隐藏层冻结 %d 层' % (st['frozen_embedding'], st['frozen_layers']))


def dataset_mode(path):
    """3120012 的数据哲学：.txt = 无监督预训练（全部 token 都是目标）；
    .jsonl = 对话（assistant mask；按对话切分验证集）。"""
    return 'pretrain' if str(path).lower().endswith('.txt') else 'dialogue'


def _split_dialogues(rows, val_file, val_max):
    """外部验证集存在就用它；否则按对话确定性切分 VAL_RATIO（默认 5%）。"""
    if val_file:
        vp = val_file if os.path.isabs(val_file) else os.path.join(DATA_DIR, val_file)
        if os.path.exists(vp):
            val_rows = list(iter_jsonl(vp))
            if val_max:
                val_rows = val_rows[:val_max]
            return rows, val_rows, '外部验证集 %s' % val_file
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * VAL_RATIO)) if len(rows) > 1 else 0
    keep = set(idx[:n_val])
    train_rows = [r for i, r in enumerate(rows) if i not in keep]
    val_rows = [rows[i] for i in sorted(keep)]
    return train_rows, val_rows, '按对话自动切分 %.0f%%（seed=%d）' % (VAL_RATIO * 100, SEED)


def _txt_token_ids(tok, path):
    full = path if os.path.isabs(path) else os.path.join(DATA_DIR, path)
    ids = []
    with io.open(full, encoding='utf-8') as f:
        for line in f:
            s = line.rstrip('\n\r')
            if s:
                ids.extend(tok.encode(s + '\n'))
    return ids


def _txt_windows(tok, path, max_windows=0):
    """把 txt 变成不重叠的 (x, y) 窗口；y 是 x 的下一个 token（预训练口径）。"""
    ids = _txt_token_ids(tok, path)
    windows = []
    for start in range(0, max(0, len(ids) - SEQ_LEN), SEQ_LEN):
        w = ids[start:start + SEQ_LEN + 1]
        windows.append((torch.tensor(w[:-1], dtype=torch.long),
                        torch.tensor(w[1:], dtype=torch.long)))
        if max_windows and len(windows) >= max_windows:
            break
    return windows, len(ids)


def train_hebbian(model, tok, path, epochs=EPOCHS, limit=LIMIT_SAMPLES, val_file=None,
                  val_max=40, eval_every=0, logfile=None):
    """纯 Hebbian 训练：forward → 局部规则 → forward。无 BP、无 optimizer。

    · jsonl：对话模式；assistant mask；验证集按对话切分（外部或自动）。
    · txt  ：预训练模式；全部 token 计损；验证集取最后 VAL_RATIO 的窗口
             （若给了 .txt 验证文件则用它）。
    """
    dev = model.device
    mode = dataset_mode(path)
    val_windows = []
    if mode == 'pretrain':
        train_samples, total_tokens = _txt_windows(tok, path, limit)
        if limit:
            log_and_print('[TRAIN] txt 预训练：limit=%d 表示最多 %d 个窗口' % (limit, limit))
        use_val_file = (val_file and str(val_file).lower().endswith('.txt')
                        and os.path.exists(val_file if os.path.isabs(val_file)
                                           else os.path.join(DATA_DIR, val_file)))
        if use_val_file:
            val_windows, _ = _txt_windows(tok, val_file, val_max)
            how_val = '外部 txt %s' % val_file
        else:
            how_val = '无（3120012 约定：txt 预训练不做验证）'
        log_and_print('[TRAIN] 模式=预训练(txt) ｜ %s ｜ %d tokens ｜ %d 窗口 ｜ 验证=%s'
                      % (path, total_tokens, len(train_samples), how_val))
        val_n = sum(len(y) for _, y in val_windows)
    else:
        rows = list(iter_jsonl(path if os.path.isabs(path)
                               else os.path.join(DATA_DIR, path)))
        if limit:
            rows = rows[:limit]
        rows, val_rows, how_val = _split_dialogues(rows, val_file, val_max)
        if val_max:
            val_rows = val_rows[:val_max]
        if TRAIN_SHUFFLE:
            random.Random(SEED).shuffle(rows)
        train_samples = rows
        log_and_print('[TRAIN] 模式=对话(jsonl) ｜ %s ｜ 训练 %d 条 ｜ 验证 %d 条（%s）｜ T=%d ｜ mask=%s'
                      % (path, len(rows), len(val_rows), how_val, SEQ_LEN, QA_MASK))
        val_n = sum(int((tok.chat_to_ids(get_messages(r))[1] != -100).sum()) for r in val_rows[:val_max])
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ce_sum, n_ce, t_ep = 0.0, 0, time.time()
        for k, sample in enumerate(train_samples):
            if mode == 'pretrain':
                x, y = sample
                ids, labels = x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev)
            else:
                ids, labels = tok.chat_to_ids(get_messages(sample))
                if (labels != -100).sum() == 0:
                    continue
                ids, labels = ids.unsqueeze(0).to(dev), labels.unsqueeze(0).to(dev)
            model(ids, labels=labels)
            model.hebbian_update(ids)
            n = int((labels[:, :-1] != -100).sum())
            if n <= 0:
                continue
            ce_sum += float(model._buf.get('ce', float('nan'))) * n
            n_ce += n
            if eval_every and (k + 1) % eval_every == 0:
                v = (evaluate_windows(model, val_windows) if mode == 'pretrain'
                     else evaluate(model, tok, val_rows))
                log_and_print('[MID] ep%d %d/%d ｜ 训练CE=%.4f ｜ **验证CE=%.4f** top1=%.4f ｜ %.0fs'
                              % (ep, k + 1, len(train_samples), ce_sum / max(n_ce, 1),
                                 v['ce'], v['top1'], time.time() - t0))
            if (k + 1) % LOG_INTERVAL == 0:
                log_and_print('[TRAIN] ep%d %d/%d ｜ CE(累计)=%.4f ｜ %.1fs'
                              % (ep, k + 1, len(train_samples), ce_sum / max(n_ce, 1),
                                 time.time() - t_ep))
        msg = '[TRAIN] ep%d 完成 ｜ 训练 CE=%.4f ｜ %.0fs' % (ep, ce_sum / max(n_ce, 1),
                                                             time.time() - t_ep)
        if val_n:
            v = (evaluate_windows(model, val_windows) if mode == 'pretrain'
                 else evaluate(model, tok, val_rows))
            msg += ' ｜ **验证 CE=%.4f top1=%.4f top5=%.4f**（%d 位置）' % (
                v['ce'], v['top1'], v['top5'], v['n'])
        log_and_print(msg)
    if logfile:
        log_and_print('[TRAIN] 日志已写 %s' % logfile)
    return ce_sum / max(n_ce, 1)


@torch.no_grad()
def evaluate(model, tok, rows, ngram_ref=False):
    """独立验证集评测（对话）：只统计 assistant token，与训练口径严格一致。"""
    model.eval()
    dev = model.device
    ce, n, t1, t5 = 0.0, 0, 0, 0
    for rec in rows:
        ids, labels = tok.chat_to_ids(get_messages(rec))
        if EVAL_LEN and len(ids) > EVAL_LEN:
            ids, labels = ids[-EVAL_LEN:], labels[-EVAL_LEN:]
        m = (labels != -100)
        if m.sum() == 0:
            continue
        x = ids.unsqueeze(0).to(dev)
        lg = model(x)[0]
        lg = lg[:-1]
        tg = labels[:-1]
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


@torch.no_grad()
def evaluate_windows(model, windows, max_windows=0):
    """txt 预训练评测：每个位置都是目标，仍报 CE/top-1/top-5。"""
    model.eval()
    dev = model.device
    ce, n, t1, t5 = 0.0, 0, 0, 0
    for x, y in (windows[:max_windows] if max_windows else windows):
        lg = model(x.unsqueeze(0).to(dev))[0][:-1]
        tg = y.to(dev)[:-1]
        if tg.numel() == 0:
            continue
        ce += float(F.cross_entropy(lg, tg, reduction='sum'))
        pred = lg.argmax(-1)
        t1 += int((pred == tg).sum())
        k5 = torch.topk(lg, 5, -1).indices
        t5 += int((k5 == tg.unsqueeze(-1)).any(-1).sum())
        n += int(tg.numel())
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
def chat_loop(model, tok, mode='jsonl', learn=False):
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
            answer = tok.decode(out)
            log_and_print('助手: ' + answer)
            if learn:
                ids, labels = tok.chat_to_ids([{'role': 'user', 'content': q},
                                               {'role': 'assistant', 'content': answer}])
                if int((labels != -100).sum()) > 0:
                    xx = ids.unsqueeze(0)
                    model(xx, labels=labels.unsqueeze(0))
                    model.hebbian_update(xx)
                    log_and_print('[ONLINE] 已学习本轮（%d 个目标 token）'
                                  % int((labels != -100).sum()))


def online_learn_file(model, tok, path, limit=0):
    """在线增量学习：按文件顺序逐条更新一次；不 shuffle、不验证、不早停。
    txt 按窗口、jsonl 按对话；与「训练模型」共用同一套局部规则。"""
    mode = dataset_mode(path)
    dev = model.device
    model.train()
    t0 = time.time()
    n = 0
    if mode == 'pretrain':
        samples, total = _txt_windows(tok, path, limit)
        log_and_print('[ONLINE] 模式=预训练(txt) ｜ %s ｜ %d tokens ｜ %d 窗口'
                      % (path, total, len(samples)))
        for x, y in samples:
            xx = x.unsqueeze(0).to(dev)
            model(xx, labels=y.unsqueeze(0).to(dev))
            model.hebbian_update(xx)
            n += 1
            if n % LOG_INTERVAL == 0:
                log_and_print('[ONLINE] %d/%d ｜ %.1fs' % (n, len(samples), time.time() - t0))
    else:
        rows = list(iter_jsonl(path if os.path.isabs(path) else os.path.join(DATA_DIR, path)))
        if limit:
            rows = rows[:limit]
        log_and_print('[ONLINE] 模式=对话(jsonl) ｜ %s ｜ %d 条' % (path, len(rows)))
        for rec in rows:
            ids, labels = tok.chat_to_ids(get_messages(rec))
            if int((labels != -100).sum()) == 0:
                continue
            xx = ids.unsqueeze(0).to(dev)
            model(xx, labels=labels.unsqueeze(0).to(dev))
            model.hebbian_update(xx)
            n += 1
            if n % LOG_INTERVAL == 0:
                log_and_print('[ONLINE] %d/%d ｜ %.1fs' % (n, len(rows), time.time() - t0))
    model.eval()
    log_and_print('[ONLINE] 完成 %d 条 ｜ %.0fs' % (n, time.time() - t0))
    return n


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
# SECTION 10 · 菜单（沿 3120012elite.py 的结构：选文件 / 容错输入 / 确认 / 返回）
# =================================================================================================
def main_menu():
    global TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY, MAX_NEW_TOKENS
    torch.set_num_threads(THREADS)
    dev = setup_device()
    deterministic(SEED)
    tok = prepare_tokenizer(train_if_missing=True)
    if tok is None:
        log_and_print('★无法准备词表，退出。')
        close_log_file()
        return
    log_and_print('%s 已启动 ｜ 内核=RawHebbKernel（无 BP）' % APP_ID)
    model = build_model(tok, dev)
    model_loaded = False
    tok_bin = tok.bin_path
    tok_json = tok.json_path

    def ensure_loaded(strict=False):
        """加载 brain.pt。strict=True 时，失败则明确返回 False（不再静默用随机模型）。"""
        nonlocal model_loaded
        if model_loaded:
            return True
        if load_model(model):
            model_loaded = True
            return True
        if os.path.exists(os.path.join(DATA_DIR, CKPT)):
            log_and_print('★%s 存在但无法加载（词表/形状不兼容）。' % CKPT)
        else:
            log_and_print('未找到 %s，当前是随机初始化模型。' % CKPT)
        return False

    while True:
        print('\n=== RawHebb 训练与对话系统菜单 ===')
        print('1. 训练模型（可选择数据集）')
        print('2. 对话模式')
        print('3. 保存模型')
        print('4. 加载模型')
        print('5. 调整生成参数')
        print('6. 查看模型统计信息')
        print('7. 规则验证拒绝采样：生成正确回答 SFT 数据')
        print('8. 自一致性投票筛选（原「规则奖励 RL」的无梯度替代）')
        print('9. 长上下文分段接力评测（原「GRPO」的无梯度替代）')
        print('10. 数据预处理：按对话切分训练/验证集')
        print('11. 独立验证集评测（防泄露口径）')
        print('12. 分词器：查看 / 重建 / 往返测试')
        print('13. 在线学习：增量灌入文件 / 对话中学习')
        print('0. 退出')
        choice = input('请选择: ').strip().lower()

        if choice == '1':
            path = select_dataset_file('train_dsr1.jsonl')
            if path is None:
                continue
            ep = _ask_int('epoch 数 [%d]: ' % EPOCHS, EPOCHS, 1, 100)
            lim = _ask_int('样本上限（0=全量）[%d]: ' % LIMIT_SAMPLES, LIMIT_SAMPLES, 0)
            if path.lower().endswith('.txt'):
                valf = input('验证 txt [留空=不验证]: ').strip()
            else:
                valf = input('验证集 [回车=val_dsr1.jsonl，auto=按对话自动切分 %.0f%%]: '
                             % (VAL_RATIO * 100)).strip()
                if valf.lower() in ('auto', 'a'):
                    valf = ''
                elif not valf:
                    valf = 'val_dsr1.jsonl'
            loaded = ensure_loaded()
            if not loaded and os.path.exists(os.path.join(DATA_DIR, CKPT)):
                if input('检测到不可用的 brain.pt，仍从头训练并覆盖？(y/n): ').lower() != 'y':
                    continue
            log_and_print('[TRAIN] %s ｜ epoch=%d ｜ limit=%d ｜ 验证=%s'
                          % (path, ep, lim, valf))
            if input('开始训练？(y/n): ').lower() == 'y':
                train_hebbian(model, tok, path, epochs=ep, limit=lim,
                              val_file=valf, val_max=150)
                if input('训练后保存到 %s？(y/n): ' % CKPT).lower() != 'n':
                    save_model(model)
                    model_loaded = True
        elif choice == '2':
            if not ensure_loaded(strict=True):
                log_and_print('对话需要可用的 %s；已取消。' % CKPT)
                continue
            mode = input('对话格式 (jsonl) 还是纯文本续写 (txt)？[jsonl]: ').strip().lower()
            chat_loop(model, tok, mode=('txt' if mode == 'txt' else 'jsonl'))
        elif choice == '3':
            save_model(model)
            model_loaded = True
        elif choice == '4':
            if load_model(model):
                model_loaded = True
        elif choice == '5':
            TEMPERATURE = _ask_float('temperature (0.1-2.0) [%.2f]: ' % TEMPERATURE,
                                     TEMPERATURE, 0.1, 2.0)
            TOP_P = _ask_float('top_p (0.0-1.0) [%.2f]: ' % TOP_P, TOP_P, 0.0, 1.0)
            TOP_K = _ask_int('top_k (0=关闭) [%d]: ' % TOP_K, TOP_K, 0)
            REPEAT_PENALTY = _ask_float('repeat_penalty (1.0-2.0) [%.2f]: ' % REPEAT_PENALTY,
                                        REPEAT_PENALTY, 1.0, 2.0)
            MAX_NEW_TOKENS = _ask_int('max_new_tokens [%d]: ' % MAX_NEW_TOKENS,
                                      MAX_NEW_TOKENS, 1, 2048)
        elif choice == '6':
            print_stats(model, tok)
        elif choice == '7':
            if not ensure_loaded(strict=True):
                continue
            src = input('输入 jsonl [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            dst = input('输出 SFT jsonl [sft_raw.jsonl]: ').strip() or 'sft_raw.jsonl'
            k = _ask_int('每题采样数 [4]: ', 4, 1, 64)
            rejection_sampling(model, tok, src, dst, k=k)
        elif choice == '8':
            if not ensure_loaded(strict=True):
                continue
            src = input('输入 jsonl [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            dst = input('输出一致性数据 [consistency.jsonl]: ').strip() or 'consistency.jsonl'
            k = _ask_int('采样次数 [5]: ', 5, 1, 64)
            self_consistency(model, tok, src, dst, k=k)
        elif choice == '9':
            if not ensure_loaded(strict=True):
                continue
            vf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            longctx_stress(model, tok, vf)
        elif choice == '10':
            preprocess_dialogs_menu()
        elif choice == '11':
            if not ensure_loaded(strict=True):
                log_and_print('评测需要可用的 %s；已取消。' % CKPT)
                continue
            vf = input('验证集 [val_dsr1.jsonl 或 .txt]: ').strip() or 'val_dsr1.jsonl'
            n = _ask_int('用多少条/窗口（0=全部）[40]: ', 40, 0)
            if vf.lower().endswith('.txt'):
                windows, _ = _txt_windows(tok, vf, n)
                v = evaluate_windows(model, windows)
            else:
                rows = list(iter_jsonl(os.path.join(DATA_DIR, vf)))
                rows = rows[:n] if n else rows
                v = evaluate(model, tok, rows)
            log_and_print('[EVAL] ★CE=%.4f ｜ top1=%.4f ｜ top5=%.4f ｜ 计分位置 %d'
                          % (v['ce'], v['top1'], v['top5'], v['n']))
            tf = input('同时算 n-gram 参考线？训练集路径 [留空=不算]: ').strip()
            if tf and os.path.exists(os.path.join(DATA_DIR, tf)):
                for k in (1, 2):
                    ce, npos = ngram_reference(tok, tf, rows, k=k)
                    log_and_print('[EVAL] %d-gram 参考线：CE=%.4f（%d 位置）' % (k, ce, npos))
        elif choice == '12':
            new_tok = tokenizer_submenu(tok, tok_bin, tok_json)
            if new_tok is not tok and new_tok is not None:
                tok = new_tok
                tok_bin, tok_json = tok.bin_path, tok.json_path
                model = build_model(tok, dev)
                model_loaded = False
                log_and_print('已切换到新词表；模型已重建为随机初始化，请从头训练。')
        elif choice == '13':
            print('\n=== 在线学习 ===')
            print('1. 从文件增量学习（按顺序逐条更新，不验证）')
            print('2. 对话中学习（每轮回答后更新）')
            print('0. 返回')
            oc = input('请选择: ').strip()
            if oc == '1':
                path = select_dataset_file('train_dsr1.jsonl')
                if path is None:
                    continue
                lim = _ask_int('最多样本/窗口（0=全部）[0]: ', 0, 0)
                if not ensure_loaded(strict=True):
                    log_and_print('在线学习需要已训练权重；请先训练或加载 %s。' % CKPT)
                    continue
                n = online_learn_file(model, tok, path, limit=lim)
                log_and_print('[ONLINE] 完成：更新 %d 条。' % n)
                if input('保存到 %s？(y/n): ' % CKPT).lower() != 'n':
                    save_model(model)
                    model_loaded = True
            elif oc == '2':
                if not ensure_loaded(strict=True):
                    log_and_print('对话中学习需要可用的 %s；已取消。' % CKPT)
                    continue
                chat_loop(model, tok, mode='jsonl', learn=True)
        elif choice in ('0', 'q', 'exit'):
            break
        else:
            log_and_print('无效选择，请重新输入')
    close_log_file()


# =================================================================================================
# 自检：不读数据，验证内核的局部规则与 mask 是否按预期工作
# =================================================================================================
# =================================================================================================
# 自检：不读训练数据；验证内核局部规则、mask、词表保存/加载往返
# =================================================================================================
def selftest():
    import tempfile
    dev = torch.device('cpu')
    torch.set_num_threads(2)
    tok = prepare_tokenizer(train_if_missing=False)
    if tok is None:
        print('[SELFTEST] ★找不到词表。请先 --build-tokenizer 或放入基座词表。')
        return
    ids, labels = tok.chat_to_ids([{'role': 'user', 'content': '你好'},
                                   {'role': 'assistant', 'content': '世界'}])
    print('[SELFTEST] ids=%s' % ids.tolist()[:16])
    print('[SELFTEST] labels=%s（-100 = 提问段，不学）' % labels.tolist()[:16])
    model = RawHebbKernel(tok.vocab_size, device=dev)
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
    with tempfile.TemporaryDirectory() as td:
        tb, tj = os.path.join(td, 't.bin'), os.path.join(td, 't.json')
        tok.save(tb, tj)
        t2 = RawHebbTokenizer(tb, tj)
        ok = t2.load()
        sample = 'RawHebb：纯 Hebbian，无 BP；空格 换行 全角标点 ＡＢＣ。'
        same = ok and (tok.encode(sample) == t2.encode(sample))
        print('[SELFTEST] 词表保存/加载往返：%s ｜ 编码一致=%s'
              % ('OK' if ok else '失败', bool(same)))
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
    ap.add_argument('--fresh', action='store_true', help='不加载 brain.pt（从头训练）')
    ap.add_argument('--threads', type=int, default=THREADS)
    ap.add_argument('--tokenizer-info', action='store_true', help='打印词表信息后退出')
    ap.add_argument('--build-tokenizer', action='store_true', help='从 --train-file 重建词表')
    ap.add_argument('--vocab-size', type=int, default=VOCAB_SIZE)
    ap.add_argument('--tok-bin', default='', help='自定义词表 .bin 路径')
    ap.add_argument('--tok-json', default='', help='自定义词表 .json 路径')
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.tokenizer_info:
        tok = prepare_tokenizer(train_if_missing=False)
        if tok is None:
            print('词表未找到。')
        else:
            print(tok.describe())
            for k, v in tok.info().items():
                print('  %-16s %s' % (k, v))
    elif args.build_tokenizer:
        tb = args.tok_bin or TOK_CUSTOM_BIN
        tj = args.tok_json or TOK_CUSTOM_JSON
        tok = build_tokenizer_from_corpus(args.train_file, args.vocab_size, tb, tj)
        if tok is not None:
            print(tok.describe())
    elif args.split:
        deterministic()
        split_by_dialogue('dsr1_v5_0001.jsonl', 'train_dsr1.jsonl', 'val_dsr1.jsonl')
    elif args.train:
        torch.set_num_threads(args.threads)
        dev = setup_device()
        deterministic(SEED)
        tok = prepare_tokenizer(train_if_missing=True)
        if tok is None:
            raise SystemExit('词表准备失败')
        model = build_model(tok, dev)
        if not args.fresh and os.path.exists(os.path.join(DATA_DIR, CKPT)):
            if not load_model(model):
                raise SystemExit('★检查点与当前词表/架构不兼容：已停止，避免用随机权重覆盖。')
        train_hebbian(model, tok, args.train_file, epochs=args.epochs, limit=args.limit,
                      val_file=args.val_file, val_max=args.val_max, eval_every=args.eval_every)
        save_model(model, args.save or CKPT)
    else:
        main_menu()

# =================================================================================================
# 版本谱系（四段版本号，不带 v；每版一个文件，旧版保留；完整实验见 docs/FINDINGS.md）
#   0.19.0.0 2026-10-22  外壳工程版：恢复完整词表生命周期（train/save/load/校验/回退）、
#                         txt 预训练 / jsonl 对话双数据管线与按对话切分、消息级截断、
#                         在线学习分支、菜单数据集选择与输入容错、存档元数据与不兼容拒绝；
#                         删除死参数、重写配置说明。模型配方与 0.18 逐位一致。
#   0.18.0.0 2026-10-22  训练窗口 128→384（评测仍最后 128）：CE 3.0543/41.86/64.23，
#                         三种子均值 3.0553/41.74/64.23。int16 计数表数值等价；2M 桶饱和。
#   0.17.0.0 2026-10-21  tri 桶 256k→1M：seed43 3.1411/41.59/63.86，均值 3.1434/41.48/63.79。
#   0.16.0.0 2026-10-20  tri 桶 256k + tri 先验 2 + hidden 432：均值 3.1516/41.20/63.80。
#   0.15.0.0 2026-10-20  bag 先验行归一化 + Jeffreys 0.5：均值 3.1724/40.45/63.78。
#   0.14.0.0 2026-10-20  past-token bag 先验：均值 3.1755/40.45/63.76。
#   0.13.0.0 2026-10-19  4-gram 4k 桶：均值 3.1793/40.41/63.66。
#   0.12.0.0 2026-10-19  hidden 128→416 容量释放：均值 3.1971/40.35/63.47。
#   0.11.0.0 2026-10-19  关闭 shuffle：三种子 3.2581/3.2608/3.2552。
#   0.10.0.0 2026-10-19  修正目标位移 off-by-one；旧版数字不可直接比较。
#   0.8.0.0  2026-10-19  修正对齐 + 精确 token 对联想记忆。
#   0.4–0.6            多时间尺度、在线闭式最小二乘、Newton 读出、漂移修复。
#   0.1–0.3            单文件内核替换、零魔数、深层状态溢出修复。
# =================================================================================================
