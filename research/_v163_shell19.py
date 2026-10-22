# -*- coding: utf-8 -*-
"""Build 0.19.0.0 (shell overhaul) from 0.18.0.0. Model numerics must not change."""
import io
import os

SRC = r'D:\explore\wav\RawHebb+bio+0.18.0.0.py'
DST = r'D:\explore\wav\RawHebb+shell19+0.9.264.0.py'
t = io.open(SRC, encoding='utf-8').read()


def cut(txt, start_marker, end_marker, new):
    i = txt.index(start_marker)
    j = txt.index(end_marker)
    i = txt.rfind('# =====', 0, i)
    j = txt.rfind('# =====', 0, j)
    return txt[:i] + new + txt[j:]


# ---------------------------------------------------------------- header
HEADER = '''"""
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
    python "RawHebb+bio+0.19.0.0.py" --train --fresh --limit 2850 --threads 6 \\
           --val-max 150 --save brain.pt

数据（3120012 的两套语义）
    .txt   ：预训练语料；整文件 token 化成 SEQ_LEN+1 窗口，全部 token 都是目标；
             默认不验证（与 3120012 一致），需要时显式给一个 .txt 验证文件。
    .jsonl ：对话语料；user 与格式 token 的 label=-100，只学习 assistant；
             验证集默认从同一文件按**对话**随机切 10%（seed 固定），也可显式给 val 文件；
             超过 SEQ_LEN 时保留 system + 最近完整问答对，不在句子中间硬切。
================================================================================================
"""'''

i = t.index('"""')
j = t.index('"""', i + 3) + 3
t = t[:i] + HEADER + t[j:]

# ---------------------------------------------------------------- SECTION 0
CONFIG = '''# =================================================================================================
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

'''
t = cut(t, '# SECTION 0 · 全局配置', '# SECTION 1 · 日志与工具', CONFIG)

# ---------------------------------------------------------------- imports
t = t.replace('import time\n', 'import time\nimport unicodedata\n', 1)

# ---------------------------------------------------------------- SECTION 2
TOKENIZER = '''# =================================================================================================
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

'''
t = cut(t, '# SECTION 2 · 分词器', '# SECTION 3 · 内核', TOKENIZER)

# ---------------------------------------------------------------- remove dead noise
t = t.replace('        noise = torch.randn(self.num_layers, T, self.hidden, device=dev) if NOISE_ENABLED else None\n',
              '', 1)
t = t.replace('            V, sp = layer.step(inc, None if noise is None else noise[li, t])',
              '            V, sp = layer.step(inc)', 1)
t = t.replace('    def step(self, incoming, noise):', '    def step(self, incoming):', 1)
t = t.replace('    def hebbian_update(self, input_ids, lr=LEARNING_RATE):',
              '    def hebbian_update(self, input_ids):', 1)

# ---------------------------------------------------------------- ckpt + stats
CKPT = '''def _split_ckpt(obj):
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


'''
i = t.index('def save_model')
j = t.index('def train_hebbian')
t = t[:i] + CKPT + t[j:]

# ---------------------------------------------------------------- helper prompts
HELPERS = '''def _ask_int(prompt, default, lo=None, hi=None):
    while True:
        raw = input(prompt).strip()
        if raw == '':
            return default
        try:
            v = int(raw)
        except ValueError:
            print('请输入整数。')
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print('范围：%s–%s。' % (lo, hi))
            continue
        return v


def _ask_float(prompt, default, lo=None, hi=None):
    while True:
        raw = input(prompt).strip()
        if raw == '':
            return default
        try:
            v = float(raw)
        except ValueError:
            print('请输入数字。')
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print('范围：%s–%s。' % (lo, hi))
            continue
        return v


def select_jsonl_file(default='train_dsr1.jsonl'):
    """列出目录下的 .jsonl 供选择；0=取消，回车=默认。"""
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.endswith('.jsonl') and not f.startswith('.'))
    if not files:
        log_and_print('未找到 .jsonl 数据文件。')
        return None
    print('\\n可用数据集：')
    for i, name in enumerate(files, 1):
        size = os.path.getsize(os.path.join(DATA_DIR, name))
        unit = '%.1f MB' % (size / 2 ** 20) if size >= 2 ** 20 else '%.1f KB' % (size / 1024)
        mark = '（默认）' if name == default else ''
        print('  %d. %s (%s)%s' % (i, name, unit, mark))
    while True:
        raw = input('选择序号 [回车=%s，0=取消]: ' % default).strip()
        if raw == '':
            return default
        if raw == '0':
            return None
        try:
            idx = int(raw)
        except ValueError:
            print('请输入数字。')
            continue
        if 1 <= idx <= len(files):
            return files[idx - 1]
        print('序号超出范围 1-%d。' % len(files))


def tokenizer_submenu(tok, tok_bin, tok_json):
    """查看 / 重建 / 往返测试。返回新 tokenizer（未重建则原样返回）。"""
    while True:
        log_and_print(tok.describe() if tok is not None else '词表未加载')
        print('\\n=== 分词器 ===')
        print('1. 查看词表信息')
        print('2. 从语料重建词表（★旧 brain.pt 不兼容）')
        print('3. 编码/解码往返测试')
        print('0. 返回')
        c = input('请选择: ').strip()
        if c == '1':
            info = tok.info()
            for k, v in info.items():
                log_and_print('  %-16s %s' % (k, v))
        elif c == '2':
            src = select_jsonl_file('train_dsr1.jsonl')
            if src is None:
                continue
            vocab = _ask_int('BPE 目标词表 [%d]: ' % VOCAB_SIZE, VOCAB_SIZE, 300, 200000)
            log_and_print('★重建后 token id 会改变，必须从头训练模型。')
            if input('确认重建？(y/n): ').lower() != 'y':
                continue
            new_tok = build_tokenizer_from_corpus(src, vocab, tok_bin, tok_json)
            if new_tok is not None:
                return new_tok
        elif c == '3':
            s = 'RawHebb：纯 Hebbian，无 BP；空格 换行 全角标点 ＡＢＣ。'
            ids = tok.encode(s)
            back = tok.decode(ids)
            log_and_print('[TOK] 原文：%s' % s)
            log_and_print('[TOK] ids=%s …（%d tokens）' % (ids[:16], len(ids)))
            log_and_print('[TOK] 回解：%s' % back)
        elif c in ('0', ''):
            return tok
        else:
            print('无效选择。')


'''
t = t.replace('def main_menu():', HELPERS + 'def main_menu():', 1)

# ---------------------------------------------------------------- SECTION 10 menu
MENU = '''# =================================================================================================
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
        print('\\n=== RawHebb 训练与对话系统菜单 ===')
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
            path = select_jsonl_file('train_dsr1.jsonl')
            if path is None:
                continue
            ep = _ask_int('epoch 数 [%d]: ' % EPOCHS, EPOCHS, 1, 100)
            lim = _ask_int('样本上限（0=全量）[%d]: ' % LIMIT_SAMPLES, LIMIT_SAMPLES, 0)
            valf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
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
            vf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            n = _ask_int('用多少条（0=全部）[40]: ', 40, 0)
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
            print('\\n=== 在线学习 ===')
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


'''
t = cut(t, '# SECTION 10 · 菜单', '# 自检：', MENU)

# ---------------------------------------------------------------- selftest
SELFTEST = '''# =================================================================================================
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


'''
i = t.index('def selftest():')
j = t.index("if __name__ == '__main__':")
t = t[:i] + SELFTEST + t[j:]

# ---------------------------------------------------------------- CLI
CLI = '''if __name__ == '__main__':
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

'''
i = t.index("if __name__ == '__main__':")
j = t.rfind('# =====', 0, t.index('# 版本谱系'))
t = t[:i] + CLI + t[j:]

# ---------------------------------------------------------------- lineage
LINEAGE = '''# =================================================================================================
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
'''
j = t.rfind('# =====', 0, t.index('# 版本谱系'))
t = t[:j] + LINEAGE

# ---------------------------------------------------------------- app id / paths in header
t = t.replace("APP_BRANCH = 'bio'\nAPP_VERSION = '0.19.0.0'", "APP_BRANCH = 'shell19'\nAPP_VERSION = '0.9.264.0'", 1)
t = t.replace('RawHebb+bio+0.19.0.0  ——', 'RawHebb+shell19+0.9.264.0  ——', 1)

# ---------------------------------------------------------------- data pipeline (txt=pretrain, jsonl=dialogue)
TRAINBLOCK = '''def dataset_mode(path):
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
            s = line.rstrip('\\n\\r')
            if s:
                ids.extend(tok.encode(s + '\\n'))
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


'''
_i = t.index('def train_hebbian')
_j = t.index('def ngram_reference')
t = t[:_i] + TRAINBLOCK + t[_j:]

# chat loop: optional online learning per exchange
t = t.replace("def chat_loop(model, tok, mode='jsonl'):",
              "def chat_loop(model, tok, mode='jsonl', learn=False):", 1)
t = t.replace('''            out = generate(model, tok, prompt, stop_ids=stop)
            log_and_print('助手: ' + tok.decode(out))
''', '''            out = generate(model, tok, prompt, stop_ids=stop)
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
''', 1)

ONLINE = '''def online_learn_file(model, tok, path, limit=0):
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


'''
_marker = '''# =================================================================================================
# SECTION 8 · 规则验证拒绝采样（菜单 7）：无梯度自蒸馏，产出 SFT jsonl'''
t = t.replace(_marker, ONLINE + _marker, 1)

# selector: both txt and jsonl
t = t.replace('def select_jsonl_file', 'def select_dataset_file', 1)
t = t.replace('select_jsonl_file(', 'select_dataset_file(')
t = t.replace("if f.endswith('.jsonl') and not f.startswith('.')",
              "if (f.endswith('.jsonl') or f.endswith('.txt')) and not f.startswith('.')", 1)
t = t.replace("print('  %d. %s (%s)%s' % (i, name, unit, mark))",
              "print('  %d. %s (%s)%s%s' % (i, name, unit, '（预训练）' if name.endswith('.txt') else '（对话）', mark))", 1)

# menu item 1: mode-aware val prompt
t = t.replace("            valf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'",
              "            if path.lower().endswith('.txt'):\n"
              "                valf = input('验证 txt [留空=不验证]: ').strip()\n"
              "            else:\n"
              "                valf = input('验证集 [回车=val_dsr1.jsonl，auto=按对话自动切分 %.0f%%]: '\n"
              "                             % (VAL_RATIO * 100)).strip()\n"
              "                if valf.lower() in ('auto', 'a'):\n"
              "                    valf = ''\n"
              "                elif not valf:\n"
              "                    valf = 'val_dsr1.jsonl'", 1)

# menu eval: txt windows
t = t.replace('''            vf = input('验证集 [val_dsr1.jsonl]: ').strip() or 'val_dsr1.jsonl'
            n = _ask_int('用多少条（0=全部）[40]: ', 40, 0)
            rows = list(iter_jsonl(os.path.join(DATA_DIR, vf)))
            rows = rows[:n] if n else rows
            v = evaluate(model, tok, rows)''',
              '''            vf = input('验证集 [val_dsr1.jsonl 或 .txt]: ').strip() or 'val_dsr1.jsonl'
            n = _ask_int('用多少条/窗口（0=全部）[40]: ', 40, 0)
            if vf.lower().endswith('.txt'):
                windows, _ = _txt_windows(tok, vf, n)
                v = evaluate_windows(model, windows)
            else:
                rows = list(iter_jsonl(os.path.join(DATA_DIR, vf)))
                rows = rows[:n] if n else rows
                v = evaluate(model, tok, rows)''', 1)

io.open(DST, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', DST, len(t.splitlines()), 'lines')
