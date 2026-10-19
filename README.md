# RawHebb

**A language model that learns without back-propagation.**
Purely forward dynamics — leaky membrane integrators, threshold spikes, and local synaptic
plasticity. No optimiser, no loss backward pass, no global gradient, no attention, no hand-tuned
magic constants.

> Branch `bio` · current release **0.10.0.0** · license **AGPL-3.0**

中文速览见 [中文说明](#中文说明)。

---

## What makes this different

Most "no-backprop" claims are really "no backprop **through the layers**". RawHebb removes the
optimiser entirely. The only things that ever change are:

| What | Rule | Signal it sees |
|---|---|---|
| Output head `W_out` | Running Gram `A=Σxxᵀ`, `B=Σxy` + one stochastic Newton step | its own probability error |
| State memory `W_m[layer]` | Hebbian pull toward the **column prototype** of the next token | its own layer state |
| Token-pair memory `C[x,y]` | Pure Hebbian count `C[x,y] += 1`; read `p(y|x)=(C+λ·p_uni)/(row+λ)` | the input stream |
| Context memory `T[key,y]` | Same count rule on a hashed prefix of the last 2 tokens | the input stream |
| Expert weights | Likelihood coordinate descent on the probability simplex | its own held-out-free likelihood |
| Layer weights `w` | Normalised hit rate above chance `1/V` | that layer's argmax |

Nothing is back-propagated into hidden weights. Hidden `W` is frozen by design, so the network is a
strictly feed-forward dynamical system whose learning is *local synaptic plasticity*. The pair/context
tables are associative memories (outer-product Hebbian counts), not gradients.

The design constraints are physical, not algorithmic:

1. **Bio-inspired first.** Membrane time constants come from biological rhythms
   (`τ = 10 ms`, sampled at `Δt = 1 ms` ⇒ `leak = exp(−Δt/τ) = 0.9048`), not from a hyper-parameter sweep.
2. **Material constraints are boundaries.** Energy, spike emission and firing rate are physical limits —
   saturation is handled by the membrane dynamics, never by a hand-filled constant.
3. **No back-propagation in any disguise** — no eligibility traces, no two-pass comparison,
   no future-answer injection.
4. **No external watchdog.** Stability comes from the dynamics; there is no "reset if it exceeds X" monitor.
5. **Refractory feedback stays.** `t+1` write / backward read / per-layer self-reference is kept.
   It is not replaced by attention or an SSM.

---

## Repository layout

```
RawHebb+bio+0.1.0.0.py      single-file application, first release (kept for history)
RawHebb+bio+0.1.1.0.py      single-file application (kept for history)
RawHebb+bio+0.2.0.0.py      single-file application (kept for history)
RawHebb+bio+0.3.0.0.py      single-file application (kept for history)
RawHebb+bio+0.4.0.0.py      single-file application (kept for history)
RawHebb+bio+0.5.0.0.py      single-file application (kept for history - see below, it regressed)
RawHebb+bio+0.6.0.0.py      single-file application (kept for history)
RawHebb+bio+0.10.0.0.py     single-file application, current
RawHebb+seq128tri+0.9.33.0.py  the exact v121 experimental branch that became 0.10.0.0
RawHebb+lab+0.9.0.0.py      v121 experiment bench (all switches)
research/                   research lineage, v121 diagnostics and branch generators
docs/FINDINGS.md            what worked, what did not, and the measured numbers
```

Each version is a **complete single file**: menu, kernel, data layer, training, evaluation and
generation. Every hyper-parameter switch lives in `SECTION 0` at the top of the file.

### The menu

The interaction structure follows the `3120012elite` framework this project grew out of; only the
kernel was replaced.

```
1. 训练模型（继续训练）      Train (continue)
2. 对话模式                 Chat (jsonl dialogue / plain-text continuation)
3. 保存模型 / 4. 加载模型    Save / Load
5. 调整生成参数              Generation parameters
6. 查看模型统计信息          Model statistics
7. 规则验证拒绝采样 → SFT     Rejection sampling → SFT data   (no gradients)
8. 自一致性投票筛选           Self-consistency voting          (gradient-free stand-in for reward RL)
9. 长上下文分段接力评测       Long-context segmented stress test (gradient-free stand-in for GRPO)
10. 退出                     Exit
11. 数据预处理：超长对话切分   Data preprocessing
12. 独立验证集评测           Held-out validation (leak-checked)
```

Items 8 and 9 replace RL/GRPO because policy gradients *are* back-propagation. The slots and their
interaction are preserved; the mechanism is swapped for the closest gradient-free equivalent, and
that substitution is stated in the menu itself rather than hidden.

---

## Quick start

```bash
# 1. kernel self-test (no data needed, < 30 s)
python "RawHebb+bio+0.10.0.0.py" --selftest

# 2. split a corpus into train/val by dialogue, with leak checks
python "RawHebb+bio+0.10.0.0.py" --split

# 3. train (add --fresh to skip the warm start from brain.pt)
python "RawHebb+bio+0.10.0.0.py" --train --train-file train.jsonl --epochs 1 --fresh

# 4. or just run the menu
python "RawHebb+bio+0.10.0.0.py"
```

Requirements: Python 3.10+, PyTorch 2.x, `tokenizers`.

---

## Data handling: mask and leakage

Real conversational corpora (e.g. `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`)
are supported, along with the older `conversations` key.

**Question-answer masking.** Labels for the user turn and all format tokens are set to `-100`;
only assistant content (and its closing `<|end|>`) counts toward the loss. This matters more than it
looks — see the bug we found and fixed in v114: the original update rule read its targets from
`input_ids[:, 1:]` and therefore *ignored* `-100` entirely, i.e. it was training the model to predict
the questions while reporting a masked loss. Unit test: `research/_v114_mask_check.py`.

**Leakage control.** The validation split is made **by dialogue**, never by random line, and four
checks are run and reported:

| check | result on the 3000-dialogue corpus |
|---|---|
| exact duplicates (SHA-1 of user+assistant) | 0 |
| cross-split SHA-1 overlap | 0 |
| validation user-prefix (first 120 chars) found in train | 0 / 150 |
| validation 8-grams that also occur in train | 7.47 % (shared Chinese phrasing, reported honestly) |

---

## Measured state of the model (read this before judging it)

Full corpus (2 850 dialogues, one epoch, CPU), held-out 150 dialogues (29 657 scored assistant tokens):

| metric | **0.10.0.0** | bigram (add-1, same mask) | unigram (add-1) | old 0.6.0.0 |
|---|---|---|---|---|
| cross-entropy | **3.269** | 3.864 | 5.778 | 5.862 |
| top-1 | **38.62 %** | 27.82 % | 5.67 % | 14.85 % |
| top-5 | **62.99 %** | 55.24 % | 16.86 % | 27.17 % |

The model is now **better than a bigram on both CE and top-1**, with no back-propagation anywhere.

**The decisive finding of this release: a target-shift bug that had been present since 0.1.0.0.**
`chat_to_ids` already returns `labels[t] = token t+1` (the standard next-token target), but the
training/evaluation code paired `logits[t]` with `labels[t+1]`, i.e. it was solving a **two-step-ahead**
task. Scoring the same 0.6.0.0 checkpoint under three alignments makes it unambiguous:

| alignment | CE | top-1 |
|---|---|---|
| old `logits[t]` vs `labels[t+1]` | 5.87 | 12.58 % |
| standard `logits[t]` vs `labels[t]` | 6.52 | 2.68 % |

After the fix (`RawHebb+align+0.8.0.0.py`), the same 500-dialogue subset jumped from
CE 5.87 / top-1 12.6 % to **5.10 / 23.7 %**. Every number before 0.10.0.0 is therefore on a different
task and must not be compared directly.

**Three local memories, all Hebbian.** The recurrent state memory is kept; on top of it are an exact
token-pair count table (V×V, `C[x,y] += 1`) and a hashed 2-token-prefix count table. The three
experts (neural base / pair / prefix) are mixed on the probability simplex by coordinate descent on
the likelihood; there is no learning rate and no hand-set mixture weight. The prior strength is one
pseudo-observation; the hash width is an architecture quantity.

**Negative results are kept in the repo** (see `docs/FINDINGS.md` and the v121 branch files):
4-gram/5-gram hashed contexts are worse than the 2-token prefix (sparsity), Kneser–Ney discounting
improves CE slightly but costs top-1, two epochs overfit, and full-content chunked training is
promising but ~30 min for 500 dialogues, so it was not scaled up.

Also recorded: GPU acceleration does **not** help this kernel (kernel-launch bound); batching makes it
faster but destroys the online local rules.

See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full chronology, including everything that failed.

---

## Versioning

Every change is a **new file**: copy the current version, edit the copy, rename with the next
version number. Old versions are never overwritten.

```
RawHebb+bio+<a>.<b>.<c>.<d>.py   (four-segment version, no `v` prefix)
```

| version | what changed |
|---|---|
| 0.1.0.0 | first release. Kernel replaced with the pure-Hebbian `RawHebbKernel`; menu structure preserved; items 8/9 swapped to gradient-free equivalents; added held-out validation. |
| 0.1.1.0 | fixes two measured defects in 0.1.0.0: (i) memory read-out normalised by RMS over the *whole* vocabulary, which amplified the few written channels by √N early in training (merged peak 44, CE 8.4 → 24.9) — now normalised only over channels that hold memory; (ii) `W_out` LMS switched to a relative step size, so it cannot diverge without an external watchdog. |
| 0.2.0.0 | zero magic numbers: every learning rate, mixture coefficient and memory decay is self-scaling. |
| 0.4.0.0 | cascade gain fix + multi-timescale read-out + online closed-form least squares. |
| 0.6.0.0 | full-overwrite stochastic Newton read-out; first version past unigram on both CE and top-1 (old alignment). |
| 0.7.x | token-identity memory experiments; all kept as negative results, none released. |
| **0.10.0.0** | **fixes the target-shift bug; adds exact token-pair and hashed-prefix Hebbian memories; three-expert likelihood mixing; SEQ_LEN=128.** Full corpus: CE **3.269**, top-1 **38.62 %**, top-5 **62.99 %**. |

---

## Citation

If you use this software, please cite the repository (a Zenodo DOI is minted from each GitHub
release). See [`CITATION.cff`](CITATION.cff).

## License

AGPL-3.0. See [`LICENSE`](LICENSE).

---

## 中文说明

**RawHebb 是一个不用反向传播训练的语言模型。**
只有前向动力学：漏电膜电位积分、阈值放电、突触局部可塑性。没有优化器、没有 loss 反传、
没有全局梯度、没有 attention、没有手填的魔数。

能改的只有四样东西，而且每一样只看自己那一份信号：输出头 `W_out`（概率空间的 LMS）、
联想记忆 `W_m`（Hebbian 列原型）、读出增益 `γ`、层权重 `w`。隐藏层 `W` 冻结，
所以整个网络是一个严格前向的动力学系统，学习完全来自**局部**突触可塑性。

**菜单沿用 `3120012elite` 的 11 项结构与交互，只换内核。** 其中原版的「奖励 RL」「GRPO」需要策略
梯度（就是反传），与纲领冲突，已替换为无梯度等价物（自一致性投票 / 长上下文分段接力评测），
并在菜单里明说，没有偷偷改。

**问答 mask 与防泄露**：user 段与格式 token 的 label 置 `-100`，只统计 assistant；
训练/验证按**对话**切分，并做精确重复、跨集 sha1、user 前缀、8-gram 四项泄露检查（结果见 README）。

**必须说清楚的现状（0.10.0.0）**：全量 2850 条 / 1 epoch 后，验证 **CE 3.269 / top-1 38.62% /
top-5 62.99%**，两个口径都超过 bigram（CE 3.864 / top-1 27.82%）。这是修掉一个从 0.1.0.0 起就存在的
**目标位移 off-by-one** 之后的结果：旧代码用 `logits[t]` 对 `labels[t+1]`，实际训练的是「隔一个
token」的任务；同一份 0.6.0.0 权重在标准对齐下只有 top-1 2.68%。所以 **0.10.0.0 之前的全部数字
都不能直接和现在比**。

现在的记忆是三支并联：慢状态原型、精确 token 对计数（V×V，Hebbian 累加）、前缀哈希计数；
三支权重由似然坐标下降在单纯形上自定，没有学习率、没有手填混合系数。负结果也保留在仓库：
4/5-gram 更稀疏更差、KN 折扣 CE 略好但 top-1 掉、2 epoch 过拟合、全内容切窗训练太慢未放大。
GPU 对这套内核**没有收益**（核启动受限），批量化会破坏在线局部规则。
完整实验史（含全部失败项）见 `docs/FINDINGS.md`。