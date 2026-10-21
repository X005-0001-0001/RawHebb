# RawHebb

**A language model that learns without back-propagation.**
Purely forward dynamics — leaky membrane integrators, threshold spikes, and local synaptic
plasticity. No optimiser, no loss backward pass, no global gradient, no attention, no hand-tuned
magic constants.

> Branch `bio` · current release **0.14.0.0** · license **AGPL-3.0**

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
RawHebb+bio+0.11.0.0.py     single-file application (kept for history)
RawHebb+bio+0.12.0.0.py     single-file application (kept for history)
RawHebb+bio+0.13.0.0.py     single-file application (kept for history)
RawHebb+bio+0.14.0.0.py     single-file application, current
RawHebb+seq128tri+0.9.33.0.py  the exact v121 experimental branch that became 0.10.0.0
RawHebb+trinoshuf+0.9.52.0.py  the exact v122 experimental branch that became 0.11.0.0
RawHebb+tri4buck64hidden416+0.9.137.0.py  the exact v123 branch that became 0.12.0.0
RawHebb+tri4buck64hidden416four4k+0.9.164.0.py  the exact v124b branch that became 0.13.0.0
RawHebb+tri4buck64hidden416four4kfourbagprior+0.9.186.0.py  the exact v135 branch that became 0.14.0.0
RawHebb+tri4buck64hidden416four8k+0.9.158.0.py  best top-5 branch (63.84 %), lower CE/top-1
RawHebb+tri4buck64hidden352+0.9.127.0.py  best top-1 branch of v123 (40.61 %)
RawHebb+lab+0.9.0.0.py      v121 experiment bench (all switches)
research/                   research lineage, v121-v124 diagnostics, branches and generators
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
python "RawHebb+bio+0.13.0.0.py" --selftest

# 2. split a corpus into train/val by dialogue, with leak checks
python "RawHebb+bio+0.13.0.0.py" --split

# 3. train (add --fresh to skip the warm start from brain.pt)
python "RawHebb+bio+0.13.0.0.py" --train --train-file train.jsonl --epochs 1 --fresh

# 4. or just run the menu
python "RawHebb+bio+0.13.0.0.py"
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

Full corpus (2 850 dialogues, one epoch, CPU), held-out 150 dialogues. The model is evaluated on
the same last-128-token windows it trains on (18 900 scored assistant tokens); the n-gram baselines
below use exactly the same windows and mask (see `research/_v122_bigram_ref.py`):

| metric | **0.14.0.0** | 0.13.0.0 | bigram (unigram-prior, same window) | bigram add-1 | unigram add-1 | old 0.6.0.0* |
|---|---|---|---|---|---|---|
| cross-entropy | **3.174** | 3.177 | 3.809 | 4.456 | 5.785 | 5.862 |
| top-1 | **40.43 %** | 40.43 % | 31.32 % | 31.39 % | 4.83 % | 14.85 % |
| top-5 | **63.77 %** | 63.69 % | 58.05 % | 58.07 % | 15.85 % | 27.17 % |

\* 0.6.0.0 was measured on the old two-step-ahead task; it is shown for lineage only and is not
directly comparable.

The model is **better than a window-matched bigram on CE, top-1 and top-5**, with no
back-propagation anywhere. 0.14.0.0 keeps the hidden-416 base and the 4k count expert, but the
four expert now uses a **bag-of-past-tokens prior** (the exact pair table accumulated over the
window, causal) instead of the trigram prior. It improves 0.13.0.0 by 0.0032 CE and 0.08 points
top-5 at equal top-1. **Seed robustness:** three seeds give CE 3.1740 / 3.1773 / 3.1753 (mean
3.1755), top-1 40.43 / 40.53 / 40.40 % (mean 40.45 %) and top-5 63.77 / 63.72 / 63.79 % (mean
63.76 %); the 0.13.0.0 means are 3.1793 / 40.41 % / 63.66 %, so all three improve. The 8k-bucket
variant still has the best single top-5 (63.84 %) and is kept as a top-5 branch. The capacity scan
remains 3.2248 / 3.2246 / 3.2182 / 3.2076 / 3.2061 / 3.2006 / 3.1946 for hidden 192 / 224 / 256 /
288 / 352 / 384 / 416, and hidden352 still has the best top-1 of the no-bag family (40.61 %).

> **Naming correction (2026-10-20).** The 0.13.0.0 / 0.14.0.0 key uses `_tri_base = 4096` and
> `FOURGRAM_BUCKETS = 4096`, so the 4-gram hash collapses to the last token: it is a last-token
> count table, not a strict 4-gram. 0.13 interpolates it with the trigram prior; 0.14 with the
> bag-of-past prior. The measured scores are unchanged; a true 4-gram key is much worse (see
> `docs/FINDINGS.md`, v129). The release labels are kept for lineage.

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

**Negative results are kept in the repo** (see `docs/FINDINGS.md` and the v121-v124 branch files):
4-gram/5-gram hashed contexts are worse than the 2-token prefix (sparsity), Kneser–Ney discounting
improves CE slightly but costs top-1, two epochs overfit, and full-content chunked training is
promising but ~30 min for 500 dialogues, so it was not scaled up. The v122 round also rejected
tri→pair backoff, count-adaptive per-position mixing, a linear (probability-space) mixture, a causal
prefix-cache expert (its fitted weight was exactly zero), and 192/256-token windows. The v123 round
scaled the hidden width from 128 to 416 and rejected reverse/length order, pair prior 2/4, tri prior
2/3/5/8, 8k/32k/128k buckets, seq96/160/192/256, layers 3/5, hidden 64, newton2, bias,
noise/spike/coupling switches, mix rounds 1/4, and all-pair scope. The v124 round rejected
last-successor copy, prompt copy, 4-gram backoff at 64k buckets (weight 1.6e-5 ≈ 0); 4k/8k buckets now work and are the 0.13.0.0 / top-5 branch, and a top-1 mixture objective
(CE collapse).

Also recorded: GPU acceleration does **not** help this kernel. On an idle machine, 500 dialogues
take 232 s on CPU and 233 s on an RTX 5080 (training CE 3.9879 vs 3.9880). Batching the forward
pass alone is slower (0.635 s/dialogue at batch 8 vs 0.466 at batch 1) because the serial
per-dialogue `W_m`/count/Newton updates dominate; batching those would change the online local
rules. See `docs/FINDINGS.md`, v136.

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
| 0.10.0.0 | fixes the target-shift bug; adds exact token-pair and hashed-prefix Hebbian memories; three-expert likelihood mixing; SEQ_LEN=128. Full corpus: CE 3.269, top-1 38.62 %, top-5 62.99 %. |
| 0.11.0.0 | fixes the training order (no shuffle). Three full-corpus seeds: CE 3.2581 / 3.2608 / 3.2552, top-1 38.70 / 38.63 / 38.86 %, top-5 63.02 / 63.04 / 63.08 %. Corrected window-matched bigram: CE 3.809, top-1 31.32 %. |
| 0.12.0.0 | scales the hidden width 128 → 416 (fixed order, tri prior 4, 64k tri buckets). Full corpus: CE 3.1946, top-1 40.34 %, top-5 63.52 %. Capacity scan 192–416; top-1 peaks at hidden 352 (40.61 %). New cache branches and a 64k-bucket 4-gram were rejected. |
| 0.13.0.0 | adds the 4k count expert (last-token table + trigram prior; the hash key collapses, see the naming correction above). Full corpus: CE 3.1772, top-1 40.43 %, top-5 63.69 %, beating 0.12.0.0 on all three. The 8k variant keeps the best top-5 (63.84 %). |
| **0.14.0.0** | **replaces the four expert's trigram prior with a bag-of-past-tokens prior** (exact pair table accumulated over the window, causal; no new counts or constants). Full corpus seed43: CE **3.1740**, top-1 **40.43 %**, top-5 **63.77 %**. Three seeds: 3.1740 / 3.1773 / 3.1753 (mean 3.1755), 40.43 / 40.53 / 40.40 % (mean 40.45 %), 63.77 / 63.72 / 63.79 % (mean 63.76 %); the 0.13 means are 3.1793 / 40.41 % / 63.66 %. |

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

**必须说清楚的现状（0.14.0.0）**：全量 2850 条 / 1 epoch 后，验证 **CE 3.1740 / top-1 40.43% /
top-5 63.77%**（hidden416 + 4k 计数专家 + past-token bag 先验；0.13 是 3.1772/40.43/63.69）。
同窗口、同 mask 的 bigram
（unigram 先验）是 CE 3.809 / top-1 31.32% / top-5 58.05%，add-1 bigram 是 CE 4.456 / top-1
31.39%；**三个指标都超过 bigram**。旧版本还修掉了从 0.1.0.0 起就存在的 **目标位移 off-by-one**：
旧代码用 `logits[t]` 对 `labels[t+1]`，实际训练的是「隔一个 token」的任务；同一份 0.6.0.0 权重
在标准对齐下只有 top-1 2.68%，所以 **0.10.0.0 之前的数字不能直接比**。

0.14.0.0 在 0.13 的 hidden416 + 4k 计数专家上，把 four 专家的先验从 trigram 分布换成
**窗口内过去 token 的 bag 分布**：复用精确 pair 表做累计计数，没有新表、没有新常数。
相对 0.13：CE -0.0032、top-5 +0.08、top-1 持平。三种子 CE 3.1740/3.1773/3.1753（均值
**3.1755**），top-1 40.43/40.53/40.40%（均值 **40.45%**），top-5 63.77/63.72/63.79%（均值
**63.76%**）；0.13 均值 3.1793/40.41%/63.66%，三项均改善。见 `docs/FINDINGS.md`。

0.13.0.0 在 hidden416 上加 4k 计数专家（last-token 表 + trigram 先验回退），比 0.12 三项全好：
CE -0.0174、top-1 +0.09、top-5 +0.17。8k 桶变体的 top-5 更高（63.84%），但 CE/top-1 略低，
作为 top-5 分支保留。4-gram 64k 桶曾因太稀疏而权重≈0；4k/8k 桶才让它生效。
**命名更正：** 0.13/0.14 的 key 因 `_tri_base = 4096` 与 `FOURGRAM_BUCKETS = 4096` 相同而塌缩成
最后一个 token，实际是「last-token 计数」专家，不是严格 4-gram；0.13 用 trigram 先验，0.14 用
bag 先验。分数不变，真 4-gram 明显更差（FINDINGS v129）。

现在的记忆是三支并联：慢状态原型、精确 token 对计数（V×V，Hebbian 累加）、前缀哈希计数；
三支权重由似然坐标下降在单纯形上自定，没有学习率、没有手填混合系数。负结果也保留在仓库：
4/5-gram 更稀疏更差、KN 折扣 CE 略好但 top-1 掉、2 epoch 过拟合、全内容切窗训练太慢未放大；
v123/v124 又排除了 order/prior/bucket/seq/layers、lastcopy/promptcopy/4-gram/top1mix。
GPU 对这套内核**没有收益**：空闲机 500 条对照，CPU 232 秒、RTX 5080 233 秒（训练 CE
3.9879/3.9880）；只批量化 forward 反而更慢（batch=8 为 0.635 秒/条，B=1 为 0.466 秒/条），
因为逐对话的 `W_m`/计数/Newton 更新是串行的；要批量化就得改写在线上规则（FINDINGS v136）。
完整实验史（含全部失败项）见 `docs/FINDINGS.md`。