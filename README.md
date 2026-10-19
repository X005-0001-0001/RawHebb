# RawHebb

**A language model that learns without back-propagation.**
Purely forward dynamics — leaky membrane integrators, threshold spikes, and local synaptic
plasticity. No optimiser, no loss backward pass, no global gradient, no attention, no hand-tuned
magic constants.

> Branch `bio` · current release **0.1.1.0** · license **AGPL-3.0**

中文速览见 [中文说明](#中文说明)。

---

## What makes this different

Most "no-backprop" claims are really "no backprop **through the layers**". RawHebb removes the
optimiser entirely. The only things that ever change are:

| What | Rule | Signal it sees |
|---|---|---|
| Output head `W_out` | LMS on the **probability** error `e_t = onehot(y_t) − softmax(logits_t)` | its own readout |
| Associative memory `W_m[layer]` | Hebbian pull toward the **column prototype** of the next token | its own layer state |
| Readout gain `γ` | local correlation `err · readout` | its own readout |
| Layer weights `w` | local correlation `err · layer_readout` | that layer's readout |

Nothing is back-propagated into hidden weights. Hidden `W` is frozen by design, so the network is a
strictly feed-forward dynamical system whose learning is *local synaptic plasticity*.

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
RawHebb+bio+0.1.1.0.py      single-file application, current
research/                 the full research lineage (v115 → v117) and its diagnostics
docs/FINDINGS.md          what worked, what did not, and the measured numbers
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
python "RawHebb+bio+0.1.1.0.py" --selftest

# 2. split a corpus into train/val by dialogue, with leak checks
python "RawHebb+bio+0.1.1.0.py" --split

# 3. train
python "RawHebb+bio+0.1.1.0.py" --train --train-file train.jsonl --epochs 1 --limit 500

# 4. or just run the menu
python "RawHebb+bio+0.1.1.0.py"
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

On a held-out set of 150 dialogues (29 771 scored assistant tokens), the best research checkpoint:

| metric | model | unigram | bigram |
|---|---|---|---|
| cross-entropy | **5.96** | 5.88 | **3.96** |
| top-1 | 2.77 % | 6.82 % | — |

The model sits at the **unigram level**. That is not yet a language model, and the repository
documents it as such rather than dressing it up. Two findings are worth stating up front because
they cost us several rounds:

**1. Most of the CE gain was calibration, not correctness.** Decomposing the trained model's logits:

| branch | CE | top-1 |
|---|---|---|
| full | 6.04 | 2.61 % |
| base head only | 7.76 | **5.63 %** |
| memory branch only | 6.09 | 1.96 % |

The base head *ranks* better than the whole model. Sweeping the memory gain `γ` without retraining:

| γ | 0 | 0.1 | 0.3 | 1.0 | 1.434 (learned) |
|---|---|---|---|---|---|
| CE | 7.76 | 7.50 | 7.04 | 6.12 | **6.04** |
| top-1 | 5.63 % | **6.34 %** | 5.14 % | 2.93 % | 2.61 % |

The local rule pushes `γ` to the CE optimum, which is the top-1 worst point. **CE and top-1 are
zero-sum here.** Every earlier version was judged on CE alone; that was a mistake, and top-1 is now a
first-class gate.

**2. The bottleneck is representation, not the merge rule.** Three probes, all run without training:

- The three deep read-outs are 0.93–0.95 cosine-similar per position and 91–94 % linearly
  reconstructible from each other — yet the existing sum is *better* than every alternative
  (de-correlation 7.3660, L3-only 7.3718, current 7.3477). The merge was never the problem.
- 1-NN retrieval on the layer states matches the linear read-out (0.45 % vs 0.43 %): there is no
  hidden information the read-out fails to extract.
- The deep state is the *best* available feature, not a liability: swapping the base head's feature
  from the slow layer (τ=80) to the fast one (τ=10) makes things dramatically worse (CE 7.92, top-1 0.05 %).

Also recorded, because negative results are results: GPU acceleration does **not** help this kernel.
It is a 200-step Python loop of small ops, so it is kernel-launch bound (3.04 s/sample on GPU vs
1.78 s on CPU). Batching makes it 15× faster but destroys the online local rules.

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
| 0.1.1.0 | fixes two measured defects in v0.1.0: (i) memory read-out normalised by RMS over the *whole* vocabulary, which amplified the few written channels by √N early in training (merged peak 44, CE 8.4 → 24.9) — now normalised only over channels that hold memory; (ii) `W_out` LMS switched to a relative step size, so it cannot diverge without an external watchdog. |

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

**必须说清楚的现状**：模型目前停在 **unigram 水平**（CE 5.96 / unigram 5.88 / bigram 3.96，
top-1 2.77%），还不是一个语言模型。仓库如实记录这一点。两条关键结论：
① 之前的 CE 改善主要来自**校准**而非正确性——分解后底板的 top-1（5.63%）反而比整个模型（2.61%）好，
γ 在 CE 与 top-1 之间零和；② 瓶颈在**表征**不在合并——三个探针都证明合并方式已经接近最优。
GPU 对这套内核**没有收益**（核启动受限），批量化会破坏在线局部规则。
完整实验史（含全部失败项）见 `docs/FINDINGS.md`。