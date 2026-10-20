# Findings — the honest chronology

Everything below was measured in this repository's own `research/` scripts. Negative results are
kept, because they are the reason several plausible-looking ideas are no longer being pursued.

---

## The setting

* corpus: 3 000 real dialogues, 2.87 M characters, 2 messages each (user + assistant)
* tokenizer: 2 754 tokens (ByteLevel BPE), inherited from the project baseline
* architecture: 4 leaky-integrator layers, H = 128, T = 200, ~8.9 M parameters
* learnable: `W_out` (LMS), `W_m` (Hebbian), `γ`, `w` — nothing else, no optimiser

Reference points on the same held-out split:

| baseline | CE | top-1 |
|---|---|---|
| uniform (`ln V`) | 7.92 | 0.04 % |
| unigram | 5.88 | 6.82 % |
| bigram | **3.96** | — |
| best model (v115, 1 epoch) | 5.96 | 2.77 % |

The model is at the unigram level. Everything below is the search for why.

---

## v111 — the lookup table

Token-level associative memory in L0 alone reached CE 6.21. It works because it degenerates into a
near-exact n-gram table keyed on recent token identity. Good CE, but not long-context modelling.

## v112 — breaking the L0 monopoly

L0 memory frozen, τ grown per layer (10/20/40/80 steps), read-out restricted to L1–L3.
Gates: ① 7.4786 < 7.52 ✔ ② 7.3477 < 7.52 ✔ ③ ΔCE +0.2838 < +0.30 ✘ ④ all layer weights positive ✔

The premise we started from was "L1/L2 are net noise because they are collinear with L3". Three
probes, all **without training**, killed that premise:

| probe | result | conclusion |
|---|---|---|
| collinearity | per-position cos 0.93–0.95; each layer 91–94 % linearly reconstructible from the other two | yes, they are collinear |
| residual information | after removing the explainable part, top-1 = 2.4× / **1.2×** / 3.2× chance | L2's private information *is* noise |
| merge scan | current sum **7.3477** beats de-correlation 7.3660, L3-only 7.3718, equal-weight 7.3725, private-only 7.6326 | the merge was never broken |

A gain sweep then showed why two layers beat three: the 3-layer sum prefers a volume factor of 0.75
(CE 7.3121) while the 2-layer sum prefers exactly 1.0. It was **volume**, not redundancy.

## v113 — energy-preserving merge

One change: divide the merged read-out by `‖w‖₂`. `V113_NORM=0` is bit-identical to v112 (verified).
Training CE 7.4286 (better by 0.05), inference CE 7.3469 (better by 0.001). Gate ③ unchanged.

**The fix worked on the trajectory and not on the endpoint.** That was the clue that the merge was
never the bottleneck.

## v114 — real corpus, question-answer mask

The dataset layer already produced `-100` on user turns, but the update rule read its targets from
`input_ids[:, 1:]` and **ignored the mask entirely**: the model was being trained to predict the
questions while reporting a masked loss. Fixed in six places; unit test confirms the user-token
memory column receives exactly zero update while assistant columns update normally.

On real data, L0 memory back on beats L0 memory off (6.3108 vs 6.8331), consistent with the k-NN
probe: at this scale the identity of the current token carries most of the next-token signal.

Full epoch (2 850 dialogues): CE 6.04, then flat. Training CE 6.02 — the model cannot even fit the
data it sees. Not overfitting, not data starvation: a capacity problem.

## v115 / v116 / v117 — locating the real ceiling

**Hypothesis 1 — "the slow state carries no token identity, feed the base head from the fast
layer".** Tested by swapping the feature layer without retraining:

| base feature | CE | top-1 |
|---|---|---|
| last layer (τ=80) | 7.76 | 5.63 % |
| L0 (τ=10) | **7.92** | **0.05 %** |

**Falsified, badly.** The deep state is the best feature available.

**Hypothesis 2 — "the memory branch swamps the base".** Decomposition of the trained model
(7 960 scored positions):

| branch | CE | top-1 |
|---|---|---|
| full | 6.0439 | 2.61 % |
| base only | 7.7622 | **5.63 %** |
| memory only | 6.0899 | 1.96 % |

Confirmed, and sharper than expected: the base head ranks better than the entire model. γ sweep
(no retraining):

| γ | 0 | 0.1 | 0.3 | 0.5 | 1.0 | 1.434 (learned) |
|---|---|---|---|---|---|---|
| CE | 7.76 | 7.50 | 7.04 | 6.66 | 6.12 | **6.04** |
| top-1 | 5.63 % | **6.34 %** | 5.14 % | 3.69 % | 2.93 % | 2.61 % |

**γ is zero-sum between CE and top-1.** The local rule `γ ← γ + lr·mean(err·readout)` is rewarded for
sharpening the distribution, not for getting the argmax right. Every version up to here was judged
on CE; that was the methodological error. top-1 is now a gate.

**Hypothesis 3 — "GPU makes this faster".** Measured, then dropped:

| configuration | throughput |
|---|---|
| CPU, B=1 | 1.78 s/sample |
| GPU, B=1 | 3.04 s/sample (slower) |
| GPU, B=32 | 0.115 s/sample (15× faster) **but learns worse** |

Batching breaks the online local rules: the effective displacement of a memory column after k
sequential writes is `1−(1−lr)^k` (0.47 at k=32, lr=0.02), while a parallel write moves it by `lr`
(0.02). Naive batching costs 1.0 CE (6.11 → 7.12); adding the exact effective step overshoots and
collapses `γ` to its lower bound (CE 7.89). A pure Hebbian online rule is inherently sequential.

Other recorded dead ends:

* **Transient read-out** `dV = V(t) − V(t−1)`: 2.0× / 1.6× / 2.8× chance vs 7.9× / 12.3× / 12.3× for `V`.
* **Shorter L0 τ** (1/2/4/10 ms): 6.5902 / 6.5199 / 6.3727 / **6.3108**. The 10-step average beats
  the instantaneous value — the hypothesis that "τ→1 catches token identity" is falsified.
* **k-NN on layer states** matches the linear read-out (0.45 % vs 0.43 %): no hidden information.

---

## What this implies for the next version

1. Gate on **CE and top-1 together**; CE alone can be improved while predictions get worse.
2. The local rule for `γ` must reward a **discrete** event (argmax correct) rather than correlation.
3. The unigram ceiling is a representation problem: the state must actually carry the identity of the
   current token. The most direct testable move is to let the associative memory be keyed by the
   current token's embedding — Hebbian association between neighbouring tokens is exactly that — and
   let the deep layers modulate rather than replace it.

---

## v118 / v119 — removing the magic numbers (and one instructive failure)

Every rule used to carry a hand-picked constant: `LR_MEM=0.02`, `MEM_DECAY_EVERY`, `LR_OUT`,
`LR_GAMMA`, `LR_W`, plus clamps on `w` (±3) and `γ` ([0.1, 10]). Those are exactly the kind of
numbers that work on one corpus and explode on the next.

**v118** replaced all of them with self-scaling forms: running-mean memory (`lr = 1/(n+1)`),
hit-rate mixing, direction-normalised `1/(t+1)` output steps, and no clamps.

It **failed**, and the failure was informative: validation CE 7.84 (≈ uniform 7.92), top-1 2.15 %,
and the mixing weights collapsed to `[0.0, 0.0, 0.0001]`.

Why: a *whole-history* running mean averages away recency. Every token column ends up close to the
global mean state vector, so the column prototypes become mutually collinear, their argmax is
essentially random, and a mix weighted by "has this layer been right" correctly concludes that no
layer has earned a voice. The system was right; the memory was the problem.

**v119** fixed the memory in the most physical way available: the forgetting curve **is** the
membrane time constant. The memory update became an exponential average with decay `(1 − leak)` of
that layer — L0 0.9048, deeper layers 0.951 / 0.975 / 0.988 — i.e. the same biological constants the
network already runs on, no new number. The mix now only counts evidence **above chance**
(`max(hit_rate − 1/V, 0)`), where `1/V` is a structural quantity.

| arm | held-out CE | held-out top-1 | magic numbers |
|---|---|---|---|
| v116 (= v115 behaviour) | 6.6031 | 2.85 % | 6 + 2 clamps |
| v118 (zero magic, whole-history mean) | 7.8409 | 2.15 % | 0 |
| **v119 (zero magic, τ-based forgetting)** | 7.6158 | **4.33 %** | 0 |

The single-file application `RawHebb+bio+0.2.0.0.py` carries the same rules and reproduces the
behaviour (250 dialogues × 2 epochs: held-out top-1 5.00 %).

**What this round establishes.** The previous versions' CE numbers were, in large part, the model
learning how sharp to be rather than how to be right. Once the rules were made self-scaling and the
reward switched to the discrete correctness signal, top-1 nearly doubled **without a single tuned
constant** — and nothing needs re-tuning when the corpus or the model size changes. The remaining
free parameters are architecture and physics (layers, width, window, τ), which is where they should
be.


---

## v120 / 0.4.0.0 — the readout was the bottleneck, and the state was overflowing

Two measurements settled it.

**1. The closed-form probe (offline, diagnostic only).** Extract the features once, then solve
ridge regression in closed form:

| features | dim | held-out CE | held-out top-1 |
|---|---|---|---|
| embedding only | 128 | 7.887 | 13.47 % |
| L0 state only (τ=10) | 128 | 7.900 | 11.45 % |
| L3 state only (τ=80) | 128 | 7.917 | 5.82 % |
| four timescales concatenated | 512 | 7.891 | 13.11 % |
| **embedding ⊕ four timescales** | 640 | **7.876** | **16.39 %** |

The trained model was at **5 %**. The information was in the features all along; the online rule was
simply not converging.

**2. The cascade was overflowing.** Norms per layer: embedding 1.0 → L0 2.5 → L1 37 → L2 1104 →
L3 **inf**. Steady-state gain is `1/(1−leak)`, so the four layers multiply to ≈ 670 000. The base
head reads the last layer — i.e. it was reading `inf`. Fix: drive gain = the layer's own `(1−leak)`,
so every layer's steady-state variance equals its input's variance. After the fix, L3's probe top-1
went from `nan` (0 %) to 5.82 %.

### What 0.4.0.0 does
1. features = embedding ⊕ all four timescales (each RMS-normalised, so the solver decides the mix);
2. `W = A⁻¹B` from running sufficient statistics, re-solved when the accumulated evidence doubles
   (doubling schedule ⇒ no parameter, O(log n) solves);
3. one global temperature calibrated by finite-difference Newton against the likelihood.

All three are local, online, gradient-free, and free of hand-set constants.

### Result (500 dialogues, CPU)

| version | held-out CE | held-out top-1 |
|---|---|---|
| 0.2.0.0 (zero magic numbers) | 6.60 | 2.85 % |
| 0.3.0.0 (gain fix only, last-layer readout) | 6.50 | 3.23 % |
| **0.4.0.0 (multi-timescale closed-form)** | **6.07** | **11.72 %** |
| *closed-form probe ceiling* | *7.88* | *16.39 %* |

Top-1 quadrupled at equal data, and the deployed rule is now within striking distance of the
offline ceiling it was derived from. The remaining CE gap to a bigram (3.96) is a calibration
question, not a representation question — which is the next thing to attack.


---

## 0.5.0.0 → 0.6.0.0 — a regression, and what it taught

0.4.0.0 had the right features and the right estimator but the wrong *scoring*: least squares
against one-hot targets ranks well (top-1 16 % offline) and predicts badly (CE 7.88), because the
fit is systematically under-confident. Replacing it with a Newton step in the cross-entropy sense
was the obvious move.

The first attempt (0.5.0.0) did this:

```
W ← W − A⁻¹B + A⁻¹R        # wrong: W is carried over from the previous window
```

Because `W` was carried over, the `−A⁻¹B` term was subtracted **once per window** and the weights
drifted. Result: held-out CE 7.24, top-1 **0.98 %** — far worse than the version it replaced.

The fix (0.6.0.0) is a full overwrite:

```
W ← A⁻¹B − A⁻¹ Σ_win (p−y)xᵀ
```

Least-squares solution as the starting point, one stochastic Newton correction with the accumulated
inverse Hessian. No drift, no learning rate, and the external temperature calibration disappears
because the fit now carries its own scale.

### Full corpus, one epoch, 6 minutes on CPU

| version | held-out CE | held-out top-1 | top-5 |
|---|---|---|---|
| v115 (previous best) | 6.039 | 2.53 % | 12.5 % |
| 0.4.0.0 (least squares + temperature) | 6.074 | 10.98 % | 21.5 % |
| **0.6.0.0** | **5.862** | **14.85 %** | **27.17 %** |
| unigram baseline | 5.875 | 6.11 % | — |
| bigram baseline | 3.961 | — | — |

This is the first configuration in the project's history to beat the unigram baseline on **both**
metrics, at 1/6th of the data of some earlier runs and with no tuned constants anywhere.

Scaling on the same configuration: 500 dialogues → CE 5.949 / top-1 12.06 %;
2 850 dialogues → CE 5.862 / top-1 14.85 %. More data still helps, and nothing had to be re-tuned.


---

## 0.7.x — a negative result: keying the memory on token identity

After 0.6.0.0 the remaining gap to a bigram (CE 3.96) is a modelling problem. A bigram table stores
*exact* token-to-token associations; our memory column is an average of slowly-varying states, so the
column prototypes end up mutually collinear and can only store a vague direction.

The idea: key the memory on the **current token's embedding** instead of the slow state, so column
`y` becomes "the average embedding of tokens that were followed by `y`" — a bigram table compressed
to `H` dimensions, with the same parameter count (`H × V`).

Four variants were tried (500 dialogues, one epoch, same seed; 0.6.0.0 scores CE 5.949 / top-1 12.06 %):

| version | memory read-out | CE | top-1 |
|---|---|---|---|
| 0.6.0.0 | slow-state key, z-scored (baseline) | **5.949** | 12.06 % |
| 0.7.0.0 | token key, z-scored | 11.14 | 1.11 % |
| 0.7.1.0 | token key, raw inner product | 47.33 | 0.49 % |
| 0.7.2.0 | token key, **cosine** | 6.773 | **14.04 %** |
| 0.7.3.0 | cosine + volume `β` fitted by likelihood | 6.751 | 13.28 % |

**The idea half-works and is worth recording.** Keying on token identity clearly improves the
*ranking* (12.06 % → 14.04 % top-1). But the probabilities stay wrong: even when the total volume
`β` is fitted by maximum likelihood on the current window (0.7.3.0), CE stays at 6.75 instead of
returning to 5.95. So the problem is not the volume — it is the *shape* of what the memory branch
emits: confidently and systematically wrong.

Two useful mechanisms from the failures:

* dividing by the standard deviation (z-score) **amplifies the noise of evidence-free columns** to
  unit variance — that alone cost 5 nats (CE 11.14);
* raw inner products blow up when columns coincide (CE 47.33). Cosine is the bounded, meaningful
  measure of "how specifically does this token follow this context".

None of 0.7.x is released; 0.6.0.0 remains the best. The next step is not to replace the slow-state
memory but to **run both keys in parallel**, weighted by above-chance hit rate with a
likelihood-fitted total volume — the same complementarity the closed-form probe found for
embedding ⊕ four timescales (16.39 %) over any single scale (13 %).

---

## v121 (2026-10-20): the target-shift bug, and the jump past bigram

### The bug

`chat_to_ids` returns `labels[t] = token t+1`, the standard next-token target. Since 0.1.0.0 the
training and evaluation code paired `logits[t]` with `labels[t+1]` instead — a **two-step-ahead**
task. The error is invisible if you only look at the reported loss, because training and evaluation
were wrong in the same way.

Scoring the *same* 0.6.0.0 checkpoint under three alignments makes it unambiguous
(`research/_v121_align_check.py`, 40 validation dialogues):

| alignment | CE | top-1 | top-5 |
|---|---|---|---|
| old `logits[t]` vs `labels[t+1]` | 5.8714 | **12.58 %** | 24.04 % |
| standard `logits[t]` vs `labels[t]` | 6.5219 | 2.68 % | 11.09 % |
| `logits[t+1]` vs `labels[t+1]` | 6.5190 | 2.68 % | 11.06 % |

The fix (`RawHebb+align+0.8.0.0.py`) changes only the target indices:
`logits[:, :-1]` vs `labels[:, :-1]` in the loss and evaluation, and `tgt = labels[:, :T-1]` in the
plasticity update. On the same 500-dialogue subset: CE 5.8714 → **5.1035**, top-1 12.58 % → **23.69 %**.

**All numbers before 0.10.0.0 are on the old task.** They remain useful as a record of the dynamics,
but they cannot be compared directly with current results.

### Three Hebbian memories

With the alignment fixed, the next bottleneck was modelling, not calibration:

| version | what it adds | CE (500) | top-1 (500) | CE (full) | top-1 (full) |
|---|---|---|---|---|---|
| align 0.8.0.0 | alignment only | 5.1035 | 23.69 % | 4.9584 | 25.60 % |
| aligncnt 0.8.0.0 | exact token-pair count table | 3.9406 | 29.60 % | 3.5751 | 31.74 % |
| tri 0.9.25.0 | + hashed 2-token-prefix counts | 3.8394 | 34.79 % | 3.3156 | 37.50 % |
| **seq128tri 0.9.33.0** | + `SEQ_LEN=128` | 3.8069 | 35.38 % | **3.2690** | **38.62 %** |

The pair table is pure Hebbian `C[x,y] += 1`; the prefix table uses the same rule on a hashed key.
Both are read out as smoothed conditional distributions; the three experts are mixed by coordinate
descent on the likelihood over the probability simplex. No learning rate, no hand-set mixture weight.

### What did not work (kept as branches)

* **4-gram / 5-gram hashed contexts**: worse (CE 3.96 / 3.94, top-1 30.9 % / 31.3 % at 500)
  because longer contexts are sparse and collide in the hash table.
* **Kneser–Ney absolute discounting**: CE slightly better (3.834 vs 3.839) but top-1 drops
  (30.9 % vs 34.8 %). The discount is estimated from count-of-counts, but it hurts ranking here.
* **Two epochs**: overfits (500: CE 4.005 vs 3.839; full: 3.442 vs 3.269), although full-data top-1
  rises slightly (38.94 %). One epoch is the better CE operating point.
* **Full-content chunked training**: uses all answer tokens (multi-window) but costs ~30 min for 500
  dialogues; at that scale it does not beat the tail-window model on both metrics.
* **Micro-variants** of the base head (intercept, extra Newton steps, coupling 0.5→1, disabling the
  dead spike path) are within noise: the pair/prefix experts dominate the readout.
* **Seed noise** on 500 dialogues is about ±0.08 CE and ±1 % top-1; full-corpus seeds 43/44/45 are
  consistent (tri: 3.316 / 3.325; seq128tri: 3.269 / 3.358).

### v121 experimental artefacts

`RawHebb+lab+0.9.0.0.py` is the switchboard (alignment, pair/prefix experts, mixture rules, bias,
Newton steps, coupling, chunking). `_v121_gen*.py` generate one standalone file per variant; all are
kept. Diagnostics: `_v121_align_check.py`, `_v121_pair_ref*.py`, `_v121_sweep.py`,
`_v121_eval150.py`, `_v121_crosseval.py`.

---

## v122 (2026-10-20): training order, and a corrected bigram reference

### Fixed training order wins, mainly by removing seed variance

The 0.10.0.0 training loop shuffled the 2 850 dialogues with seed 43. Turning shuffle off
(`TRAIN_SHUFFLE = False`, branch `trinoshuf+0.9.52.0`) gives:

| seed | shuffled full CE / top-1 | fixed-order full CE / top-1 |
|---|---|---|
| 43 | 3.2690 / 38.62 % | **3.2581 / 38.70 %** |
| 44 | 3.3582 / 36.07 % | **3.2608 / 38.63 %** |
| 45 | — | **3.2552 / 38.86 %** |

The mean gain over 0.10.0.0 is small (+0.011 CE), but the three fixed-order seeds agree to
0.006 CE, while the shuffled seeds ranged over 0.089 CE. Fixed file order is therefore the better
operating point, not a new mechanism.

### The old bigram reference had the same target-shift bug

`_v121_pair_ref*.py` paired `ids[:-1]` with `labels[1:]`. But `labels[t]` is already token `t+1`, so
that reference was scoring a **two-step-ahead** bigram. The corrected script
`research/_v122_bigram_ref.py` uses `ids[t]` vs `labels[t]` on the same last-128-token windows:

| reference (same window/mask) | CE | top-1 | top-5 |
|---|---|---|---|
| unigram add-1 | 5.785 | 4.83 % | 15.85 % |
| bigram add-1 | 4.456 | 31.39 % | 58.07 % |
| bigram + unigram prior | **3.809** | **31.32 %** | **58.05 %** |
| pair/unigram log-mixture (best w=0.71) | 3.684 | — | — |

0.11.0.0 is **3.258 / 38.70 % / 63.02 %**, so it still beats the corrected bigram on all three
metrics. The conclusion survives; the margin is smaller and now measured correctly.

### v122 negatives

| branch | change | CE (500) | top-1 (500) | verdict |
|---|---|---|---|---|
| `tribackoff` | tri smoothed with pair instead of unigram | 3.8466 | 35.10 % | worse |
| `triadapt` | per-position weight `n/(n+1)` | 3.7377 | 36.08 % | neutral |
| `trilin` | linear probability mixture instead of log-linear | 4.9340 | 26.75 % | much worse |
| `trinoshufpref` | + causal prefix-cache expert | 3.6099 | 36.38 % | prefix weight = 0 |
| `seq192tri` | window 192 | 3.7406 | 35.90 % | neutral/worse |
| `seq256tri` | window 256 | 3.6874 | 35.74 % | CE better, top-1 worse |
| `trinoshuf` | fixed order (baseline for the table) | **3.6099** | **36.38 %** | current |

The prefix-cache expert was rejected by the likelihood fit: its weight was exactly 0.0, so it had
no effect on the output. The linear mixture is a genuine failure: log-linear (geometric) pooling is
much better for this model, probably because the base and count experts are over-confident in
different regions.

### GPU

An RTX 5080 was available for this round. On the same 50-dialogue run, CPU took 6 s and GPU 47 s
(after fixing two device-placement bugs in the count updates and evaluation). The kernel launches
one tiny operation per timestep and per layer, so it is launch-bound; batching would help the GPU but
breaks the online local update. CPU remains the target.

### v122 artefacts

`_v122_gen.py` (branch generator), `_v122_eval.py` (checkpoint evaluator),
`_v122_bigram_ref.py` (corrected reference), and the branch files
`RawHebb+trinoshuf+0.9.52.0.py`, `RawHebb+trinoshufseed44+0.9.64.0.py`,
`RawHebb+trinoshufseed45+0.9.70.0.py`, plus the negative branches listed above.


## v123 (2026-10-20): capacity is the lever

After 0.11.0.0, the question was whether the remaining gap is in the local rules or in capacity.
v123 swept the hidden width, the tri prior, the hash width, the window, the layer count, and several
neutral switches, first on 500 dialogues and then on the full corpus for the capacity candidates.
The answer was unambiguous: **width is the lever**, and none of the new mechanisms tested so far
beats a wider base.

### 500-dialogue screen (fixed order, seq 128, pair + tri)

| branch | change | CE | top-1 | verdict |
|---|---|---|---|---|
| 	rinoshuf | 0.11.0.0 baseline | **3.6099** | 36.38 % | baseline |
| order reverse | reverse file order | 3.6350 | 36.35 % | worse |
| order length_asc | sort by length, ascending | 3.6098 | 36.38 % | neutral |
| order length_desc | sort by length, descending | 3.6489 | 35.27 % | worse |
| pairprior2 | pair prior 2 | 3.6093 | 36.50 % | neutral |
| pairprior4 | pair prior 4 | 3.6247 | 36.51 % | worse CE |
| 	riprior2 | tri prior 2 | 3.6070 | 36.48 % | neutral |
| 	riprior4 | tri prior 4 | 3.6050 | 36.52 % | best prior |
| 	rinoshuf + TRIGRAM_BUCKETS=64k | 64k tri buckets | 3.5999 | 36.86 % | better |
| 	rinoshuf + TRIGRAM_BUCKETS=32k | 32k | 3.6053 | 36.84 % | neutral |
| 	rinoshuf + TRIGRAM_BUCKETS=8k | 8k | 3.6160 | 36.30 % | worse |
| seq96 | window 96 | 3.6411 | 37.76 % | CE worse |
| seq160 | window 160 | 3.6203 | 37.86 % | CE worse |
| seq192 | window 192 | 3.6186 | 36.69 % | worse |
| seq256 | window 256 | 3.6144 | 35.38 % | worse |
| layers3 | 3 layers | 3.6240 | 36.21 % | worse |
| layers5 | 5 layers | 3.6238 | 36.29 % | worse |
| hidden64 | width 64 | 3.6391 | 36.37 % | worse |
| hidden192 | width 192 | 3.5953 | 36.17 % | better CE |
| 
ewton2 | 2 Newton steps | 3.6102 | 36.38 % | neutral |
| `bias` | output-head bias | 3.6107 | 36.38 % | neutral |
| 	ri4buck64hidden192 | 192 wide + 64k + tri4 | **3.5830** | 36.81 % | capacity wins |
| 	ri4buck64hidden256 | 256 wide + 64k + tri4 | 3.5873 | 37.36 % | capacity wins |
| 	ri4buck64hidden288 | 288 wide + 64k + tri4 | 3.5709 | 37.37 % | capacity wins |
| 	ri4buck64hidden352 | 352 wide + 64k + tri4 | 3.5686 | 37.58 % | capacity wins |
| 	ri4buck64hidden384 | 384 wide + 64k + tri4 | 3.5692 | 37.67 % | capacity wins |

seq64 reached 37.95 % top-1 but its CE was 3.6279; its evaluation token set is different, so it is
not directly comparable to the seq-128 rows. The 
oise/spike/couple switches were exactly neutral
(3.6099), confirming that the recurrent state is not contributing to the read-out at this scale; the
count experts and the wide base carry the result.

### Full corpus (2 850 dialogues, one epoch, 1 890 scored tokens per dialogue)

| hidden | CE | top-1 | top-5 | notes |
|---|---|---|---|---|
| 128 (0.11.0.0) | 3.2581 | 38.70 % | 63.02 % | baseline |
| 192 | 3.2248 | 40.13 % | 63.10 % | seed44 3.2278 / 40.18; seed45 3.2221 / 40.11 |
| 224 | 3.2246 | 40.21 % | 63.00 % | |
| 256 | 3.2182 | 40.54 % | 63.17 % | |
| 288 | 3.2076 | 40.20 % | 63.34 % | |
| 352 | 3.2061 | **40.61 %** | 63.26 % | best top-1 |
| 384 | 3.2006 | 40.53 % | 63.33 % | seed44 3.2073 / 40.59 |
| **416** | **3.1946** | 40.34 % | **63.52 %** | **0.12.0.0**; seeds 44/45 3.2002 / 3.1966 |

The CE gain from 128 to 416 is 0.063, the top-1 gain 1.64 points, and the top-5 gain 0.50 points.
Three hidden-416 seeds give CE 3.1946 / 3.2002 / 3.1966 (mean 3.1971), top-1 40.34 / 40.49 / 40.21 %,
top-5 63.52 / 63.29 / 63.61 %. The top-1 peak at 352 and the CE peak at 416 are within 0.01 CE /
0.3 % top-1, so both are kept as branches instead of pretending one is strictly better.

### v124: new local mechanisms, all rejected

| branch | mechanism | CE (500) | top-1 | verdict |
|---|---|---|---|---|
| lastcopy | causal last-successor cache expert | 3.5743 | 37.41 % | worse than 3.5692 |
| promptcopy | prompt-token distribution expert | 3.5845 | 37.45 % | worse |
| lastpromptcopy | both caches (6 weight slots) | 3.5846 | 37.43 % | worse |
| `fourbackoff` | hashed 4-gram, tri-prior backoff | 3.5692 | 37.66 % | fitted weight 1.6e-5 (≈0) |
| `top1mix` | fit mixture on top-1 instead of NLL | 3.8746 | 37.53 % | CE collapse |

The 4-gram table is too sparse at 500 dialogues (and the tri expert already captures the useful
context); its fitted weight is 1.6e-5, effectively zero. The top-1 objective is discontinuous and the coordinate search
collapses the calibration, so NLL remains the right local objective.

### v124b: the 4-gram was not dead, just too sparse

The 64k-bucket 4-gram got a fitted weight of 1.6e-5 and no measurable effect. Reducing the hash width
(which increases collisions and smoothing) changes the result at 500 dialogues:

| 4-gram buckets | CE (500) | top-1 | fitted weight | verdict |
|---|---|---|---|---|
| 65536 | 3.5692 | 37.66 % | 1.6e-5 | no effect |
| 32768 | (not finished) | — | — | interrupted |
| 16384 | 3.5610 | 37.77 % | 0.0618 | better |
| 8192 | 3.5587 | **37.86 %** | 0.0529 | best top-1 |
| 4096 | **3.5514** | 37.73 % | 0.0927 | best CE |
| 2048 | 3.5517 | 37.73 % | 0.0921 | plateau |

The hidden-384 baseline for this table is 3.5692 / 37.67 %. The 4k table has the best CE, the 8k table the best top-1; 2k is identical to 4k (the sweep has
plateaued). Both beat the baseline on CE and top-1. Full
2 850-dialogue runs of hidden384+8k and hidden416+8k are in progress. If they confirm, the 4-gram
backoff becomes 0.13.0.0; otherwise it stays a documented near-miss. The lesson is that the earlier
4-gram negative was a **hash-sparsity** failure, not evidence that 4-grams are useless.

### v124b full corpus: the 4-gram gain survives; 8k trades top-1 for top-5

| model | CE | top-1 | top-5 |
|---|---|---|---|
| 0.12.0.0 (hidden416, no 4-gram) | 3.1946 | 40.34 % | 63.52 % |
| hidden384 + 4-gram 8k | 3.1911 | 40.05 % | 63.65 % |
| hidden416 + 4-gram 8k | 3.1863 | 40.04 % | **63.84 %** |
| **hidden416 + 4-gram 4k (0.13.0.0)** | **3.1772** | **40.43 %** | 63.69 % |

The 4-gram backoff gives the best CE and top-1 so far. The 4k table reaches 3.1772 / 40.43 / 63.69:
-0.0174 CE, +0.09 top-1 and +0.17 top-5 versus 0.12.0.0, and -0.0199 CE versus the three-seed 0.12
mean (3.1971). Three 4k seeds: CE 3.1772 / 3.1824 / 3.1783 (mean 3.1793), top-1 40.43 / 40.49 /
40.31 % (mean 40.41 %), top-5 63.69 / 63.54 / 63.75 % (mean 63.66 %). The 8k table keeps the best top-5 (63.84 %) but gives up 0.39 top-1 and 0.0091 CE.
The 4k variant is the overall 0.13.0.0 champion; 8k is kept as the top-5 branch. The 4-gram has
finally paid off after the 64k-bucket zero-weight result: the bucket width is the architecture
quantity that controls the sparsity/generalisation trade-off.

### v129: true 4-gram / 5-gram keys are worse than the collapsed backoff

Adding FOUR_BASE = 4099 / FIVEGRAM_BASE = 4099 makes the keys use all context
tokens. At 500 dialogues (baseline collapsed-4-gram 3.5525 / 37.80 / 59.89):

| model | CE | top-1 | top-5 | four weight | five weight |
|---|---|---|---|---|---|
| true 4-gram | 3.5714 | 37.80 % | 59.33 % | 4.5e-5 | - |
| true 4+5-gram | 3.5714 | 37.80 % | 59.33 % | 3.1e-5 | 3.5e-5 |

A true 4-gram is too sparse at this data size; the **trigram prior + last-token
count** backoff (0.13) is better than a strict 4-gram. This is a negative result for
higher-order contexts, not a bug fix that improves the score.

### Correction (0.13): the "4-gram" key collapses to the last token

The 0.13 4-gram hash uses _tri_base = 4096 (next power of two above vocab) and
FOURGRAM_BUCKETS = 4096, so

    key = ((p3 * 4096 + p2) * 4096 + p1) % 4096 = p1

The table is therefore a **last-token count** (a global bigram) interpolated with the
**trigram prior** _tp. That combination is real and is what improved 0.11 -> 0.13;
it is not a true 4-gram. The 5-gram file had the same collapse, so the v125 5-gram
negative is invalid for a true 5-gram. Branches 0.9.175.0 / 0.9.176.0 use a separate
FOUR_BASE = 4099 (first prime above 4096) and FIVEGRAM_BASE = 4099 to make the
keys use all context tokens; they are the correct order-4/order-5 tests.

### v125: 5-gram is too sparse at every tested width

The 4-gram success suggested a 5-gram with 4-gram backoff. At 500 dialogues (hidden416 + 4-gram 4k,
baseline 3.5525 / 37.80 / 59.89), the 5-gram table gets essentially no mixture weight:

| 5-gram buckets | CE | top-1 | fitted weight |
|---|---|---|---|
| 4096 | 3.5524 | 37.79 % | 0.0006 |
| 2048 | 3.5526 | 37.79 % | 0.0001 |
| 1024 | 3.5521 | 37.83 % | 8.8e-6 |
| 512 | 3.5522 | 37.83 % | 7.3e-6 |

The table is populated (~62 700 counts), but each 5-gram context is almost unique; the 4-gram
expert already captures what generalises. The tiny CE/top-1 differences are noise. **5-gram is a
negative result; the 4-gram is the right order for this corpus.**

### v130: position-wise count experts are dead

Counting P(z | c1) or P(z | c2) (exact V×V tables, trigram prior) gives
3.5549 / 37.67 / 59.91 and 3.5527 / 37.80 / 59.89 at 500 dialogues, against the
3.5525 / 37.80 / 59.89 collapsed-4-gram baseline. Fitted weights are 1.2e-5 and
1.4e-5: the trigram prior already carries the context; the last-token count is the
only order-4-style table that earns weight.

### v131: count scope and an explicit unigram expert

* PAIR_SCOPE = 'all' (count user and assistant tokens, not only assistant targets):
  500 dialogues 3.5522 / 37.80 / 59.90, a 0.0003 CE / 0.01 top-5 change versus the
  mask scope. The full 2850 run gives 3.1771 / 40.42 / 63.68, versus 3.1772 / 40.43 /
  63.69 for the mask scope: no gain, so the mask scope is retained.
* UNIGRAM_EXPERT = True: 3.5525 / 37.78 / 59.89, fitted weight 2.6e-5; redundant with
  the unigram prior and dead.

### v132: trigram counts with the pair distribution as prior

The symmetric backoff chain (pair + unigram, trigram + pair, last-token + trigram)
was tested at 500 dialogues: 3.5525 / 37.77 / 59.92, versus 3.5525 / 37.80 / 59.89
for 0.13. The trigram-plus-pair expert earns a 0.0095 weight but does not change CE. The full
2850 run gives 3.1772 / 40.44 / 63.69 versus 3.1772 / 40.43 / 63.69 for 0.13: a
0.01-point top-1 change, no CE/top-5 gain. The 0.13 mixture is already the right
backoff chain; 0.13.0.0 is retained as the champion.

### v133: the four-expert prior strength is a pseudo-count

The 0.13 four expert reuses TRI_PRIOR_STRENGTH = 4; its table is a *bigram* table,
so the pair expert's PAIR_PRIOR_STRENGTH = 1 (one pseudo-observation) is the more
principled value. At 500 dialogues (baseline 3.5525 / 37.80 / 59.89):

| four prior | CE | top-1 | top-5 |
|---|---|---|---|
| 1 | 3.5514 | 37.74 % | 59.93 % |
| 2 | 3.5521 | 37.77 % | 59.88 % |

The full 2850 run gives 3.1773 / **40.53** / 63.67, versus 3.1772 / 40.43 / 63.69 for
0.13: CE +0.0001 and top-5 -0.02 are a wash, top-1 is +0.10. Prior 1 is therefore a
**top-1 tradeoff branch**, not an all-three champion; 0.13.0.0 keeps the CE/top-5 crown.

### v134: hidden352 + 4k does not recover the top-1 crown

The 0.12 hidden352 branch has the best top-1 of the no-4-gram family (40.61 %, CE
3.2061). Adding the 4k expert gives 3.1893 / 40.50 / 63.40: CE improves by 0.017,
but top-1 falls to 40.50, so it does not beat the 0.12 hidden352 top-1 branch.
The top-1 ranking remains 0.12 hidden352 40.61, fprior1 40.53, hidden352+4k 40.50.

### A methodology bug worth recording

The first v123 batch generated branch files with names such as 
ewton2+0.9.89.0.py, but the runner
selected the **first** glob match RawHebb+newton2+0.9.*.py, which was the old v121 file
RawHebb+newton2+0.9.9.0.py. Several early numbers were therefore old branches, not the new
switches. The fix was to run exact full file names and to rerun the affected branches with _v123b_
output prefixes. Exact filenames are now the rule for every branch run; the old files are kept
untouched.

### v123 / v124 artefacts

_v123_gen.py, _v123b_gen.py ... _v123e_gen.py, _v124_gen.py, _v123_release416.py,
_v123_release.py; branches RawHebb+tri4buck64hidden416+0.9.137.0.py,
RawHebb+tri4buck64hidden384+0.9.128.0.py, RawHebb+tri4buck64hidden352+0.9.127.0.py, and the
negative v124 files. Full-run logs are kept locally as _v123_full_*.out and checkpoint files as
_v123_full_*.pt.
