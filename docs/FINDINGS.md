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
