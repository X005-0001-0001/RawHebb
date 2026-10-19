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