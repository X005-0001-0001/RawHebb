# Changelog

All notable changes to RawHebb. Versions follow `RawHebb+bio+<a>.<b>.<c>.<d>.py` (four segments, no `v` prefix);
every change is a new file, older versions are kept verbatim.

## [0.11.0.0] — current
* **Fixed the training order: `TRAIN_SHUFFLE = False`.** On the full corpus, three seeds give
  CE **3.2581 / 3.2608 / 3.2552** and top-1 **38.70 / 38.63 / 38.86 %**, versus 3.2690 / 38.62 %
  for 0.10.0.0. The biggest effect is robustness: shuffled seed44 was 3.3582 / 36.07 %, fixed-order
  seed44 is 3.2608 / 38.63 %.
* **Corrected the n-gram reference.** The v121 reference scripts paired `ids[:-1]` with
  `labels[1:]`, the same target-shift bug fixed in 0.10.0.0. The corrected, window-matched
  reference (`research/_v122_bigram_ref.py`) is: bigram + unigram prior CE 3.809 / top-1 31.32 % /
  top-5 58.05 %; bigram add-1 CE 4.456; unigram add-1 CE 5.785. The model still beats bigram on all
  three metrics.
* Rejected in v122 and kept as branches: tri→pair backoff, count-adaptive per-position mixing,
  linear probability-space mixing (much worse), a causal prefix-cache expert (fitted weight exactly
  zero), and 192/256-token windows.
* Measured GPU speed: on 50 dialogues, 6 s on CPU vs 47 s on an RTX 5080. The kernel is
  launch-bound; CPU remains the target.

## [0.10.0.0]
* **Fixed a target-shift bug present since 0.1.0.0.** `chat_to_ids` already returns
  `labels[t] = token t+1`, but training/evaluation paired `logits[t]` with `labels[t+1]`, i.e. the
  model was solving a two-step-ahead task. Scoring the same 0.6.0.0 checkpoint under both alignments:
  old 12.58 % top-1 vs standard 2.68 %. All pre-0.10 numbers are on the old task and are not
  directly comparable. The fix alone (500 dialogues) moved CE/top-1 from 5.87/12.6 % to 5.10/23.7 %.
* Added two **pure Hebbian count memories**: an exact token-pair table `C[x,y] += 1` with
  add-one/unigram-prior readout, and a hashed 2-token-prefix count table. They run alongside the
  recurrent-state memory.
* Added likelihood **coordinate descent** over the expert weights on the probability simplex
  (base / pair / prefix). No learning rate, no hand-set mixture weight.
* `SEQ_LEN = 128`; full corpus (2 850 dialogues, one epoch, CPU): held-out **CE 3.269,
  top-1 38.62 %, top-5 62.99 %**, beating bigram (3.864 / 27.82 %) on both metrics.
* Negative results kept in `research/`: 4/5-gram hashed contexts are worse (sparsity), Kneser–Ney
  discounting trades CE for top-1, two epochs overfit, and full-content chunked training is too slow
  at this scale.

## [0.6.0.0]
* The read-out update is now a proper **overwrite**: start from the least-squares solution `A^-1 B`
  and add one stochastic Newton correction `-A^-1 Σ(p−y)xᵀ` using the accumulated inverse Hessian.
  The external temperature calibration is deleted — the fit carries its own scale.
* **Full corpus (2 850 dialogues, one epoch, 6 min on CPU): held-out CE 5.862, top-1 14.85 %,
  top-5 27.17 %.** This is the first configuration to beat the unigram baseline on *both* metrics
  (unigram: CE 5.875, top-1 6.11 %).
* Previous best on the same corpus was CE 6.039 / top-1 2.53 %.

## [0.5.0.0] — kept for history, it regressed
* Attempted an online Gauss-Newton read-out. Top-1 collapsed from 12.1 % to 1.0 % because the
  update was accumulated into `W` instead of overwriting it, so `-A^-1 B` was subtracted once per
  window and the weights drifted. Fixed in 0.6.0.0. Kept because the failure is instructive.

## [0.4.0.0]
* **Fixed a structural bug**: the cascade amplified the variance by `1/(1−leak)` per layer
  (10.5 × 20 × 40 × 80 ≈ 670 000), so the deep state was `inf` and the base head was reading
  garbage. Drive gain is now the layer's own `(1−leak)`; deep features went from `inf` to usable.
* **Readout is now an exact local solver**: features = embedding ⊕ all four timescales,
  `W = A⁻¹B` accumulated online from sufficient statistics, re-solved when the evidence doubles,
  plus a temperature calibrated by finite-difference Newton. No learning rate, no forgetting factor,
  no regularisation constant.
* Measured (500 dialogues, CPU, 3 epochs): held-out CE **6.07**, top-1 **11.7 %**, top-5 22.4 %
  (previous best at equal data: CE 6.60, top-1 2.85 %).
* The closed-form probe that motivated this is kept in `research/_v120_probe_features.py`.

## [0.3.0.0] — kept for history
* Layer drive gain = the layer's own `(1−leak)`, which removes the cascade's variance explosion.

## [0.2.0.0]
* **Zero magic numbers.** Removed `LR_MEM`, `MEM_DECAY_EVERY`, `LR_OUT`, `LR_GAMMA`, `LR_W`,
  `W_CLAMP`, `GAMMA_CLAMP` and the whole `γ` parameter.
  * output head: direction-normalised step of `1/(t+1)` — scale-free, bounded total displacement
  * memory: exponential average with decay **= the layer's own membrane leak** (no new constant)
  * read-out mix: normalised hit rate **above chance** (`1/V`), so Σw = 1 and the read-out is bounded
    by construction; a layer with no evidence stays silent
* Layer weight rule switched from a continuous correlation to a **discrete** signal
  (did this read-out's argmax match the truth), because the continuous one was rewarding
  distribution sharpening rather than correctness.
* Checkpoint loading is now tolerant of shape changes across versions.
* Measured on 500 dialogues, CPU: held-out **top-1 2.85 % → 4.33 %** (CE 6.60 → 7.62, which is the
  same CE/top-1 trade documented in `docs/FINDINGS.md`).

## [0.1.1.0]
* Fixed: memory read-out is normalised **only over channels that hold memory**
  (previously normalised by RMS over the whole vocabulary, which amplified the few
  written channels by √N early in training — measured merged peak 44, CE 8.4 → 24.9).
* Fixed: `W_out` LMS now uses a **relative** step size (a fixed fraction of ‖W_out‖),
  so it cannot diverge and no external watchdog is needed.
* Measured: 250 dialogues × 2 epochs = 42 s, held-out CE 6.19 → 6.12, top-1 ≈ 4 %.

## [0.1.0.0] — first release
* Kernel replaced with `RawHebbKernel`: pure Hebbian, no optimiser, no backward pass.
* Menu structure of `3120012elite` preserved; only the kernel changed.
* Items 8 and 9 (reward RL / GRPO) replaced by gradient-free equivalents
  (self-consistency voting, long-context segmented stress test), stated in the menu.
* Added menu item 12: held-out validation with leak checks.
* Added question-answer masking to every local update rule, with a unit test.
