# Changelog

All notable changes to RawHebb. Versions follow `RawHebb+bio+<a>.<b>.<c>.<d>.py` (four segments, no `v` prefix);
every change is a new file, older versions are kept verbatim.

## [0.20.0.0] - current
* **Usable shell: the 3120012 data chain ported to the RawHebb core.** 0.19.0.0 crashed on menu
  item 1 because a helper was defined inside a replaced block; that is fixed, and the approximate
  data layer is replaced by an actual port of the 3120012 logic.
* **Rendering/mask:** `<|bos|>` + tools + system/user/tool/assistant segments + `<|eos|>`;
  assistant `reasoning_content`, `content` and `tool_calls` are trained, everything else is -100.
* **Context truncation:** `_truncate_pairs` semantics - keep the leading system prompt and the
  newest complete user/assistant pairs, budgeted in real tokens (including role/end/reasoning/tool
  calls), never cutting a sentence in the middle.
* **Validation:** a record-level random 10 % sample by default (fixed seed) or an explicit val
  file; validation uses the **same `SEQ_LEN` window as training**, not a separate last-128 slice.
  `.txt` pretraining remains all-token with shuffled windows and no validation by default.
* **No new benchmark claim:** the model recipe is unchanged and was not retrained. The same 0.18
  checkpoint scores CE **3.3545** / top-1 **38.06 %** / top-5 **60.60 %** over 166 632 positions
  under the full-window protocol; these numbers are not comparable to the last-128 numbers of
  0.15-0.18.
* **Online learning** (menu 13) requires a trained checkpoint and supports file streaming or
  in-chat updates.

## [0.19.0.0] - current
* **Shell overhaul; the model recipe is unchanged.** Loading the 0.18.0.0 checkpoint reproduces
  CE **3.0543** / top-1 **41.86 %** / top-5 **64.23 %** exactly (verified with
  `research/_v162_eval_ckpt.py`), so the numbers below are identical to 0.18.
* **Vocabulary lifecycle restored** (from 3120012elite): ByteLevel BPE can be trained from the
  corpus, saved as `tokenizer.bin` + `tokenizer.json`, loaded with metadata validation, and falls
  back to the base vocabulary when present. Rebuilt vocabularies are detected on checkpoint load
  and refused explicitly instead of silently skipping most tables.
* **Two data pipelines** matching 3120012: `.txt` = unsupervised pretraining (all tokens, windows
  of `SEQ_LEN+1`, no validation by default); `.jsonl` = dialogue (assistant mask, validation split
  by dialogue at 10 % with a fixed seed, or an explicit val file).
* **Message-aware truncation:** dialogues longer than `SEQ_LEN` keep the system prompt and the
  newest complete user/assistant pairs (`_truncate_pairs` behaviour), instead of cutting a sentence
  in the middle.
* **Online learning menu branch:** stream a `.txt`/`.jsonl` incrementally (one update per sample,
  no shuffle/validation) or learn during chat (one update per exchange). Requires a trained
  checkpoint so the Newton head does not start from an ill-conditioned random state.
* **Config cleanup:** removed six never-read placebo constants (`BATCH_SIZE`, `LEARNING_RATE`,
  `WEIGHT_DECAY`, `GRAD_CLIP_NORM`, `MEM_MOD_DECAY`, `ADAPT_TAU_MS`), rewrote the configuration
  with sourced/measured comments, and added dataset selection, input validation and a tokenizer
  submenu to the menu.

## [0.18.0.0] - current
* **Full-dialogue training window.** The online counts and membrane state now see the whole
  dialogue (capped at 384 tokens) instead of its last 128 tokens; evaluation is unchanged and
  still scores the last-128-token windows, so the comparison is exact. On this corpus dialogues
  average 472 tokens (median 511), so each one contributes about four times more context. No new
  parameters, no new constants, no back-propagation.
* Full corpus seed43: CE **3.0543**, top-1 **41.86 %**, top-5 **64.23 %** (0.17: 3.1411 /
  41.59 % / 63.86 %), i.e. **-0.088 CE, +0.27 top-1, +0.44 top-5**.
* **Seeds:** CE 3.0543 / 3.0554 / 3.0561 (mean **3.0553**), top-1 41.86 / 41.65 / 41.71 % (mean
  **41.74 %**), top-5 64.23 / 64.27 / 64.19 % (mean **64.23 %**); the 0.17 means are 3.1434 /
  41.48 % / 63.79 %.
* Training at 512 tokens (the complete dialogue) ties on CE (mean 3.0542) but gives top-1 41.43 %
  and is about 30 % slower, so 384 is shipped. Screen trajectory 128 -> 192 -> 256 -> 384 -> 512:
  3.5864 -> 3.5137 -> 3.4891 -> 3.4529 -> 3.4503.
* **int16 count table validated:** a 1M int16 table reproduces float32 exactly (full seed43
  3.1411 / 41.59 % / 63.86 %) at half the memory. With int16 a 2M table fits, but it scores 3.5869
  on the screen versus 3.5864 at 1M, so the count-bucket axis is saturated at 1M.

## [0.17.0.0]
* **1M trigram count buckets.** On top of 0.16, the trigram table is widened from 256k to **1M**
  buckets (a pure count-resolution change; hidden 432, trigram prior 2 and the row-normalised bag
  prior are unchanged). Full corpus seed43: CE **3.1411**, top-1 **41.59 %**, top-5 **63.86 %**
  (0.16: 3.1496 / 41.35 % / 63.84 %).
* **Seeds:** CE 3.1411 / 3.1458 / 3.1433 (mean **3.1434**), top-1 41.59 / 41.49 / 41.36 % (mean
  **41.48 %**), top-5 63.86 / 63.78 / 63.74 % (mean **63.79 %**); the 0.16 means are 3.1516 /
  41.20 % / 63.80 %, so CE and top-1 improve with top-5 level.
* The bucket axis is still unsaturated (full CE 3.1584 at 128k, 3.1496 at 256k, 3.1411 at 1M), but a
  2M table is about 23 GB and does not fit this machine's RAM; 1M is the practical ceiling.
* Note: the 1M checkpoint is a 12.4 GB table, so `_v122_eval.py` cannot load it for a second-pass
  evaluation on a 47 GB / 24 GB-free machine. The reported numbers are the held-out evaluation
  produced inside each training run; the release file is code-identical to the branch.

## [0.16.0.0]
* **Wider trigram count table, lower trigram prior, slightly wider hidden layer.** On top of the
  0.15 row-normalised bag prior, the pure count knobs move: trigram buckets 64k -> **256k**, trigram
  prior strength 4 -> **2**, hidden 416 -> **432**. Full corpus seed43: CE **3.1496**, top-1
  **41.35 %**, top-5 **63.84 %** (0.15: 3.1705 / 40.38 % / 63.78 %).
* **Seeds:** CE 3.1496 / 3.1546 / 3.1507 (mean **3.1516**), top-1 41.35 / 41.22 / 41.02 % (mean
  **41.20 %**), top-5 63.84 / 63.77 / 63.80 % (mean **63.80 %**); the 0.15 means are 3.1724 /
  40.45 % / 63.78 %, so all three improve. The gain is in the count-table resolution: 128k buckets
  alone already give 3.1584, hidden width alone is flat (h432 3.1709, h448 3.1728, h464 worse).
* Rejected in v138: hidden 400 / 428 / 436 / 464, 32k buckets, 5 layers, 2 epochs, seq 192, a
  separate raw bag expert, four-table row normalisation, tri prior 2/8 without the bucket change,
  and the row0.5/final0.5 smoothing pair.

## [0.15.0.0]
* **Row-normalised past-context bag prior.** Each past context's pair row is normalised before the
  causal cumsum, so rare and frequent contexts each vote once; the four expert's pseudo-count is the
  Jeffreys prior 0.5. It improves 0.14.0.0 by 0.0031 CE and 0.02 points top-5 at equal mean top-1.
  Full corpus seed43: CE **3.1705**, top-1 **40.38 %**, top-5 **63.78 %** (0.14: 3.1740 / 40.43 % /
  63.77 %).
* **Seeds:** CE 3.1705 / 3.1749 / 3.1717 (mean **3.1724**), top-1 40.38 / 40.59 / 40.38 % (mean
  **40.45 %**), top-5 63.78 / 63.71 / 63.86 % (mean **63.78 %**); the 0.14 means are 3.1755 /
  40.45 % / 63.76 %.
* Rejected in v137 and kept as branches: the raw fourmix geometric pool (full 3.1747 / 40.41 /
  63.76), last-half-window bag, leaky bag, hidden448 + bag, separate raw/normalised bag experts,
  sqrt-normalised rows, and separate tri-prior experts. Full details in `docs/FINDINGS.md`.

## [0.14.0.0]
* **Bag-of-past-tokens prior for the four expert.** The exact pair table is accumulated causally
  over the window and used as the prior for the last-token count expert, replacing the trigram
  prior; no new counts, tables or constants. Full corpus seed43: CE **3.1740**, top-1 **40.43 %**,
  top-5 **63.77 %**, versus 3.1772 / 40.43 % / 63.69 % for 0.13.0.0 (CE -0.0032, top-5 +0.08,
  top-1 equal).
* **Seeds:** CE 3.1740 / 3.1773 / 3.1753 (mean **3.1755**), top-1 40.43 / 40.53 / 40.40 % (mean
  **40.45 %**), top-5 63.77 / 63.72 / 63.79 % (mean **63.76 %**). The 0.13 means are 3.1793 /
  40.41 % / 63.66 %, so all three improve.
* Rejected in v135: an independent bag expert (3.5534 at 500), the true 4-gram / 5-gram keys
  (3.5714, weight ≈0), position-wise counts, `PAIR_SCOPE='all'` (full 3.1771 / 40.42 / 63.68),
  and the geometric trigram/bag pool (500 3.5476, not run full). All are kept as branches.

## [0.13.0.0]
* **4-gram count expert, 4k hashed buckets, tri-prior backoff.** On top of the hidden-416 base:
  full corpus CE **3.1772**, top-1 **40.43 %**, top-5 **63.69 %**, versus 3.1946 / 40.34 % / 63.52 %
  for 0.12.0.0. It beats 0.12 on all three metrics. The 8k-bucket variant keeps the best top-5
  (63.84 %) but gives up 0.39 top-1 and 0.0091 CE.
* **The 64k 4-gram was a hash-sparsity failure, not a dead idea.** The 64k table got a fitted weight
  of 1.6e-5; 16k/8k/4k/2k buckets at 500 dialogues give 3.5610 / 3.5587 / 3.5514 / 3.5517, and the
  full 4k run confirms. The bucket width is an architecture quantity, not a hand-set weight.
* **Seeds:** CE 3.1772 / 3.1824 / 3.1783 (mean 3.1793), top-1 40.43 / 40.49 / 40.31 %, top-5 63.69 / 63.54 / 63.75 %.
* Rejected and kept as branches: last-successor copy, prompt copy, top-1 mixture objective, and the
  64k 4-gram. The v123 capacity scan and all earlier negative results are unchanged.
* **Naming correction (2026-10-20).** The key uses `_tri_base = 4096` and `FOURGRAM_BUCKETS = 4096`,
  so `((p3 * 4096 + p2) * 4096 + p1) % 4096 = p1`. The 4k expert is therefore a **last-token count
  table interpolated with the trigram prior**, not a strict 4-gram; the measured score is unchanged.
  A true 4-gram key (base 4099) is much worse (3.5714 at 500 dialogues), so the collapsed backoff is
  kept deliberately. The earlier 5-gram negative used the same collapse and is not a true 5-gram test.

## [0.12.0.0]
* **Scaled the hidden width 128 -> 416** (SEQ_LEN=128, fixed file order, tri prior 4, 64k tri buckets).
  Full corpus: CE **3.1946**, top-1 **40.34 %**, top-5 **63.52 %** (seed 43), versus 3.2581 /
  38.70 % / 63.02 % for 0.11.0.0. Three seeds: CE 3.1946 / 3.2002 / 3.1966 (mean 3.1971), top-1
  40.34 / 40.49 / 40.21 %, top-5 63.52 / 63.29 / 63.61 %.
* **Capacity scan (full corpus)**: hidden 192 3.2248 / 40.13; 224 3.2246 / 40.21; 256 3.2182 / 40.54;
  288 3.2076 / 40.20; 352 3.2061 / 40.61; 384 3.2006 / 40.53; 416 3.1946 / 40.34. The 352 branch has
  the best top-1; 416 has the best CE / top-5; the spread is within 0.01 CE and 0.3 % top-1.
* **Rejected in v123 / v124**: reverse and length order, pair prior 2/4, tri prior 2/3/5/8, 8k/32k/128k
  buckets, seq96/160/192/256, layers 3/5, hidden 64, newton2, bias, noise/spike/coupling switches,
  mix rounds 1/4, all-pair scope, last-successor copy, prompt copy, 4-gram backoff (fitted weight 1.6e-5 ≈ 0),
  and a top-1 mixture objective (CE collapse). All are kept as branches.

## [0.11.0.0]
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
