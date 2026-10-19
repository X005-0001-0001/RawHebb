# Changelog

All notable changes to RawHebb. Versions follow `RawHebb+bio+<a>.<b>.<c>.<d>.py` (four segments, no `v` prefix);
every change is a new file, older versions are kept verbatim.

## [0.2.0.0] — current
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
