# Changelog

All notable changes to RawHebb. Versions follow `RawHebb+bio+<a>.<b>.<c>.<d>.py` (four segments, no `v` prefix);
every change is a new file, older versions are kept verbatim.

## [0.1.1.0] — current
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
