## [0.1.0.0] — first release
* Kernel replaced with `RawHebbKernel`: pure Hebbian, no optimiser, no backward pass.
* Menu structure of `3120012elite` preserved; only the kernel changed.
* Items 8 and 9 (reward RL / GRPO) replaced by gradient-free equivalents
  (self-consistency voting, long-context segmented stress test), stated in the menu.
* Added menu item 12: held-out validation with leak checks.
* Added question-answer masking to every local update rule, with a unit test.
