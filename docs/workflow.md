# Workflow

Only what is specific to this repository. The shared rules live in
`~/dotfiles/claude/WORKFLOW.md`. Do not copy them here.

## Local overrides

<!-- Anything this repository does differently, and why. -->

No `dev` branch, and no pull requests. A `feat/*` branch starts from `main` and
merges back into `main` after the owner confirms the step works. The owner
decided this on 2026-11-01. The reason is a solo repository, 7 small steps, and
a demo deadline. The shared rule in `AGENTS.md` that sends every branch to `dev`
through a pull request does not apply here.

## Release

<!-- How a change reaches production, if it does. -->
TODO

## Traps

<!-- What breaks if an agent gets it wrong. Evaluator traps, flaky tests,
     generated files that must not be edited by hand. -->

1. Never commit `images/`, `data/`, `runs/`, or a `*.pt` file. `.gitignore`
   blocks them. Check `git status` before each commit.
2. This laptop has 2 kernel flavours. The `gke` kernel has no NVIDIA module, so
   `nvidia-smi` fails on it. Boot a `generic` kernel. Run `nvidia-smi` before
   you train.
3. `model.predict()` runs inside `torch.no_grad()`. Grad-CAM gets no gradient
   from it. Call `model.model` directly instead. See `gradcam.py`.
4. The dataset holds duplicate images. `split.py` removes them by SHA-256 hash
   before it splits. Never split before you remove the duplicates. A duplicate
   in 2 splits inflates the test score.
