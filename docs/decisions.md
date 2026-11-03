# Decisions

One line for each decision and its reason. Newest first.

## 2026-11-03

- **Hook the `Conv` block, not `conv.act`.** Ultralytics shares one `nn.SiLU`
  object across the network, so the activation has no unique module name and a
  hook on it fires for every layer. The `Conv` block output is already
  post-activation.
- **Take the Grad-CAM gradient before the softmax.** At 100% confidence the
  gradient of a softmax score is about zero and the heatmap becomes noise.
- **Draw the heatmap on the centre crop.** The preprocessing crops to 224, so a
  heatmap drawn on the full photo is misaligned.
- **One CLI, not two.** `leaf.py` lost its `main()` when `predict.py` arrived.
  Two entry points printing the same thing is duplication.
- **Report 100% with a written caveat rather than hiding it.** 2 leakage audits
  show the split is clean, so the number is real but laboratory-only. A bare
  100% in a portfolio README invites a credibility hit.
- **No crop-loss figures and no treatment advice.** The owner decided this.
  Nothing on screen should be fact-checkable against the author.
- **Deduplicate before splitting.** A duplicate across train and test inflates
  the test score silently.
- **Symlinks, not copies.** The dataset stays in one place and cannot be
  modified by accident.
- **Gradio over Streamlit.** Upload plus image and bar outputs with no frontend
  code.
- **YOLO11n-cls transfer learning over a CNN from scratch.** Higher accuracy at
  0.45 GB of VRAM, and it trains in minutes.
- **No `dev` branch and no pull requests.** Solo repository, 7 small steps, demo
  deadline. Recorded in `docs/workflow.md`.
