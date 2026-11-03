# AGENTS.md

Read this file before you change the repository. Update it when the
architecture or the workflow changes.

`CLAUDE.md` is a symlink to this file. Claude, Codex, and Kimi read the same
contract.

## What this project is

<!-- One paragraph. What it does and who uses it. -->
A leaf disease classifier for apple and grape leaves, 8 classes. It answers with
the disease name, a confidence score, and a Grad-CAM heatmap of the pixels it
used. It exists as a portfolio demo, so the screen matters as much as the model.
`README.md` is the portfolio page. `DEV_DOC.md` holds the technical detail.

## Stack

<!-- Language, framework, build tool, database. Delete the rows that do not apply. -->
| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Build | `uv` venv, `requirements.txt`, no packaging step |
| Test | The asserts inside `split.py`. No framework. |
| Model | Ultralytics YOLO11n-cls |
| Web | Gradio |

## Setup

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
# then put the dataset in images/, one folder per class
.venv/bin/python split.py
.venv/bin/python train.py
```

## Run locally

```sh
.venv/bin/python app.py      # serves http://127.0.0.1:7860
```

## Verify

Run these before you open a pull request. Put the output in the pull request
body.

```sh
nvidia-smi                                  # must print the GPU
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv/bin/python split.py                   # asserts no overlap between splits
.venv/bin/python predict.py <image>         # one answer plus the heatmap
```

There is no lint step and no type check.

## Git workflow

`main` is stable. `dev` is the integration branch. One branch for each task,
branched from `dev`, merged into `dev` by a pull request.

Full rules: `~/dotfiles/claude/WORKFLOW.md`. The rules are enforced by
`~/dotfiles/claude/hooks/guard-git.sh`, not by this paragraph.

An agent never merges a pull request.

## Agent roles

| Agent | Owns |
|---|---|
| Claude | Planning, review, architecture decisions |
| Codex | Implementation against an approved plan |
| Kimi | Frontend work and research |

One agent owns one branch. Two agents never share a branch.

## Constraints

<!-- The rules an agent must not break in this repository. Be specific. -->
- An NVIDIA GPU is required. `train.py` exits rather than running on the CPU.
- Never commit `images/`, `data/`, `runs/`, or a `*.pt` file.
- Never split the dataset before you deduplicate it.
- Add no third direct dependency without asking. There are 2.
- Put no invented figure on screen or in the README. No crop-loss numbers and
  no treatment advice.

## More

| Document | Holds |
|---|---|
| `docs/architecture.md` | Structure, modules, data flow |
| `docs/decisions.md` | Decisions taken and the reason for each |
| `docs/workflow.md` | Anything about the process that is specific to this repository |
| `~/dotfiles/claude/WORKFLOW.md` | Branch, commit, pull request, and CI rules shared by every repository |

Continuous integration follows the shared policy: light checks on a pull
request, the expensive build and deploy only on a version tag. Do not restate it
here. Record a deviation in `docs/workflow.md` with its reason.

## Current state

<!-- What is in progress now. Keep it short. Delete what is complete. -->
- All 7 build steps are complete and verified. Test top-1 is 100% on 723
  held-out images. That reflects an easy laboratory dataset, not a field result.
  Read the honesty note in `README.md` before you quote the number.
- Branches `feat/setup` through `docs/readme` stack on each other and wait to
  merge into `main`.
