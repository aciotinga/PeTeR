# Paper reproduction supplement

This repository reproduces every empirical result in the submitted paper:
Tables 1–5, Figures 2–4, the DEBD significance claims, the Appendix C
Circuit-Wasserstein calculation, and the MNIST headline values.

The checked-in scientific implementation is not rewritten by the reproduction
CLI. `reproduce.py` validates inputs and launches the original scripts with the
current Python interpreter from the repository root. Generated datasets,
models, evaluations, tables, and figures are ignored by Git.

## Quick start

Paste one block. Install PyTorch from https://pytorch.org/get-started/locally/
before any PyJuice step.

### DEBD only

SparC + PyJuice. Downloads DEBD, then runs corruptions, PeTeR, RL-TPM, attacks, eval, CW (Table 3), and Table 4 rebenchmark.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision   # platform/CUDA-specific
python -m pip install -r requirements-sparc-paper.txt -r requirements-pyjuice-paper.txt
python reproduce.py doctor --strict --profile all
python reproduce.py debd all --download -j 100
```

### MNIST only

SparC only. Exact paper path via included `mnist/hclt_mnist_blocksize4.json` (prepare → PeTeR → FGSM → eval).

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sparc-paper.txt
python reproduce.py doctor --strict --profile sparc
python reproduce.py mnist all
```

### Both (full paper artifacts + verify)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision   # platform/CUDA-specific
python -m pip install -r requirements-sparc-paper.txt -r requirements-pyjuice-paper.txt
python reproduce.py doctor --strict --profile all
python reproduce.py debd all --download -j 100
python reproduce.py mnist all
python reproduce.py artifacts
python reproduce.py verify-paper
```

Use `python reproduce.py --dry-run ...` to print commands without running them.
Stages skip completed outputs when the underlying script supports resume.

## Included and regenerated files

Included immutable inputs:

- `example_pcs/`: the 28 submitted DEBD MLE-PCs.
- `mnist/hclt_mnist_blocksize4.json`: the submitted MNIST MLE-PC required for
  the exact paper likelihoods.
- `sweeps/tpe/`: complete DEBD and MNIST TPE journals, trial records, failures,
  summaries, and winners.
- `manifests/`: dataset hashes, exact experiment settings, paper-result oracle,
  provenance, and scientific-source hashes.

Reviewers regenerate:

- downloaded datasets under `original_datasets/`;
- random and model-aware corruptions under `corrupted_datasets/` and
  `adversarial_datasets/`;
- PeTeR, RL-TPM, and CW auxiliary PCs;
- evaluation and runtime caches;
- `paper_outputs/`.

Do not delete the included MNIST reference PC when cleaning generated outputs.

## Environments

### SparC profile

PeTeR, attacks, evaluations, runtime measurement, plotting, and table rendering
use SparC. These stages support Windows and Linux. The submitted experiments
ran on Linux.

Pinned paper profile:

```bash
python -m pip install -r requirements-sparc-paper.txt
python reproduce.py doctor --strict --profile sparc
```

Unpinned convenience profile:

```bash
python -m pip install -r requirements.txt
python reproduce.py doctor --profile sparc
```

The submitted SparC version was `sparc-pc==0.6.1`.

### PyJuice profile

RL-TPM, from-scratch MNIST MLE learning, and Appendix C auxiliary-PC fitting
use PyJuice and PyTorch. Install PyTorch/Torchvision using the official command
for the machine's CUDA version, then install one profile:

```bash
# Paper profile
python -m pip install -r requirements-pyjuice-paper.txt
python reproduce.py doctor --strict --profile pyjuice

# Or unpinned convenience profile
python -m pip install -r requirements-pyjuice.txt
python reproduce.py doctor --profile pyjuice
```

The submitted PyJuice version was `pyjuice==2.4.3`. These stages are
Linux-tested and Linux-recommended, but the CLI does not impose an OS lock.
The `debd cw` command computes distances immediately after PyJuice fitting, so
that environment also needs the SparC profile.

### Paper hardware

- PeTeR: Rocky Linux 8.10, 2 × AMD EPYC 7713, 2 TB DDR4 RAM, no GPU; 128 CPU
  cores were used in parallel.
- PyJuice baselines: 2 × NVIDIA L40S GPUs.
- Submitted runtime table: Intel i9-13950HX CPU.

Table 4 is intentionally rebenchmarked on the reviewer's hardware. Its values
are not expected to equal the submitted Intel measurements.

## DEBD data

One command downloads the UCLA-StarAI archive, caches it under `data/debd/`,
hash-validates every split against `manifests/debd.json`, and writes
`original_datasets/`:

```bash
python reproduce.py debd download
```

Source: <https://github.com/UCLA-StarAI/Density-Estimation-Datasets>

Re-fetch with `python reproduce.py debd download --force`. A local checkout or
archive still works:

```bash
python reproduce.py debd import --source /path/to/archive-or-directory
```

Layout after import:

```text
original_datasets/<dataset>/<dataset>.train.data
original_datasets/<dataset>/<dataset>.valid.data
original_datasets/<dataset>/<dataset>.test.data
```

Line-ending normalization is used only for validation so Windows and Linux
downloads validate identically. Invalid or incomplete archives are rejected.

## Exact DEBD workflow

The combined command is:

```bash
python reproduce.py debd all --download -j 100
```

It requires a combined SparC/PyJuice environment. To use separate environments,
run the stages in this order.

In the SparC environment:

```bash
python reproduce.py debd download
python reproduce.py debd corrupt -j 100
python reproduce.py debd peter -j 100
```

In the PyJuice environment:

```bash
python reproduce.py debd rltpm -j 1
```

Back in the SparC environment:

```bash
python reproduce.py debd attack -j 100
python reproduce.py debd evaluate -j 100
```

For Appendix C, use an environment containing PyJuice, PyTorch, and SparC:

```bash
python reproduce.py debd cw -j 1
```

Filtered CW runs are supported for diagnostics, but canonical Table 3 is
written only when all 28 datasets and all three positive budgets are present.
Undefined or missing distances are never silently omitted from its means.

Finally, rebenchmark Table 4 in the SparC environment:

```bash
python reproduce.py debd runtime
```

Most stages accept repeatable filters, for example:

```bash
python reproduce.py --dry-run debd peter --dataset nltcs --k 1
python reproduce.py debd evaluate --dataset nltcs --k 1 -j 1
```

`debd attack` accepts the same repeatable dataset/budget filters and `--force`
through the wrapper. `debd runtime` intentionally benchmarks all 28 datasets.

### DEBD protocol and hyperparameters

Data and attacks:

- corruption budgets: `epsilon ∈ {1, 3, 5}`;
- random evaluation: 10 copies per dataset/budget;
- each copy samples exactly `epsilon` bit indices with replacement per row;
- deterministic seed: first 32 bits of
  `MD5("<dataset>|K<epsilon>|r<replicate>")`;
- adversarial evaluation: greedy model-aware bit flips, at most `epsilon` per
  row, allowing a previously selected bit to be selected again;
- candidate-row cap: 65,536.

Included MLE-PCs:

- Hidden Chow-Liu Tree structure;
- block size / latent count 4;
- submitted description: 1,000 EM iterations on the training split;
- the included JSON circuits, not a new fit, are the DEBD starting point.

PeTeR:

- one TPE study per dataset and budget;
- TPE search: log-uniform learning rate `[1e-8, 0.5]` and ratio `[0.1, 100]`;
- 10 startup trials and 50 target trials;
- objective: final likelihood on the MLE-PC adversarial test set;
- each trial and final DEBD run: 500 OGDA iterations after a 100-step Q warm
  start;
- 100 theta samples, deterministic sampling, evaluation every iteration;
- `eta_phi = learning_rate × ratio`, `eta_lambda = 10`;
- Circuit-Wasserstein settings: `metric_p=1.0`, `scale_factor=1.0`;
- exact per-dataset winners are read from the included
  `study_summary.json` files, not copied into Python source.

RL-TPM:

- HCLT with block size 4 and seed 0;
- 1,000 EM epochs, batch size 512;
- optimizer learning rate 0.1, pseudocount 0.1;
- multi-linear schedule with rates `[0.9, 0.1, 0.05]` at epochs
  `[0, 100, 500]`;
- greedy train-time corruption with the requested budget;
- fixed settings: this repository contains no RL-TPM hyperparameter sweep.

Appendix C:

- fit auxiliary HCLT-4 MLE-PCs for `K={0,1,3,5}`;
- `K=0` fits the clean test set; positive K fits each PeTeR adversarial set;
- seed 0, 1,000 EM epochs, batch 512, and the same optimizer/scheduler as
  RL-TPM;
- compare each positive-K circuit to K=0 with `metric_p=1.0` and
  `scale_factor=1.0`;
- expected means: `12.08`, `17.66`, and `21.37`.

## Exact MNIST path

The exact paper path uses the included reference MLE-PC:

```bash
python reproduce.py mnist all
```

Equivalent individual stages:

```bash
python reproduce.py mnist prepare
python reproduce.py mnist peter
python reproduce.py mnist attack
python reproduce.py mnist evaluate
```

`mnist prepare` downloads the official test images from the PyTorch-hosted
MNIST mirror into `data/MNIST/raw/`. It exports all 10,000 images and a
500-image PeTeR/FGSM subset.

MNIST corruption and evaluation:

- tune target: Gaussian `sigma=0.100`, replicate 0, first 500 test images;
- evaluation: `sigma=0.001, 0.002, ..., 0.010`, 10 deterministic copies per
  sigma, all 10,000 test images;
- pixel rule:
  `clip(round(pixel + 255 × Normal(0, sigma²)), 0, 255)`;
- deterministic seed: first 32 bits of
  `MD5("mnist|sigma<0.000>|r<replicate>")`;
- final PeTeR winner: learning rate `2.7001862237401832e-06`, ratio
  `0.17230491130528763`;
- TPE trials use 500 PeTeR iterations; the final paper circuit uses 1,000;
- epsilon-1 FGSM uses one central finite-difference sign step against each
  model's own likelihood, on the 500-image subset.

Expected submitted values:

- MLE clean: `-724`;
- MLE at sigma 0.010: `-3287`;
- PeTeR clean: `-1423`;
- PeTeR at sigma 0.010: `-2993`;
- MLE on its own FGSM set: `-4734`;
- PeTeR on its own FGSM set: `-4663`;
- mean absolute intensity change at sigma 0.010: `1.17`.

The manuscript displays these MNIST headlines by truncating toward zero rather
than by nearest-value rounding. The verifier applies that submitted display
policy while retaining full-precision values in `verification.json`.

## From-scratch MNIST MLE path

The original MLE learner has no explicit random seed, so a new circuit is not
expected to reproduce the submitted likelihoods exactly. It also overwrites
the included reference path. Run it only when a from-scratch fit is desired:

```bash
python reproduce.py mnist prepare
python reproduce.py mnist learn --yes-overwrite-reference
python reproduce.py mnist peter --force
python reproduce.py mnist attack --force
python reproduce.py mnist evaluate --force
```

Learning settings:

- PyJuice HCLT, 4 latents, 256 categories per pixel;
- 350 epochs, batch size 512;
- EM learning rate 0.1, pseudocount 0.1;
- multi-linear rates `[0.9, 0.1, 0.05]` at epochs `[0, 100, 350]`;
- training loader shuffled, with no explicit seed.

The unchanged learner asks Torchvision to cache its train/test download under
`../data` relative to the repository. This is separate from
`prepare_mnist_data.py`'s reviewer-facing `data/MNIST/raw/` test-image cache.

Restore `mnist/hclt_mnist_blocksize4.json` from the supplement before returning
to the exact paper path.

## Render and verify the paper

After DEBD and MNIST evaluation:

```bash
python reproduce.py artifacts
python reproduce.py verify-paper
```

Canonical outputs:

- Table 1: `paper_outputs/table1.tex`, the 10-dataset epsilon-5 MLE/PeTeR
  comparison from `table.py`;
- Table 2: `paper_outputs/table2.tex`, the ordered 8-dataset main comparison
  from `table1.py`;
- Table 3: `paper_outputs/table3.tex` and `table3.json`, CW means from
  `learn_mle_adv.py` and `print_distances.py`;
- Table 4: `paper_outputs/table4.tex`, reviewer-hardware runtime from
  `runtime.py` and `table2.py`, with OS/CPU/Python/method details in
  `paper_outputs/table4_provenance.json`;
- Table 5: `paper_outputs/table5.tex`, all 28 DEBD datasets from `table1.py`;
- Figure 2: `paper_outputs/figure2.pdf` from `plot2.py`;
- Figure 3: `paper_outputs/figure3.pdf`, plus separate random/adversarial PDFs,
  from `plot1.py`;
- Figure 4: `paper_outputs/figure4.png` from `mnist_visualizer.py`.

Figure 1 is manuscript artwork rather than an experimental output.

`verify-paper` writes `paper_outputs/verification.json`. It checks Tables 1, 2,
3, and 5 at their displayed precision; full Figure 2 aggregate/Figure 3 source
arrays; Figure 4's deterministic source rows; DEBD win counts `27/23/20`;
exact two-sided paired Wilcoxon p-values and three-way Bonferroni values; the
MNIST sigma-0.010 Wilcoxon/ten-way-Bonferroni significance claim; MNIST
headlines and pixel change; rendered outputs; and scientific-source hashes.
Table 4's 28-row shape, method, and reviewer provenance are checked, while its
hardware-dependent numbers remain exempt from equality.

For a cache-only diagnostic before figures are rendered:

```bash
python reproduce.py verify-paper --allow-missing-artifacts
```
