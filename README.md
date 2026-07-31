# Paper reproduction supplement

This repository reproduces every empirical result in the submitted paper:
Tables 1–5, Figures 2–4, the DEBD significance claims, the Appendix C
Circuit-Wasserstein calculation, and the MNIST headline values.

## Quick start

Paste one block. Install PyTorch from https://pytorch.org/get-started/locally/
before any PyJuice step. Depending on your pip version it *might* work automatically, but not guaranteed.

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

### Both (full paper artifacts, all tables and figures)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision   # platform/CUDA-specific
python -m pip install -r requirements-sparc-paper.txt -r requirements-pyjuice-paper.txt
python reproduce.py doctor --strict --profile all
python reproduce.py debd all --download -j 100
python reproduce.py mnist all
python reproduce.py artifacts
```

The above commands should recreate all of our results. Runtime results will differ depending on your hardware. Randomness is (potentially) OS-dependent; PyJuice requires Linux. See our appendix for information on OS version, packages, etc.

Use `python reproduce.py --dry-run ...` to print commands without running them.

