# Compared methods (tabular)

`experiments/tabular/run_baselines.py` runs every compared method on the two tabular benchmarks, at its defaults and
tuned with the same budget, on exactly the data FLARE sees. This directory holds what it needs:

| File | Content |
|---|---|
| `tabular.py` | builders, default configurations, Optuna search spaces and rule counts of all methods |
| `tuning.py` | the tuning protocol (TPE, 30 trials, 80/20 split of the training rows, macro-F1) |
| `rrl.py`, `rlnet.py`, `drnet.py`, `neurules.py` | adapters that call the authors' code (cloned by `fetch_external.py`) |
| `gradtree.py` | GradTree in PyTorch (our implementation; the authors' is TensorFlow) |
| `workers/` | CLASSY and GRANDE, run as subprocesses of their own environments |
| `fetch_external.py`, `external.py` | clone the authors' repositories at the commits of the paper |

`graph.py` holds the graph baselines (SGC, GCN, GAT, and the tables of node and neighbour variables that the other methods read; runner: `experiments/graphs/run_baselines.py`) and `cbm_heads.py` the heads of the concept bottleneck models (LogicCBM, DCR and the settings of every head; runner: `experiments/cbm/run_heads.py`).

## Methods and sources

| Method | Source | Licence |
|---|---|---|
| XGBoost, CatBoost | `xgboost`, `catboost` (pip) | Apache-2.0 |
| LightGBM | `lightgbm` (pip) | MIT |
| RF, CART, LR | `scikit-learn` | BSD-3 |
| EBM | `interpret` (pip) | MIT |
| FIGS, RuleFit | `imodels` (pip) | MIT |
| RIPPER | `wittgenstein` (pip) | MIT |
| CORELS | `corels` (pip, `pip install -e ".[corels]"`: a source distribution that needs a C++ compiler and GMP) | GPL-3.0 (an optional install, never imported by the other methods) |
| CLASSY | `rulelist` 0.2.0 (pip, own environment) | MIT |
| GRANDE | `GRANDE` 0.1.6 (pip, own environment) | MIT |
| RRL | github.com/12wang3/rrl @ f8d0886b | MIT |
| RL-Net | github.com/luciledierckx/RLNet @ 2e7ad950 | MIT |
| DR-Net | github.com/Joeyonng/decision-rules-network @ ac246983 | MIT |
| NeuRules | github.com/Schascha1/neurules @ b7776cee | none published |
| GradTree | `gradtree.py` (our PyTorch version of github.com/s-marton/GradTree) | this repository |

`python -m baselines.fetch_external` clones RRL, RL-Net, DR-Net and NeuRules into `baselines/external/` (git-ignored; set
`FLARE_EXTERNAL_DIR` to use another directory) and prints their licences. The NeuRules repository has no licence file,
so this repository only contains an adapter and never a copy of its code: a method whose clone is missing is skipped by
the harness with a message. Only `neurules.py` of that clone is used (the network); its training loop is written out in
`baselines/neurules.py` because the authors' `training.py` imports their dataset loaders (pmlb, torchvision, imodels).
The adapter also applies the authors' `init="const"` rule-to-class weights, which their published network ignores
(see the comment in `neurules.py`).

Notes on the adapters of the authors' code: RRL needs a one-line fix of its binarisation layer for current PyTorch
(`.view` on a non-contiguous tensor, patched at import in `rrl.py`); DR-Net imports `torchvision` and `tqdm`; RL-Net's
spectral-norm penalty falls back to float64 when the float32 SVD does not converge.

## Environments

Main environment (what the paper's tabular baselines were run and checked with): Python 3.12, `numpy` 2.3,
`pandas` 2.3, `scikit-learn` 1.9, `torch` 2.9, `torchvision` 0.24, `optuna` 5.0.0, `xgboost` 3.2.0, `lightgbm` 4.7.0,
`catboost` 1.2.10, `interpret` 0.7.8, `imodels` 2.0.4, `wittgenstein` 0.3.5, `corels` 1.1.29 (its PyPI source
distribution needs a C++ compiler and GMP, and may need its `setup.py` adjusted for recent NumPy). TPE draws depend on
the Optuna version: with another one the tuned configurations differ.

CLASSY needs older numpy and pandas than the rest: Python 3.10, `numpy` 1.26, `pandas` 1.5, `scikit-learn` 1.3,
`numba`, `gmpy2`, `rulelist` 0.2.0 (it uses `DataFrame.iteritems` and `nanquantile(interpolation=)`), `optuna`.
GRANDE needs TensorFlow: Python 3.11, `GRANDE` 0.1.6, `scikit-learn` 1.5.2 (`category_encoders` breaks on 1.6 and later),
`optuna`. Create each environment, then point the harness at its interpreter:

    export FLARE_CLASSY_PYTHON=/path/to/classy-env/bin/python
    export FLARE_GRANDE_PYTHON=/path/to/grande-env/bin/python

## Running

    python -m baselines.fetch_external
    python -m experiments.tabular.run_baselines --benchmark cc18 --method XGBoost --mode tuned --dataset diabetes --seed 33 --out results/baselines
    python -m experiments.tabular.run_baselines manifest --benchmark cc18 --out results/baselines
    python -m experiments.tabular.run_baselines --list results/baselines/jobs/cc18.txt --index 1 --out results/baselines

Each line writes `rows/<method>_<mode>_<dataset>_seed<S>.csv` with one row per outer fold. Lines already finished are
not run again. Tuned mode: 30 trials or one hour per fold, whichever comes first (`--trials`, `--tune_timeout`).

Multi-class problems with the binary-only methods (RIPPER, CORELS: one model per class, the row goes to the class
whose model fires, ties and no-fires to the most frequent of the candidates by training frequency; DR-Net: one
network per class and the largest output; RuleFit: scikit-learn's one-vs-rest).
