# FLARE: Fuzzy Learned-Antecedent Rule Evidence

FLARE is a differentiable rule learner. It learns many short fuzzy rules, most of them with a single condition, and
adds their evidence to decide the class: every rule that holds for a sample gives points to the classes, weighted by how
much it holds. The conditions are fuzzy labels of the variables ("age IS high", "income IS at least medium") whose
partitions are learned with the rules, and categorical variables with thousands of categories get a learned space
where their labels are regions. The rules are searched by gradient descent: the forward pass is the discrete rule base
that is deployed, and the gradient of a relaxed version is grafted onto it.

This repository contains the method, the compared methods and the scripts that reproduce the results of the paper.

```
flare/          the method (a PyTorch package; every module says which part of the paper it implements)
baselines/      the compared methods: builders, search spaces and adapters to the authors' code
experiments/    data loaders, runners, job lists and the scripts of the tables and figures
tests/          unit tests
```

## Install

    pip install -e .                     # the method: numpy, pandas, scikit-learn, scipy, torch
    pip install -e ".[experiments]"      # + OpenML, Optuna, matplotlib: the scripts of the paper
    pip install -e ".[baselines]"        # + the compared methods that come from pip (see baselines/README.md)
    pip install -e ".[corels]"           # + CORELS, which builds from source (a C++ compiler and GMP)
    pip install -e ".[graphs,cbm]"       # + what the graph and the concept bottleneck experiments read (h5py, torchvision)

Python 3.10 or later. A GPU is used when PyTorch sees one (the training step is then replayed as a CUDA graph); every
result can also be computed on a CPU, more slowly.

## Use

```python
from flare import FLAREClassifier, flare, flare_explainability

model = FLAREClassifier(flare()).fit(X_train, y_train, categorical=mask)   # mask: True for categorical columns (codes 0..K-1)
model.predict(X_test)
model.predict_proba(X_test)
print(model.print_rules())
```

`flare()` is the configuration of the paper that aims at performance (8 averaged models of 120 rules, 100 of them with
one condition), `flare_explainability()` the one that aims at a small rule base (one model of 40 rules). Every option
is in `flare/config.py`, e.g. `flare(rules=60, labels=5)`.

## Data

The repository holds no data. The tabular benchmarks are downloaded from OpenML on first use (42 OpenML-CC18 datasets,
`experiments/data/cc18.json`, and the 38 classification tasks of TabArena v0.1, `experiments/data/tabarena.json`) and
cached in `$FLARE_DATA_DIR` (default `~/.cache/flare`). The graph and image datasets are described in
`experiments/README.md`: what is downloaded automatically and what you have to download yourself and pass as a path.

## Reproducing the paper

`experiments/README.md` lists, for every table and figure of the paper, the commands that produce its runs and the
script that builds it from them. In short, for the tabular results:

    python -m baselines.fetch_external                       # the authors' code of RRL, RL-Net, DR-Net and NeuRules
    python -m experiments.jobs --out results                 # job lists: one command per line, each one resumable
    python -m experiments.run_jobs results/jobs/flare.txt --workers 4
    python -m experiments.run_jobs results/jobs/flare_tuned.txt --workers 4
    python -m experiments.run_jobs results/jobs/baselines_cc18.txt --workers 32
    python -m experiments.run_jobs results/jobs/baselines_tabarena.txt --workers 32
    python -m experiments.paper.table1 --results results --out results/tables          # Table 1 and its paired tests
    python -m experiments.paper.fig_size --results results --out results/figures       # Figure 2
    python -m experiments.paper.fig_scalability --results results --out results/figures   # Figure 3

The tuned runs dominate the cost (30 configurations or one hour per fold for every method): plan for several thousand
CPU hours and a few hundred GPU hours, or run the lines of a job list on a cluster (each line is independent).

## Tests

    pip install -e ".[experiments,dev]"
    pytest                                   # unit tests (CPU; the tests of an optional part skip when its extra is missing)

## Licence

MIT (`LICENSE`). The compared methods keep their own licences (`baselines/README.md`); the authors' code is cloned at
run time, not distributed here.
