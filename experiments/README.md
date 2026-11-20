# Reproducing the paper

Every command runs from the repository root and resumes: a run whose result file is complete is skipped, so a command
can be repeated after a crash, and the lines of a job list can be spread over several machines or a cluster. All the
runs write CSV rows below a results folder (`results/` here); the table and figure scripts read every `rows/*.csv`
below the folders they are given.

| Result in the paper | Runs | Built by |
|---|---|---|
| Table 1 (CC18 and TabArena) and the paired comparisons of the appendix | `jobs/flare.txt`, `jobs/flare_tuned.txt`, `jobs/baselines_cc18.txt`, `jobs/baselines_tabarena.txt` | `python -m experiments.paper.table1 --results results --out results/tables` |
| Figure 2, size of the rule base against performance (CC18) | `jobs/flare.txt` (four sizes of FLARE), `jobs/baselines_cc18.txt` | `python -m experiments.paper.fig_size --results results --out results/figures` |
| Figure 3, rank against the size of the dataset | as Table 1 | `python -m experiments.paper.fig_scalability --results results --out results/figures` |
| Training time of one fold (appendix) | `jobs/flare.txt`, `jobs/baselines_cc18.txt` (default rows) | `python -m experiments.paper.times --results results` |
| Concept bottleneck models (CUB, SUN Attribute, Derm7pt) | `experiments/cbm/GUIDE.md` | `experiments/cbm/run_heads.py` writes the table's rows |
| Graph node classification (Cora, Citeseer, PubMed, EMOGI) | `experiments.graphs.run_flare`, `experiments.graphs.run_baselines` | `python -m experiments.graphs.summarise results/graphs` |

## Tabular benchmarks

```bash
pip install -e ".[experiments,baselines]"
python -m baselines.fetch_external                      # the authors' code of RRL, RL-Net, DR-Net and NeuRules
python -m experiments.jobs --out results                # results/jobs/*.txt, one command per line
python -m experiments.run_jobs results/jobs/flare.txt --workers 4
python -m experiments.run_jobs results/jobs/flare_tuned.txt --workers 4
python -m experiments.run_jobs results/jobs/baselines_cc18.txt --workers 32
python -m experiments.run_jobs results/jobs/baselines_tabarena.txt --workers 32
```

* **Data**: the 42 OpenML-CC18 datasets (`data/cc18.json`) and the 38 classification tasks of TabArena v0.1
  (`data/tabarena.json`) are downloaded from OpenML on first use into `$FLARE_DATA_DIR` (default `~/.cache/flare`).
  CC18: 5 stratified folds shuffled with seeds 33, 34 and 35. TabArena: the task's own outer splits, seed 33.
  `experiments.paper.table1` also downloads the per-split results that TabArena published for its gradient-boosted
  trees, random forest, EBM and logistic regression.
* **Preprocessing** (`flare/preprocessing.py`, shared by every method): imputation from the training rows, 20% of the
  training rows held out for validation (FLARE chooses its epoch on them; the other methods are fitted on the same 80%),
  standardised real variables, one-hot categoricals for the baselines.
* **Tuning**: every method of Table 1 on CC18, FLARE included, is tuned per fold with Optuna's TPE sampler, 30
  configurations or one hour, each scored by macro-F1 on rows of the training fold that it was not fitted on, and the
  best one is refitted as the default run is. The baselines hold out 20% of the 80% training part they are fitted on
  (`baselines/tuning.py`); FLARE holds out 20% of the whole training fold and runs on the rest as it does on a fold,
  with its own 80/20 split for choosing the epoch (`experiments/tabular/tune_flare.py`). The test rows are never read.
  FLARE-explainability is not tuned. On TabArena every method runs at its defaults.
* **Single runs**, e.g. to try one dataset:

      python -m experiments.tabular.run_flare --benchmark cc18 --dataset diabetes --config flare --seed 33 --out results/flare
      python -m experiments.tabular.tune_flare --dataset diabetes --seed 33 --fold 0 --out results/flare_tuned
      python -m experiments.tabular.run_baselines --benchmark cc18 --method XGBoost --mode tuned --dataset diabetes --seed 33 --out results/baselines

* **Cost**: the tuned lines dominate: about one GPU hour per fold for FLARE (630 folds) and up to one CPU hour per fold
  and method for the baselines. The untuned FLARE runs take seconds to minutes per dataset on a GPU.

## Graphs

```bash
pip install -e ".[experiments,baselines,graphs]"
python -m experiments.graphs.run_flare --out results/graphs            # the four graphs, seeds 33, 34 and 35
python -m experiments.graphs.run_baselines --out results/graphs
python -m experiments.graphs.summarise results/graphs
```

Cora, Citeseer and PubMed (the Planetoid files) are downloaded from https://github.com/kimiyoung/planetoid on first use;
EMOGI's multi-omics graph (CPDB) from https://owww.molgen.mpg.de/~sasse/EMOGI/ (only `CPDB_multiomics.h5` is
extracted from the archive). Both go to `$FLARE_DATA_DIR`. FLARE reads, for each node, its own variables and the share
of its neighbours that satisfy each condition on them (`flare/graph.py`); the variant of each graph and its epochs are
listed in `experiments/graphs/run_flare.py`.

## Concept bottleneck models

See `experiments/cbm/GUIDE.md`: the three image datasets have to be downloaded by hand (their licences do not allow
redistribution) and their folders passed with `--root`. The pipeline builds the manifests, trains the Inception v3
concept detectors (GPU), exports the predicted and the cross-fitted concepts and runs every head.

## Checking an installation

    pytest                                  # unit tests on small synthetic data (CPU)
    python -m experiments.tabular.run_flare --benchmark cc18 --dataset diabetes --config flare-explainability --fold 0 --out /tmp/flare_check
