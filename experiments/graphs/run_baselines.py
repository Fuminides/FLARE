"""The compared methods on the graph node-classification tasks, on exactly the nodes FLARE sees
(experiments/data/graphs.py, seeds 33, 34 and 35).

    LR         logistic regression on the node's own variables
    LR-nb      the same plus the mean of the neighbours and log degree
    SGC-1/2/3  logistic regression on the variables propagated over 1, 2 or 3 hops
    GCN        two-layer GCN on standardised variables
    GCN-standard, GAT-standard    GCN and GAT as published, with their own preprocessing and training (citation graphs)
    RF, EBM    random forest and EBM on the node's variables, the means of its neighbours at one and two hops and log degree
    RIPPER, CORELS, CLASSY, RRL, RL-Net, NeuRules, DR-Net, GradTree, GRANDE, FIGS, RuleFit, CART
               the rule and tree learners of the tabular benchmarks at their defaults on that same table

    python -m experiments.graphs.run_baselines --datasets cora --methods SGC-3 GCN-standard --seeds 33 --out results/graphs
    python -m experiments.graphs.run_baselines --methods RF RuleFit --out results/graphs

Output: <out>/rows/<method>_<dataset>_seed<S>.csv, one row with the scores (accuracy, macro-F1, balanced accuracy and, on EMOGI,
the areas under the precision-recall and ROC curves) and the number of rules of the rule learners. A job whose row exists is not
run again; a method that cannot run here (a package or environment missing, GCN-standard on EMOGI) is skipped with a message.
The progress line of every job is "[k/N] ...". Methods that need the authors' code or an environment of their own are
described in baselines/README.md.
"""
import argparse
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from baselines import graph as models
from experiments.data import graphs
from experiments.tabular.run_baselines import unavailable

SEEDS = (33, 34, 35)
METHODS = ['LR', 'LR-nb', 'SGC-1', 'SGC-2', 'SGC-3', 'GCN', 'GCN-standard', 'GAT-standard', 'RF', 'EBM'] + models.RULE_LEARNERS
STANDARD = ('GCN-standard', 'GAT-standard')             # the published protocols are those of the citation graphs


def probabilities(method, data, train, val, test, seed):
    """(test probabilities, number of rules or NaN) of `method`."""
    balanced = data.metric == 'auprc'
    size = float('nan')
    if method in ('LR', 'LR-nb'):
        P, _ = models.logistic(data, train, val, test, method == 'LR-nb', balanced)
    elif method.startswith('SGC-'):
        P, _ = models.sgc(data, train, val, test, int(method[-1]), balanced)
    elif method == 'GCN':
        P = models.gcn(data, train, val, test, seed, hidden=64 if data.name == 'emogi' else 16, epochs=500 if data.name == 'emogi' else 200,
                       balanced=balanced)
    elif method == 'GCN-standard':
        P, _ = models.gcn_standard(data, train, val, test, seed)
    elif method == 'GAT-standard':
        P, _ = models.gat_standard(data, train, val, test, seed)
    elif method in ('RF', 'EBM'):
        P, _ = models.table_probabilities(method, data, train, val, test, seed)
    else:
        P, _, size = models.rule_learner(method, data, train, val, test, seed)
    return P, size


def run_job(data, method, seed, out):
    """Fit `method` on the nodes of `data` for `seed`; the row is written to <out>/rows and returned."""
    path = os.path.join(out, 'rows', f'{method}_{data.name}_seed{seed}.csv')
    if os.path.exists(path):
        done = pd.read_csv(path)
        if 'error' not in done or done['error'].isna().all():
            print(f'[reused] {path}', flush=True)
            return done
    row = {'dataset': data.name, 'collection': 'graph', 'seed': seed, 'method': method}
    start = time.time()
    try:
        train, val, test = graphs.split_indices(data, seed)
        P, size = probabilities(method, data, train, val, test, seed)
        row.update(models.graph_scores(data.metric, data.y[test], P), complexity=size)
    except Exception as error:                              # recorded, never silently dropped
        row['error'] = f'{type(error).__name__}: {error}'[:300]
        traceback.print_exc()
    row['seconds'] = time.time() - start
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)
    return frame


def main(argv=None):
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=graphs.NAMES, choices=graphs.NAMES)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    parser.add_argument('--out', default='results/graphs')
    args = parser.parse_args(argv)
    jobs = [(d, m, s) for d in args.datasets for m in args.methods for s in args.seeds
            if not (m in STANDARD and d not in graphs.PLANETOID)]
    skipped = {m: unavailable(m) for m in args.methods if unavailable(m)}
    for method, reason in skipped.items():
        print(f'[skipped] {method}: {reason}', flush=True)
    jobs = [job for job in jobs if job[1] not in skipped]
    failed, loaded = 0, {}
    for k, (name, method, seed) in enumerate(jobs, 1):
        if name not in loaded:
            loaded = {name: graphs.load_graph(name)}
        row = run_job(loaded[name], method, seed, args.out).iloc[0]
        failed += 'error' in row and not pd.isna(row['error'])
        shown = row['auprc'] if name == 'emogi' and 'auprc' in row else row.get('test_acc', np.nan)
        print(f"[{k}/{len(jobs)}] {method} {name} seed={seed} {'auprc' if name == 'emogi' else 'acc'}={shown:.4f} seconds={row['seconds']:.0f}"
              + (f" FAILED {row['error']}" if 'error' in row and not pd.isna(row['error']) else ''), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main(sys.argv[1:])
