"""FLARE on the graph node-classification tasks: Cora, Citeseer, PubMed and EMOGI.

Each graph has its own variant of the nodes' rows (flare/graph.py):

    share    the node's variables and the share of its neighbours that satisfy each condition on them
    degree   `share` and log10(1 + degree) as one more variable (EMOGI, Citeseer)
    hops2    `share` for the neighbours at one hop and at two hops (Cora, PubMed)

and its number of epochs (100 on EMOGI, 200 on the citation graphs), as listed in PAPER; `--variant` runs another one.
The partitions of all the variables are trained (flare/graph.py; `--fixed_sets` keeps them as initialised). The nodes
are split as published (Planetoid) or as the authors of EMOGI did, with a validation part drawn from their training
genes with the seed (experiments/data/graphs.py).

    python -m experiments.graphs.run_flare --datasets cora --seeds 33 --out results/graphs
    python -m experiments.graphs.run_flare --out results/graphs                  # the four graphs, seeds 33, 34 and 35
    python -m experiments.graphs.run_flare --datasets cora --variant share --epochs 50 --bags 2

Output: <out>/rows/FLARE_<variant>_<dataset>_seed<S>.csv (one row: accuracy, macro-F1, balanced accuracy, and area under the
precision-recall curve and ROC AUC on EMOGI, rules, seconds) and the printed rules in <out>/rules/. A job whose row exists
is not run again. The progress line of every job is "[k/N] ...".
"""
import argparse
import os
import sys
import time
import warnings

import pandas as pd

from baselines.graph import graph_scores
from experiments.data import graphs
from flare.graph import HOP_NAMES, fit_graph, graph_config, neighbourhood_features, predict_proba
from flare.printing import rules_text
from flare.train import score_fold

SEEDS = (33, 34, 35)
VARIANTS = {'share': dict(hops=1, degree=False), 'degree': dict(hops=1, degree=True), 'hops2': dict(hops=2, degree=False)}
PAPER = {'emogi': ('degree', 100), 'cora': ('hops2', 200), 'citeseer': ('degree', 200), 'pubmed': ('hops2', 200)}   # variant, epochs


def prepare(data, variant, seed):
    """(features, y, train, val, test): the rows of the nodes of `data` for a variant, and its split for the seed."""
    options = VARIANTS[variant]
    data = graphs.with_degree_column(data) if options['degree'] else data
    train, val, test = graphs.split_indices(data, seed)
    features = neighbourhood_features(data.X, data.categorical, data.adjacency, train, hops=options['hops'],
                                      hop_names=[data.network_names[0]] + HOP_NAMES[1:],
                                      category_names=data.category_names, quantify_categorical=data.quantify_categorical)
    return features, data.y, train, val, test


def run_job(data, variant, epochs, seed, bags=8, out='results/graphs', device=None, learn_sets=True, verbose=False, overrides=None):
    """Train and score one (graph, variant, seed); the row is written to <out>/rows and returned."""
    # Settings other than the defaults (the graph's epochs, 8 models, trained partitions) are part of the file name,
    # so a short test never stands in for a full run.
    changed = ('' if epochs == PAPER.get(data.name, (None, epochs))[1] else f'_ep{epochs}') + ('' if bags == 8 else f'_bags{bags}') \
        + ('' if learn_sets else '_fixedsets')
    tag = f'FLARE_{variant}_{data.name}_seed{seed}{changed}'
    path = os.path.join(out, 'rows', tag + '.csv')
    if os.path.exists(path):
        print(f'[reused] {path}', flush=True)
        return pd.read_csv(path)
    start = time.time()
    features, y, train, val, test = prepare(data, variant, seed)
    config = graph_config(balanced=data.metric == 'auprc', epochs=epochs, bags=bags, **(overrides or {}))
    result = fit_graph(features, y, train, val, test, config, seed, learn_sets=learn_sets, device=device, verbose=verbose)
    row = {'dataset': data.name, 'collection': 'graph', 'seed': seed, 'method': 'FLARE', 'variant': variant + changed, 'epochs': epochs, 'bags': bags,
           'variables': features.X.shape[1], 'route': result.route}
    scored = score_fold(result)
    row.update(graph_scores(data.metric, y[test], predict_proba(result)))
    row.update(n_rules=scored['n_rules'], n_deciding_rules=scored['n_deciding_rules'], n_ants=scored['n_ants'],
               n_deciding_ants=scored['n_deciding_ants'], train_acc=scored['train_acc'], time2train=result.time2train,
               seconds=time.time() - start)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.join(out, 'rules'), exist_ok=True)
    with open(os.path.join(out, 'rules', tag + '.txt'), 'w') as f:
        f.write(rules_text(result.model, feat_names=features.names, class_names=data.class_names) + '\n')
    frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)
    return frame


def main(argv=None):
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=graphs.NAMES, choices=graphs.NAMES)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--variant', choices=sorted(VARIANTS), default=None, help="default: the paper's variant of each graph")
    parser.add_argument('--epochs', type=int, default=None, help="default: the paper's number of epochs of each graph")
    parser.add_argument('--bags', type=int, default=8, help='averaged models')
    parser.add_argument('--fixed_sets', action='store_true', help='keep the partitions of the variables fixed instead of training them')
    parser.add_argument('--out', default='results/graphs')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    total, done = len(args.datasets) * len(args.seeds), 0
    for name in args.datasets:
        data = graphs.load_graph(name)
        variant, epochs = args.variant or PAPER[name][0], args.epochs or PAPER[name][1]
        for seed in args.seeds:
            frame = run_job(data, variant, epochs, seed, args.bags, args.out, learn_sets=not args.fixed_sets, verbose=args.verbose)
            done += 1
            row = frame.iloc[0]
            print(f"[{done}/{total}] FLARE {variant} {name} seed={seed} acc={row['test_acc']:.4f} f1={row['test_f1']:.4f}"
                  + (f" auprc={row['auprc']:.4f}" if 'auprc' in row and not pd.isna(row['auprc']) else '')
                  + f" rules={row['n_rules']} seconds={row['seconds']:.0f}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
