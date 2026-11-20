"""The table of the graph experiments from the rows written by run_flare.py and run_baselines.py: for every method and graph, the
mean and standard deviation (over seeds, n - 1 in the denominator) of the graph's metric in %: area under the
precision-recall curve on EMOGI (the rows that have one), accuracy otherwise. A cell is left empty when no seed finished.

    python -m experiments.graphs.summarise results/graphs
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

DATASETS = ['emogi', 'cora', 'citeseer', 'pubmed']


def summarise(folder):
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, 'rows', '*.csv')))], ignore_index=True)
    rows = rows[rows['error'].isna()] if 'error' in rows else rows
    variant = rows['variant'] if 'variant' in rows else pd.Series(np.nan, index=rows.index)
    metric = rows['auprc'].fillna(rows['test_acc']) if 'auprc' in rows else rows['test_acc']
    rows = rows.assign(score=100 * metric,
                       name=rows['method'] + variant.map(lambda v: '' if pd.isna(v) else f' ({v})'))
    grouped = rows.groupby(['name', 'dataset'])['score']
    table = grouped.mean().round(2).astype(str) + ' +- ' + grouped.std().round(2).astype(str) + ' (' + grouped.count().astype(str) + ')'
    table = table.unstack('dataset')
    return table[[d for d in DATASETS if d in table] + [d for d in table if d not in DATASETS]].fillna('')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='?', default='results/graphs', help='the --out folder of the graph runners')
    print(summarise(parser.parse_args(argv).results).to_string())


if __name__ == '__main__':
    main(sys.argv[1:])
