"""Table "Training time in seconds for one fold of the CC18 datasets".

    python -m experiments.paper.times --results results

Median and mean over every fold (42 datasets, seeds 33-35, 5 folds) of the time to train each method once with its
default configuration: `time2train` of FLARE's rows, `fit_time` of the baselines' default rows. The times are not
directly comparable: FLARE and the methods written in PyTorch use a GPU when there is one, the rest run on one CPU
thread.
"""
import argparse
import sys

import pandas as pd

from experiments.paper.common import read_rows

METHODS = ['FLARE', 'FLARE-explainability', 'XGBoost', 'LightGBM', 'CatBoost', 'RIPPER', 'CORELS', 'RRL', 'RL-Net',
           'NeuRules', 'DR-Net', 'GradTree', 'GRANDE', 'CLASSY']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', nargs='+', default=['results'])
    args = parser.parse_args(argv)
    rows = read_rows(args.results)
    rows = rows[(rows.collection == 'cc18') & rows.method.isin(METHODS)]
    seconds = rows['time2train'].where(rows.method.str.startswith('FLARE'), rows.get('fit_time'))
    table = seconds.groupby(rows.method).agg(['median', 'mean', 'count']).reindex(METHODS).round(1)
    print(table.to_string())


if __name__ == '__main__':
    main(sys.argv[1:])
