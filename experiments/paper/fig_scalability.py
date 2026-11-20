"""Figure "Rank against the size of the dataset" (Figure 3 of the paper).

    python -m experiments.paper.fig_scalability --results results --out results/figures

The 20 methods of Table 1 are ranked on each of the 42 CC18 datasets and the 38 TabArena tasks as in that table
(macro-F1 on CC18, the error of each TabArena task, average ranks for ties, the worst rank for an unfinished entry). On
CC18 FLARE is its default configuration here (the other methods keep the configuration of Table 1). The datasets of
the two benchmarks are pooled and grouped by their training rows (0.8 of the rows on CC18, one split on TabArena); each
point is the mean rank of a method in a group, with the standard error of that mean as a bar. Writes
fig_scalability.pdf / .png and fig_scalability.json (the numbers behind it, including the ranks by number of
variables and the Spearman correlations of FLARE's rank with the rows and the variables).
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

from experiments.data import openml as data
from experiments.paper.common import read_rows
from experiments.paper.table1 import METHODS, cc18_table, tabarena_errors

ROW_BINS, ROW_GROUPS = [0, 1000, 5000, 20000, 1e9], ['up to 1,000', '1,000 to 5,000', '5,000 to 20,000', 'over 20,000']
VARIABLE_BINS, VARIABLE_GROUPS = [0, 15, 40, 1e9], ['up to 15', '15 to 40', 'over 40']


def ranks(rows):
    cc, _ = cc18_table(rows, flare='FLARE')
    errors = tabarena_errors(rows)
    out = {'CC18': cc.rank(axis=1, ascending=False, method='average').fillna(len(METHODS)),
           'TabArena': errors.rank(axis=1, ascending=True, method='average').fillna(len(METHODS))}
    size = {'CC18': {d['name']: (0.8 * d['rows'], d['features']) for d in data.cc18_manifest()},
            'TabArena': {k: (t['n_samples_train_per_fold'], t['n_features']) for k, t in data.tabarena_manifest().items()}}
    every = pd.concat([r[METHODS].assign(benchmark=b, rows=[size[b][d][0] for d in r.index],
                                         variables=[size[b][d][1] for d in r.index]) for b, r in out.items()])
    every['rows_group'] = pd.cut(every.rows, ROW_BINS, labels=ROW_GROUPS)
    every['variables_group'] = pd.cut(every.variables, VARIABLE_BINS, labels=VARIABLE_GROUPS)
    return every


def summary(every):
    by_rows = every.groupby('rows_group', observed=False)
    flare = every[['benchmark', 'rows', 'variables', 'FLARE']].rename(columns={'FLARE': 'rank'})
    return dict(
        row_groups=ROW_GROUPS, datasets_by_training_rows=[int(n) for n in by_rows.size()],
        mean_rank_of_each_method_by_training_rows={m: by_rows[m].mean().round(2).tolist() for m in METHODS},
        sem_of_each_method_by_training_rows={m: by_rows[m].sem().round(2).tolist() for m in METHODS},
        flare_mean_rank_by_variables=every.groupby('variables_group', observed=False)['FLARE'].mean().round(2).tolist(),
        variable_groups=VARIABLE_GROUPS,
        spearman_rank_and_p={f'{b}, {c}': [round(float(v), 3) for v in spearmanr(d['rank'], d[c])]
                             for b, d in [*flare.groupby('benchmark'), ('both', flare)] for c in ('rows', 'variables')})


def draw(D, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, vermillion, orange, purple, black = '#0072B2', '#D55E00', '#E69F00', '#CC79A7', '#000000'   # Okabe-Ito
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'pdf.fonttype': 42, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
    # method -> (colour, line style, marker, marker size): a marker and a line style each, so it also reads in grey
    shown = {'FLARE': (blue, '-', 'o', 4.6), 'CatBoost': (black, '--', '^', 4.2), 'Random forest': (black, ':', 'v', 4.2),
             'Logistic regression': (purple, '-.', 'X', 4.4), 'RuleFit': (vermillion, '-', 's', 3.8), 'RRL': (orange, '-', 'D', 3.6)}
    groups, counts = D['row_groups'], D['datasets_by_training_rows']
    fig, ax = plt.subplots(figsize=(2.75, 2.2))
    for k, (name, (colour, style, marker, size)) in enumerate(shown.items()):
        main = name == 'FLARE'
        x = [i + (k - (len(shown) - 1) / 2) * 0.05 for i in range(len(groups))]     # side by side: the bars do not overlap
        ax.errorbar(x, D['mean_rank_of_each_method_by_training_rows'][name], yerr=D['sem_of_each_method_by_training_rows'][name],
                    color=colour, ls=style, lw=1.8 if main else 1.0, marker=marker, ms=size, mfc=colour, mec='white', mew=0.6,
                    elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=5 if main else 3, label=name)
    ax.set_xticks(range(len(groups)))
    short = {'up to 1,000': 'up to\n1,000', '1,000 to 5,000': '1,000 to\n5,000', '5,000 to 20,000': '5,000 to\n20,000',
             'over 20,000': 'over\n20,000'}
    ax.set_xticklabels([f'{short[g]}\n($n$ = {n})' for g, n in zip(groups, counts)])
    ax.set_xlim(-0.35, len(groups) - 0.65)
    ax.set_ylim(15.5, 0.5)                                  # rank 1 at the top
    ax.set_yticks([1, 5, 10, 15])
    ax.set_xlabel('Training rows')
    ax.set_ylabel('Mean rank (1 is the best)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend([h[0] for h in handles], labels, loc='upper center', ncol=3, frameon=False, handlelength=2.4,
               columnspacing=1.0, labelspacing=0.25, borderaxespad=0.1)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.845, bottom=0.29)
    for extension in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_scalability.{extension}'), dpi=300)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', nargs='+', default=['results'])
    parser.add_argument('--out', default='results/figures')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    D = summary(ranks(read_rows(args.results)))
    with open(os.path.join(args.out, 'fig_scalability.json'), 'w') as f:
        json.dump(D, f, indent=1)
    draw(D, args.out)
    print(json.dumps({k: D[k] for k in ('datasets_by_training_rows', 'flare_mean_rank_by_variables', 'spearman_rank_and_p')}))
    print('FLARE by training rows:', D['mean_rank_of_each_method_by_training_rows']['FLARE'])


if __name__ == '__main__':
    main(sys.argv[1:])
