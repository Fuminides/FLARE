"""Figure "Size of the rule base against performance on CC18" (Figure 2 of the paper).

    python -m experiments.paper.fig_size --results results --out results/figures

Each point is a method: its mean macro-F1 over the 42 CC18 datasets (seeds 33-35, 5 folds) against the mean number of
rules of the same trained models. Both coordinates use exactly the same completed rows.

* FLARE at four sizes, joined by a line: a single model of 20 rules, FLARE-explainability (one model, 40 rules), one
  model of 120 rules and FLARE (8 averaged models of 120 rules), all at their default values. The two intermediate
  sizes come from experiments/tabular/run_flare.py with `--config flare-explainability --set rules=20
  single_condition_rules=15 --name "FLARE-20 rules"` and `--set rules=120 single_condition_rules=100 --name "FLARE-one
  model"`.
* The rule and tree learners with a rule count (`complexity` in their rows, see baselines/tabular.py): CORELS,
  RIPPER, FIGS, RuleFit and CART in their tuned configuration (as in Table 1); NeuRules, RRL, RL-Net, DR-Net, GradTree
  and CLASSY at their defaults, because their tuned runs do not record the size.
Writes fig_size.pdf / .png and fig_size.json.
"""
import argparse
import json
import os
import sys

import numpy as np

from experiments.paper.common import expected_cc18, read_rows

FLARE = [('20 rules', 'FLARE-20 rules'), ('FLARE-explainability', 'FLARE-explainability'),
         ('one model', 'FLARE-one model'), ('FLARE', 'FLARE')]
TUNED = ['CORELS', 'RIPPER', 'FIGS', 'RuleFit', 'CART']
DEFAULT = ['NeuRules', 'RRL', 'RL-Net', 'DR-Net', 'GradTree', 'CLASSY']
FAMILY = {'CORELS': 'Exact search', 'RIPPER': 'Greedy', 'CART': 'Greedy', 'FIGS': 'Greedy', 'RuleFit': 'Greedy', 'CLASSY': 'Greedy',
          'NeuRules': 'Differentiable', 'RRL': 'Differentiable', 'RL-Net': 'Differentiable', 'DR-Net': 'Differentiable',
          'GradTree': 'Differentiable'}


def point(rows, method, size_column):
    """Mean macro-F1 (%) and size over the datasets on which `method` finished every seed and fold."""
    expected = expected_cc18()
    rows = rows[(rows.collection == 'cc18') & (rows.method == method)]
    rows = rows[np.isfinite(rows.test_f1) & np.isfinite(rows[size_column])] if size_column in rows else rows.iloc[:0]
    kept = [g for d, g in rows.groupby('dataset') if d in expected and set(zip(g.seed, g.fold)) == expected[d]]
    if not kept:
        return None
    per = [(g.test_f1.mean() * 100, g[size_column].mean()) for g in kept]
    return dict(f1=round(float(np.mean([p[0] for p in per])), 2), size=round(float(np.mean([p[1] for p in per])), 1),
                datasets=len(per))


def collect(rows):
    flare = [dict(name=name, **p) for name, method in FLARE if (p := point(rows, method, 'n_rules'))]
    sized = [dict(name=m, configuration='tuned', **p) for m in TUNED if (p := point(rows, m + '_tuned', 'complexity'))]
    sized += [dict(name=m, configuration='default', **p) for m in DEFAULT if (p := point(rows, m, 'complexity'))]
    return dict(flare=flare, sized=sized)


def draw(D, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    blue, sky, vermillion, green, orange, black, grey = '#0072B2', '#56B4E9', '#D55E00', '#009E73', '#E69F00', '#000000', '#4d4d4d'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'pdf.fonttype': 42, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
    families = {'FLARE': (blue, 'o', 4.6), 'Greedy': (vermillion, 's', 4.0), 'Exact search': (green, '^', 4.8),
                'Differentiable': (orange, 'D', 3.8)}
    # where each label goes: (dx, dy) in points and its alignment
    place = {'CORELS': (-5, 0, 'right', 'center'), 'NeuRules': (-5, -1, 'right', 'top'), 'RL-Net': (5, -3, 'left', 'top'),
             'RIPPER': (5, -1, 'left', 'top'), 'DR-Net': (5, 0, 'left', 'center'), 'GradTree': (5, 0, 'left', 'center'),
             'RRL': (5, 0, 'left', 'center'), 'FIGS': (2, -4, 'center', 'top'), 'RuleFit': (5, 1, 'left', 'bottom'),
             'CART': (2, -6, 'left', 'top'), 'CLASSY': (0, 4, 'center', 'bottom'),
             'FLARE-explainability': (-3, 4, 'right', 'bottom'), 'FLARE': (0, 5, 'center', 'bottom')}
    fig, ax = plt.subplots(figsize=(2.75, 2.2))
    ax.set_xscale('log')
    colour, marker, size = families['FLARE']
    ax.plot([p['size'] for p in D['flare']], [p['f1'] for p in D['flare']], color=colour, lw=1.5, zorder=4)
    for p in D['flare']:
        named = p['name'] in place                     # the two intermediate sizes are light, without a label
        ax.plot(p['size'], p['f1'], ls='none', marker=marker, ms=size, mfc=colour if named else sky, mec='white', mew=0.6, zorder=5)
        if named:
            dx, dy, ha, va = place[p['name']]
            label = 'FLARE-\nexplainability' if p['name'] == 'FLARE-explainability' else p['name']
            ax.annotate(label, (p['size'], p['f1']), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, color=black,
                        fontsize=7.5, fontweight='bold', zorder=6, multialignment='right')
    for p in D['sized']:
        colour, marker, size = families[FAMILY[p['name']]]
        ax.plot(p['size'], p['f1'], ls='none', marker=marker, ms=size, mfc=colour, mec='white', mew=0.6, zorder=3)
        dx, dy, ha, va = place[p['name']]
        ax.annotate(p['name'], (p['size'], p['f1']), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, color=grey, fontsize=7)
    ax.set_xlabel('Number of rules')
    ax.set_ylabel('Macro-F1 (%)')
    ax.set_xlim(2.5, 1500)
    ax.set_ylim(56, 79.5)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(['10', '100', '1,000'])
    ax.minorticks_off()
    ax.set_yticks([60, 65, 70, 75])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=c if name == 'FLARE' else 'none', lw=1.5, marker=m, ms=s, mfc=c, mec='white', mew=0.6, label=name)
               for name, (c, m, s) in families.items()]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, handlelength=1.6, handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.25, borderaxespad=0.1)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.845, bottom=0.17)
    for extension in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_size.{extension}'), dpi=300)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', nargs='+', default=['results'])
    parser.add_argument('--out', default='results/figures')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    D = collect(read_rows(args.results))
    with open(os.path.join(args.out, 'fig_size.json'), 'w') as f:
        json.dump(D, f, indent=1)
    draw(D, args.out)
    print(json.dumps(D, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
