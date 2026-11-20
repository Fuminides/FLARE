"""Job lists of the tabular runs: one shell command per line, each one dataset x seed (x fold for tuned FLARE).

    python -m experiments.jobs --out results          # writes results/jobs/*.txt
    python -m experiments.run_jobs results/jobs/flare.txt --workers 4

Every command resumes: a line whose result file is complete is skipped, so a list can be run again after a crash or
split across machines (run lines 1..N of a list with any scheduler; each line is independent). The lists:

    flare.txt            FLARE and FLARE-explainability, CC18 (42 datasets x seeds 33-35) and TabArena (38 tasks, seed 33),
                         and the two smaller FLARE sizes on CC18
    flare_tuned.txt      tuned FLARE on CC18, one line per dataset, seed and fold (630 lines)
    baselines_cc18.txt   every compared method on CC18, at its defaults and tuned
    baselines_tabarena.txt  the rule and tree learners that TabArena did not publish, at their defaults, seed 33

The tuned lines dominate the cost. FLARE runs on a GPU when one is visible; the baselines use one CPU thread each.
"""
import argparse
import os
import shlex
import sys

from baselines import tabular
from experiments.data import openml as data

FLARE_SIZES = {'FLARE-20 rules': 'rules=20 single_condition_rules=15',
               'FLARE-one model': 'rules=120 single_condition_rules=100'}


def flare_lines(out):
    lines = []
    for dataset in data.cc18_names():
        for seed in data.SEEDS:
            for config in ('flare', 'flare-explainability'):
                lines.append(f'python -m experiments.tabular.run_flare --benchmark cc18 --dataset {dataset} --config {config} '
                             f'--seed {seed} --out {out}/flare')
            for name, options in FLARE_SIZES.items():
                lines.append(f'python -m experiments.tabular.run_flare --benchmark cc18 --dataset {dataset} '
                             f'--config flare-explainability --set {options} --name {shlex.quote(name)} --seed {seed} --out {out}/flare')
    for task in data.tabarena_manifest():
        for config in ('flare', 'flare-explainability'):
            lines.append(f'python -m experiments.tabular.run_flare --benchmark tabarena --dataset {task} --config {config} '
                         f'--seed 33 --out {out}/flare')
    return lines


def flare_tuned_lines(out):
    return [f'python -m experiments.tabular.tune_flare --dataset {d} --seed {s} --fold {f} --out {out}/flare_tuned'
            for d in data.cc18_names() for s in data.SEEDS for f in range(data.N_FOLDS)]


def baseline_lines(benchmark, methods, modes, seeds, out):
    return [f'python -m experiments.tabular.run_baselines --benchmark {benchmark} --method {m} --mode {mode} '
            f'--dataset {d} --seed {s} --out {out}/baselines'
            for m in methods for mode in modes
            for d in (data.cc18_names() if benchmark == 'cc18' else data.tabarena_manifest()) for s in seeds]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='results')
    args = parser.parse_args(argv)
    every = tabular.METHODS + tabular.INTERPRETABLE + tabular.REFERENCES
    lists = {'flare': flare_lines(args.out), 'flare_tuned': flare_tuned_lines(args.out),
             'baselines_cc18': baseline_lines('cc18', every, ('default', 'tuned'), data.SEEDS, args.out),
             'baselines_tabarena': baseline_lines('tabarena', tabular.TABARENA_METHODS, ('default',), (33,), args.out)}
    os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
    for name, lines in lists.items():
        with open(os.path.join(args.out, 'jobs', name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{args.out}/jobs/{name}.txt: {len(lines)} lines')


if __name__ == '__main__':
    main(sys.argv[1:])
