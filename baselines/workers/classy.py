"""CLASSY (Proenca, Grunwald, Leeuwen and Moerland, 2022, `rulelist` 0.2.0) on one fold inside its own environment.

CLASSY needs numpy / pandas 1.x, older than the rest of the repository, so experiments/tabular/run_baselines.py runs it
in a subprocess of the interpreter named in $FLARE_CLASSY_PYTHON (baselines/README.md lists the packages). Reads
<dir>/in.npz (X, y, X_test), takes the defaults or tunes as run_baselines.py does in-process, and writes
<dir>/out.npz (predictions on X and X_test, fit seconds, configuration, trials done, validation macro-F1, rules).

    $FLARE_CLASSY_PYTHON baselines/workers/classy.py --dir D --mode tuned --seed 33 --tune_seed 33
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root
from baselines import tabular                                                                        # noqa: E402
from baselines.tuning import fit_predict, tune                                                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--mode', choices=['default', 'tuned'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tune_seed', type=int, required=True)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--timeout', type=int, default=3600)
    a = p.parse_args()
    d = np.load(os.path.join(a.dir, 'in.npz'))
    X, y, X_test = d['X'], d['y'], d['X_test']
    params, n, val = dict(tabular.DEFAULTS['CLASSY']), 0, float('nan')
    if a.mode == 'tuned':
        params, n, val = tune('CLASSY', a.tune_seed, X, y, a.trials, a.timeout)
    (p_train, p_test), model, seconds = fit_predict('CLASSY', params, a.seed, X, y, [X, X_test])
    np.savez(os.path.join(a.dir, 'out.npz'), pred_full=p_train, pred_test=p_test, fit_seconds=seconds,
             params=json.dumps(params, default=str), n_trials=n, val_f1=val, complexity=tabular.n_rules('CLASSY', model))


if __name__ == '__main__':
    main()
