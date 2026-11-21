"""FLARE from the command line, on a CSV file.

    python -m flare fit train.csv --target class                       # prints the rules, saves model.pt
    python -m flare fit train.csv --target class --test test.csv --config flare-explainability --rules rules.txt
    python -m flare fit train.csv --target class --categorical colour city --set rules=60 epochs=50
    python -m flare predict model.pt new.csv --out predictions.csv --proba

`fit` trains FLARE on every column but the target. Columns whose values are not numbers are categorical; give
--categorical to name them yourself (e.g. integer codes). Missing values may be left empty. With --test it reports the
accuracy and macro-F1 on another CSV with the same columns. `predict` writes one row per input row with the predicted
class (and, with --proba, the probability of each class).
"""
import argparse
import sys

import numpy as np
import pandas as pd

from .config import PRESETS, parse_options
from .estimator import FLAREClassifier


def fit(args):
    data = pd.read_csv(args.data, sep=args.sep)
    if args.target not in data:
        raise SystemExit(f'{args.target!r} is not a column of {args.data}')
    y = data.pop(args.target)
    config = PRESETS[args.config](**parse_options(args.set))
    model = FLAREClassifier(config, seed=args.seed, device=args.device, verbose=args.verbose)
    model.fit(data, y, categorical=args.categorical)
    text = model.rules_text()
    if args.rules:
        with open(args.rules, 'w') as f:
            f.write(text + '\n')
    else:
        print(text)
    if args.test:
        from sklearn.metrics import accuracy_score, f1_score
        test = pd.read_csv(args.test, sep=args.sep)
        y_test = test.pop(args.target)
        predicted = model.predict(test)
        print(f'test accuracy {accuracy_score(y_test.astype(str), predicted.astype(str)):.4f}, '
              f'macro-F1 {f1_score(y_test.astype(str), predicted.astype(str), average="macro"):.4f}')
    model.save(args.model)
    n_rules = sum(line.startswith('IF ') for line in text.splitlines())
    print(f'{n_rules} rules; model saved to {args.model}' + (f', rules to {args.rules}' if args.rules else ''))


def predict(args):
    model = FLAREClassifier.load(args.model, device=args.device or 'cpu')
    data = pd.read_csv(args.data, sep=args.sep)
    if args.target and args.target in data:
        data = data.drop(columns=args.target)
    out = pd.DataFrame({'prediction': model.predict(data)})
    if args.proba:
        P = model.predict_proba(data)
        for c, name in enumerate(model.classes_):
            out[f'p_{name}'] = np.round(P[:, c], 6)
    if args.out:
        out.to_csv(args.out, index=False)
    else:
        out.to_csv(sys.stdout, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m flare', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit', help='train FLARE on a CSV file')
    p.add_argument('data', help='CSV file with the training rows')
    p.add_argument('--target', required=True, help='the column with the class')
    p.add_argument('--categorical', nargs='*', default=None, help='the categorical columns (default: the non-numeric ones)')
    p.add_argument('--config', default='flare', choices=sorted(PRESETS))
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='other options, e.g. rules=60 epochs=50 (flare/config.py)')
    p.add_argument('--test', help='CSV file with test rows (same columns) to score the model on')
    p.add_argument('--model', default='model.pt', help='where to save the model (default: model.pt)')
    p.add_argument('--rules', help='write the rules to this file instead of printing them')
    p.add_argument('--seed', type=int, default=33)
    p.add_argument('--device', default=None, help='cpu or cuda (default: cuda when available)')
    p.add_argument('--sep', default=',', help='field separator of the CSV files')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(run=fit)
    p = sub.add_parser('predict', help='predict the rows of a CSV file with a saved model')
    p.add_argument('model', help='a model saved by fit')
    p.add_argument('data', help='CSV file with the same columns as the training file (the target may be there)')
    p.add_argument('--target', default=None, help='a column to ignore (the class, if the file has it)')
    p.add_argument('--out', help='CSV file for the predictions (default: print them)')
    p.add_argument('--proba', action='store_true', help='also write the probability of each class')
    p.add_argument('--device', default=None)
    p.add_argument('--sep', default=',')
    p.set_defaults(run=predict)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except ValueError as error:
        raise SystemExit(f'error: {error}')


if __name__ == '__main__':
    main(sys.argv[1:])
