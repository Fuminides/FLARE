"""The heads of a concept bottleneck model on the concept exports of one dataset.

    python -m experiments.cbm.run_heads --dataset cub --export work/cub/export --regime independent --out results/cbm
    python -m experiments.cbm.run_heads --dataset cub --export work/cub/export --regime sequential --out results/cbm
    python -m experiments.cbm.run_heads --dataset derm7pt --export work/derm7pt/export --heads FLARE LR --seeds 33

Every head is trained on the training split of a source of concepts and scored on the test split of the concepts that the
detector of seed k predicts (`predicted_seed<k>`), with head seed 33 + k, so each of the three seeds pairs a head with a
detector:

    independent   the head is trained on the annotated concepts (`oracle`)
    sequential    the head is trained on the cross-fitted concepts (`crossfit_seed<k>`)

The validation split (of the training source) picks FLARE's epoch and the epoch or temperature of LogicCBM and DCR; RRL,
RL-Net and DR-Net, GradTree and the logistic regression do not use it. Rows are appended to
<out>/<dataset>_<regime>.csv as each head finishes, and a (head, seed) already in the file is not run again. The
printed rules of FLARE go to <out>/rules/. Progress lines are `[k/N]`.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from baselines import cbm_heads
from experiments.cbm import datasets, flare_head

FIRST_SEED = 33                       # head seed 33 + k pairs with the detector of seed k
HEADS = ['FLARE'] + cbm_heads.HEADS
COLUMNS = ['dataset', 'regime', 'head', 'seed', 'detector_seed', 'train_source', 'test_source', 'f1', 'acc', 'seconds',
           'n_rules', 'unique_conditions', 'n_deciding_rules']


def sources(regime, seed):
    """(training source, test source) of a head seed in a regime."""
    k = seed - FIRST_SEED
    return ('oracle' if regime == 'independent' else f'crossfit_seed{k}'), f'predicted_seed{k}'


def run_head(head, names, n_classes, splits, train_source, test_source, seed, device=None):
    """Train one head and score it on the test split of `test_source`. Returns (the result row without its labels,
    FLARE's FitResult or None)."""
    start = time.time()
    y_test = splits['test']['y']
    if head == 'FLARE':
        result = flare_head.fit_flare(names, splits, train_source, test_source, seed, device=device)
        acc, f1 = flare_head.score_flare(result, splits['test'][test_source], y_test)
        return {'f1': f1, 'acc': acc, 'seconds': time.time() - start, **flare_head.rule_counts(result)}, result
    model = cbm_heads.ConceptHead(head, n_classes, seed, device).fit(
        splits['train'][train_source], splits['train']['y'], splits['val'][train_source], splits['val']['y'])
    predicted = model.predict(splits['test'][test_source])
    return {'f1': model.macro_f1(y_test, predicted), 'acc': float(np.mean(predicted == y_test)),
            'seconds': time.time() - start}, None


def done_rows(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {(r['head'], int(r['seed'])) for r in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=sorted(datasets.BUILDERS))
    parser.add_argument('--export', required=True, help='the export folder of the dataset (<work>/export of pipeline.py)')
    parser.add_argument('--regime', default='independent', choices=['independent', 'sequential'])
    parser.add_argument('--heads', nargs='+', default=HEADS, choices=HEADS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[FIRST_SEED, FIRST_SEED + 1, FIRST_SEED + 2])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=os.path.join('results', 'cbm'))
    args = parser.parse_args(argv)

    needed = sorted({s for seed in args.seeds for s in sources(args.regime, seed)})
    names, n_classes, splits = datasets.load_export(args.export, needed)
    heads = [h for h in args.heads if h == 'FLARE' or cbm_heads.runs_on(h, n_classes)]
    for h in set(args.heads) - set(heads):
        print(f'{h} is not run on {n_classes} classes', flush=True)
    os.makedirs(os.path.join(args.out, 'rules'), exist_ok=True)
    path = os.path.join(args.out, f'{args.dataset}_{args.regime}.csv')
    finished = done_rows(path)
    jobs = [(head, seed) for seed in args.seeds for head in heads]
    print(f'{args.dataset} {args.regime}: {len(names)} concepts, {n_classes} classes, train {splits["train"]["y"].shape[0]} rows, '
          f'{len(jobs)} runs -> {path}', flush=True)
    for done, (head, seed) in enumerate(jobs, 1):
        if (head, seed) in finished:
            print(f'[{done}/{len(jobs)}] {head} seed={seed} already in {path}', flush=True)
            continue
        train_source, test_source = sources(args.regime, seed)
        row, result = run_head(head, names, n_classes, splits, train_source, test_source, seed, args.device)
        if result is not None:
            with open(os.path.join(args.out, 'rules', f'{args.dataset}_{args.regime}_{head}_seed{seed}.txt'), 'w') as f:
                f.write(flare_head.printed_rules(result, names) + '\n')
        append_row(path, {'dataset': args.dataset, 'regime': args.regime, 'head': head, 'seed': seed, 'detector_seed': seed - FIRST_SEED,
                          'train_source': train_source, 'test_source': test_source, **row})
        print(f"[{done}/{len(jobs)}] {head} seed={seed} f1={row['f1']:.4f} acc={row['acc']:.4f} seconds={row['seconds']:.0f}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
