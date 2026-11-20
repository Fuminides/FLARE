"""From the images to the concept exports: the four steps that precede the heads (see GUIDE.md).

    python -m experiments.cbm.pipeline prepare  cub --root /data/CUB_200_2011 --work work/cub
    python -m experiments.cbm.pipeline train    --work work/cub --seeds 0 1 2
    python -m experiments.cbm.pipeline export   --work work/cub --seeds 0 1 2
    python -m experiments.cbm.pipeline crossfit --work work/cub --seeds 0 1 2

`prepare` reads the dataset (cub, sun or derm7pt), writes the manifest (datasets.py) and the oracle concepts to
<work>/export. `train` fits one detector per seed (detector.py, the recipe of the dataset) in <work>/detectors.
`export` writes the concepts that each detector predicts for every image, `predicted_seed<k>_<split>.npz`. `crossfit`
trains, for each seed k, five more detectors, each on four fifths of the training images, and writes
`crossfit_seed<k>_train.npz`: the concepts of every training image predicted by a detector that did not see it. A head
trained on them sees concepts as noisy as those it will meet at test time.

Every step resumes: a detector or a fold whose checkpoint exists is not trained again. Progress lines are `[k/N]`.
`--limit` keeps a few images of every split and `--epochs` / `--batch_size` replace the recipe's: a smoke test only.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

from experiments.cbm import datasets
from experiments.cbm import detector as det
from experiments.cbm.datasets import SPLITS, Manifest

FOLDS = 5


def paths(work):
    return {'manifest': os.path.join(work, 'manifest.json'), 'export': os.path.join(work, 'export'),
            'detectors': os.path.join(work, 'detectors'), 'crossfit': os.path.join(work, 'crossfit')}


def load_rows(manifest, split, decode=False):
    """The Rows of a split; the recipe that keeps its images in memory (Derm7pt) gets them decoded."""
    files, C = manifest.rows(split)
    return det.Rows(files, C, det.load_images(files) if decode else None)


def say(message):
    print(message, flush=True)


def prepare(dataset, root, work, limit=0):
    manifest = datasets.BUILDERS[dataset](root)
    if limit:
        manifest = manifest.limited(limit)
    where = paths(work)
    manifest.save(where['manifest'])
    datasets.write_oracle_export(manifest, where['export'])
    say(f"{dataset}: {len(manifest.concept_names)} concepts, {len(manifest.class_names)} classes, "
        + ', '.join(f'{s} {len(manifest.splits[s]["ids"])}' for s in SPLITS) + f' -> {work}')
    return manifest


def _setup(work):
    where = paths(work)
    manifest = Manifest.load(where['manifest'])
    return where, manifest, det.RECIPES[manifest.dataset]


def train(work, seeds, device, workers, epochs=None, batch_size=None, pretrained=True):
    where, manifest, recipe = _setup(work)
    os.makedirs(where['detectors'], exist_ok=True)
    decode = recipe.kind == 'derm'
    train_rows, val_rows = load_rows(manifest, 'train', decode), load_rows(manifest, 'val', decode)
    for done, seed in enumerate(seeds, 1):
        path = os.path.join(where['detectors'], f'seed{seed}.pt')
        if os.path.exists(path):
            say(f'[{done}/{len(seeds)}] seed={seed} reused')
            continue
        start = time.time()
        model = det.fit_detector(recipe, train_rows, val_rows, seed, device, workers, epochs, batch_size, pretrained, log=say)
        det.save_detector(path + '.partial', model, manifest.dataset, manifest.concept_names, seed)
        os.replace(path + '.partial', path)                      # a run killed while saving is not taken as finished
        say(f'[{done}/{len(seeds)}] seed={seed} detector trained in {time.time() - start:.0f} s')


def export(work, seeds, device, workers, batch_size=64):
    where, manifest, _ = _setup(work)
    for done, seed in enumerate(seeds, 1):
        model, recipe, names = det.load_detector(os.path.join(where['detectors'], f'seed{seed}.pt'), device)
        if names != manifest.concept_names:
            raise ValueError(f'the detector of seed {seed} was trained on other concepts than the manifest has')
        for split in SPLITS:
            rows = load_rows(manifest, split, recipe.kind == 'derm')
            part = manifest.splits[split]
            P = det.predict(recipe, model, rows, device, batch_size, workers)
            datasets.save_split(os.path.join(where['export'], f'predicted_seed{seed}_{split}.npz'), P, part['y'], part['ids'])
        say(f'[{done}/{len(seeds)}] seed={seed} concepts predicted')


def crossfit(work, seeds, device, workers, folds=FOLDS, epochs=None, batch_size=None, pretrained=True, batch_predict=64):
    """Out-of-fold concepts of the training images. For seed k the training rows are split in `folds` parts with
    KFold(shuffle, random_state=k); the detector of part f (seed 100 k + f) trains on the other parts and predicts part
    f. The validation and test concepts of the source are those of the full detector k."""
    from sklearn.model_selection import KFold
    where, manifest, recipe = _setup(work)
    os.makedirs(where['crossfit'], exist_ok=True)
    decode = recipe.kind == 'derm'
    train_rows, val_rows = load_rows(manifest, 'train', decode), load_rows(manifest, 'val', decode)
    part = manifest.splits['train']
    total, done = len(seeds) * folds, 0
    for seed in seeds:
        out_of_fold = np.full(train_rows.C.shape, np.nan, dtype=np.float32)
        for fold, (fit, held) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(train_rows.C)):
            path = os.path.join(where['crossfit'], f'seed{seed}_fold{fold}.pt')
            if not os.path.exists(path):
                model = det.fit_detector(recipe, train_rows.subset(fit), val_rows, 100 * seed + fold, device, workers, epochs,
                                         batch_size, pretrained, log=lambda message: None)
                det.save_detector(path + '.partial', model, manifest.dataset, manifest.concept_names, 100 * seed + fold)
                os.replace(path + '.partial', path)
            model, _, _ = det.load_detector(path, device)
            out_of_fold[held] = det.predict(recipe, model, train_rows.subset(held), device, batch_predict, workers)
            done += 1
            say(f'[{done}/{total}] seed={seed} fold={fold} held-out concepts predicted')
        datasets.save_split(os.path.join(where['export'], f'crossfit_seed{seed}_train.npz'), out_of_fold, part['y'], part['ids'])
        for split in ('val', 'test'):                       # the heads see the full detector's concepts there
            shutil.copyfile(os.path.join(where['export'], f'predicted_seed{seed}_{split}.npz'),
                            os.path.join(where['export'], f'crossfit_seed{seed}_{split}.npz'))
        say(f'seed={seed} out-of-fold concept AUROC {det.mean_auroc(train_rows.C, out_of_fold):.4f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='step', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('dataset', choices=sorted(datasets.BUILDERS))
    p.add_argument('--root', required=True, help="the dataset's folder: CUB_200_2011, a folder holding SUNAttributeDB/ and images/ (SUN), or the Derm7pt release")
    p.add_argument('--limit', type=int, default=0, help='keep this many images of every split (a smoke test)')
    steps = {name: sub.add_parser(name) for name in ('train', 'export', 'crossfit')}
    for name, p in steps.items():
        p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--workers', type=int, default=4, help='processes that load images')
    for name in ('train', 'crossfit'):
        steps[name].add_argument('--epochs', type=int, default=None, help="replaces the recipe's (a smoke test)")
        steps[name].add_argument('--batch_size', type=int, default=None, help="replaces the recipe's (a smoke test)")
        steps[name].add_argument('--no_pretrained', action='store_true', help='random weights instead of ImageNet (tests)')
    steps['crossfit'].add_argument('--folds', type=int, default=FOLDS)
    for p in (sub.choices[name] for name in sub.choices):
        p.add_argument('--work', required=True, help='the folder of this dataset (manifest, detectors and exports)')
    args = parser.parse_args(argv)
    if args.step == 'prepare':
        prepare(args.dataset, args.root, args.work, args.limit)
    elif args.step == 'train':
        train(args.work, args.seeds, args.device, args.workers, args.epochs, args.batch_size, not args.no_pretrained)
    elif args.step == 'export':
        export(args.work, args.seeds, args.device, args.workers)
    else:
        crossfit(args.work, args.seeds, args.device, args.workers, args.folds, args.epochs, args.batch_size, not args.no_pretrained)


if __name__ == '__main__':
    main(sys.argv[1:])
