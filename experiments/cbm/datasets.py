"""The three image datasets of the concept bottleneck experiments, as manifests, and the layout of the concept exports.

A manifest lists, for every image of each split (train / val / test), its path, its binary concepts and its class:

    CUB (per image)   200 bird species, 112 concepts: the subset of the 312 attributes that Koh et al. (2020) keep, with
                      the annotation of each image (not the majority of its class). The official train / test split;
                      20% of the training images of every class, drawn with seed 0, are the validation split.
    SUN Attribute     717 scene categories, 102 concepts: an attribute is present when at least two of the three
                      workers voted for it (vote fraction >= 0.5). Each category's 20 images are split at random into
                      80% / 10% / 10% (seed 0).
    Derm7pt           5 diagnosis groups, 12 concepts: the grouped criteria of the 7-point checklist, and the authors'
                      train / validation / test indexes.

The detectors (detector.py) read the images of a manifest and produce the *predicted* concepts. The heads
(run_heads.py) read an export: a folder with the concept names and, for each source and split, the concepts of every
row in the order of the manifest.

    concept_names.json                  the names of the concepts
    oracle_<split>.npz                  C (the annotated concepts), y (class), ids (image or case number)
    predicted_seed<k>_<split>.npz       the concepts predicted by the detector of seed k
    crossfit_seed<k>_<split>.npz        for the training split, the concepts of each image predicted by a detector that
                                        did not train on it (cross-fitting); validation and test are the predicted ones

Image files are never copied: a manifest keeps their paths relative to the dataset's own folder.
"""
import json
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

SPLITS = ('train', 'val', 'test')

# The 112 attributes (0-based indexes of CUB's attribute list) that Koh et al. keep: those that, in the class-level
# annotation, are present in at least 10 classes after the majority vote (their CUB/generate_new_data.py).
KOH_ATTRIBUTES = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99,
    101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151,
    152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
    194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235,
    236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274,
    277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
]

DERM_CLASSES = ['BCC', 'NEV', 'MEL', 'MISC', 'SK']
DERM_NEVI = {'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus', 'recurrent nevus',
             'reed or spitz nevus'}
DERM_MISC = {'dermatofibroma', 'lentigo', 'melanosis', 'miscellaneous', 'vascular lesion'}
# name, criterion (column of meta.csv) and the values of the criterion that make the concept present
DERM_CONCEPTS = [('pigment_network_typical', 'pigment_network', {'typical'}),
                 ('pigment_network_atypical', 'pigment_network', {'atypical'}),
                 ('streaks_regular', 'streaks', {'regular'}),
                 ('streaks_irregular', 'streaks', {'irregular'}),
                 ('pigmentation_regular', 'pigmentation', {'diffuse regular', 'localized regular'}),
                 ('pigmentation_irregular', 'pigmentation', {'diffuse irregular', 'localized irregular'}),
                 ('regression_structures_present', 'regression_structures', {'blue areas', 'white areas', 'combinations'}),
                 ('dots_and_globules_regular', 'dots_and_globules', {'regular'}),
                 ('dots_and_globules_irregular', 'dots_and_globules', {'irregular'}),
                 ('blue_whitish_veil_present', 'blue_whitish_veil', {'present'}),
                 ('vascular_structures_regular', 'vascular_structures',
                  {'arborizing', 'comma', 'hairpin', 'within regression', 'wreath'}),
                 ('vascular_structures_irregular', 'vascular_structures', {'dotted', 'linear irregular'})]
DERM_SPLIT_FILES = {'train': 'train_indexes.csv', 'val': 'valid_indexes.csv', 'test': 'test_indexes.csv'}


@dataclass
class Manifest:
    dataset: str                         # 'cub', 'sun' or 'derm7pt'
    image_root: str                      # the folder the image paths are relative to
    concept_names: list
    class_names: list
    splits: dict = field(default_factory=dict)   # {split: {'ids': [N], 'paths': [N], 'C': [N, K] 0/1, 'y': [N]}}

    def rows(self, split):
        part = self.splits[split]
        return [os.path.join(self.image_root, p) for p in part['paths']], np.asarray(part['C'], dtype=np.float32)

    def limited(self, per_split):
        """The same manifest with `per_split` rows of each split, spread over it (for smoke tests)."""
        out = Manifest(self.dataset, self.image_root, self.concept_names, self.class_names)
        for split, part in self.splits.items():
            n = len(part['ids'])
            index = np.arange(0, n, max(n // per_split, 1))[:per_split]
            out.splits[split] = {key: [value[i] for i in index] for key, value in part.items()}
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {'dataset': self.dataset, 'image_root': self.image_root, 'concept_names': self.concept_names,
                   'class_names': self.class_names,
                   'splits': {s: {k: np.asarray(v).tolist() if k in ('C', 'y', 'ids') else list(v) for k, v in part.items()}
                              for s, part in self.splits.items()}}
        with open(path, 'w') as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            payload = json.load(f)
        out = cls(payload['dataset'], payload['image_root'], payload['concept_names'], payload['class_names'])
        out.splits = payload['splits']
        return out


def _split_part(ids, paths, C, y):
    return {'ids': [int(i) for i in ids], 'paths': list(paths), 'C': [[int(v) for v in row] for row in C],
            'y': [int(v) for v in y]}


# --- CUB ----------------------------------------------------------------------------------------------------------------

def _read_pairs(path, cast=str):
    mapping = {}
    with open(path, encoding='utf-8') as f:
        for line in f.read().splitlines():
            key, value = line.split(maxsplit=1)
            mapping[int(key)] = cast(value)
    return mapping


def _cub_concept_name(name):
    return name.replace('::', '_').replace('(', '').replace(')', '').replace('-', '_').replace(' ', '_')


def _cub_attribute_names(root, ids):
    """{attribute id: name}: from attributes.txt (in attributes/, in the root, or next to it, where the official archive
    puts it), or generic names when the file is missing."""
    for candidate in (os.path.join(root, 'attributes', 'attributes.txt'), os.path.join(root, 'attributes.txt'),
                      os.path.join(os.path.dirname(os.path.abspath(root)), 'attributes.txt')):
        if os.path.exists(candidate):
            return _read_pairs(candidate)
    warnings.warn('attributes.txt not found: the concepts get generic names (attribute_<id>)')
    return {i: f'attribute_{i}' for i in ids}


def build_cub(root, val_fraction=0.2, seed=0):
    """Manifest of a CUB_200_2011 folder: per-image annotations of the 112 attributes of Koh et al."""
    for name in ('images.txt', 'image_class_labels.txt', 'train_test_split.txt', 'classes.txt',
                 os.path.join('attributes', 'image_attribute_labels.txt'), 'images'):
        if not os.path.exists(os.path.join(root, name)):
            raise FileNotFoundError(f'{root} is not a CUB_200_2011 folder: {name} is missing')
    images = _read_pairs(os.path.join(root, 'images.txt'))
    class_of = _read_pairs(os.path.join(root, 'image_class_labels.txt'), int)
    is_train = _read_pairs(os.path.join(root, 'train_test_split.txt'), int)
    classes = _read_pairs(os.path.join(root, 'classes.txt'))
    present = defaultdict(dict)                                   # image id -> {attribute id: 0 / 1}
    attribute_ids = set()
    with open(os.path.join(root, 'attributes', 'image_attribute_labels.txt'), encoding='utf-8') as f:
        for line in f.read().splitlines():
            image_id, attribute_id, value = (int(v) for v in line.split()[:3])
            present[image_id][attribute_id] = value
            attribute_ids.add(attribute_id)
    names = _cub_attribute_names(root, sorted(attribute_ids))
    # CUB numbers its attributes from 1; Koh's list is 0-based
    chosen = [i + 1 for i in KOH_ATTRIBUTES if i + 1 in attribute_ids]
    class_ids = sorted(set(class_of.values()))
    label = {c: i for i, c in enumerate(class_ids)}

    # Validation images: a fraction of the training images of each class (classes in order of first appearance).
    by_class = defaultdict(list)
    for image_id in class_of:
        if is_train[image_id] == 1:
            by_class[class_of[image_id]].append(image_id)
    rng = random.Random(seed)
    train_ids, val_ids = [], []
    for ids in by_class.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_fraction))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    test_ids = [i for i in class_of if is_train[i] == 0]

    manifest = Manifest('cub', os.path.join(os.path.abspath(root), 'images'), [_cub_concept_name(names[i]) for i in chosen],
                        [classes[c] for c in class_ids])
    for split, ids in (('train', sorted(train_ids)), ('val', sorted(val_ids)), ('test', sorted(test_ids))):
        manifest.splits[split] = _split_part(ids, [images[i] for i in ids],
                                             [[present[i].get(a, 0) for a in chosen] for i in ids],
                                             [label[class_of[i]] for i in ids])
    return manifest


# --- SUN Attribute ------------------------------------------------------------------------------------------------------

def _sun_annotations(root):
    from scipy.io import loadmat
    db = os.path.join(root, 'SUNAttributeDB')
    images = [str(cell[0][0]) for cell in loadmat(os.path.join(db, 'images.mat'))['images']]
    attributes = [str(cell[0][0]) for cell in loadmat(os.path.join(db, 'attributes.mat'))['attributes']]
    votes = loadmat(os.path.join(db, 'attributeLabels_continuous.mat'))['labels_cv']
    if len(votes) != len(images) or votes.shape[1] != len(attributes):
        raise ValueError('the SUN annotation files are not aligned')
    return images, attributes, votes


def build_sun(root, val_fraction=0.1, test_fraction=0.1, vote_threshold=0.5, seed=0):
    """Manifest of the SUN Attribute database: `root` holds SUNAttributeDB/ (the .mat files) and images/."""
    image_names, attribute_names, votes = _sun_annotations(root)
    image_root = os.path.join(os.path.abspath(root), 'images')
    missing = [n for n in image_names if not os.path.exists(os.path.join(image_root, n))]
    if missing:
        raise FileNotFoundError(f'{len(missing)} SUN images are missing under {image_root}, e.g. {missing[:3]}')
    class_names = sorted({n.rsplit('/', 1)[0] for n in image_names})
    label = {c: i for i, c in enumerate(class_names)}
    by_class = {c: [] for c in class_names}
    for index, name in enumerate(image_names):
        by_class[name.rsplit('/', 1)[0]].append(index)
    rng = random.Random(seed)
    chosen = {s: [] for s in SPLITS}
    for c in class_names:
        indices = sorted(by_class[c])
        rng.shuffle(indices)
        n_test, n_val = int(round(len(indices) * test_fraction)), int(round(len(indices) * val_fraction))
        chosen['test'] += indices[:n_test]
        chosen['val'] += indices[n_test:n_test + n_val]
        chosen['train'] += indices[n_test + n_val:]
    manifest = Manifest('sun', image_root, ['_'.join(n.replace('/', ' ').replace('-', ' ').split()) for n in attribute_names],
                        class_names)
    for split in SPLITS:
        index = sorted(chosen[split])
        manifest.splits[split] = _split_part([i + 1 for i in index], [image_names[i] for i in index],
                                             (votes[index] >= vote_threshold).astype(int),
                                             [label[image_names[i].rsplit('/', 1)[0]] for i in index])
    return manifest


# --- Derm7pt ------------------------------------------------------------------------------------------------------------

def derm_group(diagnosis):
    """The diagnosis group (index into DERM_CLASSES) of a Derm7pt diagnosis, following the authors' grouping."""
    if diagnosis == 'basal cell carcinoma':
        return 0
    if diagnosis in DERM_NEVI:
        return 1
    if diagnosis.startswith('melanoma'):
        return 2
    if diagnosis in DERM_MISC:
        return 3
    if diagnosis == 'seborrheic keratosis':
        return 4
    raise ValueError(f'unknown Derm7pt diagnosis {diagnosis!r}')


def build_derm7pt(root):
    """Manifest of the Derm7pt release: `root` holds meta/ (meta.csv and the index files) and images/. The images
    used are the dermoscopic ones (column `derm`)."""
    import pandas as pd
    meta = pd.read_csv(os.path.join(root, 'meta', 'meta.csv'))
    y = np.array([derm_group(d) for d in meta.diagnosis])
    columns = []
    for name, criterion, values in DERM_CONCEPTS:
        known = set().union(*(v for _, c, v in DERM_CONCEPTS if c == criterion)) | {'absent'}
        if set(meta[criterion]) - known:
            raise ValueError(f'{criterion}: the values {sorted(set(meta[criterion]) - known)} are not in the grouping')
        columns.append(meta[criterion].isin(values).astype(int).to_numpy())
    C = np.stack(columns, axis=1)
    manifest = Manifest('derm7pt', os.path.join(os.path.abspath(root), 'images'), [n for n, _, _ in DERM_CONCEPTS], list(DERM_CLASSES))
    for split, file in DERM_SPLIT_FILES.items():
        rows = pd.read_csv(os.path.join(root, 'meta', file))['indexes'].to_numpy()
        manifest.splits[split] = _split_part(meta.case_num.to_numpy()[rows], meta.derm.to_numpy()[rows], C[rows], y[rows])
    return manifest


BUILDERS = {'cub': build_cub, 'sun': build_sun, 'derm7pt': build_derm7pt}


# --- exports ------------------------------------------------------------------------------------------------------------

def save_split(path, C, y, ids):
    """One npz of an export: the concepts are clipped to [0, 1]."""
    np.savez(path, C=np.clip(np.asarray(C, dtype=np.float32), 0.0, 1.0), y=np.asarray(y, dtype=np.int64),
             ids=np.asarray(ids, dtype=np.int64))


def write_oracle_export(manifest, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, 'concept_names.json'), 'w') as f:
        json.dump(manifest.concept_names, f)
    for split, part in manifest.splits.items():
        save_split(os.path.join(export_dir, f'oracle_{split}.npz'), part['C'], part['y'], part['ids'])
    return export_dir


def has_sources(export_dir, sources):
    return all(os.path.exists(os.path.join(export_dir, f'{source}_{split}.npz')) for source in sources for split in SPLITS)


def load_export(export_dir, sources):
    """Concept names, number of classes and {split: {'y', 'ids', <source>: C}} for sources such as 'oracle' or
    'predicted_seed0'. The rows of every source must be those of the others."""
    with open(os.path.join(export_dir, 'concept_names.json')) as f:
        names = json.load(f)
    splits = {}
    for split in SPLITS:
        entry = {}
        for source in sources:
            path = os.path.join(export_dir, f'{source}_{split}.npz')
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path}: the export has no source {source!r} (see experiments/cbm/GUIDE.md)')
            arrays = np.load(path)
            if 'ids' in entry and not np.array_equal(entry['ids'], arrays['ids']):
                raise ValueError(f'{source}_{split}: the rows are not those of the other sources')
            entry.update({source: arrays['C'].astype(np.float32), 'y': arrays['y'], 'ids': arrays['ids']})
        splits[split] = entry
    return names, int(max(entry['y'].max() for entry in splits.values())) + 1, splits
