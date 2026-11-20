"""The four graphs of the node-classification experiments, downloaded on first use.

* **cora, citeseer, pubmed**: the Planetoid citation networks (Yang et al., 2016) with their public split: 20 labelled
  nodes per class for training (140, 120 and 60 nodes), the next 500 nodes for validation and the 1,000 test nodes of
  `test.index`. The raw files `ind.<name>.*` come from github.com/kimiyoung/planetoid and are stored in
  `$FLARE_DATA_DIR/planetoid` (default `~/.cache/flare/planetoid`); they are Python pickles, so only that repository's
  URLs are fetched. Cora (2,708 nodes x 1,433 words) and Citeseer (3,327 x 3,703) hold 0 / 1 word indicators, served
  as binary categorical variables, so a condition reads "word_12 IS present" and only the category "present" is
  counted among the neighbours. PubMed (19,717 x 500) holds TF-IDF weights, which stay real. Citeseer has 15 test
  nodes that are missing from the raw files (inserted as empty rows, the usual fix) and 48 isolated nodes.
  Metric: accuracy on the test nodes.
* **emogi**: cancer driver genes (Schulte-Sasse et al., 2021, Nature Machine Intelligence): 13,627 genes with 64 omics
  variables, the symmetrised CPDB protein-interaction network and the authors' fixed split over their 2,983 labelled
  genes (796 drivers). The authors' container `CPDB_multiomics.h5` comes from the archive `EMOGI_model_CPDB.tar.gz`
  (https://owww.molgen.mpg.de/~sasse/EMOGI/), of which only that file is extracted (1.5 GB) into
  `$FLARE_DATA_DIR/emogi`. Their baselines fit on training plus validation genes, so a run pools those and draws its own
  validation part (`split_indices`). Metric: area under the precision-recall curve on the 746 test genes.

Every loader returns a `GraphData`.
"""
import os
import pickle
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get('FLARE_DATA_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'flare'))
NAMES = ['emogi', 'cora', 'citeseer', 'pubmed']
PLANETOID = ['cora', 'citeseer', 'pubmed']
PLANETOID_URL = 'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/'
PLANETOID_FILES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
EMOGI_URL = 'https://owww.molgen.mpg.de/~sasse/EMOGI/EMOGI_model_CPDB.tar.gz'
EMOGI_MEMBER = 'EMOGI_CPDB/CPDB_multiomics.h5'


@dataclass
class GraphData:
    """X: the variables of every node. y: the class of a labelled node, -1 for an unlabelled one. categorical: one flag per
    column. adjacency: SciPy sparse matrix, symmetric, binary, without self loops. split: the fixed train / validation /
    test node positions. quantify_categorical: which categories of a categorical column get a neighbourhood share, 'all'
    or 'nonzero' (binary indicators, where "absent" is the complement of "present")."""
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical: list
    adjacency: object
    split: dict
    metric: str = 'accuracy'
    quantify_categorical: str = 'all'
    class_names: list = None
    category_names: dict = None
    network_names: list = field(default_factory=lambda: ['partners'])
    # True: the published baselines fit on train + validation, so a run pools them and draws its own validation part.
    # False: the published split already has a validation set and it is used as published.
    pool_validation: bool = False

    @property
    def n_classes(self):
        return int(self.y.max()) + 1


def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.part'
    print(f'[download] {url}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(temporary, 'wb') as out:
        shutil.copyfileobj(response, out)
    os.replace(temporary, path)


def planetoid_files(name, folder):
    """Paths of the raw files of a Planetoid graph; the missing ones are downloaded."""
    paths = {}
    for suffix in PLANETOID_FILES:
        paths[suffix] = os.path.join(folder, f'ind.{name}.{suffix}')
        if not os.path.exists(paths[suffix]):
            _download(PLANETOID_URL + f'ind.{name}.{suffix}', paths[suffix])
    return paths


def load_planetoid(name, data_dir=None):
    """The Planetoid version of a citation network with its public split."""
    import scipy.sparse as sp
    paths = planetoid_files(name, os.path.join(data_dir or DATA_DIR, 'planetoid'))
    parts = {}
    for suffix in PLANETOID_FILES[:-1]:
        with open(paths[suffix], 'rb') as handle:
            parts[suffix] = pickle.load(handle, encoding='latin1')
    with open(paths['test.index']) as handle:
        test_order = np.array([int(line.strip()) for line in handle])
    test_sorted = np.sort(test_order)

    tx, ty = parts['tx'], parts['ty']
    if name == 'citeseer':
        # Citeseer's test block has isolated nodes that are missing from tx / ty: insert empty rows for them.
        full = np.arange(test_sorted.min(), test_sorted.max() + 1)
        tx_full = sp.lil_matrix((len(full), parts['x'].shape[1]), dtype=np.float32)
        tx_full[test_sorted - test_sorted.min(), :] = tx
        ty_full = np.zeros((len(full), parts['y'].shape[1]))
        ty_full[test_sorted - test_sorted.min(), :] = ty
        tx, ty = tx_full.tocsr(), ty_full

    features = sp.vstack((parts['allx'], tx)).tolil()
    features[test_order, :] = features[test_sorted, :]
    labels = np.vstack((parts['ally'], ty))
    labels[test_order, :] = labels[test_sorted, :]
    n_nodes = features.shape[0]

    rows, cols = [], []
    for source, targets in parts['graph'].items():
        rows.extend([source] * len(targets))
        cols.extend(targets)
    adjacency = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    y = np.full(n_nodes, -1, dtype=int)
    labelled = labels.sum(axis=1) > 0
    y[labelled] = labels[labelled].argmax(axis=1)
    n_train = parts['y'].shape[0]
    split = {'train': np.arange(n_train), 'val': np.arange(n_train, n_train + 500), 'test': test_sorted}

    values = np.asarray(features.todense(), dtype=float)
    # 0 / 1 word indicators (Cora, Citeseer) are binary categorical variables; TF-IDF weights (PubMed) stay real.
    binary = bool(np.isin(values, (0.0, 1.0)).all())
    X = pd.DataFrame(values.astype(int) if binary else values, columns=[f'word_{j}' for j in range(values.shape[1])])
    return GraphData(name=name, X=X, y=y, categorical=[binary] * X.shape[1], adjacency=adjacency, split=split, metric='accuracy',
                     quantify_categorical='nonzero' if binary else 'all', class_names=[f'class_{c}' for c in range(labels.shape[1])],
                     network_names=['neighbours'],
                     category_names={j: ['absent', 'present'] for j in range(X.shape[1])} if binary else None)


def emogi_container(data_dir=None):
    """Path of the authors' container CPDB_multiomics.h5; downloaded and extracted from their archive when it is missing.
    Set FLARE_DATA_DIR to a folder that already holds emogi/CPDB_multiomics.h5 to skip the download (22 MB, 1.5 GB once
    extracted)."""
    path = os.path.join(data_dir or DATA_DIR, 'emogi', 'CPDB_multiomics.h5')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'[download] {EMOGI_URL}', flush=True)
        try:
            with urllib.request.urlopen(EMOGI_URL, timeout=120) as response, tarfile.open(fileobj=response, mode='r|gz') as archive:
                for member in archive:                   # a stream: the container is copied as it passes, nothing else is stored
                    if member.name == EMOGI_MEMBER:
                        with archive.extractfile(member) as source, open(path + '.part', 'wb') as out:
                            shutil.copyfileobj(source, out)
                        os.replace(path + '.part', path)
                        break
                else:
                    raise FileNotFoundError(f'{EMOGI_MEMBER} is not in {EMOGI_URL}; put CPDB_multiomics.h5 in {os.path.dirname(path)}')
        finally:
            if os.path.exists(path + '.part'):
                os.remove(path + '.part')
    return path


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_emogi(data_dir=None, block=1024):
    """The EMOGI driver-gene graph. y is 0 / 1 for the 2,983 labelled genes and -1 for the others; the dense
    [13,627, 13,627] network of the container is read in blocks of rows, so it is never held in memory at once."""
    import h5py
    import scipy.sparse as sp
    with h5py.File(emogi_container(data_dir), 'r') as f:
        features = _decode(f['feature_names'][:])
        genes = _decode(f['gene_names'][:, 1])
        X = pd.DataFrame(f['features'][:], index=genes, columns=[c.replace(': ', '_') for c in features])
        n = X.shape[0]
        rows, cols = [], []
        for start in range(0, n, block):
            chunk = np.asarray(f['network'][start:start + block], dtype=np.float32)
            r, c = np.nonzero(chunk)
            rows.append(r + start)
            cols.append(c)
        masks = {k: f[f'mask_{k}'][:].astype(bool).ravel() for k in ('train', 'val', 'test')}
        y = np.full(n, -1, dtype=int)
        for k in ('train', 'val', 'test'):
            y[masks[k]] = f[f'y_{k}'][:].ravel()[masks[k]].astype(int)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adjacency = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    split = {k: np.flatnonzero(mask) for k, mask in masks.items()}
    return GraphData(name='emogi', X=X.reset_index(drop=True), y=y, categorical=[False] * X.shape[1], adjacency=adjacency, split=split,
                     metric='auprc', class_names=['not a driver', 'driver'], network_names=['partners'], pool_validation=True)


def load_graph(name, data_dir=None):
    if name == 'emogi':
        return load_emogi(data_dir)
    if name in PLANETOID:
        return load_planetoid(name, data_dir)
    raise KeyError(f'unknown graph {name!r}; known: {NAMES}')


def with_degree_column(data):
    """A copy of `data` with log10(1 + degree) appended as a real column, so that a rule can read the degree of a node
    beside the share of its neighbours with a property."""
    import dataclasses
    degree = np.asarray(data.adjacency.sum(axis=1)).ravel()
    X = data.X.copy()
    X['log_degree'] = np.log10(degree + 1.0)
    return dataclasses.replace(data, X=X, categorical=list(data.categorical) + [False])


def split_indices(data, seed=33, val_proportion=0.2):
    """(training, validation, test) node positions. A graph whose public split has a validation set (Planetoid) is used as
    published; one whose baselines fit on training plus validation nodes (EMOGI) is pooled, and a stratified validation
    part is drawn from the pool with the seed, so the test nodes are still touched only once."""
    train, val, test = (np.asarray(data.split[k]) for k in ('train', 'val', 'test'))
    if data.pool_validation:
        from sklearn.model_selection import train_test_split
        pooled = np.concatenate([train, val])
        train, val = train_test_split(pooled, test_size=val_proportion, random_state=seed, stratify=data.y[pooled])
    return np.asarray(train), np.asarray(val), np.asarray(test)
