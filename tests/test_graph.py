"""FLARE and its baselines on graphs, on a small synthetic graph (the real graphs are downloaded on first use)."""
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from baselines import graph as models
from experiments.data import graphs
from experiments.graphs import run_baselines, run_flare, summarise
from flare import graph
from flare.partitions import quantile_partitions
from flare.labels import trapezoid_constants, trapezoid_memberships


def synthetic_graph(n=240, seed=0, binary=False, pool=False):
    """Nodes with 4 variables; a node is class 1 when most of its neighbours have a high first variable."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((n, n)) < 0.04, 1)
    adjacency = sp.csr_matrix((upper | upper.T).astype(np.float32))
    X = rng.normal(size=(n, 4))
    high = (X[:, 0] > 0.3).astype(float)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    share = np.where(degree > 0, adjacency @ high / np.maximum(degree, 1), 0.0)
    y = (share > 0.4).astype(int)
    if binary:
        X = (X > 0).astype(int)
    order = rng.permutation(n)
    n_train, n_val = int(n * 5 / 12), int(n / 6)
    split = {'train': np.sort(order[:n_train]), 'val': np.sort(order[n_train:n_train + n_val]), 'test': np.sort(order[n_train + n_val:])}
    frame = pd.DataFrame(X, columns=[f'v{j}' for j in range(4)])
    return graphs.GraphData(name='synthetic', X=frame, y=y, categorical=[binary] * 4, adjacency=adjacency, split=split,
                            metric='auprc' if pool else 'accuracy', quantify_categorical='nonzero' if binary else 'all',
                            class_names=['low', 'high'], network_names=['neighbours'], pool_validation=pool,
                            category_names={j: ['absent', 'present'] for j in range(4)} if binary else None)


def test_quantifier_partition_is_a_ruspini_partition():
    partition = graph.quantifier_partition()
    constants = trapezoid_constants(torch.tensor(partition, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    memberships = trapezoid_memberships(torch.linspace(0, 1, 101).unsqueeze(1), constants)
    torch.testing.assert_close(memberships.sum(dim=2), torch.ones(101, 1), atol=1e-6, rtol=0)
    assert partition[0, 2] == pytest.approx(0.2) and partition[2, 0] == pytest.approx(0.6)       # SOME ends at 0.2, MOST starts at 0.8


def test_delta_partitions_reproduce_the_quantile_trapezoids_and_can_be_trained():
    X = np.random.default_rng(1).normal(size=(300, 3)).astype(np.float32)
    partitions = quantile_partitions(X, 3)
    expected = trapezoid_memberships(torch.tensor(X), trapezoid_constants(torch.tensor(partitions, dtype=torch.float32).unsqueeze(0)))
    fixed, learned = graph.DeltaPartitions(partitions, learn=False), graph.DeltaPartitions(partitions, learn=True)
    torch.testing.assert_close(fixed(torch.tensor(X)), expected, atol=2e-6, rtol=0)
    assert not fixed.deltas.requires_grad and list(fixed.parameters()) == []
    learned(torch.tensor(X)).sum().backward()
    assert learned.deltas.grad is not None and learned.deltas.shape == (3, 9)


def test_shares_are_the_mean_truth_of_the_neighbours():
    data = synthetic_graph(60, seed=2)
    features = graph.neighbourhood_features(data.X, data.categorical, data.adjacency, data.split['train'], hops=1,
                                            hop_names=['neighbours'])
    n_own = data.X.shape[1]
    assert features.X.shape == (60, n_own + n_own * 3) and not features.categorical[n_own:].any()
    own = features.X[:, :n_own].astype(np.float64)
    assert abs(own[data.split['train']].mean()) < 1e-6                                          # standardised with the training nodes
    truth = graph.DeltaPartitions(features.own_partitions, learn=False)(torch.tensor(features.X[:, :n_own]))
    truth = truth.flatten(1).numpy()                                                            # [nodes, variable x label]
    degree = np.asarray(data.adjacency.sum(axis=1)).ravel()
    mean = np.where(degree[:, None] > 0, (data.adjacency @ truth) / np.maximum(degree, 1)[:, None], 0.0)
    np.testing.assert_allclose(features.X[:, n_own:], mean, atol=1e-6)
    assert features.names[n_own] == 'neighbours HAVE v0 IS Low' and features.names[n_own + 3] == 'neighbours HAVE v1 IS Low'


def test_two_hops_and_binary_variables():
    data = synthetic_graph(60, seed=3, binary=True)
    features = graph.neighbourhood_features(data.X, data.categorical, data.adjacency, data.split['train'], hops=2,
                                            hop_names=['neighbours', 'partners-of-partners'], category_names=data.category_names,
                                            quantify_categorical='nonzero')
    # one share per binary variable and hop (the category "present"); the own columns stay 0 / 1 and categorical
    assert features.X.shape == (60, 4 + 4 + 4) and features.categorical[:4].all() and not features.categorical[4:].any()
    assert features.cardinalities == [2, 2, 2, 2] and features.own_partitions is None
    assert features.names[4:] == [f'{hop} HAVE v{j} IS present' for hop in ('neighbours', 'partners-of-partners') for j in range(4)]
    one_hop, two_hops = graph.hop_matrices(data.adjacency, 2)
    np.testing.assert_allclose(features.X[:, 4:8], one_hop @ data.X.values, atol=1e-6)
    reach = ((data.adjacency @ data.adjacency) > 0).astype(np.float32).tolil()
    reach.setdiag(0)
    np.testing.assert_allclose(features.X[:, 8:], graph.row_normalise(reach.tocsr()) @ data.X.values, atol=1e-6)


def test_graph_inputs_follow_the_inputs_contract():
    data = synthetic_graph(60, seed=4)
    features = graph.neighbourhood_features(data.X, data.categorical, data.adjacency, data.split['train'], hops=1)
    inputs = features.inputs_factory(True)('cpu')
    n_real = 4 + 12
    assert inputs.n_variables == n_real and inputs.n_conditions == 3
    assert inputs(torch.tensor(features.X)).shape == (60, n_real, 3)
    assert inputs.crossing_parameters() == [] and inputs.space_parameters() == [] and len(list(inputs.parameters())) == 2
    assert inputs.condition_text(0, 1, 'v0') == 'v0 IS Medium' and inputs.condition_text(4, 1, 'neighbours HAVE v0 IS Low') == \
        'MANY neighbours HAVE v0 IS Low'
    assert len(list(features.inputs_factory(False)('cpu').parameters())) == 0


def test_fit_graph_trains_a_union_of_models_that_read_the_neighbourhood():
    data = synthetic_graph()
    features = graph.neighbourhood_features(data.X, data.categorical, data.adjacency, data.split['train'], hops=1,
                                            hop_names=['neighbours'])
    train, val, test = (data.split[k] for k in ('train', 'val', 'test'))
    config = graph.graph_config(rules=12, single_condition_rules=8, epochs=8, bags=2)
    result = graph.fit_graph(features, data.y, train, val, test, config, random_seed=33, device='cpu')
    assert result.route == 'sequential' and result.model.nrules == 24 and len(result.losses) == 8
    probabilities = graph.predict_proba(result)
    assert probabilities.shape == (len(test), 2) and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert (probabilities.argmax(axis=1) == data.y[test]).mean() > 0.6
    # the same seed gives the same model
    again = graph.fit_graph(features, data.y, train, val, test, config, random_seed=33, device='cpu')
    np.testing.assert_array_equal(graph.predict_proba(again), probabilities)
    assert again.losses == result.losses


def test_graph_config_is_flare_with_one_step_size():
    config = graph.graph_config(balanced=True)
    assert (config.labels, config.slots, config.label_runs, config.bags, config.rules) == (3, 2, 'none', 8, 120)
    assert config.lr_structure == config.lr_interactions == 1.0 and config.k_lower_train == 0.1 and config.class_weight == 'balanced'
    assert graph.graph_config().class_weight == 'none'


@pytest.fixture
def toy(monkeypatch):
    data = synthetic_graph(pool=True)
    monkeypatch.setattr(graphs, 'load_graph', lambda name: data)
    return data


def test_pooled_split_keeps_the_test_nodes_and_draws_the_validation_part_with_the_seed(toy):
    train, val, test = graphs.split_indices(toy, 33)
    assert sorted(np.concatenate([train, val])) == sorted(np.concatenate([toy.split['train'], toy.split['val']]))
    np.testing.assert_array_equal(test, toy.split['test'])
    assert len(val) == 28 and graphs.split_indices(toy, 33)[1].tolist() == val.tolist() and graphs.split_indices(toy, 34)[1].tolist() != val.tolist()
    published = synthetic_graph(pool=False)
    assert all(np.array_equal(a, published.split[k]) for a, k in zip(graphs.split_indices(published), ('train', 'val', 'test')))


def test_run_flare_writes_a_row_and_the_rules(toy, tmp_path, capsys):
    frame = run_flare.run_job(toy, 'degree', 6, 33, bags=2, out=str(tmp_path), device='cpu',
                              overrides=dict(rules=12, single_condition_rules=8))
    row = frame.iloc[0]
    assert row['method'] == 'FLARE' and row['variant'] == 'degree_bags2' and row['route'] == 'sequential' and row['variables'] == 5 + 15
    assert {'test_acc', 'test_f1', 'test_bal_acc', 'auprc', 'auroc', 'n_rules', 'n_deciding_rules'} <= set(frame.columns)
    assert 0 <= row['auprc'] <= 1
    rules = (tmp_path / 'rules' / 'FLARE_degree_synthetic_seed33_bags2.txt').read_text()
    assert rules.startswith('# model 1 of 2\nIF ') and ('SOME' in rules or 'MANY' in rules or 'MOST' in rules or 'IS' in rules)
    run_flare.run_job(toy, 'degree', 6, 33, bags=2, out=str(tmp_path), device='cpu')
    assert 'reused' in capsys.readouterr().out


def test_run_baselines_rows_progress_and_summary(toy, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run_baselines.main(['--datasets', 'emogi', '--methods', 'LR', 'LR-nb', 'SGC-2', 'GCN', 'RF', 'CART', '--seeds', '33', '34',
                            '--out', str(tmp_path)])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert '[1/12]' in out and '[12/12]' in out
    rows = pd.concat([pd.read_csv(f) for f in (tmp_path / 'rows').glob('*.csv')])
    assert len(rows) == 12 and 'error' not in rows
    assert {'test_acc', 'test_f1', 'auprc', 'auroc'} <= set(rows.columns) and rows['auprc'].between(0, 1).all()
    assert rows[rows['method'] == 'CART']['complexity'].gt(0).all()
    table = summarise.summarise(str(tmp_path))
    assert list(table.index) == ['CART', 'GCN', 'LR', 'LR-nb', 'RF', 'SGC-2'] and table.loc['LR', 'synthetic'].endswith('(2)')


def test_the_published_protocols_are_for_the_citation_graphs(toy, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_baselines.main(['--datasets', 'emogi', '--methods', 'GCN-standard', '--seeds', '33', '--out', str(tmp_path)])
    assert not list((tmp_path).glob('rows/*'))


@pytest.mark.parametrize('method', ['GCN-standard', 'GAT-standard'])
def test_published_gcn_and_gat_train_on_a_small_graph(method):
    data = synthetic_graph(80)
    data.name = 'cora'
    data.X = data.X.abs()
    train, val, test = (data.split[k] for k in ('train', 'val', 'test'))
    P, epochs = (models.gcn_standard if method == 'GCN-standard' else models.gat_standard)(data, train, val, test, 33)
    assert P.shape == (len(test), 2) and np.allclose(P.sum(axis=1), 1, atol=1e-4) and 11 <= epochs


# --- the data files ------------------------------------------------------------------------------------------------

def write_planetoid(folder, name, n_nodes=30, n_features=6, n_classes=3, n_train=6, n_test=8, seed=0):
    """A miniature set of raw Planetoid files: the first n_train nodes are x, the nodes before the test block are allx
    (labelled and unlabelled) and the last n_test nodes are tx, whose rows follow the shuffled order of test.index."""
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix((rng.random((n_nodes, n_features)) < 0.4).astype(np.float32))
    y = np.eye(n_classes)[rng.integers(0, n_classes, n_nodes)]
    n_all = n_nodes - n_test
    test_index = rng.permutation(np.arange(n_all, n_nodes))
    graph_dict = {i: [int(j) for j in rng.integers(0, n_nodes, 3)] for i in range(n_nodes)}
    files = {'x': X[:n_train], 'y': y[:n_train], 'allx': X[:n_all], 'ally': y[:n_all],
             'tx': X[test_index], 'ty': y[test_index], 'graph': graph_dict}
    for suffix, value in files.items():
        with open(folder / f'ind.{name}.{suffix}', 'wb') as handle:
            pickle.dump(value, handle)
    (folder / f'ind.{name}.test.index').write_text('\n'.join(str(i) for i in test_index))
    return X, y


def test_planetoid_files_are_parsed_into_a_graph_with_the_public_split(tmp_path):
    folder = tmp_path / 'planetoid'
    folder.mkdir()
    # 600 nodes so that the validation block of 500 nodes fits after the 6 training nodes
    X, y = write_planetoid(folder, 'cora', n_nodes=600, n_features=6, n_train=6, n_test=50)
    data = graphs.load_planetoid('cora', str(tmp_path))
    assert data.X.shape == (600, 6) and data.categorical == [True] * 6 and data.quantify_categorical == 'nonzero'
    assert data.n_classes == 3 and (data.adjacency != data.adjacency.T).nnz == 0 and data.adjacency.diagonal().sum() == 0
    assert len(data.split['train']) == 6 and len(data.split['val']) == 500 and len(data.split['test']) == 50
    np.testing.assert_array_equal(data.X.values, X.toarray())                    # the test rows are put back in the order of test.index
    np.testing.assert_array_equal(np.eye(3)[data.y], y)


def test_missing_planetoid_files_are_downloaded_from_the_public_repository(tmp_path, monkeypatch):
    fetched = []

    def fake_download(url, path):
        fetched.append(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
    monkeypatch.setattr(graphs, '_download', fake_download)
    paths = graphs.planetoid_files('cora', str(tmp_path / 'planetoid'))
    assert sorted(fetched) == sorted(f'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.{s}' for s in graphs.PLANETOID_FILES)
    graphs.planetoid_files('cora', str(tmp_path / 'planetoid'))
    assert len(fetched) == len(paths)                                              # files that exist are not fetched again


def test_the_emogi_container_is_extracted_from_the_archive_stream(tmp_path, monkeypatch):
    import io
    import tarfile
    archive = tmp_path / 'EMOGI_model_CPDB.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, content in (('EMOGI_CPDB/cv_1/y_train.npy', b'other'), (graphs.EMOGI_MEMBER, b'the container')):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    monkeypatch.setattr(graphs, 'EMOGI_URL', archive.as_uri())
    path = graphs.emogi_container(str(tmp_path / 'data'))
    assert path == str(tmp_path / 'data' / 'emogi' / 'CPDB_multiomics.h5') and open(path, 'rb').read() == b'the container'
    assert sorted(os.listdir(tmp_path / 'data' / 'emogi')) == ['CPDB_multiomics.h5']          # nothing else is stored
    monkeypatch.setattr(graphs, 'EMOGI_URL', (tmp_path / 'missing.tar.gz').as_uri())
    assert graphs.emogi_container(str(tmp_path / 'data')) == path                              # an existing file is not fetched again
    with tarfile.open(tmp_path / 'empty.tar.gz', 'w:gz'):
        pass
    monkeypatch.setattr(graphs, 'EMOGI_URL', (tmp_path / 'empty.tar.gz').as_uri())
    with pytest.raises(FileNotFoundError, match='CPDB_multiomics.h5'):
        graphs.emogi_container(str(tmp_path / 'other'))
    assert not os.path.exists(tmp_path / 'other' / 'emogi' / 'CPDB_multiomics.h5.part')
