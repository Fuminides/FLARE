"""Reruns of reference runs (slow).

The reference runs are fold 0 of each configuration, trained on the CPU with seed 33 (the directory is given by the
environment variable FLARE_REFERENCE_DIR; each job <name> has <name>/<dataset>/<run>/results.csv, rules_fold0.txt and
losses_fold0.csv). The tests rerun the same case with this package and compare: same accuracy, F1, number of rules and
printed rules, the same loss at every epoch, probabilities to 1e-6.

An exact match assumes the settings of the reference runs: CPU, two threads (tests/conftest.py), the same PyTorch major
version. A case is skipped if the reference or its dataset is not available.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

REFERENCE = os.environ.get('FLARE_REFERENCE_DIR')

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not REFERENCE, reason='FLARE_REFERENCE_DIR is not set')]

# job: (benchmark, dataset, configuration, Gumbel, overrides of the configuration, options of fit_fold, route)
CASES = {
    'expl_diabetes': ('cc18', 'diabetes', 'flare-explainability', False, {}, {}, 'single'),
    'flare_diabetes': ('cc18', 'diabetes', 'flare', False, {}, {}, 'staged'),
    'flare_diabetes_seq': ('cc18', 'diabetes', 'flare', False, {}, {'max_parallel_elements': 0}, 'sequential'),
    'flare_tictactoe': ('cc18', 'tic-tac-toe', 'flare', False, {}, {}, 'parallel'),
    'flare_creditg': ('cc18', 'credit-g', 'flare', False, {}, {}, 'staged'),
    'expl_diabetes_conc': ('cc18', 'diabetes', 'flare-explainability', True, {}, {}, 'single'),
    'flare_diabetes_conc': ('cc18', 'diabetes', 'flare', True, {}, {}, 'staged'),
    # The threshold is lowered so that the category spaces and the recalibration are exercised (anneal has categoricals
    # of up to 8 categories).
    'flare_tabarena': ('tabarena', 'anneal', 'flare', False, {'category_space_min': 5}, {}, 'staged'),
    'expl_tabarena': ('tabarena', 'anneal', 'flare-explainability', False, {'category_space_min': 5}, {}, 'single'),
}
EXACT = ['test_acc', 'f1', 'n_rules', 'unique_ants', 'n_deciding_rules']
CLOSE = ['test_bal_acc', 'test_roc_auc', 'test_logloss', 'train_acc', 'train_f1', 'n_ants', 'n_deciding_ants', 'n_deciding_overlap',
         'test_roc_auc_cal', 'test_logloss_cal', 'cal_temperature']


def reference_files(job):
    runs = glob.glob(os.path.join(REFERENCE, job, '*', '*', 'results.csv'))
    if not runs:
        pytest.skip(f'no reference run {job}')
    folder = os.path.dirname(runs[0])
    return (pd.read_csv(runs[0], sep=';', float_precision='round_trip').iloc[0], open(os.path.join(folder, 'rules_fold0.txt')).read().rstrip('\n'),
            pd.read_csv(os.path.join(folder, 'losses_fold0.csv'), sep=';', float_precision='round_trip'))


@pytest.mark.parametrize('job', list(CASES))
def test_rerun_matches_reference(job):
    from experiments.data import openml as data
    from experiments.tabular.run_flare import make_config, run_fold
    reference, rules, losses = reference_files(job)
    benchmark, dataset, config_name, gumbel, overrides, options, route = CASES[job]
    config = make_config(config_name, benchmark, gumbel, overrides)
    try:
        X, y, categorical, names, folds = data.load(benchmark, dataset)
    except Exception as error:                           # noqa: BLE001 - no network and no cache
        pytest.skip(f'dataset {benchmark}:{dataset} not available ({type(error).__name__})')
    train_idx, test_idx = folds(33)[0]
    row, text, result = run_fold(X, y, categorical, names, train_idx, test_idx, config, 33, 0, device='cpu', **options)

    assert result.route == route
    assert len(result.losses) == len(losses)
    np.testing.assert_allclose(result.losses, losses['train_loss'], rtol=0, atol=1e-9, err_msg='training loss per epoch')
    np.testing.assert_allclose(result.val_accs, losses['val_acc'], rtol=0, atol=1e-12, err_msg='validation accuracy per epoch')
    for key in EXACT:
        assert row[key] == pytest.approx(reference[key], abs=0, rel=0), key
    for key in CLOSE:
        if key in reference and not pd.isna(reference[key]):
            assert row[key] == pytest.approx(reference[key], abs=1e-6), key
    if route == 'sequential':        # the reference lists only the first of the models trained one after another
        text = text.split('\n# model 2 of ')[0].split('\n', 1)[1]
    assert text == rules, 'printed rules differ'


def test_tuned_flare_matches_reference():
    """Tuned FLARE on diabetes, seed 33, fold 0, with three trials: the same proposals of the TPE sampler, the same holdout
    scores, the same chosen options and the same refit. Needs Optuna."""
    pytest.importorskip('optuna')
    from experiments.data import openml as data
    from experiments.tabular.run_flare import make_config
    from experiments.tabular.tune_flare import tune_fold
    rows = glob.glob(os.path.join(REFERENCE, 'tuned', 'rows', 'FLARE_tuned_diabetes_seed33_fold0.csv'))
    if not rows:
        pytest.skip('no reference run tuned')
    reference = pd.read_csv(rows[0], float_precision='round_trip').iloc[0]
    trials = pd.read_csv(os.path.join(REFERENCE, 'tuned', 'trials', 'FLARE_tuned_diabetes_seed33_fold0.csv'), float_precision='round_trip')
    try:
        X, y, categorical, names, folds = data.load('cc18', 'diabetes')
    except Exception as error:                           # noqa: BLE001
        pytest.skip(f'dataset not available ({type(error).__name__})')
    train_idx, test_idx = folds(33)[0]
    row, table = tune_fold(X, y, categorical, names, train_idx, test_idx, 33, 0, make_config('flare', 'cc18'), trials=3, timeout=3600)
    np.testing.assert_array_equal(table['value'], trials['value'])                       # the holdout F1 of every trial
    for name in [c for c in trials if c.startswith('params_')]:
        np.testing.assert_array_equal(table[name], trials[name], err_msg=name)           # what the sampler proposed
    assert row['params'] == reference['params']
    assert row['val_f1'] == reference['val_f1'] and row['n_trials'] == reference['n_trials'] == 3
    assert row['f1'] == reference['f1'] and row['test_acc'] == reference['test_acc'] and row['n_rules'] == reference['n_rules']


# Graph node classification (flare/graph.py) against a reference run, seed 33, with few epochs and averaged models to keep
# it to minutes (4 epochs and 3 models on Cora; 40 epochs and 2 models on EMOGI). Folder <job>: case.json (graph, variant,
# seed, epochs, bags), row.json (scores and rule counts of the deployed model), losses.csv (loss and validation accuracy of
# the first model per epoch), predictions.csv (test probabilities, ';'-separated) and rules.npz (rules of the union).
GRAPH_CASES = ['graph_cora_hops2', 'graph_emogi_degree']


@pytest.mark.parametrize('job', GRAPH_CASES)
def test_graph_rerun_matches_reference(job):
    from baselines.graph import graph_scores
    from experiments.data import graphs
    from experiments.graphs.run_flare import prepare
    from flare.graph import fit_graph, graph_config, predict_proba
    from flare.train import score_fold
    folder = os.path.join(REFERENCE, job)
    if not os.path.exists(os.path.join(folder, 'case.json')):
        pytest.skip(f'no reference run {job}')
    case = json.load(open(os.path.join(folder, 'case.json')))
    reference = json.load(open(os.path.join(folder, 'row.json')))
    losses = pd.read_csv(os.path.join(folder, 'losses.csv'), sep=';', float_precision='round_trip')
    if case['dataset'] == 'emogi' and not os.path.exists(os.path.join(graphs.DATA_DIR, 'emogi', 'CPDB_multiomics.h5')):
        pytest.skip('the EMOGI container is not in the data directory (it would be downloaded and extracted: 1.5 GB)')
    try:
        data = graphs.load_graph(case['dataset'])
    except Exception as error:                           # noqa: BLE001 - no network and no cache
        pytest.skip(f"graph {case['dataset']} not available ({type(error).__name__})")
    features, y, train, val, test = prepare(data, case['variant'], case['seed'])
    config = graph_config(balanced=data.metric == 'auprc', epochs=case['epochs'], bags=case['bags'])
    result = fit_graph(features, y, train, val, test, config, case['seed'], device='cpu')
    row, P = score_fold(result), predict_proba(result)
    scores = graph_scores(data.metric, y[test], P)

    assert result.route == 'sequential'
    np.testing.assert_allclose(result.losses, losses['train_loss'], rtol=0, atol=1e-9, err_msg='training loss per epoch')
    np.testing.assert_allclose(result.val_accs, losses['val_acc'], rtol=0, atol=1e-12, err_msg='validation accuracy per epoch')
    assert scores['test_acc'] == reference['accuracy'] == row['test_acc']
    assert scores['test_f1'] == pytest.approx(reference['macro_f1'], abs=1e-12)
    assert row['f1'] == pytest.approx(reference['f1_score'], abs=1e-12)
    assert (row['n_rules'], row['n_deciding_rules']) == (reference['rules'], reference['deciding_rules'])
    assert row['n_deciding_ants'] == pytest.approx(reference['antecedents'], abs=1e-12)
    if data.metric == 'auprc':
        assert scores['auprc'] == pytest.approx(reference['auprc'], abs=1e-9)
    predictions = pd.read_csv(os.path.join(folder, 'predictions.csv'), sep=';').sort_values('node')
    np.testing.assert_array_equal(predictions['node'].to_numpy(), np.sort(test))
    np.testing.assert_allclose(P[np.argsort(test)], predictions[[c for c in predictions if c.startswith('p')]].to_numpy(), rtol=0, atol=1e-6)
    rules, weights, consequents = result.model.extract_rules()
    expected = np.load(os.path.join(folder, 'rules.npz'))
    np.testing.assert_array_equal(rules, expected['rules'])
    np.testing.assert_array_equal(consequents, expected['consequents'])
    np.testing.assert_allclose(weights, expected['weights'], rtol=0, atol=1e-6)
