"""The experiment scripts on a small synthetic dataset (the real benchmarks need OpenML)."""
import numpy as np
import pandas as pd
import pytest

from experiments.tabular import run_flare, tune_flare


def toy_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n), 'c': rng.integers(0, 3, size=n).astype(float)})
    y = pd.Series(((X['a'] + (X['c'] == 1) + 0.3 * rng.normal(size=n)) > 0.4).astype(int), name='class')
    names = {'classes': ['no', 'yes'], 'categories': {'c': ['red', 'green', 'blue']}}
    from experiments.data import openml as data
    return X, y, np.array([False, False, True]), names, lambda seed: data.cc18_folds(y, seed, n_folds=3)


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(run_flare.data, 'load', lambda benchmark, name: toy_dataset())


SMALL = dict(rules=8, single_condition_rules=5, slots=2, labels=5, bags=1, parallel_bags=False, epochs=3, cuda_graph=False,
             steps_per_epoch=0)


def test_run_flare_writes_a_row_and_the_rules_of_every_fold(toy, tmp_path, capsys):
    frame = run_flare.run_one('cc18', 'toy', 'flare', 33, str(tmp_path), overrides=SMALL)
    assert len(frame) == 3 and list(frame['fold']) == [0, 1, 2]
    assert {'dataset', 'seed', 'fold', 'config', 'test_acc', 'f1', 'test_bal_acc', 'test_roc_auc', 'test_logloss', 'n_rules', 'n_ants',
            'time2train', 'unique_ants'} <= set(frame.columns)
    assert 'test_logloss_cal' not in frame                                        # recalibration is for TabArena
    out = capsys.readouterr().out
    assert '[1/3]' in out and '[3/3]' in out
    rules = (tmp_path / 'rules' / 'FLARE_toy_seed33' / 'rules_fold1.txt').read_text()
    assert rules.startswith('IF ') and 'THEN Class' in rules
    again = run_flare.run_one('cc18', 'toy', 'flare', 33, str(tmp_path), overrides=SMALL)          # a complete file is reused
    assert 'reused' in capsys.readouterr().out and len(again) == 3


def test_run_flare_one_fold_and_the_explainability_configuration(toy, tmp_path):
    frame = run_flare.run_one('cc18', 'toy', 'flare-explainability', 33, str(tmp_path), only_fold=2,
                              overrides=dict(epochs=2, steps_per_epoch=0, cuda_graph=False))
    assert len(frame) == 1 and frame['fold'][0] == 2 and frame['method'][0] == 'FLARE-explainability'
    assert (tmp_path / 'rows' / 'FLARE-explainability_toy_seed33_fold2.csv').exists()


def test_tuned_flare_writes_the_row_with_its_options(toy, tmp_path):
    pytest.importorskip('optuna')
    frame = tune_flare.run_one('cc18', 'toy', 33, str(tmp_path), trials=2, only_fold=0,
                               overrides=dict(epochs=2, steps_per_epoch=0, cuda_graph=False))
    row = frame.iloc[0]
    assert row['n_trials'] == 2 and 0 <= row['val_f1'] <= 1 and row['method'] == 'FLARE_tuned'
    params = __import__('json').loads(row['params'])
    assert {'lr', 'temp', 'vl', 'rules', 'single_condition_rules', 'lr_structure', 'lr_interactions', 'crossings_lr', 'class_weight',
            'point_bags'} == set(params)
    assert (tmp_path / 'trials').exists()
