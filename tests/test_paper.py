"""The table scripts on synthetic rows (no network, no training)."""
import numpy as np
import pandas as pd

from experiments.data import openml as data
from experiments.paper import common, table1


def synthetic_rows(tmp_path, methods, datasets, score):
    rows = [dict(dataset=d, collection='cc18', seed=s, fold=f, method=m, test_f1=score(m, d), fit_time=1.0)
            for m in methods for d in datasets for s in data.SEEDS for f in range(data.N_FOLDS)]
    (tmp_path / 'rows').mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / 'rows' / 'x.csv', index=False)
    return common.read_rows([str(tmp_path)])


def test_rank_matrix_gives_unfinished_entries_the_worst_rank():
    scores = pd.DataFrame({'a': [0.9, 0.5], 'b': [0.8, np.nan], 'c': [0.8, 0.7]})
    r = common.rank_matrix(scores, ascending=False)
    assert r.loc[0].tolist() == [1.0, 2.5, 2.5] and r.loc[1].tolist() == [2.0, 3.0, 1.0]


def test_complete_mean_drops_a_dataset_with_a_missing_fold():
    expected = {'d1': {(33, 0), (33, 1)}, 'd2': {(33, 0), (33, 1)}}
    rows = pd.DataFrame(dict(method='m', dataset=['d1', 'd1', 'd2'], seed=33, fold=[0, 1, 0], x=[1.0, 3.0, 5.0]))
    per = common.complete_mean(rows, 'x', expected)
    assert per.loc['d1', 'm'] == 2.0 and np.isnan(per.loc['d2', 'm'])


def test_table1_cc18_columns(tmp_path):
    names = data.cc18_names()[:3]
    methods = ['FLARE_tuned', 'FLARE-explainability', 'CART_tuned', 'RF_tuned']
    value = {'FLARE_tuned': 0.8, 'FLARE-explainability': 0.7, 'CART_tuned': 0.6, 'RF_tuned': 0.81}
    rows = synthetic_rows(tmp_path, methods, names, lambda m, d: value[m])
    cc, std = table1.cc18_table(rows)
    assert cc.loc[names[0], 'FLARE'] == 80.0 and cc.loc[names[0], 'Random forest'] == 81.0
    assert cc['FIGS'].isna().all()                              # not run: no score, the worst rank
    ranks = common.rank_matrix(cc.loc[names], ascending=False).mean()       # the 3 datasets that were run
    assert ranks['Random forest'] == 1.0 and ranks['FLARE'] == 2.0
    paired = table1.paired(cc.dropna(axis=1, how='all').reindex(columns=table1.METHODS), False, 0.5)
    assert paired.set_index('method').loc['CART', 'W'] == 3
