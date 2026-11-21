"""python -m flare fit / predict on a small CSV."""
import numpy as np
import pandas as pd

from flare.__main__ import main


def test_fit_and_predict_from_csv_files(tmp_path, capsys):
    rng = np.random.default_rng(0)
    n = 240
    data = pd.DataFrame({'age': rng.normal(40, 10, n).round(), 'colour': rng.choice(['red', 'green', 'blue'], n)})
    data['label'] = np.where((data.age > 40) | (data.colour == 'red'), 'yes', 'no')
    data.loc[::23, 'age'] = np.nan
    data.iloc[:200].to_csv(tmp_path / 'train.csv', index=False)
    data.iloc[200:].to_csv(tmp_path / 'test.csv', index=False)
    model, rules = tmp_path / 'model.pt', tmp_path / 'rules.txt'
    main(['fit', str(tmp_path / 'train.csv'), '--target', 'label', '--test', str(tmp_path / 'test.csv'), '--model', str(model),
          '--rules', str(rules), '--device', 'cpu', '--set', 'epochs=5', 'bags=2', 'rules=8', 'single_condition_rules=6'])
    out = capsys.readouterr().out
    assert 'test accuracy' in out and model.exists() and 'IF ' in rules.read_text()
    main(['predict', str(model), str(tmp_path / 'test.csv'), '--target', 'label', '--proba', '--out', str(tmp_path / 'p.csv')])
    predictions = pd.read_csv(tmp_path / 'p.csv')
    assert len(predictions) == 40 and set(predictions.prediction) <= {'yes', 'no'}
    np.testing.assert_allclose(predictions[['p_no', 'p_yes']].sum(axis=1), 1, atol=1e-5)
