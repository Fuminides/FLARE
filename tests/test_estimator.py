import numpy as np
import pandas as pd

from flare import FLAREClassifier
from helpers import small_config


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'age': rng.normal(size=n), 'size': rng.normal(size=n), 'kind': rng.integers(0, 3, size=n).astype(float)})
    y = np.where(X['age'] + (X['kind'] == 1) + 0.3 * rng.normal(size=n) > 0.5, 'old', 'young')
    X.loc[::17, 'size'] = np.nan
    return X, y


def test_fit_predict_and_rules():
    X, y = toy()
    clf = FLAREClassifier(small_config(epochs=20, bags=2), seed=3, device='cpu').fit(X, y, categorical=[False, False, True])
    assert list(clf.classes_) == ['old', 'young']
    pred, proba = clf.predict(X), clf.predict_proba(X)
    assert set(pred) <= {'old', 'young'} and proba.shape == (200, 2) and np.allclose(proba.sum(axis=1), 1)
    assert (pred == y).mean() > 0.7
    text = clf.print_rules(class_names=list(clf.classes_))
    assert text.count('IF ') == len(np.unique(clf.rules_[0], axis=0)) or text.count('IF ') > 0
    assert 'age IS' in text or 'size IS' in text or 'kind IS' in text
    rules, points, consequents = clf.rules_
    assert rules.shape == (16, 3) and points.shape == (16, 2)                 # two models of 8 rules


def test_missing_values_in_new_rows_are_filled_from_the_training_rows():
    X, y = toy()
    clf = FLAREClassifier(small_config(epochs=3), seed=1, device='cpu').fit(X, y, categorical=[False, False, True])
    filled = X.copy()
    filled['size'] = filled['size'].fillna(X['size'].median())
    np.testing.assert_allclose(clf.decision_function(X), clf.decision_function(filled))


def test_a_given_validation_set_is_used_and_calibration_gives_probabilities():
    X, y = toy()
    clf = FLAREClassifier(small_config(epochs=5, calibrate=True), seed=1, device='cpu')
    clf.fit(X.iloc[:150], y[:150], categorical=[False, False, True], X_val=X.iloc[150:], y_val=y[150:])
    assert len(clf.fold_.y_val) == 50 and clf.calibration_[0] >= 1.0
    assert clf.predict_proba(X.iloc[150:]).shape == (50, 2)


def test_the_estimator_follows_the_scikit_learn_protocol():
    from sklearn.base import clone
    clf = FLAREClassifier(small_config(), seed=5)
    assert clone(clf).get_params()['seed'] == 5
