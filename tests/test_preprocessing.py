import numpy as np
import pandas as pd

from flare.preprocessing import category_effect, preprocess_fold, preprocess_split, transform_rows


def test_binary_scores_are_the_shrunk_class_one_rate():
    codes = np.array([0] * 10 + [1] * 10 + [2] * 2)
    y = np.array([1] * 9 + [0] + [0] * 10 + [1, 1])
    scores, unseen = category_effect(codes, y, 2, m=10.0)
    prior = y.mean()
    assert scores[0] == (9 + 10 * prior) / 20 and scores[1] == (0 + 10 * prior) / 20
    assert unseen == prior
    assert scores[1] < scores[0] and abs(scores[2] - prior) < abs(1.0 - prior)       # a rare category shrinks to the prior


def test_multiclass_order_follows_the_main_axis_of_the_class_proportions():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(6), 200)                    # categories 0..5 move class mass from class 0 to class 2 in steps
    probs = [np.array([1 - t, 0.0, t]) * 0.8 + 0.2 / 3 for t in np.linspace(0, 1, 6)]
    y = np.concatenate([rng.choice(3, 200, p=p / p.sum()) for p in probs])
    scores, _ = category_effect(codes, y, 3)
    s = np.array([scores[c] for c in range(6)])
    assert np.all(np.diff(s) > 0) or np.all(np.diff(s) < 0)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'c': rng.integers(0, 60, n).astype(float), 'x': rng.normal(size=n), 'z': rng.normal(size=n)})
    return X, pd.Series((X['c'] < 30).astype(int))


def test_category_space_ranks_categories_by_training_frequency_only():
    X, y = _frame()
    tr, te = np.arange(300), np.arange(300, 400)
    X_te = X.iloc[te].copy()
    X_te.iloc[0, 0] = 999.0                                                       # a category training never saw
    fold = preprocess_fold(X.iloc[tr], y.iloc[tr], X_te, y.iloc[te], [True, False, False], seed=1, category_space=32)
    info = fold.category_space[0]
    K = len(info['counts'])
    assert list(fold.categorical) == [False] * 3 and np.all(np.diff(info['counts']) <= 0)
    assert fold.X_test[0, 0] == K and fold.X_train[:, 0].max() < K


def test_scaler_and_effects_are_fitted_on_training_rows_only():
    X, y = _frame()
    tr, te = np.arange(300), np.arange(300, 400)
    flipped = y.copy()
    flipped.iloc[te] = 1 - flipped.iloc[te]                 # test labels must not move the training encoding
    a = preprocess_fold(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], [True, False, False], seed=1, category_space=32)
    b = preprocess_fold(X.iloc[tr], y.iloc[tr], X.iloc[te], flipped.iloc[te], [True, False, False], seed=1, category_space=32)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.X_test, b.X_test)
    assert abs(a.X_train[:, 1].mean()) < 1e-12 and abs(a.X_train[:, 1].std() - 1) < 1e-2


def test_preprocess_split_equals_preprocess_fold_on_the_same_split():
    from sklearn.model_selection import train_test_split
    X, y = _frame()
    tr, te = np.arange(300), np.arange(300, 400)
    fold = preprocess_fold(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], [True, False, False], seed=5, category_space=32)
    X_tr, X_va, y_tr, y_va = train_test_split(X.iloc[tr], y.iloc[tr], test_size=0.2, random_state=5, stratify=y.iloc[tr])
    split = preprocess_split(X_tr.values, y_tr, X_va.values, y_va, X.iloc[te].values, y.iloc[te], [True, False, False], category_space=32)
    for name in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'categorical'):
        np.testing.assert_array_equal(getattr(fold, name), getattr(split, name))
    assert fold.n_classes == split.n_classes
    np.testing.assert_array_equal(fold.category_space[0]['counts'], split.category_space[0]['counts'])
    np.testing.assert_allclose(fold.category_space[0]['effects'], split.category_space[0]['effects'])


def test_transform_rows_reproduces_the_encoding_of_the_fitted_rows():
    X, y = _frame()
    tr, te = np.arange(300), np.arange(300, 400)
    original = np.array([True, False, False])
    fold = preprocess_fold(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], original, seed=5, category_space=32)
    np.testing.assert_allclose(transform_rows(fold, X.iloc[te].values, original), fold.X_test)
    unseen = X.iloc[te].values.copy()
    unseen[0, 0] = 777.0
    assert transform_rows(fold, unseen, original)[0, 0] == len(fold.category_space[0]['codes'])
