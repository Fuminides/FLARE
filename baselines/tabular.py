'''The baseline methods, their search spaces and their complexity (number of rules).

`build(name, params, seed, n_classes)` returns an sklearn-style estimator (fit / predict) on the encoded fold matrix:
the real columns standardised as FLARE sees them and the categorical columns one-hot encoded (experiments/tabular/
run_baselines.py). `DEFAULTS[name]` is the package's own configuration ({} = its defaults); `space(name, trial)` draws
one configuration for Optuna. CLASSY and GRANDE run in environments of their own (baselines/workers), so GRANDE has a
search space here but no builder; CLASSY has both, imported only inside its environment.

Groups: gradient-boosted trees (XGBoost, LightGBM, CatBoost); discrete rule learners (RIPPER, CORELS, CLASSY);
differentiable rule learners (RRL, RL-Net, NeuRules, DR-Net); differentiable trees (GradTree, GRANDE); interpretable
models (FIGS, RuleFit, CART); references (RF with 300 trees, EBM with 14 outer bags and 10 interaction pairs, logistic
regression).
'''
import numpy as np

GBDT = ['XGBoost', 'LightGBM', 'CatBoost']
DISCRETE_RULES = ['RIPPER', 'CORELS', 'CLASSY']
DIFF_RULES = ['RRL', 'RL-Net', 'NeuRules', 'DR-Net']
DIFF_TREES = ['GradTree', 'GRANDE']
METHODS = GBDT + DISCRETE_RULES + DIFF_RULES + DIFF_TREES
# TabArena: its published results already hold the GBDTs, EBM and the black boxes; these are the rule learners and
# interpretable trees it lacks.
INTERPRETABLE = ['FIGS', 'RuleFit', 'CART']
TABARENA_METHODS = DISCRETE_RULES + DIFF_RULES + INTERPRETABLE + DIFF_TREES
REFERENCES = ['RF', 'EBM', 'LR']       # random forest (300 trees), EBM (14 outer bags, 10 pairs), logistic regression
DEFAULTS = {name: {} for name in METHODS + INTERPRETABLE + REFERENCES}
DEFAULTS.update(RF={'balanced': True}, EBM={'balanced': True})     # the references use balanced class weights


def space(name, trial):
    '''One configuration of `name` drawn by an Optuna trial. Ranges follow the methods' papers and the usual tabular
    benchmarks (Grinsztajn et al. 2022, GRANDE); class weighting is a choice wherever the method supports it.'''
    s = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    f, i, c = s
    if name == 'XGBoost':
        return dict(n_estimators=i('n_estimators', 100, 1000, log=True), max_depth=i('max_depth', 2, 10),
                    learning_rate=f('learning_rate', 1e-3, 0.3, log=True), subsample=f('subsample', 0.5, 1.0),
                    colsample_bytree=f('colsample_bytree', 0.5, 1.0), min_child_weight=f('min_child_weight', 0.1, 10, log=True),
                    reg_lambda=f('reg_lambda', 1e-3, 10, log=True), balanced=c('balanced', [False, True]))
    if name == 'LightGBM':
        return dict(n_estimators=i('n_estimators', 100, 1000, log=True), num_leaves=i('num_leaves', 8, 128, log=True),
                    learning_rate=f('learning_rate', 1e-3, 0.3, log=True), subsample=f('subsample', 0.5, 1.0),
                    colsample_bytree=f('colsample_bytree', 0.5, 1.0), min_child_samples=i('min_child_samples', 5, 100, log=True),
                    reg_lambda=f('reg_lambda', 1e-3, 10, log=True), balanced=c('balanced', [False, True]))
    if name == 'CatBoost':
        return dict(iterations=i('iterations', 100, 1000, log=True), depth=i('depth', 3, 8),
                    learning_rate=f('learning_rate', 1e-3, 0.3, log=True), l2_leaf_reg=f('l2_leaf_reg', 1, 10, log=True),
                    balanced=c('balanced', [False, True]))
    if name == 'RIPPER':
        return dict(k=i('k', 1, 3), prune_size=f('prune_size', 0.2, 0.5), dl_allowance=i('dl_allowance', 16, 128, log=True),
                    n_discretize_bins=i('n_discretize_bins', 5, 15))
    if name == 'CORELS':
        return dict(c=f('c', 1e-4, 0.05, log=True), max_card=i('max_card', 1, 3), bins=i('bins', 3, 9),
                    min_support=f('min_support', 0.005, 0.1, log=True))
    if name == 'CLASSY':
        return dict(max_depth=i('max_depth', 2, 5), beam_width=c('beam_width', [25, 50, 100]),
                    n_cutpoints=i('n_cutpoints', 3, 9), max_rules=c('max_rules', [10, 20, 40]))
    if name == 'RRL':
        return dict(hidden_width=c('hidden_width', [32, 64, 128, 256]), learning_rate=f('learning_rate', 5e-4, 2e-2, log=True),
                    weight_decay=f('weight_decay', 1e-6, 1e-2, log=True), binarization_nodes=i('binarization_nodes', 3, 10))
    if name == 'RL-Net':
        return dict(n_rules=i('n_rules', 10, 50, log=True), n_bins=i('n_bins', 5, 15), learning_rate=f('learning_rate', 1e-3, 5e-2, log=True),
                    lambda_and=f('lambda_and', 1e-5, 1e-2, log=True), l2_lambda=f('l2_lambda', 1e-6, 1e-2, log=True))
    if name == 'NeuRules':
        return dict(n_rules=i('n_rules', 5, 30), learning_rate=f('learning_rate', 1e-3, 5e-2, log=True),
                    min_support=f('min_support', 0.05, 0.3), coverage_weight=f('coverage_weight', 0.1, 1.0))
    if name == 'DR-Net':
        return dict(rules=i('rules', 20, 100, log=True), lr=f('lr', 1e-3, 3e-2, log=True), and_lam=f('and_lam', 1e-4, 1e-1, log=True),
                    bins=i('bins', 5, 15))
    if name == 'GradTree':
        return dict(depth=i('depth', 3, 8), lr_index=f('lr_index', 1e-3, 0.1, log=True), lr_values=f('lr_values', 1e-3, 0.1, log=True),
                    lr_leaf=f('lr_leaf', 1e-3, 0.1, log=True))
    if name == 'FIGS':
        return dict(max_rules=i('max_rules', 5, 50, log=True))
    if name == 'RuleFit':
        return dict(max_rules=i('max_rules', 10, 100, log=True))
    if name == 'CART':                                      # depth up to 32: the default tree is not limited
        return dict(max_depth=i('max_depth', 2, 32, log=True), min_samples_leaf=i('min_samples_leaf', 1, 50, log=True),
                    balanced=c('balanced', [False, True]))
    if name == 'RF':
        return dict(max_features=c('max_features', ['sqrt', 'log2', 0.25, 0.5, 0.75, 1.0]), min_samples_leaf=i('min_samples_leaf', 1, 50, log=True),
                    max_depth=i('max_depth', 2, 32, log=True), balanced=c('balanced', [False, True]))
    if name == 'EBM':                                       # pairs stay a bounded count: interpret's '3x' does not finish
        return dict(interactions=i('interactions', 0, 20), learning_rate=f('learning_rate', 5e-3, 5e-2, log=True),
                    max_leaves=i('max_leaves', 2, 4), min_samples_leaf=i('min_samples_leaf', 2, 20, log=True),
                    smoothing_rounds=c('smoothing_rounds', [0, 75, 200, 500]), balanced=c('balanced', [False, True]))
    if name == 'LR':
        return dict(C=f('C', 1e-3, 1e3, log=True), balanced=c('balanced', [False, True]))
    if name == 'GRANDE':
        return dict(depth=i('depth', 3, 7), n_estimators=c('n_estimators', [256, 512, 1024]),
                    learning_rate_weights=f('learning_rate_weights', 1e-4, 0.25, log=True),
                    learning_rate_index=f('learning_rate_index', 1e-3, 0.2, log=True),
                    learning_rate_values=f('learning_rate_values', 1e-3, 0.2, log=True),
                    learning_rate_leaf=f('learning_rate_leaf', 1e-3, 0.2, log=True), dropout=c('dropout', [0.0, 0.25, 0.5]),
                    selected_variables=f('selected_variables', 0.5, 1.0), use_class_weights=c('use_class_weights', [False, True]))
    raise KeyError(name)


def _weights(y, balanced):
    if not balanced:
        return None
    classes, counts = np.unique(y, return_counts=True)
    w = {k: len(y) / (len(classes) * n) for k, n in zip(classes, counts)}
    return np.array([w[v] for v in y])


class _Weighted:
    '''Passes class-balanced sample weights to fit when the configuration asks for them.'''

    def __init__(self, model, balanced):
        self.model, self.balanced = model, balanced

    def fit(self, X, y):
        w = _weights(y, self.balanced)
        self.model.fit(X, y) if w is None else self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).reshape(-1)

    def predict_proba(self, X):
        return _proba(self.model, X)


class _LabelEncoded:
    '''Fits on 0..K-1 labels (XGBoost needs them) and maps predictions back.'''

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.model.fit(X, yi)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.model.predict(X)).reshape(-1).astype(int)]

    def predict_proba(self, X):
        return _proba(self.model, X)


class _OneVsRest:
    '''Binary-only rule learners (RIPPER, CORELS) on any number of classes: one learner per class; a row goes to the
    class whose learner fires (`score`), ties and no-fires to the most frequent of those classes in training.'''

    def __init__(self, make):
        self.make = make

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.prior_ = counts / counts.sum()
        targets = self.classes_[1:] if len(self.classes_) == 2 else self.classes_
        self.models_ = [self.make().fit(X, (y == c).astype(int)) for c in targets]
        return self

    def predict(self, X):
        S = np.column_stack([m.score(X) for m in self.models_])
        if len(self.classes_) == 2:
            return self.classes_[(S[:, 0] > 0.5).astype(int)]
        S = S + 1e-6 * self.prior_                                # tie-break towards the frequent class
        return self.classes_[S.argmax(axis=1)]


def _frame(X):
    import pandas as pd
    return pd.DataFrame(X, columns=[f'x{j}' for j in range(X.shape[1])])


class _Ripper:
    def __init__(self, params, seed):
        self.params, self.seed = params, seed

    def fit(self, X, y):
        import pandas as pd
        import wittgenstein as lw
        self.model = lw.RIPPER(random_state=self.seed, **self.params)
        # wittgenstein reads the class column's name from the labels: an unnamed Series fails with KeyError: None
        self.model.fit(_frame(X), pd.Series(y, name='target'), pos_class=1)
        return self

    def score(self, X):
        return np.asarray(self.model.predict(_frame(X)), dtype=float)

    def n_rules(self):
        return len(self.model.ruleset_)


class _Corels:
    '''CORELS on quantile-thresholded columns ("x >= q" for `bins` - 1 quantiles of each column), fitted on the
    training rows.'''

    def __init__(self, params, seed):
        self.params, self.seed = dict(params), seed

    def fit(self, X, y):
        from corels import CorelsClassifier
        bins = self.params.pop('bins', 5)
        self.cuts_ = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1])) for j in range(X.shape[1])]
        self.model = CorelsClassifier(n_iter=100000, verbosity=[], **self.params)
        self.model.fit(self._binary(X), y)
        self.params['bins'] = bins
        return self

    def _binary(self, X):
        return np.column_stack([X[:, j] >= t for j, cuts in enumerate(self.cuts_) for t in cuts]).astype(np.uint8)

    def score(self, X):
        return np.asarray(self.model.predict(self._binary(X)), dtype=float)

    def n_rules(self):
        return len(self.model.rl().rules) - 1


class _Classy:
    def __init__(self, params, seed):
        self.params, self.seed = params, seed

    def fit(self, X, y):
        import pandas as pd
        from rulelist import RuleList
        if not hasattr(pd.DataFrame, 'iteritems'):        # rulelist 0.2.0 predates pandas 2, which renamed it items
            pd.DataFrame.iteritems = pd.DataFrame.items
        self.model = RuleList(task='prediction', target_model='categorical', **self.params)
        self.model.fit(_frame(X), pd.Series(y.astype(str), name='target'))
        return self

    def predict(self, X):
        pred = self.model.predict(_frame(X))
        return np.asarray(pred).reshape(-1).astype(float).astype(int)


class _Adapter:
    '''Estimators whose predict returns the labels they were fitted on.'''

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).reshape(-1)

    def predict_proba(self, X):
        return _proba(self.model, X)


class _NumpySeededAdapter(_Adapter):
    '''Scope NumPy's legacy RNG to one fit, then restore the caller's state.

    This is needed for estimators which accept a seed but do not forward it to
    every stochastic sub-estimator.  Scoping it to the fit leaves the RNG stream
    seen by the other methods unchanged.
    '''

    def __init__(self, model, seed):
        super().__init__(model)
        self.seed = seed

    def fit(self, X, y):
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            return super().fit(X, y)
        finally:
            np.random.set_state(state)


def _proba(model, X):
    '''The wrapped model's class probabilities [N, K] in its classes_ order, or None when it has none.'''
    if not hasattr(model, 'predict_proba'):
        return None
    try:
        P = model.predict_proba(X)
    except Exception:
        return None
    return None if P is None else np.asarray(P, dtype=float)


def probabilities(model, X, n_classes):
    '''[N, n_classes] probabilities of labels 0..n_classes-1 from a fitted model of this module, and whether they are
    the model's own (False: one-hot of its predictions, for methods without probabilities).'''
    P = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    if P is not None and P.ndim == 2:
        inner = model
        while hasattr(inner, 'model') and not hasattr(inner, 'classes_'):
            inner = inner.model
        classes = np.asarray(getattr(inner, 'classes_', np.arange(P.shape[1]))).astype(int)
        if P.shape[1] == len(classes) and classes.max() < n_classes:
            out = np.zeros((len(P), n_classes))
            out[:, classes] = P
            return out, True
    pred = np.asarray(model.predict(X)).astype(int)
    return np.eye(n_classes)[pred], False


def build(name, params, seed, n_classes):
    p = dict(params)
    balanced = p.pop('balanced', False)
    if name == 'XGBoost':
        from xgboost import XGBClassifier
        return _LabelEncoded(_Weighted(XGBClassifier(random_state=seed, n_jobs=1, tree_method='hist', verbosity=0, **p), balanced))
    if name == 'LightGBM':
        from lightgbm import LGBMClassifier
        if 'subsample' in p:
            p['subsample_freq'] = 1
        return _Weighted(LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, class_weight='balanced' if balanced else None, **p), False)
    if name == 'CatBoost':
        from catboost import CatBoostClassifier
        return _Adapter(CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False,
                                           auto_class_weights='Balanced' if balanced else None, **p))
    if name == 'RIPPER':
        return _OneVsRest(lambda: _Ripper(p, seed))
    if name == 'CORELS':
        return _OneVsRest(lambda: _Corels(p, seed))
    if name == 'CLASSY':
        return _Classy(p, seed)
    if name == 'RRL':
        from baselines.rrl import RRLExternal
        return _Adapter(RRLExternal(random_state=seed, **p))
    if name == 'RL-Net':
        from baselines.rlnet import RLNetExternal
        return _Adapter(RLNetExternal(random_state=seed, **p))
    if name == 'NeuRules':
        from baselines.neurules import NeuRules
        return _Adapter(NeuRules(random_state=seed, **p))
    if name == 'DR-Net':
        from baselines.drnet import DRNetModel
        return _Adapter(DRNetModel(seed, **p))
    if name == 'GradTree':
        from baselines.gradtree import GradTreeClassifier
        return _Adapter(GradTreeClassifier(random_state=seed, **p))
    if name == 'FIGS':
        from imodels import FIGSClassifier
        # imodels stores random_state but does not pass it to the regression
        # stumps used during fitting; those stumps draw from NumPy's global RNG.
        return _NumpySeededAdapter(FIGSClassifier(random_state=seed, **{'max_rules': 30, **p}), seed)
    if name == 'RuleFit':
        from imodels import RuleFitClassifier
        from sklearn.multiclass import OneVsRestClassifier      # imodels RuleFit is binary-only
        return _Adapter(OneVsRestClassifier(RuleFitClassifier(random_state=seed, **{'max_rules': 30, **p})))
    if name == 'CART':
        from sklearn.tree import DecisionTreeClassifier
        return _Adapter(DecisionTreeClassifier(random_state=seed, class_weight='balanced' if balanced else None, **p))
    if name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return _Adapter(RandomForestClassifier(300, random_state=seed, n_jobs=1, class_weight='balanced' if balanced else None, **p))
    if name == 'EBM':
        from interpret.glassbox import ExplainableBoostingClassifier
        return _Weighted(ExplainableBoostingClassifier(random_state=seed, n_jobs=1, **{'interactions': 10, **p}), balanced)
    if name == 'LR':
        from sklearn.linear_model import LogisticRegression
        return _Adapter(LogisticRegression(max_iter=5000, class_weight='balanced' if balanced else None, **p))
    raise KeyError(name)


def n_rules(name, model):
    '''Rules of the fitted rule learners (RIPPER, CORELS, RuleFit: summed over the one-vs-rest learners; CART: leaves),
    else NaN.'''
    try:
        if name in ('RIPPER', 'CORELS'):
            return float(sum(m.n_rules() for m in model.models_))
        if name == 'CLASSY':                                # rulelist 0.2.0: rules of the list, without the final ELSE
            return float(model.model.number_rules)
        if name == 'CART':
            return float(model.model.get_n_leaves())
        if name == 'RuleFit':
            return float(np.nansum([getattr(sub, 'complexity_', np.nan) for sub in model.model.estimators_]))
        if name in ('FIGS', 'NeuRules'):
            return float(model.model.complexity_)
        # the differentiable learners: rules with at least one literal
        if name == 'RRL':                                   # nodes of the first logical layer (conjunctions and disjunctions)
            union = model.model.net_.layer_list[1]
            return float(sum((layer.W.detach().cpu().numpy() > 0.5).any(axis=0).sum() for layer in (union.con_layer, union.dis_layer)))
        if name == 'RL-Net':
            W = model.model.model_.and_layer.masked_weight().detach().cpu().numpy()
            return float((np.abs(W) > 1e-6).any(axis=1).sum())
        if name == 'DR-Net':                                # rules the OR layers use, summed over the per-class networks
            return float(sum(len(positive) + len(negative) > 0 for positive, negative, _ in model.model.rules()))
        if name == 'GradTree':                              # leaves
            return float(2 ** model.model.depth)
    except Exception:
        pass
    return float('nan')
