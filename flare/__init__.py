"""FLARE: Fuzzy Learned-Antecedent Rule Evidence, a differentiable rule learner.

    from flare import FLAREClassifier, flare, flare_explainability
    model = FLAREClassifier(flare()).fit(X, y, categorical=mask)
    print(model.print_rules())
"""
from .config import FLAREConfig, flare, flare_explainability  # noqa: F401
from .estimator import FLAREClassifier  # noqa: F401
