"""Age-group ensemble: one soft-voting ensemble per AQ-10 version, rows routed by age.

Adults and adolescents answer different AQ-10 questionnaires, so a1..a10 mean different
things in each group. Training one model per group keeps those meanings separate; the
router combines them into a single scikit-learn estimator that can be cross-validated,
pickled and called from the app like any other classifier.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .aq10 import ADOLESCENT, ADOLESCENT_MIN_AGE, ADULT, ADULT_MIN_AGE, GROUPS
from .data import CATEGORICAL_COLS, FEATURES, SCORE_COLS


def aq10_total(scores) -> np.ndarray:
    """Sum of the 10 item scores, the questionnaire's own summary, as one column."""
    return np.asarray(scores, dtype=float).sum(axis=1, keepdims=True)


def _aq10_total_name(transformer, input_features) -> list[str]:
    return ["aq10_total"]


def make_group_model(random_state: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("items", "passthrough", SCORE_COLS),
            # Without the total, the models must rebuild it from the items and end up
            # under-confident for people right at the cutoff.
            ("total", FunctionTransformer(aq10_total, feature_names_out=_aq10_total_name), SCORE_COLS),
            ("ethnicity", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("age", StandardScaler(), ["age"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ensemble = VotingClassifier(
        estimators=[
            ("logreg", LogisticRegression(max_iter=2000)),
            ("forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=random_state)),
            ("boosting", GradientBoostingClassifier(random_state=random_state)),
        ],
        voting="soft",
    )
    return Pipeline([("preprocess", preprocess), ("ensemble", ensemble)])


def route_by_age(ages) -> np.ndarray:
    ages = np.asarray(ages, dtype=float)
    if (ages < ADOLESCENT_MIN_AGE).any():
        raise ValueError(f"Ages below {ADOLESCENT_MIN_AGE} are outside the training data")
    return np.where(ages < ADULT_MIN_AGE, ADOLESCENT, ADULT)


class AgeGroupEnsemble(ClassifierMixin, BaseEstimator):
    """Fits a separate ensemble for adults and adolescents and routes each row by its age."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = self._validate_frame(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError(f"Expected binary 0/1 labels, got {self.classes_}")

        groups = route_by_age(X["age"])
        self.models_ = {}
        for group in GROUPS:
            mask = groups == group
            if len(np.unique(y[mask])) < 2:
                raise ValueError(f"{group} training rows need both classes")
            self.models_[group] = make_group_model(self.random_state).fit(X.loc[mask], y[mask])

        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "models_")
        X = self._validate_frame(X)
        groups = route_by_age(X["age"])
        proba = np.zeros((len(X), len(self.classes_)))
        for group, model in self.models_.items():
            mask = groups == group
            if mask.any():
                proba[mask] = model.predict_proba(X.loc[mask])
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def known_ethnicities(self) -> list[str]:
        check_is_fitted(self, "models_")
        categories = set()
        for model in self.models_.values():
            encoder = model.named_steps["preprocess"].named_transformers_["ethnicity"]
            categories.update(encoder.categories_[0])
        return sorted(categories)

    @staticmethod
    def _validate_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AgeGroupEnsemble expects a pandas DataFrame with named columns")
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        return X[FEATURES].reset_index(drop=True)
