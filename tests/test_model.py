import numpy as np
import pytest
from sklearn.base import clone

from asd.aq10 import ADOLESCENT, ADULT
from asd.config import DATA_PATH
from asd.data import FEATURES, GROUP, TARGET, load_dataset
from asd.model import AgeGroupEnsemble


@pytest.fixture(scope="module")
def data():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="module")
def fitted(data):
    return AgeGroupEnsemble(random_state=0).fit(data[FEATURES], data[TARGET])


def test_predict_proba_shape_and_range(fitted, data):
    proba = fitted.predict_proba(data[FEATURES])
    assert proba.shape == (len(data), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(fitted.predict(data[FEATURES])) <= {0, 1}


@pytest.mark.parametrize("group", [ADULT, ADOLESCENT])
def test_rows_are_routed_to_their_group_model(fitted, data, group):
    rows = data.loc[data[GROUP] == group, FEATURES]
    expected = fitted.models_[group].predict_proba(rows)
    assert np.allclose(fitted.predict_proba(rows), expected)


def test_routing_handles_mixed_and_shuffled_rows(fitted, data):
    shuffled = data[FEATURES].sample(frac=1, random_state=1)
    proba = fitted.predict_proba(shuffled)
    for group in (ADULT, ADOLESCENT):
        mask = (data.loc[shuffled.index, GROUP] == group).to_numpy()
        assert np.allclose(proba[mask], fitted.models_[group].predict_proba(shuffled[mask]))


def test_column_order_does_not_matter(fitted, data):
    X = data[FEATURES]
    assert np.allclose(fitted.predict_proba(X[FEATURES[::-1]]), fitted.predict_proba(X))


def test_rejects_missing_columns(fitted, data):
    with pytest.raises(ValueError):
        fitted.predict_proba(data[FEATURES].drop(columns="age"))


def test_known_ethnicities(fitted, data):
    assert fitted.known_ethnicities() == sorted(data["ethnicity"].unique())


def test_is_cloneable():
    assert clone(AgeGroupEnsemble(random_state=7)).get_params() == {"random_state": 7}
