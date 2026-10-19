from asd.aq10 import REFERRAL_CUTOFF
from asd.config import DATA_PATH
from asd.data import BINARY_COLS, FEATURES, GROUP, SCORE_COLS, TARGET, build_features, load_dataset
from asd.model import route_by_age


def test_load_dataset_cleans_values():
    df = load_dataset(DATA_PATH)
    assert len(df) == 808
    assert list(df.columns) == FEATURES + [TARGET, GROUP]
    for col in BINARY_COLS:
        assert set(df[col]) <= {0, 1}
    assert "others" not in set(df["ethnicity"])
    assert (df["ethnicity"] == df["ethnicity"].str.strip()).all()


def test_age_routing_matches_dataset_source():
    df = load_dataset(DATA_PATH)
    assert (route_by_age(df["age"]) == df[GROUP].to_numpy()).all()


def test_labels_are_the_aq10_cutoff():
    # Documents why the model cannot be more than an AQ-10 reproduction on this data.
    df = load_dataset(DATA_PATH)
    assert ((df[SCORE_COLS].sum(axis=1) > REFERRAL_CUTOFF).astype(int) == df[TARGET]).all()


def test_build_features_matches_training_encoding():
    df = load_dataset(DATA_PATH)
    first = df.iloc[0]
    row = build_features([int(first[c]) for c in SCORE_COLS], first["age"], male=bool(first["gender"]),
                         jaundice=bool(first["jaundice"]), family_asd=bool(first["autism"]),
                         used_app_before=bool(first["used_app_before"]), ethnicity=first["ethnicity"])
    assert list(row.columns) == FEATURES
    assert row.iloc[0].tolist() == first[FEATURES].tolist()
    assert row.dtypes.equals(df[FEATURES].dtypes)
