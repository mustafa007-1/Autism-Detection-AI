"""Loading, validating and cleaning the screening dataset, and building model inputs.

Training and the app both go through `build_features` so they always produce the same
columns with the same encoding.
"""
from pathlib import Path

import pandas as pd

SCORE_COLS = [f"a{i}_score" for i in range(1, 11)]
BINARY_COLS = ["gender", "jaundice", "autism", "used_app_before"]  # autism = family member with ASD
FEATURES = SCORE_COLS + ["age"] + BINARY_COLS + ["ethnicity"]
CATEGORICAL_COLS = ["ethnicity"]
TARGET = "asd_status"
GROUP = "dataset_source"

_BINARY_MAPS = {
    "gender": {"m": 1, "f": 0},
    "jaundice": {"yes": 1, "no": 0},
    "autism": {"yes": 1, "no": 0},
    "used_app_before": {"yes": 1, "no": 0},
}


def clean_ethnicity(values: pd.Series) -> pd.Series:
    # The raw data has 'Middle Eastern ' (trailing space) and both 'Others' and 'others'.
    return values.astype(str).str.strip().str.title()


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the CSV and return FEATURES + TARGET + GROUP, cleaned and validated."""
    df = pd.read_csv(path)

    missing = set(FEATURES + [TARGET, GROUP]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[FEATURES + [TARGET, GROUP]].copy()
    for col, mapping in _BINARY_MAPS.items():
        raw = df[col].astype(str).str.strip().str.lower()
        unknown = set(raw) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected values in {col!r}: {sorted(unknown)}")
        df[col] = raw.map(mapping)
    df["ethnicity"] = clean_ethnicity(df["ethnicity"])
    df["age"] = pd.to_numeric(df["age"], errors="raise")

    for col in SCORE_COLS + [TARGET]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col!r} must contain only 0/1")
    if df[FEATURES + [TARGET, GROUP]].isna().any().any():
        raise ValueError("Dataset contains missing values")

    return df.reset_index(drop=True)


def build_features(scores: list[int], age: float, male: bool, jaundice: bool,
                   family_asd: bool, used_app_before: bool, ethnicity: str) -> pd.DataFrame:
    """One model-input row from app answers, encoded exactly like `load_dataset`."""
    if len(scores) != len(SCORE_COLS):
        raise ValueError(f"Expected {len(SCORE_COLS)} item scores, got {len(scores)}")
    row = dict(zip(SCORE_COLS, scores))
    row.update(age=float(age), gender=int(male), jaundice=int(jaundice),
               autism=int(family_asd), used_app_before=int(used_app_before))
    row["ethnicity"] = clean_ethnicity(pd.Series([ethnicity])).iloc[0]
    return pd.DataFrame([row], columns=FEATURES)
