"""Train the age-group ensemble, evaluate it honestly and save it for app.py / graphs.py.

Run:  python train_model.py
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from asd.aq10 import GROUPS, REFERRAL_CUTOFF
from asd.config import CV_FOLDS, DATA_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from asd.data import FEATURES, GROUP, SCORE_COLS, TARGET, load_dataset
from asd.model import AgeGroupEnsemble, route_by_age


def evaluate(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    print("Loading data...")
    df = load_dataset(DATA_PATH)
    routed = route_by_age(df["age"])
    if not (routed == df[GROUP].to_numpy()).all():
        raise ValueError("Age-based routing disagrees with dataset_source; check the age boundaries")

    X, y = df[FEATURES], df[TARGET].to_numpy()
    # Stratify on group x label so both splits keep the adult/adolescent and ASD/no-ASD mix.
    strata = (df[GROUP] + "_" + df[TARGET].astype(str)).to_numpy()
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                                           stratify=strata, random_state=RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Cross-validating on {len(train_idx)} training rows ({CV_FOLDS} folds)...")
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_validate(AgeGroupEnsemble(RANDOM_STATE), X_train, y_train,
                        cv=list(folds.split(X_train, strata[train_idx])),
                        scoring=["accuracy", "balanced_accuracy", "f1", "roc_auc"])
    cv_summary = {m: {"mean": float(cv[f"test_{m}"].mean()), "std": float(cv[f"test_{m}"].std())}
                  for m in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]}

    print("Training final model...")
    model = AgeGroupEnsemble(RANDOM_STATE).fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    test_groups = df[GROUP].to_numpy()[test_idx]
    # The labels are defined by the AQ-10 cutoff, so this rule is the baseline to compare against.
    rule_pred = (X_test[SCORE_COLS].sum(axis=1).to_numpy() > REFERRAL_CUTOFF).astype(int)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "cross_validation": cv_summary,
        "test": evaluate(y_test, y_pred, y_prob),
        "test_by_group": {g: evaluate(y_test[test_groups == g], y_pred[test_groups == g],
                                      y_prob[test_groups == g]) for g in GROUPS},
        "aq10_rule_baseline_test_accuracy": accuracy_score(y_test, rule_pred),
        "label_equals_aq10_rule_on_all_rows": bool(
            ((df[SCORE_COLS].sum(axis=1) > REFERRAL_CUTOFF).astype(int) == df[TARGET]).all()),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "test_index": test_idx.tolist(), "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))

    print_report(metrics)
    print(f"Saved model to {MODEL_PATH}\nSaved metrics to {METRICS_PATH}")


def print_report(m: dict):
    line = "=" * 60
    print(f"\n{line}\nMODEL EVALUATION\n{line}")
    print(f"Train rows: {m['n_train']}   Test rows: {m['n_test']}")
    print("\nCross-validation on training set (mean +/- std):")
    for name, s in m["cross_validation"].items():
        print(f"  {name:<18} {s['mean']:.3f} +/- {s['std']:.3f}")

    columns = ["n", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = {"All": m["test"], **m["test_by_group"]}
    table = pd.DataFrame([{"group": k, **{c: v[c] for c in columns}} for k, v in rows.items()])
    print("\nHeld-out test set:")
    print(table.set_index("group").to_string(float_format="{:.3f}".format))
    cm = m["test"]["confusion_matrix"]
    print(f"\nConfusion matrix (all): TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")

    print(f"\nAQ-10 rule (score > {REFERRAL_CUTOFF}) test accuracy: {m['aq10_rule_baseline_test_accuracy']:.3f}")
    if m["label_equals_aq10_rule_on_all_rows"]:
        print("NOTE: every label in this dataset equals the AQ-10 cutoff rule. The model can only\n"
              "      learn that rule, so high accuracy does not show it detects autism beyond the\n"
              "      questionnaire. Clinically diagnosed labels are needed for that.")
    print(line)


if __name__ == "__main__":
    main()
