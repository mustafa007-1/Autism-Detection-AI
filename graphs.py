"""Plot evaluation graphs for the saved model on its held-out test set.

Run after train_model.py:  python graphs.py   (images are written to graphs/)
"""
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from asd.aq10 import GROUPS, REFERRAL_CUTOFF
from asd.config import DATA_PATH, GRAPHS_DIR, MODEL_PATH, RANDOM_STATE
from asd.data import FEATURES, GROUP, SCORE_COLS, TARGET, load_dataset

LABELS = ["No ASD traits", "ASD traits"]
# The dataset's 'autism' column means "immediate family member with autism".
DISPLAY_NAMES = {"autism": "family_member_with_asd"}


def save(fig, name: str):
    path = GRAPHS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Created {path}")


def plot_confusion_matrices(y, pred, groups):
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(11, 4.5))
    for ax, g in zip(axes, GROUPS):
        mask = groups == g
        ConfusionMatrixDisplay.from_predictions(y[mask], pred[mask], labels=[0, 1], display_labels=LABELS,
                                                cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{g} (n={mask.sum()})")
    fig.suptitle("Confusion matrix on held-out test data")
    save(fig, "confusion_matrix.png")


def plot_roc(y, prob, groups):
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y, prob, name="All", ax=ax)
    for g in GROUPS:
        mask = groups == g
        RocCurveDisplay.from_predictions(y[mask], prob[mask], name=g, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC curve on held-out test data")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")


def plot_feature_importance(model, X, y):
    # Permutation importance works for any model, including the voting ensemble.
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=30,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    names = [DISPLAY_NAMES.get(f, f) for f in FEATURES]
    imp = pd.DataFrame({"feature": names, "mean": result.importances_mean, "std": result.importances_std})
    imp = imp.sort_values("mean").tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"], imp["mean"], xerr=imp["std"], color="#3b75af")
    ax.set_xlabel("Drop in ROC AUC when the feature is shuffled")
    ax.set_title("Permutation feature importance (test data)")
    save(fig, "feature_importance.png")


def plot_score_distribution(df):
    totals = df[SCORE_COLS].sum(axis=1)
    counts = pd.crosstab(totals, df[TARGET]).reindex(index=range(11), columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts[0], label=LABELS[0], color="#7fb3d5")
    ax.bar(counts.index, counts[1], bottom=counts[0], label=LABELS[1], color="#c0392b")
    ax.axvline(REFERRAL_CUTOFF + 0.5, color="black", linestyle="--", label=f"Cutoff (score > {REFERRAL_CUTOFF})")
    ax.set_xticks(range(11))
    ax.set_ylim(0, counts.sum(axis=1).max() * 1.15)  # headroom so the tallest bar isn't clipped
    ax.set_xlabel("AQ-10 total score")
    ax.set_ylabel("People")
    ax.set_title("Labels follow the AQ-10 cutoff exactly (all data)")
    ax.legend()
    save(fig, "aq10_score_distribution.png")


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: {MODEL_PATH} not found. Run 'python train_model.py' first.")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    df = load_dataset(DATA_PATH)
    test = df.iloc[bundle["test_index"]]
    X, y = test[FEATURES], test[TARGET].to_numpy()
    groups = test[GROUP].to_numpy()
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    GRAPHS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrices(y, pred, groups)
    plot_roc(y, prob, groups)
    plot_feature_importance(model, X, y)
    plot_score_distribution(df)
    print("All graphs saved.")


if __name__ == "__main__":
    main()
