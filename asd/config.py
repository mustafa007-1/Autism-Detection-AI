"""Paths and constants shared by training, plotting and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "autism_final_processed.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "asd_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
GRAPHS_DIR = ROOT / "graphs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
