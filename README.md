# Autism Trait Screening (AQ-10)

A Streamlit app that screens for autism traits with the AQ-10 questionnaire, backed by a
machine-learning ensemble trained on 808 AQ-10 records (704 adults, 104 adolescents aged 12–16).

> This is a screening aid, not a diagnostic tool.

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python train_model.py      # trains, evaluates, writes models/asd_model.joblib + models/metrics.json
python graphs.py           # writes evaluation plots to graphs/
streamlit run app.py       # opens the screening app
pytest                     # runs the test suite
```

## How it works

| Path | Purpose |
|---|---|
| `asd/aq10.py` | The AQ-10 Adult and Adolescent questionnaires and their published scoring keys |
| `asd/data.py` | Loads, validates and cleans the CSV; builds model inputs for the app with the same encoding |
| `asd/model.py` | `AgeGroupEnsemble`: one soft-voting ensemble (logistic regression + random forest + gradient boosting) per age group, routing each person by age; the AQ-10 total is added as a feature |
| `train_model.py` | Stratified 80/20 split, 5-fold cross-validation, held-out test metrics per age group |
| `graphs.py` | Confusion matrices, ROC curves, permutation feature importance, AQ-10 score distribution |
| `app.py` | The screening form |
| `tests/` | Unit tests for scoring, data cleaning, routing, and end-to-end app tests |

**Age groups.** Adults (17+) answer the self-report AQ-10 Adult; ages 12–16 get the
parent-report AQ-10 Adolescent. These boundaries match the training data. The two
questionnaires ask different questions, so each has its own model. The training data has no
one under 12, so the app doesn't accept those ages.

**Scoring.** Each item is answered on a 4-point scale and scores 1 point when the answer is on
the autistic-trait side. That side is "agree" for some items and "disagree" for others; the
direction for each item follows the published scoring key.

## Results

On a stratified 80/20 split (646 training rows, 162 test rows):

| | Accuracy | ROC AUC |
|---|---|---|
| 5-fold cross-validation (training set) | 100% | 1.000 |
| Held-out test set: adults (141) | 100% | 1.000 |
| Held-out test set: adolescents (21) | 100% | 1.000 |

Full metrics are in `models/metrics.json`; plots are in `graphs/`.

## Limitation: what the model can learn

In this dataset, `asd_status` is exactly "AQ-10 total score above 6" for every row; the labels
come from the questionnaire's referral cutoff, not from clinical diagnoses. Any model trained on
it can at best reproduce that rule, so its high accuracy shows it learned the questionnaire, not
that it detects autism beyond it. `graphs/aq10_score_distribution.png` shows this. Making the ML
add real value needs data labelled by clinical diagnosis.

## Data

`autism_final_processed.csv` combines two public datasets from the UCI Machine Learning
Repository, both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- Thabtah, F. (2017). *Autism Screening Adult* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5F019
- Thabtah, F. (2017). *Autistic Spectrum Disorder Screening Data for Adolescent* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5V89T

Changes from the originals: the two datasets are merged into one file with a `dataset_source`
column and an `engineered_score` column added, columns are renamed, the label is encoded as
0/1, and missing values are filled.

## Author

**Mustafa Waheed**

## License

The code is licensed under the MIT License; see [LICENSE](LICENSE). The dataset is not covered
by it and stays under CC BY 4.0 (see [Data](#data)).
