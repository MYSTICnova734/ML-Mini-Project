# ML Mini-Project: Mortgage Prepayment Prediction

Binary classification on the Freddie Mac Single-Family Loan-Level sample (2019 originations, 50,000 loans):

* `1` = loan **fully prepaid within 24 months** of origination
* `0` = no prepayment within 24 months

Four classifiers are compared on the **same processed data and the same train/test split**:
Logistic Regression, SVM, GDA, Neural Network.

| Notebook | Owner | Status |
|---|---|---|
| `01_data_preprocessing.ipynb` | Shared (both verify) | done |
| `02_logistic_regression.ipynb` | Person 1 | done |
| `03_svm.ipynb` | Person 2 | to do |
| `04_gda.ipynb` | Person 2 | to do |
| `05_neural_network.ipynb` | Person 1 | done |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place the Freddie Mac sample zip here (it is git-ignored because it is large):

```
data/raw/sample_2019.zip     # contains sample_orig_2019.txt and sample_perf_2019.txt
```

The preprocessing notebook unzips it automatically. The files have no header row; column names are assigned by position in the notebook.

## How to run (order matters)

```bash
cd notebooks
jupyter notebook
```

1. Run `01_data_preprocessing.ipynb` first (Kernel -> Restart & Run All). It writes `data/processed/`.
2. Then run `02_logistic_regression.ipynb` and `05_neural_network.ipynb` (and Person 2's notebooks) in any order.

All notebooks use relative paths (`../data/...`), so start Jupyter from inside `notebooks/` or open the notebooks from there.
`data/processed/` is committed so teammates can start modelling immediately; it is reproducible (`SEED = 42`).

## What the shared preprocessing does (and why)

* **Target:** from the performance file's *Zero Balance Code*: `1` if code `01` (prepaid in full) occurs at loan age <= 24 months, else `0`. A fixed 24-month window gives every loan the same observation period.
* **Features (14 raw -> 20 after encoding), all known at origination:** credit score, DTI, LTV, interest rate, loan amount, term, MI %, units, borrowers, first-time-buyer flag, occupancy, channel, property type, loan purpose.
* **Excluded:** every performance-file column (they are observed after the outcome = leakage), loan ID, CLTV (0.995 correlated with LTV), MSA / zip3 / state / seller (many categories, small gain), first-payment date, constant or almost-empty columns.
* **Cleaning:** Freddie Mac "not available" codes (9999 / 999 / 99) -> missing -> median of the training set. 889 loans whose performance history contains repeated loan ages (two overlapping histories under one loan ID) are removed. 49,111 loans remain.
* **Split:** 80/20 stratified, `random_state=42`, done **before** fitting the imputer / encoder / scaler. Those are fitted on train only. Result: 39,288 train / 9,823 test.
* **Class balance:** 52.4 % / 47.6 % -> no resampling or class weights needed.

### Known differences from the reference paper
The paper's top feature is `current_UPB`. In this data it is exactly 0 on every prepayment row and never 0 otherwise, so it leaks the answer and explains the paper's ~99.9 % accuracy. The paper also predicts per loan-month and splits rows randomly. This project uses one row per loan and origination-only features, so accuracies are **not comparable** to the paper. Realistic accuracy with origination-only features is in the mid-60 % range (majority-class baseline: 52.4 %).

## Files for Person 2 (SVM, GDA)

```python
import pandas as pd
X_train = pd.read_csv("../data/processed/X_train.csv")   # imputed, one-hot encoded, numeric columns scaled
y_train = pd.read_csv("../data/processed/y_train.csv")["target"]
X_test  = pd.read_csv("../data/processed/X_test.csv")
y_test  = pd.read_csv("../data/processed/y_test.csv")["target"]
```

* Do **not** re-split or re-preprocess. `data/processed/preprocessing_summary.json` lists the settings, the numeric / binary / one-hot column groups and the feature order.
* RBF-kernel SVM is slow on ~39k rows. If you subsample, subsample **only the training set**, always evaluate on the full `X_test`, and document it.
* GDA assumes roughly Gaussian features; the 0/1 columns are not Gaussian. State this as a limitation (or also try only the numeric columns listed in the summary JSON) and keep the same `y_test`.

## Results format (shared by all models)

Each model saves, using its own tag (`logistic_regression`, `svm`, `gda`, `neural_network`):

* `results/metrics/<tag>_metrics.json` - keys: `model`, `task`, `n_train`, `n_test`, `n_features`, `split_seed`, `hyperparameters`, `baseline_majority_accuracy_test`, `train_metrics`, `test_metrics` (each with `accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp`)
* `results/metrics/<tag>_test_predictions.csv` - columns `y_true, y_pred, y_prob`
* `results/confusion_matrices/<tag>_confusion_matrix.csv`
* `results/plots/<tag>_*.png`

Precision, recall and F1 are for class 1 (Prepayment). Combine all four after everyone has finished:

```python
import json, glob, pandas as pd
rows = []
for f in glob.glob("../results/metrics/*_metrics.json"):
    r = json.load(open(f)); rows.append({"model": r["model"], **r["test_metrics"]})
print(pd.DataFrame(rows).set_index("model").round(4))
```

No model is declared better until all four are compared.
