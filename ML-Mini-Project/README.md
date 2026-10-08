# Mortgage Prepayment Prediction: Logistic Regression, SVM, GDA and Neural Network

## 1. Project Overview
This project predicts whether a mortgage loan will be fully prepaid within 24 months of origination, using the Freddie Mac Single-Family Loan-Level sample (2019 originations, 50,000 loans). Four classification approaches are compared on the same processed data and the same held-out test set: Logistic Regression, Support Vector Machine (three kernels), Gaussian Discriminant Analysis (linear and quadratic) and a feed-forward neural network.

## 2. Problem Statement
Binary classification of each loan:

* `1` = loan fully prepaid within 24 months of origination
* `0` = no prepayment within 24 months

The target is derived from the performance file's Zero Balance Code (`01` = prepaid in full) at loan age <= 24 months. Features come only from information known at origination.

## 3. Objective
Compare four classification approaches that differ in decision-boundary flexibility and modelling assumptions, under identical preprocessing, split and evaluation metrics.

## 4. Dataset
* **Source:** Freddie Mac Single-Family Loan-Level sample, 2019 originations (50,000 loans). The raw files are not in this repository (see "How to Run"); the download link is not recorded in the repository.
* **Origination file:** one row per loan, known on the day the loan was made.
* **Performance file:** one row per loan per month; used only to build the target.
* **Retained loans:** 49,111. Class balance: 25,749 positive (52.4 %), 23,362 negative.

## 5. Data Preprocessing
Implemented in `notebooks/01_data_preprocessing.ipynb`; settings are saved in `data/processed/preprocessing_summary.json`.

* **Leakage prevention:** all performance-file columns are excluded from the features. `current_UPB` is exactly 0 on every prepayment row and never 0 otherwise, so it would reveal the label. The reference paper's ~99.9 % accuracy is therefore not comparable with this project.
* **Removed records:** 889 loans whose performance history has repeated loan ages (overlapping histories under one loan ID) were removed. 0 loans were dropped for unknown outcome. 49,111 loans remain.
* **Missing values:** Freddie Mac "not available" codes (9999 / 999 / 99) are set to missing, then imputed with the **training-set median**. Only 18 credit-score and 29 DTI values were missing.
* **Feature selection (14 raw -> 20 after encoding):**
  * Numeric (standardised): `credit_score`, `dti`, `ltv`, `interest_rate`, `upb`, `term`, `mi_pct`, `units`, `borrowers`
  * Binary: `first_time_buyer`
  * Categorical (one-hot): `occupancy`, `channel`, `property_type`, `purpose`
  * Excluded: loan ID, CLTV (highly correlated with LTV; 0.966 in the executed notebook output), MSA, zip3, state, seller, first-payment date, constant or almost-empty columns.
* **Encoding:** one-hot with `drop_first=True`; test columns aligned to the training columns.
* **Scaling:** `StandardScaler` on the nine numeric columns, fitted on the training set only. Binary and one-hot columns are left unscaled.
* **Split:** 80/20 stratified, `random_state=42`, performed **before** fitting the imputer, encoder and scaler. Result: 39,288 training / 9,823 test loans.
* **Class balance:** 52.4 % positive in both train and test. No resampling or class weights. Majority-class baseline accuracy: 0.5243.

## 6. Models
All models load the saved files in `data/processed/` and do not re-split or re-preprocess.

### Logistic Regression
* **Purpose:** interpretable linear baseline.
* **Details:** L2 penalty, lbfgs solver, `max_iter=1000`, threshold 0.5.
* **Tuning:** 5-fold cross-validation on the training set over C in {0.01, 0.1, 1, 10}, scored by ROC-AUC; C = 1 selected (CV AUC about 0.693 for every C).
* **Limitation:** linear in the features; cannot represent the rise-then-fall of prepayment rate with interest rate.

### Support Vector Machine
* **Purpose:** non-linear margin classifier via kernels. Three kernels with settings fixed in advance (no grid search), trained on the full training set:
  * RBF: C = 1.0, gamma = "scale" (the main SVM result, tag `svm`)
  * Polynomial: degree 3, C = 1.0, gamma = "scale", coef0 = 1
  * Sigmoid: C = 1.0, gamma = "scale", coef0 = 0
* **Details:** ROC-AUC computed from `decision_function` scores. Training times in the saved run: 398.5 s (RBF), 429.2 s (polynomial), 424.5 s (sigmoid).
* **Limitations:** slow on about 39k rows; untuned. The untuned sigmoid kernel performs close to chance.

### Gaussian Discriminant Analysis
* **Purpose:** generative classifier with Gaussian class-conditional densities.
* **Variants:** LDA (shared covariance) and QDA (class-specific covariance, `reg_param=0.1`, fixed in advance after a training-only covariance check; no fit warnings).
* **Limitations:** assumes roughly Gaussian features, but 11 of the 20 columns are 0/1; maximum absolute feature correlation is 0.82; QDA's regularisation was not tuned.

### Feed-Forward Neural Network
* **Purpose:** non-linear model that can learn feature interactions.
* **Architecture:** 20 -> 32 (ReLU) -> 16 (ReLU) -> 1 (sigmoid); 1,217 trainable parameters.
* **Training:** binary cross-entropy, Adam (learning rate 0.001), batch size 64, at most 50 epochs, early stopping on validation loss (patience 5, best weights restored), 15 % of the training set held out for validation. The logged run stopped after 20 epochs (best validation loss at epoch 15).
* **Limitations:** harder to interpret; architecture not tuned; results can vary slightly between runs and machines even with fixed seeds.

## 7. Experimental Setup
* Identical features, identical stratified split, identical held-out test set of **9,823 loans** for all models.
* Positive class = 1 (prepayment). Precision, recall and F1 are for class 1; default threshold 0.5.
* ROC-AUC is computed from continuous scores (probabilities, decision scores or log-odds), not hard predictions.
* No model was tuned on the test set. Hyperparameter selection used only the training data (Logistic Regression CV; NN early stopping on a validation split). SVM and GDA settings were fixed in advance.
* `Model comparison/06_compare_models.ipynb` recomputes all metrics from saved test predictions and checks that every model used the same test rows.

## 8. Results
Test-set results (best per metric in bold):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6474 | 0.6621 | 0.6685 | 0.6653 | 0.7004 |
| SVM (RBF) | 0.6534 | 0.6608 | 0.6963 | 0.6781 | 0.7065 |
| SVM (Polynomial) | 0.6540 | 0.6518 | 0.7299 | **0.6887** | 0.7068 |
| SVM (Sigmoid) | 0.5448 | 0.5652 | 0.5715 | 0.5683 | 0.5293 |
| GDA - LDA | 0.6472 | 0.6622 | 0.6676 | 0.6649 | 0.7003 |
| GDA - QDA | 0.6289 | 0.6175 | **0.7680** | 0.6846 | 0.6877 |
| Neural Network | **0.6573** | **0.6756** | 0.6664 | 0.6710 | **0.7094** |

Majority-class baseline accuracy: 0.5243. Confusion-matrix counts for every model are in `results/confusion_matrices/` and `Model comparison/model_comparison.md`.

The neural network is highest on accuracy, precision and ROC-AUC; QDA has the highest recall and the polynomial SVM the highest F1. Five of the seven variants (Logistic Regression, LDA, SVM-RBF, SVM-Polynomial, Neural Network) cluster within accuracy 0.647-0.657 and ROC-AUC 0.700-0.709, all well above the 0.524 baseline. The neural network exceeds Logistic Regression by about 0.010 accuracy and 0.009 ROC-AUC. This is a small difference from a single split and seed, so no strong claim of superiority is made.

The weakest variants are the untuned sigmoid SVM (ROC-AUC 0.529) and QDA (lowest ROC-AUC, 0.688, among the working models).

## 9. Visualizations
Final comparison plots (in `Model comparison/results/plots/`):

* `final_roc_curves.png` - ROC curves of all seven variants
* `final_metrics_comparison.png` - five metrics per model
* `final_confusion_matrices.png` - confusion matrices of all variants

Per-model plots (in `results/plots/`): `<model>_roc_curve.png` and `<model>_confusion_matrix.png` for `logistic_regression`, `svm`, `svm_polynomial`, `svm_sigmoid`, `gda_linear`, `gda_quadratic` and `neural_network` (the neural network has no separate files for other variants), plus:

* `logistic_regression_coefficients.png`
* `neural_network_training_curves.png`
* `svm_kernel_comparison_metrics.png`, `svm_kernel_comparison_roc_curves.png`
* `preprocessing_prepay_by_rate.png`

GDA confusion-matrix images are also in `results/confusion_matrices/`.

## 10. Discussion
* **Linear vs non-linear:** Logistic Regression and LDA are practically identical (accuracy 0.6474 vs 0.6472; AUC 0.7004 vs 0.7003). Non-linear models (RBF/polynomial SVM, Neural Network) are only about one accuracy point better, suggesting limited non-linear signal in origination-only features. The repository's explanation (unobserved factors such as future mortgage rates) was not tested.
* **Precision/recall trade-off:** QDA predicts "prepay" for about 65 % of test loans (true share 52 %), lowering false negatives (1,195 vs 1,712 for LDA) but raising false positives (2,450 vs 1,754). Its ROC-AUC is lower, so it does not rank loans better. The polynomial SVM shows a milder version of this shift; the Neural Network has the fewest false positives (1,648).
* **Interpretability:** Logistic Regression provides coefficients; the Neural Network does not. The co-op coefficient is large but based on very few loans.
* **Overfitting:** train and test metrics are close for Logistic Regression (accuracy 0.6445 / 0.6474) and the Neural Network (0.6588 / 0.6573; AUC 0.7158 / 0.7094). Training metrics are not saved for SVM or GDA, so no conclusion is drawn for them.

## 11. Limitations
* Single year (2019 originations), 50,000-loan sample, origination-only features.
* One train/test split and one seed; no confidence intervals; fixed 0.5 threshold.
* GDA's Gaussian assumption is only approximately met (11 of 20 columns are 0/1; correlated features).
* Limited hyperparameter tuning: only Logistic Regression's C was cross-validated; SVM, QDA regularisation and the NN architecture were fixed, not tuned.
* SVM training took about 400-430 s per kernel on 39,288 rows.
* Results are not comparable with the reference paper (per-loan-month rows, leakage-prone `current_UPB`).

## 12. Project Structure
```
ML-Mini-Project/                      (repository root; contains a stub README.md and the project folder)
└── ML-Mini-Project/                  (project folder)
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   ├── raw/                      (.gitkeep only; Freddie Mac files are git-ignored)
    │   └── processed/                (X_train/X_test/y_train/y_test .csv, preprocessing_summary.json)
    ├── notebooks/
    │   ├── 01_data_preprocessing.ipynb
    │   ├── 02_logistic_regression.ipynb
    │   ├── 03_svm.ipynb
    │   ├── 04_gda.ipynb
    │   └── 05_neural_network.ipynb
    ├── Model comparison/
    │   ├── 06_compare_models.ipynb
    │   ├── model_comparison.csv / model_comparison.md
    │   └── results/
    │       ├── metrics/              (model_comparison.csv / .md)
    │       └── plots/                (final_roc_curves, final_metrics_comparison, final_confusion_matrices)
    └── results/
        ├── metrics/                  (per-model metrics JSON/CSV, test predictions, svm_kernel_comparison.csv)
        ├── confusion_matrices/       (per-model CSVs; PNGs for GDA)
        ├── plots/                    (per-model ROC / confusion matrix and other PNGs)
        └── predictions/              (gda_linear, gda_quadratic, svm predictions)
```

## 13. How to Run
1. **Clone and enter the project folder**
```bash
   git clone https://github.com/MYSTICnova734/ML-Mini-Project.git
   cd ML-Mini-Project/ML-Mini-Project
```
2. **Environment and dependencies**
```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
```
3. **Raw data (only needed to re-run preprocessing).** Download the Freddie Mac 2019 sample and place it as `data/raw/sample_2019.zip` (containing `sample_orig_2019.txt` and `sample_perf_2019.txt`). The download link is not recorded in this repository. The processed files in `data/processed/` are committed, so steps 5-7 work without the raw data.
4. **Start Jupyter from inside `notebooks/`** (notebooks 01, 02 and 05 use `../` relative paths):
```bash
   cd notebooks
   jupyter notebook
```
5. **Preprocessing:** run `01_data_preprocessing.ipynb` (Restart & Run All). It rewrites `data/processed/` (seed 42).
6. **Models:** run `02_logistic_regression.ipynb`, `03_svm.ipynb`, `04_gda.ipynb` and `05_neural_network.ipynb` in any order. They write to `results/metrics/`, `results/confusion_matrices/` and `results/plots/`. The SVM notebook takes roughly 20 minutes (three kernels, about 400 s each in the saved run).
7. **Comparison:** run `Model comparison/06_compare_models.ipynb` after all model notebooks. It detects the project root automatically, recomputes metrics from the saved predictions and writes `model_comparison.csv/.md` and the `final_*.png` plots into `results/`. The repository also keeps earlier copies under `Model comparison/results/`.

Note: neural-network results can differ slightly between runs and machines.

## 14. Technologies Used
Python, Jupyter, pandas, NumPy, scikit-learn (Logistic Regression, SVC, LDA/QDA, metrics, preprocessing), TensorFlow/Keras (neural network), Matplotlib. Versions are only constrained by `requirements.txt` (scikit-learn >= 1.3, tensorflow >= 2.15).

## 15. References
1. Freddie Mac Single-Family Loan-Level Dataset, 2019 sample (origination and performance files).
2. Reference paper discussed in the preprocessing notebook: Not available in the current repository (no title, authors or DOI recorded).
