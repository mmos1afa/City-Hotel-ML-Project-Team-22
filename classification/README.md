# Classification — Milestone 1 Progress Report

**Project:** CSE382 — City Hotel Booking Demand (H2)  
**Task:** Predict whether a booking is canceled (`IsCanceled`: `1` = canceled, `0` = not canceled)  
**Current dataset:** `data/cleaned/H2_cleaned2.csv` — **duplicates retained**  
**Prediction point:** Booking creation (provisional; see Leakage Audit)  
**Milestone 1 status:** EDA, feature selection, preprocessing, reproducible splits, three untuned classifier experiments, feature-importance exploration, and a preliminary leakage checker. **Final test set not evaluated.**

## 1. Overview

We developed a classification workflow that estimates cancellation risk from booking information. We use 12 candidate booking-time predictors and compare a simple linear model with two tree ensembles. Our aim in Milestone 1 is to establish a reproducible baseline and understand the data, not select a final production model.

We assume the prediction is requested **when a booking is created**, before cancellation or check-in. This matters because a feature may be highly predictive but invalid if its recorded value was only known after booking creation. Some fields in the published dataset are not guaranteed to be snapshots from the booking-creation date; we record this uncertainty in Section 7.

## 2. Data and cleaning decisions

The required source is `H2.csv` (City Hotel): **79,330 records and 31 source variables**. The team performs common cleaning in `clean_data.py`, outside our classification module. Our classification scripts then read a prepared dataset.

| Dataset | Rows | Cancellation rate | Use |
|---|---:|---:|---|
| `data/precleaned/H2.csv` | 79,330 | ~41.73% | Original City Hotel source |
| `data/cleaned/H2_cleaned2.csv` | 79,162 | **41.79%** | **Current classification dataset; duplicates retained** |
| `data/cleaned/H2_cleaned3.csv` | 53,273 | **30.10%** | Temporary deduplication experiment, **not** the current dataset |

Shared cleaning excludes **167 zero-guest records** and **one record with ADR = 5400**, and performs category/missing-value normalization. These are team cleaning choices, not proof that every excluded record is invalid. `H2_cleaned2.csv` retains **25,889 repeated full rows** because identical-looking records cannot automatically be identified as the same real booking. We tested removing duplicates; this substantially changed the label distribution and reduced the Logistic Regression validation scores. We reverted to `H2_cleaned2.csv` for the current work.

**Configuration:** `eda.py`, `features.py`, and `split.py` must all reference `H2_cleaned2.csv`. `train_baseline.py` uses `DATA_PATH` imported from `split.py`. 

**Supporting tables:** `outputs/classification/tables/dataset_baseline_comparison.csv`.

## 3. Files, main functions, and data flow

```text
classification/
├── eda.py                Descriptive EDA and saved plots/tables
├── features.py           Select predictors, engineer totals, create X and y
├── preprocessing.py      Define an unfitted transformation pipeline
├── split.py              80/20 holdout, CV folds, saved split manifest
├── train_baseline.py     Model pipelines, untuned CV results, importance study
├── leakage_checker.py    Programmatic split / fit integrity checks
└── README.md             Classification progress report
```

| File / main function | What it does |
|---|---|
| `eda.py` — `summarize()` | Groups bookings and calculates counts, cancellations, and cancellation rates; saves CSV files. |
| `eda.py` — `bar_chart()`, `line_chart()` | Creates cancellation-rate figures and selected feature-interaction plots. |
| `eda.py` — `main()` | Loads `H2_cleaned2.csv`, produces engineered totals for description, and runs the EDA analyses. |
| `features.py` — `select_classification_features(df)` | Returns `X` containing the 12 approved predictors and `y = IsCanceled`; computes `TotalGuests` and `TotalNights`; normalizes text categories. |
| `preprocessing.py` — `build_preprocessor(scale_numeric=True)` | Builds an **unfitted** `ColumnTransformer` for imputation, scaling, one-hot encoding, and binary passthrough. |
| `split.py` — `split_data(X, y)` | Creates the stratified **80% development / 20% test** split using seed 42. |
| `split.py` — `get_feature_groups(X)` | Assigns group IDs to identical 12-feature patterns (not to unique customers). |
| `split.py` — `make_cv_folds(...)` | Produces either stratified or group-aware five-fold development-only CV assignments. |
| `split.py` — `main()` | Saves split membership and fold assignments to `split_manifest.csv`. |
| `train_baseline.py` — `build_model(model_name)` | Creates an independent pipeline for Logistic Regression, Random Forest, or Histogram Gradient Boosting. |
| `train_baseline.py` — `evaluate_cv(...)` | Fits/evaluates the selected pipeline across development folds; reports accuracy, precision, recall, F1, ROC-AUC and fit time. |
| `train_baseline.py` — `to_dense()` | Converts sparse encoded data to dense when needed by Histogram Gradient Boosting. |
| `train_baseline.py` — `analyze_feature_importance(...)` | Trains on one development training fold and measures **permutation importance on its validation fold**. |
| `train_baseline.py` — `main()` | Runs the selected baselines, saves CV tables, and produces an importance table/plot. It does not score the final test set. |
| `leakage_checker.py` — `LeakageChecker.assert_no_leakage()` | Audits approved inputs, disjoint row IDs, split manifest, group-fold purity, fitted-scaler statistics, and a synthetic negative control. |

`train_baseline.py` initially handled only Logistic Regression. Our later tree-model experiment extends the same file with model selection and permutation importance; the original baseline remains the reference.

```text
H2_cleaned2.csv
    ├── eda.py ───────────────────────────────> EDA tables and figures
    └── features.py ──> X, y
                          └── split.py ───────> 80% development | 20% reserved test
                                  └── train_baseline.py
                                        ├── training fold: FIT transformer + model
                                        ├── validation fold: TRANSFORM + PREDICT
                                        ├── five-fold metrics, model comparisons
                                        └── importance on a development validation fold
    leakage_checker.py audits the declared inputs and split/fold safeguards
```

**Supporting table:** `outputs/classification/tables/split_manifest.csv`.

## 4. EDA and our 12 selected features

We first investigated relationships between booking characteristics and cancellation. The following rates are from **`H2_cleaned2.csv`**, which retains duplicates. They justify testing these features; they do **not** prove causality, independent predictive value, or absence of prediction-time leakage.

| Feature | Type | Why we included it; observed EDA evidence |
|---|---|---|
| `LeadTime` | Numerical | Advance-booking time strongly distinguishes outcomes: **12.18%** cancellation for 0–7 days versus **64.07%** for 181+ days. |
| `TotalGuests` | Numerical, engineered | Represents booked party size (`Adults + Children + Babies`); 2 guests had **44.90%** cancellations versus **34.51%** for 0–1 guests. |
| `TotalNights` | Numerical, engineered | Represents stay length (weekday + weekend nights); 1-night stays had **30.94%** cancellation versus **46.54%** for 2–3 nights. |
| `ArrivalDateMonth` | Categorical | Captures seasonality: **37.06%** in March versus **46.31%** in April; month patterns vary by year. |
| `MarketSegment` | Categorical | Large segment differences: **Groups 68.92%** versus **Direct 17.39%**. |
| `DistributionChannel` | Categorical | Channel differs by cancellation outcome: **TA/TO 45.08%** versus **Direct 18.22%**; may overlap with `MarketSegment`. |
| `CustomerType` | Categorical | Booking/customer types differ: **Contract 48.13%** versus **Transient-Party 28.12%**. |
| `IsRepeatedGuest` | Binary | Repeated guests show **22.21%** cancellation versus **42.29%** for first-time guests; history timing must be verified. |
| `PreviousCancellations` | Numerical | Previous cancellation history could indicate risk; the **96.31%** rate for exactly one previous cancellation is unusually high and requires a leakage/timing check. |
| `PreviousBookingsNotCanceled` | Numerical | Prior non-canceled bookings could indicate loyalty/reliability: **42.48%** for zero versus **7.96%** for one; timing remains uncertain. |
| `ReservedRoomType` | Categorical | The **originally reserved** room type might reflect booking preferences: type A **43.60%** versus type D **35.29%**. |
| `Meal` | Categorical | Booked meal package may distinguish behavior: BB **42.84%** versus SC **37.66%**; very small categories need caution. |

We also examined **`LeadTime × MarketSegment`**, `ArrivalDateMonth × ArrivalDateYear`, `MarketSegment × ArrivalDateYear`, and prior cancellation history by repeated-guest status. These analyses check whether one-variable associations depend on other booking characteristics or time periods.

### Features excluded or postponed

| Source fields | Our decision |
|---|---|
| `ReservationStatus`, `ReservationStatusDate` | **Excluded:** directly reveal the final reservation outcome/status. |
| `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList` | **Excluded:** can reflect events after booking creation. |
| `DepositType`, `ADR` | **Excluded in this baseline:** uncertain timing / possible target-proxy concerns; reassess only with a defensible prediction-time definition. |
| `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` | **Postponed:** requests may have changed during the booking lifecycle. |
| `Agent`, `Company`, `Country` | **Postponed:** high-cardinality identifiers/categories and missingness, rather than assumed irrelevant. |
| Additional arrival date columns | **Postponed:** month used as the initial, compact seasonal representation. |
| `Adults`, `Children`, `Babies`, weekday/weekend nights | Represented by `TotalGuests` and `TotalNights` in the initial feature shortlist. |

**Supporting figures:** `outputs/classification/figures/lead_time.png`, `total_guests.png`, `total_nights.png`, `arrival_month.png`, `market_segment.png`, `repeated_guest.png`, `previous_cancellations.png`, `lead_time_by_segment.png`, `arrival_month_by_year.png`, `segment_by_year.png` (all under the same `figures/` directory).  
**Supporting tables:** `outputs/classification/tables/lead_time.csv`, `total_guests.csv`, `total_nights.csv`, `arrival_month.csv`, `market_segment.csv`, `distribution_channel.csv`, `customer_type.csv`, `repeated_guest.csv`, `previous_cancellations.csv`, `previous_non_canceled.csv`, `reservedroomtype.csv`, `meal.csv`, `one_previous_by_repeated.csv` (all under `tables/`). `reservedroomtype.csv` and `meal.csv` are supplementary results not generated by the current basic `eda.py` run.

## 5. Preprocessing and evaluation design

### Preprocessing

| Input kind | Current processing | Rationale |
|---|---|---|
| Five numerical features | Median imputation → `StandardScaler` | Handles missing numbers robustly and makes scales suitable for the linear baseline. |
| Six categorical features | Whitespace normalization → missing label / `SimpleImputer` → `OneHotEncoder(handle_unknown="ignore")` | Produces numerical inputs without falsely ordering categories; handles unseen categories. |
| One binary feature | `IsRepeatedGuest` as 0/1 (passthrough) | Already numeric and binary. |
| Derived totals | Create `TotalGuests` and `TotalNights` in `features.py` | Reduces several raw fields to reproducible party-size / stay-duration inputs. |

`build_preprocessor()` **defines** transformations but does not fit them. During cross-validation, `cross_validate()` fits the **entire `Pipeline(preprocessing → classifier)` separately on each training fold**; validation rows are transformed using only that fold's fitted preprocessing. Our baseline **does not** fit a scaler or encoder once on all 79,162 rows. The separate risk from cleaning the shared CSV before splitting is recorded in Section 7.

For this first controlled comparison, we kept standardized inputs for Logistic Regression, Random Forest and Histogram Gradient Boosting even though tree models do not need numerical scaling. Histogram Gradient Boosting also uses a sparse-to-dense conversion after preprocessing. We can examine model-specific transformations later.

### Splitting and metrics

- **Final holdout:** 80% development (**63,329**) and 20% reserved test (**15,833**), stratified by `IsCanceled`, `random_state=42`.
- **Primary CV:** `StratifiedKFold(5, shuffle=True, random_state=42)` **within development only**.
- **Robustness CV:** `StratifiedGroupKFold(5, shuffle=True, random_state=42)` using matching 12-feature vectors as groups **within development only**.
- **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, and training time. Canceled (`1`) is the positive class. Model selection has not used the final test set.

The group-aware CV split does not make the final holdout group-disjoint, and matching features do not necessarily identify matching customers.

**Supporting table:** `outputs/classification/tables/split_manifest.csv`.

## 6. Baseline experiments and observations

### 6.1 Three untuned models on the current dataset

We began with Logistic Regression (`max_iter=1000`, default regularization), then tested **Random Forest** and **Histogram Gradient Boosting** using their default learning settings and fixed random seeds. Histogram Gradient Boosting is a boosting ensemble. No hyperparameter-search procedure has been performed yet.

**Stratified 5-fold CV (development only):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7346 | 0.7762 | 0.5126 | 0.6174 | 0.7797 |
| Random Forest | **0.7921** | 0.7676 | **0.7206** | **0.7434** | **0.8663** |
| Histogram Gradient Boosting | 0.7770 | **0.8710** | 0.5474 | 0.6722 | 0.8569 |

**Group-aware 5-fold CV (development only):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7333 | 0.7738 | 0.5112 | 0.6156 | 0.7770 |
| Random Forest | 0.7189 | 0.6994 | **0.5747** | 0.6306 | 0.7766 |
| Histogram Gradient Boosting | **0.7587** | **0.8490** | 0.5139 | **0.6399** | **0.8274** |

**Interpretation:** Random Forest leads on ordinary stratified F1, recall and ROC-AUC, but its ROC-AUC falls **0.0897** under group-aware CV. Histogram Gradient Boosting retains a higher group-aware ROC-AUC (**0.8274**), while Logistic Regression changes little. Repeated feature patterns may affect how well these models appear to generalize; differences in fold composition also contribute. **We have not selected a final winner.**

### 6.2 Temporary duplicate-removal experiment

We ran the same Logistic Regression baseline on `H2_cleaned3.csv`, then reverted to `H2_cleaned2.csv`:

| Dataset / CV | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `cleaned2` stratified | 0.7346 | 0.7762 | 0.5126 | 0.6174 | 0.7797 |
| `cleaned3` stratified | 0.7196 | 0.6193 | 0.1781 | 0.2766 | 0.7113 |

Since removing duplicates also reduces the observed cancellation proportion from **41.79% to 30.10%**, this is **not a controlled same-holdout comparison**. The lower scores alone do not establish that duplicate retention is universally better.

### 6.3 Exploratory permutation importance

Using Random Forest and **one development validation fold**, we shuffled each original predictor three times and measured the change in validation ROC-AUC. This operates on the full model pipeline, so each categorical feature receives **one** importance value rather than separate values for each one-hot category.

| Highest importance | Mean ROC-AUC decrease | Lowest importance | Mean ROC-AUC decrease |
|---|---:|---|---:|
| `LeadTime` | 0.1726 | `PreviousBookingsNotCanceled` | 0.0028 |
| `MarketSegment` | 0.1195 | `IsRepeatedGuest` | 0.0012 |
| `CustomerType` | 0.1098 | `DistributionChannel` | 0.0171 |
| `ArrivalDateMonth` | 0.0886 | `TotalGuests` | 0.0282 |

The strong importance of lead time, segment, and customer type supports our EDA-based choices. Near-zero importance **does not automatically justify deleting** a feature: correlated variables can substitute for one another, and results may differ across folds. These rankings are model-dependent and cannot validate prediction-time availability.

**Supporting tables:** `outputs/classification/tables/logreg_cleaned2_stratified_summary.csv`, `logreg_cleaned2_group_aware_summary.csv`, `random_forest_stratified_summary.csv`, `random_forest_group_aware_summary.csv`, `hist_gradient_boosting_stratified_summary.csv`, `hist_gradient_boosting_group_aware_summary.csv`, `dataset_baseline_comparison.csv`, `random_forest_permutation_importance.csv`; fold-level results use corresponding `*_folds.csv` names (all in `tables/`).  
**Supporting figure:** `outputs/classification/figures/random_forest_permutation_importance.png` (generated by the extended training script, if the figure was saved).  

## 7. Preliminary leakage audit

We audited both **prediction-time leakage** (what the hotel could know when creating a booking) and **experimental leakage** (what data enters model fitting and validation).

| Risk or safeguard | Findings and current status |
|---|---|
| Direct target / post-outcome features | `ReservationStatus` and `ReservationStatusDate` excluded. Post-booking fields such as `AssignedRoomType` and `BookingChanges` also excluded. |
| History and booking-category timing | `PreviousCancellations`, `PreviousBookingsNotCanceled`, and `IsRepeatedGuest` require evidence that values reflect **only earlier bookings**. Other categories might have changed before arrival. This is **not certified** by the checker. |
| Extreme EDA association | Exactly one previous cancellation corresponds to an unusually high outcome rate; investigate before treating it as reliably available at prediction time. |
| Fold-level fitting | **PASS:** traced Logistic Regression pipelines fit imputation/scaling only on each fold's training rows. No validation/test rows were used in those fitted fold statistics. |
| Feature allowlist | **PASS:** only the 12 designated predictors enter the feature matrix. |
| Source-row separation | **PASS:** no original row index occurs in both development and final holdout. |
| Repeated feature patterns in the final holdout | **WARN:** **9,611 of 15,833 test rows (60.7%)** have a selected-feature vector also present in development; these are not necessarily identical bookings. |
| Group-aware CV | **PASS:** identical selected-feature groups cannot span training and validation within each group-aware fold. |
| Saved manifest verification | **PASS:** recorded split/fold assignments match the current checked configuration. |
| Negative-control leakage test | **PASS:** a deliberately invalid **synthetic** train+test fit raises `LeakageError`. |
| Upstream full-data cleaning | The shared cleaned CSV contains transformations performed **before splitting**, including a dataset-wide median fill for four `Children` values. This is an information-flow limitation even though downstream CV scaling is fold-specific. |
| Full-dataset EDA and test blindness | Target relationships were inspected across the full dataset before the holdout was reserved. Future feature/model choices must not use held-out test scores. |

`LeakageChecker.assert_no_leakage()` verifies the checks it can observe and raises `LeakageError` for specified failures. Cross-holdout predictor matches are a **warning** by default (optional strict mode rejects them). Passing this checker does **not** prove the upstream cleaning or booking-time availability is leakage-free. In particular, **the current sklearn pipeline is not fitting its scaler/encoder on the entire dataset**; the known concern is upstream of the pipeline and in the source semantics.

**Supporting table:** `outputs/classification/tables/split_manifest.csv`.  
**Supporting audit output:** `outputs/classification/leakage_audit_log.txt` 

## 8. Our Milestone 2 implementation plan

We will build on the current three-classifier baseline experiments rather than recreate them.

1. **Strengthen evaluation and feature decisions.** Choose a consistent development-only model-selection protocol, investigate group-aware robustness and duplicate sensitivity, repeat permutation importance across folds, and test whether removing weak or uncertain booking-history predictors improves performance. Avoid using the final holdout to make these decisions.
2. **Tune the three classifiers we already tested.** Use `RandomizedSearchCV` or a small `GridSearchCV` on the development folds: Logistic Regression (`C`, class weighting), Random Forest (tree depth, leaf size, feature subsampling, tree count), and Histogram Gradient Boosting (learning rate, leaf count, iterations, regularization). Record every search range, best configuration, metrics, and run time. Consider another library only if it adds a useful comparison.
3. **Investigate thresholds and dimensionality reduction.** Examine cancellation decision thresholds using development predictions. Fit PCA within the training pipeline, report explained variance, and compare **supervised models** with/without PCA. Apply supervised LDA to classification and compare original/PCA/LDA features for at least one classifier; since `IsCanceled` has two classes, LDA produces at most one component.
4. **Choose and finalize models.** Compare baseline and tuned models using precision, recall, F1, ROC-AUC, robustness and efficiency. Select a justified cancellation-risk trade-off, refit on the available development data, and evaluate the selected pipeline  on the reserved test set. Save the fitted preprocessing and classifier together for inference.
