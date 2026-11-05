# Classification — Milestone 1 Progress Report

**Project:** CSE382 City Hotel Booking Demand (H2)  
**Task:** Predict booking cancellation (`IsCanceled`: `1` = canceled; `0` = not canceled)  
**Current classification dataset:** `data/cleaned/H2_cleaned2.csv` (**duplicates retained**)  
**Status:** EDA, initial feature selection, split strategy, preprocessing, untuned Logistic Regression, and a preliminary leakage checker are implemented. **The final test set has not been evaluated.**

## 1. Overview and prediction point

Our classification module estimates whether a City Hotel booking will eventually be canceled. We assume predictions are made **when the booking is created**, before check-in or cancellation. We selected information expected to be available at that point; however, the dataset does not guarantee that all recorded fields are historical booking-creation snapshots. The availability of guest-history and some booking-category fields remains provisional.

The module uses **12 selected predictors** (5 numerical, 6 categorical, 1 binary) and begins with an **untuned Logistic Regression** baseline. Its primary evaluation uses development-only stratified five-fold cross-validation; group-aware cross-validation is an additional robustness check.

## 2. Data source, cleaning choices, and dataset variants

The required original dataset is **H2.csv (City Hotel)**, with **79,330 rows and 31 source columns**. The team's shared cleaning is performed outside this classification module; classification reads the prepared CSV rather than cleaning raw H2 again.

| Dataset | Rows | Cancellation rate | Duplicates | How we used it |
|---|---:|---:|---|---|
| `data/precleaned/H2.csv` | 79,330 | — | Original source | Source verification/shared cleaning input |
| `H2_cleaned2.csv` | 79,162 | 41.79% | 25,889 exact repeated rows | **Current primary dataset**, EDA, CV baseline, leakage checker |
| `H2_cleaned3.csv` | 53,273 | 30.10% | Exact duplicate rows removed | **Temporary comparison experiment only**; reverted afterward |

**Documented shared cleaning decisions:** remove 167 zero-guest rows and one extreme `ADR = 5400` row; normalize selected missing/category values; the cleaned source also filled four missing `Children` entries using a dataset-wide median. These are chosen preparation rules, not proof that all removed bookings were erroneous. **We retained duplicates for the primary analysis** because repeated source rows are not necessarily duplicate real-world bookings, and their removal substantially changes the cancellation distribution. We tested the duplicate-free version but did not adopt it.

**Current code configuration:** `classification/eda.py`, `classification/features.py`, and `classification/split.py` should point to `H2_cleaned2.csv`; `train_baseline.py` imports `DATA_PATH` from `split.py`. The previously uploaded versions of `features.py` and `split.py` still showed `H2_cleaned3.csv` from the temporary test; **verify those two path constants are reverted in the actual repository before generating outputs**.

## 3. Module structure and how the files work together

```text
classification/
├── eda.py               # Descriptive EDA; creates summary tables and plots
├── features.py          # Selects 12 inputs and derives two totals
├── preprocessing.py     # Defines UNFITTED transformations for model inputs
├── split.py             # Creates holdout, CV folds, and split manifest
├── train_baseline.py    # Fits/evaluates Logistic Regression within CV
├── leakage_checker.py   # Audits feature allowlist, split and fold fit integrity
└── README.md            # This report
```

**Main functions and functionality**

| File / function | Responsibility |
|---|---|
| `eda.py` → `summarize()` | Groups bookings by one or more features; saves booking counts, cancellations, and cancellation rates as CSV tables. |
| `eda.py` → `bar_chart()`, `line_chart()` | Saves cancellation-rate figures, including feature interactions and year-by-year comparisons. |
| `eda.py` → `main()` | Loads **cleaned2**, engineers descriptive totals, organizes EDA groups, and generates all implemented tables/figures. Does not fit a model. |
| `features.py` → `select_classification_features(df)` | Produces `X` (12 approved predictors) and `y` (`IsCanceled`); computes `TotalGuests` and `TotalNights`; strips categorical whitespace and handles missing text labels. |
| `preprocessing.py` → `build_preprocessor(scale_numeric=True)` | Returns an **unfitted** `ColumnTransformer` for numerical, categorical, and binary features. `scale_numeric=False` is available for later model comparisons. |
| `split.py` → `split_data(X, y)` | Makes an 80% development / 20% test split, stratified by `IsCanceled`, with seed `42`. |
| `split.py` → `get_feature_groups(X)` | Assigns the same group ID to identical **12-feature** predictor patterns. These IDs are not guest IDs. |
| `split.py` → `make_cv_folds(...)` | Returns either five stratified folds or five stratified **group-aware** folds within development data. |
| `split.py` → `main()` | Generates and saves `outputs/classification/tables/split_manifest.csv` with subset/fold assignments. |
| `train_baseline.py` → `build_model()` | Wraps the unfitted preprocessor and `LogisticRegression(max_iter=1000)` in a single sklearn `Pipeline`. |
| `train_baseline.py` → `evaluate_cv(...)` | Runs five-fold `cross_validate` and returns per-fold accuracy, precision, recall, F1, ROC-AUC, and fit time. |
| `train_baseline.py` → `main()` | Evaluates both CV strategies and saves fold-by-fold and summary CSV files; does **not** score the test set. |
| `leakage_checker.py` → `LeakageChecker.assert_no_leakage()` | Runs the integrity checks described in Section 8, raising `LeakageError` for specified violations. |

**Execution/data flow:**

```text
H2_cleaned2.csv
       ├──> eda.py ──────────────────> outputs/classification/{tables,figures}/
       └──> features.py (X, y)
                   └──> split.py (80% development / 20% untouched test)
                               └──> train_baseline.py
                                     ├── CV training fold: FIT preprocessing + Logistic Regression
                                     ├── CV validation fold: TRANSFORM + PREDICT (no refit)
                                     └── saved CV tables (not final test results)
                    leakage_checker.py verifies the selected split/fit safeguards
```

## 4. EDA and feature selection — 12 selected predictors

Our `eda.py` examines cancellation-rate differences for each selected topic and interactions such as lead time by market segment, month by year, and previous cancellations by guest history. The observations below come from the **duplicate-retaining `H2_cleaned2.csv`**. EDA supports *why we considered* a feature; it does not establish causality or guarantee that the feature was known at prediction time.

| Feature | Type | Why we selected it / observed evidence | Screenshot evidence to insert |
|---|---|---|---|
| `LeadTime` | Numeric | Longer advance bookings have higher cancellation: **12.18%** at 0–7 days versus **64.07%** at 181+ days. | **[NOTE: Add screenshot of `figures/lead_time.png` or `tables/lead_time.csv`; highlight the 0–7 and 181+ groups.]** |
| `TotalGuests` | Numeric, engineered | Booking party size may matter: **44.90%** cancellation for 2 guests versus **34.51%** for 0–1. Formula: `Adults + Children + Babies`. | **[NOTE: Add screenshot of `figures/total_guests.png` or `tables/total_guests.csv`; highlight 2 vs 0–1 guests.]** |
| `TotalNights` | Numeric, engineered | Planned stay length differs by outcome: **30.94%** at 1 night versus **46.54%** at 2–3 nights. Formula: weekend + weekday nights. | **[NOTE: Add screenshot of `figures/total_nights.png` or `tables/total_nights.csv`; highlight 1 and 2–3 nights.]** |
| `ArrivalDateMonth` | Categorical | Seasonality may matter: March **37.06%** versus April **46.31%**; the pattern changes between years. | **[NOTE: Add screenshot of `figures/arrival_month.png` and, preferably, `figures/arrival_month_by_year.png` to show the year-specific limitation.]** |
| `MarketSegment` | Categorical | Market segments differ sharply: **Groups 68.92%** versus **Direct 17.39%**. | **[NOTE: Add screenshot of `figures/market_segment.png` or `tables/market_segment.csv`; highlight Groups and Direct.]** |
| `DistributionChannel` | Categorical | Booking channel may influence cancellation: **TA/TO 45.08%** versus **Direct 18.22%**. Possible overlap with market segment. | **[NOTE: Add screenshot of `tables/distribution_channel.csv`; highlight TA/TO and Direct.]** |
| `CustomerType` | Categorical | Contract versus Transient-Party show different rates: **48.13%** versus **28.12%**. | **[NOTE: Add screenshot of `tables/customer_type.csv`; highlight Contract and Transient-Party.]** |
| `IsRepeatedGuest` | Binary | Returning guests show lower cancellation: **22.21%** versus **42.29%** for first-time guests. Timing must be checked. | **[NOTE: Add screenshot of `figures/repeated_guest.png` or `tables/repeated_guest.csv`.]** |
| `PreviousCancellations` | Numeric | Historical cancellation may signal future cancellation; exactly **1 previous cancellation → 96.31%** here, an unusually strong and suspicious pattern requiring further audit. | **[NOTE: Add screenshot of `figures/previous_cancellations.png`, plus `tables/one_previous_by_repeated.csv` to show the unusual relationship.]** |
| `PreviousBookingsNotCanceled` | Numeric | Prior completed bookings may reflect reliability: **0 previous non-canceled → 42.48%**, **1 → 7.96%**. Timing remains uncertain. | **[NOTE: Add screenshot of `tables/previous_non_canceled.csv`; highlight rows 0 and 1.]** |
| `ReservedRoomType` | Categorical | The **originally reserved**, rather than later assigned, room type may relate to behavior: type A **43.60%** versus D **35.29%**. | **[NOTE: Add screenshot of `tables/reservedroomtype.csv`; highlight A and D.]** |
| `Meal` | Categorical | Initially booked meal plan may relate to booking type: BB **42.84%** versus SC **37.66%**; small classes should be treated cautiously. | **[NOTE: Add screenshot of `tables/meal.csv`; highlight BB and SC, and note the small FB group.]** |

**Additional useful EDA evidence:** `tables/lead_time_by_segment.csv` / `figures/lead_time_by_segment.png` demonstrate feature interactions; `tables/segment_by_year.csv` / `figures/segment_by_year.png` test whether segment patterns persist across years. **[NOTE: Add one interaction figure screenshot to show that our EDA goes beyond one-variable comparisons.]**

**Excluded/postponed options and reasons:**

| Columns | Decision |
|---|---|
| `ReservationStatus`, `ReservationStatusDate` | **Excluded:** reveal final reservation outcome/status (direct leakage). |
| `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList` | **Excluded:** may depend on events after booking creation. |
| `DepositType`, `ADR` | **Excluded for the initial classifier:** strong timing/target-proxy concerns; avoid using information not validated at the stated prediction point. |
| `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` | **Postponed:** requests may change after booking creation. |
| `Agent`, `Company`, `Country` | **Postponed:** identifier/high-cardinality or missingness complications. |
| Additional arrival-date details | **Postponed:** first baseline uses month as a compact seasonal input. |
| Raw adults/children/babies and weekend/weekday night columns | **Replaced by engineered totals** in our initial 12-feature shortlist; components are not claimed to be useless. |

## 5. Preprocessing and validation choices

| Choice | Our implementation | Why |
|---|---|---|
| Feature engineering | `TotalGuests`, `TotalNights` computed in `features.py` | Captures party size and stay length without treating component counts as separate predictors in the first baseline. |
| Missing numeric values | Median imputation | Simple, less sensitive to extreme values than the mean. |
| Numeric scaling | `StandardScaler` after imputation | Logistic Regression benefits from comparably scaled numerical predictors; `scale_numeric=False` supports later tree-model tests. |
| Categorical preparation | Strip whitespace, fill unknown text, `OneHotEncoder(handle_unknown="ignore")` | Prevents spacing artifacts and allows unseen categories at validation/inference time. |
| Binary handling | Pass `IsRepeatedGuest` through as 0/1 | No one-hot encoding required for a single indicator. |
| Holdout | 80/20 stratified random split, seed `42` | Preserves the target distribution and reserves a final evaluation sample. |
| Primary validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | Provides repeatable, class-balanced development-fold comparisons. |
| Robustness validation | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | Keeps identical 12-feature patterns together *within each development fold*. |
| Baseline | `LogisticRegression(max_iter=1000)` with default regularization | An interpretable, untuned linear reference before trying more complex models. |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC; per-fold mean and standard deviation | Cancellation detection needs more than accuracy; recall captures missed cancellations. |

**What is and is not fitted on the full data:** `features.py` applies deterministic sums/text normalization before splitting. `preprocessing.py` **only constructs an unfitted transformer**. `train_baseline.py` places that transformer inside sklearn's `Pipeline`, which `cross_validate` refits **separately on the training rows of each fold**. Validation rows are transformed by the fitted fold pipeline; final-test rows are not used in CV. Thus, the current baseline **does not fit a common StandardScaler or OneHotEncoder on the whole dataset**. The separate upstream shared-cleaning issue is recorded in the leakage audit.

**Reproducibility:** `split.py` writes `outputs/classification/tables/split_manifest.csv` containing row indices, development/test membership, and fold labels. It is an assignment record, not a fitted model. Group-aware CV within development does **not** make the final random holdout group-disjoint.

## 6. Untuned baseline results and interpretation

Results below are **five-fold development-validation means**, not final test scores. The original dataset (`cleaned2`) is our **current baseline**; `cleaned3` is only an experiment.

| Dataset / CV | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **cleaned2 — stratified (primary)** | **0.7346** | **0.7762** | **0.5126** | **0.6174** | **0.7797** |
| cleaned2 — group-aware | 0.7333 | 0.7738 | 0.5112 | 0.6156 | 0.7770 |
| cleaned3 — stratified (exploratory) | 0.7196 | 0.6193 | 0.1781 | 0.2766 | 0.7113 |
| cleaned3 — group-aware (exploratory) | 0.7181 | 0.6087 | 0.1777 | 0.2750 | 0.7101 |

**Primary baseline variability (cleaned2, stratified):** accuracy SD **0.0049**, precision **0.0101**, recall **0.0083**, F1 **0.0077**, ROC-AUC **0.0040**. Sources: `tables/logreg_cleaned2_stratified_folds.csv`, `tables/logreg_cleaned2_stratified_summary.csv`, `tables/logreg_cleaned2_group_aware_summary.csv`, and `tables/dataset_baseline_comparison.csv` where available.

**Interpretation:** At threshold `0.50`, the primary classifier identifies about **51%** of cancellations, despite **78%** precision. Thus, recall is a key improvement target. Group-aware validation gives similar means but can vary more across folds. Removing duplicates also changes the outcome frequency (41.79% → 30.10%), so scores across the two datasets are **not** a controlled same-holdout comparison and cannot prove that keeping duplicates always generalizes better.

**Optional diagnostic evidence already produced for cleaned2:** `figures/cleaned2_logreg_oof_confusion_matrix.png`, `figures/cleaned2_logreg_oof_precision_recall.png`, `tables/logreg_cleaned2_threshold_diagnostics.csv`. The out-of-fold threshold experiment shows, for example, at threshold `0.30` recall **0.8565** and precision **0.5242**; it is **diagnostic only**, not a finalized threshold.

**[NOTE: Add screenshot of `tables/logreg_cleaned2_stratified_summary.csv` (primary model result), a screenshot of `tables/dataset_baseline_comparison.csv` (duplicate experiment), and optionally `figures/cleaned2_logreg_oof_confusion_matrix.png`.]**

## 7. Leakage audit — prediction-time and experimental integrity

**Prediction-time audit:** We deliberately excluded direct outcome columns. The remaining 12 features are provisional because the published dataset may include values assembled or updated after booking creation. In particular, `PreviousCancellations`, `PreviousBookingsNotCanceled`, and `IsRepeatedGuest` must represent **history known before the booking**; the extraordinary one-previous-cancellation rate is an unresolved warning, not proof of leakage. Booking-category fields (`CustomerType`, `Meal`, channels) also need availability checks.

**Automated `LeakageChecker` results on current `H2_cleaned2.csv`:**

| Check | Result / what it proves |
|---|---|
| Feature allowlist | **PASS:** exactly the 12 approved predictors enter `X`; forbidden outcome columns are not present. |
| Source-row disjointness | **PASS:** development and test have disjoint source row indices. |
| Predictor-pattern overlap | **WARN:** **9,611 / 15,833 (60.7%)** test records have the same selected-feature pattern as at least one development record. Legitimate bookings may match; this does not prove the source rows are identical. |
| Saved split manifest | **PASS:** current development/test and stratified fold assignments match the saved manifest. |
| Group-aware CV | **PASS:** selected-feature patterns do not cross train/validation boundaries *inside group-aware folds*. |
| Actual fold fitting | **PASS:** traced model fits use only their designated training indices; scaler counts and means match the training fold. |
| Negative control | **PASS:** a deliberately leaking **synthetic** train+test fit raises `LeakageError`, showing the detector can detect that kind of violation. |

**Four safeguards implemented in code:** (1) disjoint source-row checks (plus fingerprint warning), (2) allowlist of selected predictor columns, (3) split-manifest and actual fitted-row verification, (4) fitted-scaler inspection plus negative-control test. `leakage_checker.py` raises `LeakageError` for violations it can detect; `--strict-fingerprints` also treats any cross-holdout predictor-pattern match as an error. The normal run deliberately **warns** rather than failing on these matches.

**Known limitations:**

- The shared `cleaned2` file was partially prepared **before** our split, including a dataset-wide numerical imputation for four `Children` entries; this is a genuine *upstream* information-flow concern even though our CV scalers themselves are fit correctly.
- Full-dataset exploratory feature–outcome inspection happened before final holdout reservation. Future selection must not be driven by repeated test-set results.
- A source-row split is not necessarily a customer-level split; no reliable customer ID is available. Same predictor patterns appear on both sides of the final holdout.
- The checker cannot prove booking-creation-time availability, detect every indirect target proxy, or certify how the shared CSV was originally cleaned. Its fit checks apply to the **new models it audits**, not necessarily every prior unsaved run.

**[NOTE: Add screenshot of terminal output from `python -m classification.leakage_checker`, showing the PASS results, the 9,611-pattern WARN, and the limitations line.]**

## 8. Milestone 1 scope, setup, and reproducibility

**Classification work completed:** source-feature EDA, 12-feature selection and justification, categorical/numeric preprocessing definitions, seeded 80/20 split, both five-fold CV strategies, one untuned supervised classifier, baseline/duplicate experiment reporting, and preliminary leakage checks. The final test set remains reserved for later evaluation.

**Team-level requirement check (to complete in the main project progress report):** the project specification also requires shared H2 inspection and cleaning, **one untuned ADR regression model**, **one initial clustering algorithm**, preliminary leakage risks for all three tasks, a modular GitHub repository, member contributions, and environment/run instructions. **The classification README does not certify regression/clustering results**; the responsible teammates must supply their actual metrics and findings.

**Run from the repository root:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m classification.eda
python -m classification.features
python -m classification.split
python -m classification.train_baseline
python -m classification.leakage_checker
```

Make sure both `classification/features.py` and `classification/split.py` select **`H2_cleaned2.csv`** before the run; `eda.py` already uses this variant. `train_baseline.py` automatically uses the path imported from `split.py`. `outputs/classification/tables/` contains the EDA summaries, split assignments, and CV metrics; `outputs/classification/figures/` contains EDA and diagnostic figures. Some optional diagnostic files were generated by a separate development diagnostic run rather than by `train_baseline.py` itself.

**Reproduction/ownership details to fill in before submitting the team ZIP:**

| Item | Value to enter |
|---|---|
| GitHub repository URL | **[NOTE: Insert team's repository link.]** |
| Submitted commit SHA | **[NOTE: Insert `git rev-parse HEAD` from the submitted version.]** |
| Classification contributors | **[NOTE: Insert each actual contributor's name, GitHub username, and work completed.]** |
| Regression/clustering contributors | **[NOTE: Insert confirmed names, work and baseline results in the team report.]** |
| Environment | Python environment and `requirements.txt` from the main repository. |

The Milestone 1 submission is one repository ZIP with source, dependencies, the report, relevant CSV/figure outputs, repository link and commit SHA. Do **not** include fitted model binaries in the submission ZIP.

## 9. Our plan for Milestone 2

We will **extend the existing working classification pipeline**, not rebuild it from scratch.

1. **Improve our evaluation and feature decisions.** Establish one consistent comparison protocol, including a same-holdout experiment for duplicate-retaining versus deduplicated *training* data. Investigate the unusual guest-history relationships and compare models **with and without** history-related features. Keep the final test set out of these choices.
2. **Add two complementary classifiers.** Implement **Random Forest** (nonlinear bagged trees) and **XGBoost** (boosted trees), alongside Logistic Regression. Run all three first with reasonable untuned configurations using our development-fold validation and shared metrics.
3. **Tune and compare.** Use `RandomizedSearchCV` (or a small `GridSearchCV` when suitable) inside preprocessing/model pipelines to tune Logistic Regression regularization/class weighting, Random Forest depth/leaf size/tree count, and XGBoost depth/learning rate/estimators. Record search spaces, best settings, CV scores and training times. Investigate cancellation thresholds on development predictions rather than choosing them using final test scores.
4. **Test PCA and LDA where useful.** Fit dimensionality reduction *inside* training folds after appropriate encoding/scaling. Compare **at least two supervised models** with and without PCA; report explained variance and justify the retained components. For classification, compare original features, PCA, and **LDA** for at least one classifier (binary LDA has at most **one** discriminant component).
5. **Choose and evaluate final models.** Compare baseline and tuned models using recall, precision, F1, ROC-AUC, stability and complexity; agree on a cancellation-detection trade-off. Then fit the selected pipeline on development data and evaluate **once** on the reserved test set. Save the trained preprocessing + estimator pipeline for inference.
6. **Build the unified application with the team.** We plan to use **Streamlit**: a dataset/EDA page, a cancellation-prediction form with probability, an ADR-prediction form, a clustering/segment view, and a results page for baseline/tuning/model comparisons. The app will **load saved pipelines** for prediction and provide access to training/tuning runs or their recorded results, rather than silently retraining on every user interaction.
7. **Integrate and test.** Coordinate feature-input formats and artifact paths with regression/clustering teammates, document reproducible commands, verify example predictions and visuals, and prepare the final comparison report and demonstration.

**Milestone 2 deadline:** December 11, 2026. The classification-specific plans above are ours; the team must separately complete and integrate the required regression and clustering model comparisons, optimization and leakage audits.

## 10. Evidence index

All paths below are relative to the repository root. The existing generated outputs provide the data for screenshots requested above.

| Purpose | File(s) |
|---|---|
| Feature-level EDA | `outputs/classification/tables/{lead_time,total_guests,total_nights,arrival_month,market_segment,distribution_channel,customer_type,repeated_guest,previous_cancellations,previous_non_canceled,reservedroomtype,meal}.csv` |
| EDA plots | `outputs/classification/figures/{lead_time,total_guests,total_nights,arrival_month,market_segment,repeated_guest,previous_cancellations}.png` |
| Interaction EDA | `outputs/classification/figures/{lead_time_by_segment,arrival_month_by_year,segment_by_year}.png` |
| Primary fold metrics | `outputs/classification/tables/logreg_stratified_folds.csv`, `logreg_stratified_summary.csv`, `logreg_group_aware_summary.csv` (these generic names reflect whichever dataset was run last) |
| Version-labeled comparisons | `outputs/classification/tables/logreg_cleaned2_stratified_summary.csv`, `logreg_cleaned2_group_aware_summary.csv`, `dataset_baseline_comparison.csv` |
| CV assignments | `outputs/classification/tables/split_manifest.csv` |
| Additional development diagnostics | `outputs/classification/figures/cleaned2_logreg_oof_confusion_matrix.png`, `cleaned2_logreg_oof_precision_recall.png` and `outputs/classification/tables/logreg_cleaned2_threshold_diagnostics.csv` |
| Leakage-check evidence | Terminal output of `python -m classification.leakage_checker` (or `outputs/classification/leakage_audit_log.txt`, if saved) |

*Note: some evidence tables/figures (e.g. `meal.csv`, `reservedroomtype.csv`, and optional diagnostics) were generated as supplementary analysis and are not all produced by the current `eda.py` / `train_baseline.py`. Preserve these output files when assembling the final submission.*
