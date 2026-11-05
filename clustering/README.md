# Clustering — Milestone 1 Progress Report

**Project:** CSE382 — City Hotel Booking Demand (H2)  
**Task:** Group bookings with similar booking characteristics using unsupervised learning.  
**Current configured dataset:** `data/cleaned/H2_cleaned3.csv` — deduplicated version.  
**Current selected candidate:** K-Means with **K=8**; **K=3** remains the simple baseline.  
**Status:** Feature selection, preprocessing, K-Means training, internal evaluation, cluster profiling, and recorded K comparisons are implemented. DBSCAN, Gaussian Mixture Models (GMM), automated tuning, and visualization remain unfinished.

## 1. Overview

We developed a clustering workflow to discover groups of hotel bookings with similar lead times, stay lengths, party composition, and booking categories. These groups can help the team describe booking patterns and investigate possible customer segments.

Unlike classification, clustering does not use `IsCanceled` as a target. We do not train the model to predict cancellation, and cluster numbers do not represent cancellation risk. A cluster label is an arbitrary group identifier; a business name must be justified using its original booking values and category distributions.

This report follows the structure of the classification progress report supplied as a reference. Its implementation details come from the clustering source files, and its numerical experiment results come from [TESTING_NOTES.md](TESTING_NOTES.md). Those recorded experiments were not rerun while preparing this document.

## 2. Data and cleaning decisions

The shared cleaning work lives in [`../clean_data.py`](../clean_data.py), outside the clustering package. Its code fills missing `Children` values with the dataset median, fills missing `Country` with `Unknown`, removes zero-guest bookings, parses reservation-status dates, removes ADR=5400, and normalizes `Agent`/`Company` placeholders. It creates a duplicate-retaining version and a separate version with exact duplicate rows removed.

| Dataset | Role in clustering | Available evidence |
|---|---|---|
| `data/precleaned/H2.csv` | Original City Hotel source | Present in this checkout. |
| `data/cleaned/H2_cleaned2.csv` | Earlier experiments with duplicates retained | Testing notes record 79,162 bookings. |
| `data/cleaned/H2_cleaned3.csv` | Current input selected by `main.py`; exact duplicates removed | Experiments and profiles are recorded in the testing notes. |

The two cleaned CSV files are currently absent from `data/cleaned/`. Consequently, the current workflow needs its configured cleaned input before it can run end to end.

Duplicate removal changes how frequently booking patterns occur and therefore changes the clustering problem. Identical rows are not necessarily proof of the same real booking. We treat deduplication as an experimental choice, and do not compare scores across the two datasets as if only K changed.

**Path limitation:** The current shared cleaning script resolves paths relative to the repository's parent and uses `data/H2.csv`, rather than this checkout's `data/precleaned/H2.csv`. Its paths need review before using it to regenerate the expected cleaned files. The clustering workflow does not call this script automatically.

## 3. Files, functions, and why we created them

```text
clustering/
|-- __init__.py
|-- features.py
|-- preprocessing.py
|-- models.py
|-- evaluation.py
|-- profiling.py
|-- tuning.py
|-- visualization.py
|-- TESTING_NOTES.md
`-- README.md
```

| File | What it does and why it exists | Current status / output |
|---|---|---|
| `__init__.py` | Marks the folder as a Python package so other scripts can import `clustering.*`. | Empty package marker; no analysis output. |
| `features.py` | Defines the approved numeric, binary, and categorical inputs in one place. This keeps training and profiling consistent and prevents unrelated source columns from entering the distance calculation. | Implemented; returns feature configuration and an 11-column DataFrame. |
| `preprocessing.py` | Converts mixed booking fields into numerical features suitable for distance-based clustering, and provides artifact save/load helpers. | Implemented; returns fitted preprocessing and an encoded/scaled DataFrame. |
| `models.py` | Separates model creation and training from data preparation, allowing the same prepared matrix to be reused for experiments. | K-Means implemented; DBSCAN and GMM functions are placeholders. |
| `evaluation.py` | Measures separation/compactness and cluster sizes, then combines experiment results into a table. This supports choosing K using several criteria. | Implemented for the current K-Means workflow. |
| `profiling.py` | Reconnects labels to readable, unscaled booking values so clusters can be explained. | Implemented; returns size, median, repeated-guest, and category tables plus descriptions. |
| `tuning.py` | Reserves interfaces for systematic parameter searches and ranking experiments. | All five functions are placeholders; no automated tuning output yet. |
| `visualization.py` | Reserves interfaces for a two-dimensional PCA view and metric comparison plots. | All three functions are placeholders; no figures generated yet. |
| `TESTING_NOTES.md` | Records datasets, K comparisons, profile observations, and reasons for keeping or rejecting candidates. | Contains historical experiment results and a template for future runs. |
| `README.md` | Explains the workflow, decisions, results, and limitations for readers and team members. | This progress report. |

### Main functions

| Function | Behavior and returned output |
|---|---|
| `features.get_feature_config()` | Returns copies of the three feature lists. |
| `features.select_clustering_features(df)` | Selects the 11 inputs in a consistent order and returns a copy. |
| `preprocessing.build_preprocessor()` | Creates an unfitted `ColumnTransformer`: numeric scaling, binary passthrough, and categorical one-hot encoding. |
| `preprocessing.fit_transform_features(df)` | Fits preprocessing and returns `(preprocessor, X_processed)`; processed columns have generated feature names and retain the input row index. |
| `preprocessing.transform_features(X_new, fitted_preprocessor)` | Applies existing fitted transformations to new rows; returns the transformer's matrix output rather than a named DataFrame. |
| `preprocessing.save_clustering_artifacts(artifacts, artifact_dir)` | Saves each supplied dictionary value as `<name>.joblib`. It runs only when explicitly called. |
| `preprocessing.load_clustering_artifacts(artifact_dir)` | Loads `*.joblib` files into a dictionary keyed by filename stem. |
| `models.make_kmeans(config)` | Creates an unfitted K-Means estimator from the supplied settings. |
| `models.fit_kmeans(X_processed, config)` | Fits K-Means and returns the fitted estimator and one label per input row. |
| `evaluation.compute_cluster_metrics(X, labels, algorithm)` | Returns silhouette, Davies-Bouldin, and fitted K-Means inertia; rejects fewer than two clusters or as many clusters as rows. |
| `evaluation.summarize_cluster_sizes(labels)` | Returns `cluster`, `count`, and `percentage`, sorted by cluster label. |
| `evaluation.compare_models(results)` | Converts run dictionaries into one DataFrame, placing `model` first when supplied. It does not rank or choose a winner. |
| `profiling.attach_cluster_labels(df, labels)` | Copies original data and assigns labels by row position; checks that row and label counts match. |
| `profiling.build_cluster_profiles(labeled_df)` | Returns cluster sizes, numeric medians, repeated-guest percentages, and within-cluster distributions for all four categorical features. |
| `profiling.describe_cluster_differences(profiles)` | Returns short descriptions containing cluster size, lead time, stay summary, adults, and children. Business names remain human-chosen. |

The remaining model interfaces are `make_dbscan()`, `fit_dbscan()`, `make_gmm()`, and `fit_gmm()`. The tuning interfaces are `tune_kmeans()`, `rank_experiments()`, `estimate_eps_candidates()`, `tune_dbscan()`, and `tune_gmm()`. The visualization interfaces are `fit_pca_projection()`, `plot_clusters_2d()`, and `plot_metric_comparison()`. Each currently contains `pass` and returns no useful result.

### Supporting files outside this folder

| File / directory | Connection to clustering |
|---|---|
| `../main.py` | Loads the cleaned CSV, selects/preprocesses features, fits K-Means, calculates metrics, builds profiles, and prints results. Also exposes a K-comparison function. |
| `../clean_data.py` | Shared upstream cleaning; review its paths before running. |
| `../config.yaml` | Placeholder configuration. The current pipeline does **not** read this file. |
| `../requirements.txt` | Lists runtime dependencies: NumPy, pandas, scikit-learn, Matplotlib, PyYAML, and joblib. Pytest is not listed here. |
| `../tests/test_clustering.py` | Four placeholder tests for leakage, feature matrices, label alignment, and reproducibility. |
| `../outputs/clustering/` | Intended location for generated tables/figures; currently contains only `.gitkeep`. |
| `../artifacts/clustering/` | Intended location for fitted objects; currently contains only `.gitkeep`. |

## 4. Our 11 selected features

| Feature | Type | Why we included it |
|---|---|---|
| `LeadTime` | Numeric | Distinguishes last-minute bookings from advance planners. |
| `StaysInWeekendNights` | Numeric | Represents the weekend component of stay duration. |
| `StaysInWeekNights` | Numeric | Represents the weekday component and helps distinguish short and long stays. |
| `Adults` | Numeric | Captures adult party size, such as solo, couple, and larger-party bookings. |
| `Children` | Numeric | Helps identify family booking patterns. |
| `Babies` | Numeric | Captures a rare but potentially meaningful party-composition pattern. |
| `IsRepeatedGuest` | Binary | Distinguishes repeated-guest bookings from first-time bookings. |
| `Meal` | Categorical | Represents booked meal-package preferences. |
| `ReservedRoomType` | Categorical | Captures the originally reserved room category. |
| `MarketSegment` | Categorical | Represents the commercial segment through which a booking is made. |
| `DistributionChannel` | Categorical | Represents how a booking reaches the hotel. |

The feature selector uses an explicit allowlist. Fields such as `IsCanceled`, `ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, and `BookingChanges` do not enter the clustering matrix. This keeps groups focused on the selected booking characteristics rather than final reservation outcomes.

These choices express our intended booking-time interpretation; the code does not prove that every stored value was available at booking creation. In particular, the timing of repeated-guest information and potentially updated categories still needs source-semantic review. We have not implemented a clustering leakage audit.

## 5. Preprocessing and data flow

| Input kind | Processing | Reason |
|---|---|---|
| Six numeric features | `StandardScaler` | Reduces domination by features with larger numerical scales, such as lead time. |
| One binary feature | Passthrough | Keeps `IsRepeatedGuest` as 0/1. |
| Four categorical features | `OneHotEncoder(handle_unknown="ignore")` | Creates numerical category indicators without imposing an artificial order; unseen categories can be transformed. |
| Other source columns | Dropped by the transformer | Keeps the distance calculation restricted to the approved feature set. |

There is no imputer, log transform, or category-text normalization in the clustering preprocessor. It relies on suitable upstream data; missing numeric values can prevent K-Means fitting. One-hot encoding expands the 11 source fields into a larger numerical matrix. `fit_transform_features()` converts sparse output to a dense DataFrame, which can increase memory use.

```text
Prepared H2_cleaned3.csv
    -> features.py: select 11 booking inputs
    -> preprocessing.py: fit scaling and encoding
    -> models.py: fit K-Means and assign labels
         |-> evaluation.py: metrics and cluster sizes
         `-> profiling.py: labels + original rows -> readable summaries
    -> main.py: print selected results
```

The current exploratory workflow fits preprocessing and K-Means on the full configured dataset. There is no train/test split, cross-validation, or held-out generalization measurement. Internal metrics describe structure in the fitted data. New-booking support would use the saved preprocessor followed by the fitted model, but that inference workflow is not wired into `main.py`.

## 6. K-Means settings and evaluation

The common experiment settings are `init="k-means++"`, `n_init=10`, `max_iter=300`, and `random_state=42`. Multiple starts reduce sensitivity to initialization; the fixed seed supports repeatability for a fixed dataset, environment, and configuration.

**Configuration detail:** `main.py` overrides the cluster count with `SELECTED_K=8`. The dictionary named `BASELINE_KMEANS_CONFIG` currently contains `n_clusters=7`, while `make_kmeans()` falls back to 3 if that setting is missing. `config.yaml` also says 3 but is unused. Some baseline comments/docstrings are stale: running `main.py` currently trains **K=8**.

| Metric / check | Meaning and interpretation |
|---|---|
| Silhouette | Compares within-cluster closeness with separation from other clusters. Higher is better; values near zero indicate overlapping structure. The current function samples up to 5,000 rows with seed 42. |
| Davies-Bouldin | Measures cluster similarity using compactness and separation. Lower is better; calculated on the full processed matrix. |
| Inertia | Sum of squared distances to assigned K-Means centers. Lower is useful within a consistent setup, but it generally decreases as K increases and cannot choose K alone. |
| Cluster sizes | Shows whether a result has a dominant group or tiny groups requiring investigation. Balance alone does not establish quality. |
| Original-value profiles | Helps determine whether groups describe useful and distinct booking patterns. |

The current metric function requires a fitted model with `inertia_`. It is not a general DBSCAN/GMM evaluator, and DBSCAN noise handling has not been implemented.

## 7. Recorded experiments and observations

The full K tables are in [TESTING_NOTES.md](TESTING_NOTES.md). The following results summarize that record; they are not newly generated outputs.

### 7.1 Earlier duplicate-retaining data: `H2_cleaned2.csv`

| K | Silhouette | Davies-Bouldin | Inertia | Smallest cluster | Largest cluster |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.1413 | 1.9945 | 474,212.04 | 6.45% | 49.40% |
| 4 | 0.2074 | 1.2971 | 413,210.80 | 0.47% | 73.61% |
| 8 | 0.2034 | 1.3147 | 280,176.25 | 0.47% | 27.65% |
| 9 | 0.2026 | 1.1814 | 260,265.66 | 0.003% (2 bookings) | 25.30% |

K=4 had the highest recorded silhouette but left most bookings in one cluster. K=8 was retained as a more balanced candidate. Several larger K values produced two-booking groups, raising concerns about outlier isolation or excessive segmentation.

### 7.2 Current configured data: `H2_cleaned3.csv`

| K | Silhouette | Davies-Bouldin | Inertia | Smallest cluster | Largest cluster |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.1534 | 1.9447 | 316,779.58 | 9.20% | 47.33% |
| 5 | 0.1811 | 1.5062 | 249,918.91 | 0.69% | 50.19% |
| 7 | 0.1731 | 1.4315 | 210,999.79 | 0.69% | 29.69% |
| 8 | 0.1800 | 1.4102 | 200,461.25 | 0.69% | 27.69% |
| 9 | 0.1800 | 1.2609 | 186,912.35 | 0.004% (2 bookings) | 27.69% |
| 14 | 0.1867 | 1.3490 | 149,015.23 | 0.004% (2 bookings) | 16.92% |

We keep K=3 as the simple reference and K=8 as the current candidate. K=8 improves Davies-Bouldin over K=3 and spreads the larger groups more evenly than K=5, while retaining a similar silhouette to K=5. Its silhouette of 0.1800 still suggests substantial overlap; this is not evidence of perfectly separated customer types. Higher K can improve some metrics while creating very small groups, so a numerical maximum is not enough to select the final model.

### 7.3 What the profiles suggest

| K | Recorded profile observations |
|---:|---|
| 3 | Family group (9.20%, median one child), short-stay group (47.33%), and earlier long-stay group (43.47%). |
| 5 | Advance planners (22.84%, median lead time 170 days), short-stay one-adult bookings (17.59%), a general group (50.19%), baby bookings (0.69%), and families (8.69%). |
| 8 | More detailed patterns: short-stay one-adult/repeated guests, advance planners, general bookings, baby bookings, two-child families, short-stay couples, three-adult bookings, and long stays. |

The notes identify a persistent 369-booking group with median one baby at K=5 and K=8. That supports investigating a rare booking segment, but does not prove the group is stable or useful. Standardizing rare count features can make them influential in Euclidean distance. Full category distributions, outlier sensitivity, and repeated-seed stability still need review before final naming or selection.

## 8. Understanding the outputs

`run_clustering_baseline()` in `main.py` returns the following dictionary, despite currently using the selected K=8 candidate:

| Key | Output and how to read it |
|---|---|
| `preprocessor` | Fitted scaling/encoding rules needed to transform future bookings consistently. |
| `model` | Fitted K-Means estimator containing cluster centers and inertia. |
| `labels` | One cluster number per booking, in input row order. Numbers are identifiers, not rankings. |
| `metrics` | Dictionary containing `silhouette`, `davies_bouldin`, and `inertia`. |
| `cluster_sizes` | Counts and percentages for each label; percentages describe shares of the fitted dataset. |
| `profiles` | Dictionary of seven tables: `cluster_sizes`, `numeric_medians`, `repeated_guest_percentage`, `Meal_percentages`, `ReservedRoomType_percentages`, `MarketSegment_percentages`, and `DistributionChannel_percentages`. |
| `descriptions` | Text summaries keyed by cluster identifier. |

The command-line entry point prints metrics, cluster sizes, numeric medians, repeated-guest percentages, and descriptions. It calculates categorical profile tables but does not print them. The description's “median stay” currently adds the separate weekday and weekend medians; this is not necessarily the median of each booking's total nights.

`run_clustering_comparison()` returns a DataFrame with `model`, `K`, the three metrics, and smallest/largest cluster percentages. Its default range is K=7 through K=8 inclusive. It is not called by `main()` and does not automatically reproduce the full historical search.

**Saved output status:** `main.py` does not save CSV tables, images, or model artifacts. The output and artifact directories currently contain only `.gitkeep`. The joblib helpers exist, but must be explicitly called. No saved numerical outputs or figures should be claimed beyond the recorded testing notes.

## 9. Running the workflow and tests

Run from the repository root in PowerShell. The project `.venv` has pytest installed from the recent test setup; install the runtime dependencies before executing the clustering workflow:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After placing the prepared data at `data/cleaned/H2_cleaned3.csv`:

```powershell
.\.venv\Scripts\python.exe main.py
```

To run the clustering tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_clustering.py -v
```

The latest test run reported **4 passed in 0.05s**. However, `test_no_leakage`, `test_feature_matrix`, `test_label_alignment`, and `test_reproducibility` all contain only `pass`. They verify no behavior and do not establish clustering correctness. The experiment comparisons in `TESTING_NOTES.md` are separate from these automated tests.

## 10. Next implementation steps

1. Confirm the team's dataset and duplicate policy, fix shared cleaning paths, and preserve the exact data/version used for comparisons.
2. Consolidate K settings and remove stale baseline comments so the documented configuration matches execution.
3. Replace placeholder tests with checks for feature exclusion/order, finite processed values, label alignment, artifact round trips, and repeatability.
4. Save run settings, metrics, cluster-size tables, full profiles, fitted preprocessing, and the fitted model. Record dependency versions and runtime.
5. Complete the tuning functions and check multiple seeds, tiny-group behavior, feature weighting, and outlier sensitivity before choosing a final K.
6. Implement DBSCAN and GMM with algorithm-appropriate evaluation, including noise coverage for DBSCAN and BIC/AIC for GMM.
7. Implement PCA visualization and comparison plots. Treat two-dimensional PCA as a view of the result; it does not establish separation in the full feature space.
8. Review complete profile tables with the team and justify provisional segment names before making business recommendations.
