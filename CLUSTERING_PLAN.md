# Task C — Unsupervised Clustering
**CSE382 | City Hotel (H2) | Implementation Plan**

**Dataset:** H2 only (79,330 bookings). **Scope:** clustering across Milestones 1 and 2. This document records implementation decisions, not code. Algorithm choices below are recommendations; the official specification requires **at least three algorithms, each baseline and tuned**, justified preprocessing, internal metrics, 2D visualization, cluster interpretation, and leakage control.

## 1. Objective and Data Boundaries

**What is fitted?** Each booking is transformed into a numeric vector; K-Means learns centroids, DBSCAN discovers density-connected groups, and GMM estimates mixture parameters. No labeled target is provided. Cluster IDs are arbitrary; we interpret the groups after fitting. This is still model training, just unsupervised.

**How are bookings represented?** Rows = bookings; columns = selected numeric features plus encoded categorical indicators. For example, `Meal=BB` becomes a one-hot indicator, and `LeadTime` is transformed/scaled. The final feature matrix may have more columns than the original selected fields. PCA can project that matrix into two dimensions for plotting, without determining the clusters.


Discover interpretable **booking segments** (not necessarily unique customer identities) to describe differences in booking timing, stay patterns, guest composition, and booking channels. Clustering is **unsupervised**: no outcome labels are used to build or select clusters.

**Discovery point:** assume the moment a booking is first created. Use only fields available then. Because H2 is a retrospective dataset, confirm that fields such as room preference, deposit type, and special requests reflect their original values; otherwise exclude them or report a sensitivity experiment.

**Never use** `IsCanceled` or `ADR` as clustering inputs, in PCA, or during tuning. These are the other tasks' targets; including them makes similarity depend on known outcomes. They may be examined *after* clustering for descriptive analysis, without revising the clusters.

## 2. Verified H2 Dataset Audit

The **original uploaded H2.csv** (not the team-cleaned version) contains **79,330 rows and 31 columns**. Standard pandas CSV parsing shows **4 missing `Children` values and 24 missing `Country` values**. The file has **25,902 exact duplicate rows**, but these are not proof of duplicate booking events: removing them can change segment prevalence and density. Discuss and document the team’s deduplication policy before clustering.

Other checks from this source file: **167 rows have zero total guests** (treating missing `Children` as zero only for this audit); **331 have zero total nights**; `LeadTime` ranges from **0 to 629 days**. Investigate and log these rows before deciding whether to correct, retain, or exclude them. `Meal` values have trailing whitespace (for example `BB       `): strip categorical whitespace consistently. The source uses text `NULL` placeholders in `Country`, `Agent`, and `Company`; ensure the shared loader maps these correctly. `ArrivalDateMonth` contains English month names, not integers.

**Important:** These findings describe the raw file. Recheck row counts, missingness, categories, and duplicates on the **team-approved cleaned snapshot**; do not assume the numbers remain the same.

## 3. Feature Selection and Leakage Audit

| Decision | Fields | Reason |
|---|---|---|
| **Core: booking and stay** | `LeadTime`, `StaysInWeekendNights`, `StaysInWeekNights` | Advance planning, seasonality, and stay duration/type |
| **Core: guest composition** | `Adults`, `Children`, `Babies`, `IsRepeatedGuest` | Group size, family bookings, returning guests |
| **Core: preferences** | `Meal`, `ReservedRoomType` | Requested room, services, and amenities |
| **Core: booking channel** | `MarketSegment`, `DistributionChannel` | How bookings are sourced and arranged |
| **Exclude: outcomes** | `IsCanceled`, `ADR`, `ReservationStatus`, `ReservationStatusDate` | Targets or final booking results |
| **Exclude: later changes** | `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList` | May reflect events after booking creation |
| **Exclude by default** | `Agent`, `Company`, `ArrivalDateYear`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth` | High-cardinality IDs, period effects, or redundant timing detail |
| **Optional, after audit** | `ArrivalDateMonth`, `DepositType`, `CustomerType`, `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`, `Country`, `PreviousCancellations`, `PreviousBookingsNotCanceled` | Check booking-time provenance; geography needs careful encoding and history must predate the booking |

`ReservedRoomType` represents the **requested** room; `AssignedRoomType` represents the room eventually provided. The first is potentially useful at creation, the second is not. Verify creation-time availability of all retained fields; the table is a proposed feature contract, not proof of availability.

**Derived features:** optionally use `TotalNights`, `WeekendShare`, or `HasChildren`, but avoid simultaneously including redundant originals that would overweight the same behavior.

## 4. Shared Data-Cleaning Handoff vs. Clustering Preprocessing

**Single source of truth:** the data-cleaning team delivers `data/cleaned/H2_cleaned.csv`. Clustering **does not redo shared cleaning** or silently change the number of rows. The raw-file findings in Section 2 are audit questions for the cleaning team, not instructions to remove rows ourselves.

| Shared cleaning team owns | Clustering team owns |
|---|---|
| Consistent column names/types and category spellings/whitespace (`Meal` included) | Select booking-time features and exclude leakage-prone fields |
| Missing-value conventions and justified cleaning of invalid records | Decide whether a remaining missing feature needs a **fitted** imputer; report it rather than silently re-cleaning |
| Duplicate investigation and documented retention/removal policy | Preserve booking-to-label alignment and check how duplicate policy affects cluster sizes |
| Reproducible cleaned CSV, cleaning report, schema, row count | Encode chosen categorical features and scale chosen numeric features |
| Reusable encoding/scaling helper functions **if desired** | Fit and save the clustering-specific encoder/scaler and any PCA transformation |

**Why not one globally encoded/scaled CSV?** Classification, regression, and clustering may select different features or use different representations. Share the **code/utilities**, not a mandatory transformed matrix. For example, K-Means needs meaningful distances, so nominal categories should normally be one-hot encoded; tree models may not need scaling. *Data standardization* (consistent values and types) is shared; *statistical standardization* (`StandardScaler`, approximately mean 0 and standard deviation 1) is a model-specific fitted transformation.

**Handoff acceptance checks:** verify the delivered filename/schema, row count, null counts, category values, zero-guest/zero-night policy, duplicate policy, and whether any columns were dropped. Do not assume that all missing values have been imputed or that all duplicates are erroneous. Keep a stable row identifier/index. If the cleaning team supplies a final CSV with no missing values, the clustering pipeline does **not** need to fill them again; use explicit checks and only add a fallback imputer if the handoff contract permits it.

**Clustering-specific feature matrix:**

1. Start with these **11 original fields**: `LeadTime`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `Meal`, `ReservedRoomType`, `MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`.
2. Confirm each retained value is available at **booking creation**; drop it if its provenance cannot be justified. Never include `IsCanceled`, `ADR`, `ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, or `BookingChanges` as inputs. Keep `DaysInWaitingList` out initially.
3. Numeric: optionally `log1p(LeadTime)` (only for nonnegative values), then fit `StandardScaler` or `RobustScaler` on the selected numeric fields. Categorical: one-hot encode `Meal`, `ReservedRoomType`, `MarketSegment`, `DistributionChannel`; treat `IsRepeatedGuest` as binary. Avoid scaling the one-hot indicators indiscriminately.
4. Review **feature-group balance**: many dummy columns can outweigh a few numeric variables. Document any weights or category grouping. Consider `ArrivalDateMonth` as cyclic seasonality, `DepositType`, or special requests only in subsequent justified experiments.
5. Save the fitted transformer and its feature names/order. Use the **same representation and benchmark rows** for fair K-Means/DBSCAN/GMM comparisons; fit transformations only on the designated fitting data if using a separate holdout or validation protocol.

**Why scale?** Euclidean distance squares feature differences: a 200-day `LeadTime` difference contributes 40,000, versus 4 for a two-adult difference, before scaling. Scaling helps prevent units alone from defining similarity. PCA/LDA are **not** part of the shared cleaning output: PCA is optional for clustering input compression and used for 2D plots; LDA requires labels and is not part of unsupervised Task C.

## 5. PCA: When and Why

PCA has **two separate roles**:

- **Optional input compression:** fit PCA *before clustering* as a separate experiment to reduce dimensionality. Compare its clustering quality with the original engineered features; PCA can remove useful information.
- **Required 2D visualization:** after obtaining cluster labels, project the processed data onto two principal components and color points by cluster. This plot **illustrates** the result; it does not create the labels.

For compression, examine cumulative explained variance and initially test the smallest number of components retaining **90–95%** variance. Validate the choice using clustering quality and interpretability, not variance alone. Record the variance shown by PC1 and PC2 on plots.

**M1:** document the PCA plan; an exploratory plot is optional. **M2:** deliver the final 2D cluster visualizations and a justified PCA comparison. Coordinate with the team: the specification's separate supervised PCA/LDA requirements are not fulfilled by clustering plots alone.

## 6. Algorithms and Tuning

**Recommended set: K-Means + DBSCAN + Gaussian Mixture Model (GMM).** These compare centroid-based, density-based, and probabilistic clustering. Hierarchical clustering is a valid alternative to GMM.

| Algorithm | Baseline | M2 optimization | Main consideration |
|---|---|---|---|
| **K-Means** | `K=3` as an explicitly provisional baseline; fixed seed and multiple initializations | Try `K=2–12`; compare elbow/inertia, silhouette, Davies–Bouldin, cluster sizes, and stability | Fast enough for full H2 in normal circumstances; prefers compact clusters |
| **DBSCAN** | Set `min_samples=10`; choose initial `eps` from a k-distance plot | Search nearby `eps` values and `min_samples` (e.g. 5, 10, 20, 40) | Detects irregular groups and noise (`-1`); can be costly on 79k rows and high-dimensional data |
| **GMM** | `K=3`, diagonal covariance, fixed seed | Test `K=2–12`, `diag`/`tied`/feasible `full` covariance, regularization; select using BIC/AIC plus internal metrics | Produces soft membership probabilities; Gaussian assumption is approximate for encoded categorical data |

**Hierarchical alternative:** run Agglomerative Clustering on a reproducible, representative **5k–10k subset**, because pairwise-distance memory can become prohibitive on 79,330 records. Compare linkage settings and create a dendrogram where practical. If chosen as the third algorithm, report baseline and tuned results just like the others.

**Practical comparison rule:** benchmark all three on a shared sample and identical processed features. Also report full-data K-Means (and GMM if feasible) separately. For DBSCAN, report the percentage classified as noise; do not reward an apparently good score achieved by discarding most records.

## 7. Evaluation and Optimization

Do **not** use Accuracy, F1, or R²: there are no ground-truth cluster labels.

| Metric | Interpretation | Preferred direction |
|---|---|---|
| **Inertia / WCSS** | Sum of squared distances to assigned K-Means centroids; useful for elbow analysis | Lower for a fixed K; always decreases as K grows |
| **Silhouette Score** | Compares cohesion within a cluster against separation from other clusters; ranges from −1 to +1 | Higher |
| **Davies–Bouldin Index** | Average similarity of each cluster to its most similar alternative, combining spread and separation | Lower |
| **BIC / AIC (GMM)** | Fit-versus-complexity criteria for probabilistic mixture models | Lower, on the same representation and rows |

Use **Optuna or a documented parameter grid** with fixed seeds and a clearly defined search space. Reject degenerate solutions (one cluster, tiny unusable clusters, or excessive DBSCAN noise). Select a model based on **metrics, stability, interpretability, and computational cost** rather than a single score. Record every baseline and tuned configuration, runtime, and result.

## 8. Cluster Profiling and Business Meaning

After fitting, attach cluster labels to the **original unscaled rows** using stable row identifiers. For each cluster, report:

- Booking count and percentage of the dataset.
- Median `LeadTime`, stay length, adults/children, and special requests.
- Distributions of room type, meal, channel, market segment, and arrival season.
- Notable differences from the overall dataset and DBSCAN noise coverage where relevant.

Name segments **only after observing their actual characteristics**. Possible *hypothetical* names include **Advance Planners**, **Last-Minute Short Stays**, and **Family Leisure Bookings**. Do not call a cluster “high-value” without separate evidence; `ADR` is not used to construct the clusters.

## 9. Repository and Integration

```text
project/
├── README.md                     # Setup, run commands, GitHub link
├── requirements.txt              # Python dependencies
├── config.yaml                   # Paths, seeds, feature lists, search settings
├── main.py                       # Shared CLI/application entry point
├── data/
│   ├── precleaned/               # Original/initial H2 data snapshot
│   └── cleaned/                  # H2_cleaned.csv + shared cleaning report/schema
├── clustering/
│   ├── __init__.py               # Python package marker
│   ├── features.py               # Feature lists, schema validation, leakage guard
│   ├── preprocessing.py          # Clustering encoder/scaler; optional fitted imputer
│   ├── models.py                 # K-Means, DBSCAN, GMM constructors/fitters
│   ├── tuning.py                 # Search spaces, run experiments, select configs
│   ├── evaluation.py             # Metrics, noise policy, comparison tables
│   ├── profiling.py              # Unscaled cluster summaries and naming evidence
│   └── visualization.py          # PCA projection and result plots
├── artifacts/
│   └── clustering/               # Locally saved fitted models/pipelines
├── outputs/
│   └── clustering/               # Result tables and figures
└── tests/
    └── test_clustering.py        # Leakage, pipeline, and reproducibility checks
```

The clustering module should expose **run analysis, compare baseline/tuned results, view plots and profiles, and load saved artifacts** through `main.py`. DBSCAN does not natively predict clusters for new records; the application may display its fitted analysis instead of promising unsupported inference. Keep clustering preprocessing separate from the team's supervised pipelines.

## 10. Milestone Checklist

| Milestone | Deadline | Clustering deliverables |
|---|---|---|
| **M1** | **5 Nov 2026** | Clean-data agreement; feature/leakage audit; reusable preprocessing; baseline **K-Means**; initial metrics, sizes, and interpretation; GitHub commits and progress-report contribution |
| **M2** | **11 Dec 2026** | **Three algorithms**, each baseline + tuned; search spaces and results table; Silhouette/DBI; PCA 2D plots; profiles/personas; saved local artifacts; CLI/GUI integration; final report and reproducible run instructions |
| **Live discussion** | **18–24 Dec 2026** | Demonstrate clustering analysis and explain choices, leakage prevention, tuning, limitations, and individual contribution |

**Submission caveat:** The official PDF requires fitted artifacts to be **available locally for the discussion**, but **excluded from the submitted ZIP**. Include the commands needed to regenerate them. Commit work incrementally and identify the submitted commit SHA in the README.

## 11. Step-by-Step Coding Roadmap: Files and Functions

**Function names below are proposed interfaces, not implemented code.** Each function should have clear inputs/outputs and docstrings. Keep all clustering code inside `clustering/`; `main.py` orchestrates rather than duplicates it. **M1 = Steps 1–7; M2 = Steps 8–12.**

| Step | File | Functions to implement | Responsibility / expected output |
|---|---|---|---|
| **1. Accept handoff (M1)** | `clustering/features.py` | `validate_cleaned_schema(df)`, `audit_clustering_input(df)` | Read the **shared cleaned CSV** via `main.py`; verify required columns/types, row count, remaining nulls, and agreed duplicate policy. Raise/report issues; do not silently fix shared data. |
| **2. Select features (M1)** | `clustering/features.py` | `get_feature_config()`, `select_clustering_features(df)` | Apply the 11-column core selection, reject leakage fields, preserve row index. Return feature DataFrame `X_raw` and metadata. |
| **3. Build representation (M1)** | `clustering/preprocessing.py` | `build_preprocessor(numeric_cols, categorical_cols, config)`, `fit_transform_features(X_raw, preprocessor)`, `transform_features(X_new, fitted_preprocessor)` | One-hot encode selected nominal features, scale numeric features, optionally log-transform lead time, check finite numeric output and feature order. **Only impute if needed and allowed by the cleaned-data contract.** Return `X_processed` and fitted transformer. |
| **4. Fit K-Means (M1)** | `clustering/models.py` | `make_kmeans(config)`, `fit_kmeans(X_processed, config)` | Start with provisional `K=3`, fixed seed and `n_init`. Return fitted model and aligned cluster labels. |
| **5. Score (M1)** | `clustering/evaluation.py` | `compute_cluster_metrics(X, labels, algorithm, sample_size, seed)`, `summarize_cluster_sizes(labels)` | Compute sampled silhouette, Davies–Bouldin, sizes, and K-Means inertia where applicable; handle invalid one-cluster results. Return metric dictionary/table. |
| **6. Explain (M1)** | `clustering/profiling.py` | `attach_cluster_labels(df, labels)`, `build_cluster_profiles(labeled_df)`, `describe_cluster_differences(profiles)` | Align labels to **unscaled** cleaned rows; report booking timing, stays, party composition and category distributions. Name clusters only from observed evidence. |
| **7. Run baseline (M1)** | `main.py`, `tests/test_clustering.py` | `run_clustering_baseline(config)`, tests `test_no_leakage()`, `test_feature_matrix()`, `test_label_alignment()` | One CLI command loads the shared cleaned file and runs Steps 1–6; save metrics/profiles under `outputs/clustering/`. This completes the M1 baseline. |
| **8. Tune K-Means (M2)** | `clustering/tuning.py` | `tune_kmeans(X, search_config)`, `rank_experiments(results)` | Test K (e.g., 2–12), compare metrics, stability, cluster sizes and runtime; retain baseline vs tuned settings. |
| **9. Fit/tune DBSCAN (M2)** | `clustering/models.py`, `clustering/tuning.py` | `make_dbscan(config)`, `fit_dbscan(X, config)`, `estimate_eps_candidates(X, config)`, `tune_dbscan(X, search_config)` | Use k-distance guidance, tune `eps`/`min_samples`; report noise (`-1`) share and computational limits. Sample if needed. |
| **10. Fit/tune GMM (M2)** | `clustering/models.py`, `clustering/tuning.py` | `make_gmm(config)`, `fit_gmm(X, config)`, `tune_gmm(X, search_config)` | Compare components, covariance types, regularization, BIC/AIC and clustering metrics; return labels and membership probabilities. |
| **11. Visualize/compare (M2)** | `clustering/visualization.py`, `clustering/evaluation.py`, `clustering/profiling.py` | `fit_pca_projection(X)`, `plot_clusters_2d(coords, labels)`, `plot_metric_comparison(results)`, `compare_models(results)` | **PCA for visualization only** unless input compression is tested separately; create plots, side-by-side metrics, and final profiles on consistent benchmark data. |
| **12. Integrate/save (M2)** | `main.py`, `clustering/preprocessing.py`, `README.md`, `tests/test_clustering.py` | `run_clustering_comparison(config)`, `save_clustering_artifacts(...)`, `load_clustering_artifacts(...)`, `test_reproducibility()` | CLI/GUI integration, reproducible commands, saved fitted preprocessing/model artifacts for local demo. Exclude fitted artifacts from the submitted ZIP. |

**Supporting files:** `config.yaml` stores cleaned CSV path, selected feature groups, seed, algorithm settings and output paths; `requirements.txt` lists runtime dependencies; `clustering/__init__.py` identifies the package; `outputs/clustering/` contains metrics/plots/profiles; `artifacts/clustering/` contains local fitted transformers/models. Avoid putting independent cleaning logic inside the clustering module.

**Data flow and contracts:** `H2_cleaned.csv` → `validate_cleaned_schema()` → `select_clustering_features()` → `fit_transform_features()` → `fit_kmeans()` → `compute_cluster_metrics()` → `build_cluster_profiles()`. Every output label must correspond to exactly one retained input row in the same order. Save the **fitted** transformer with the model; never fit a new scaler when assigning new bookings.

**M1 first coding session:** implement `features.py`, then `preprocessing.py`, then the K-Means functions in `models.py`. Verify shapes and alignment before writing evaluation and profiling. The cleaned CSV is **pending**, so any tests on raw H2 are temporary and must be rerun after handoff.

## Immediate Next Steps

1. Obtain the team-approved cleaned file and compare its schema and row count with raw H2.
2. Agree on duplicate handling and zero-guest/zero-night treatment; log decisions.
3. Freeze the 11 leakage-safe core features; implement **clustering-specific** encoding/scaling on top of the shared cleaned data.
4. Fit baseline K-Means, inspect cluster sizes and Silhouette/DBI, then profile clusters using original unscaled values.
5. Do **not** name clusters before examining the resulting profiles.

## Definition of Done

Task C is complete when another teammate can reproduce the same data preparation and clustering results, inspect baseline-versus-tuned comparisons for three algorithms, understand the feature/leakage decisions, view 2D plots and meaningful cluster profiles, and access the analysis from the final application.

*Source: `CSE382_Project_Description_Milestones.pdf`, especially Sections 10–18. Parameter ranges, feature choices, and file layout above are engineering recommendations, not mandatory settings from the specification.*
