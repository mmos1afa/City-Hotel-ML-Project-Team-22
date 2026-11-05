# Clustering Testing Notes

This file records K-Means experiments. Compare runs only when they use the same
dataset, selected features, preprocessing, and random seed.

## Common setup

- Selected inputs: 11 booking-time features from `clustering/features.py`.
- Preprocessing: numeric features use `StandardScaler`; `IsRepeatedGuest` stays
  as 0 or 1; categorical columns use one-hot encoding.
- K-Means settings: `init="k-means++"`, `n_init=10`, `max_iter=300`, and
  `random_state=42`.
- Metrics: higher silhouette is better; lower Davies-Bouldin is better.
- Inertia is recorded but is not used alone to choose K, because it always
  falls when more clusters are added.

## Test set A: `data/cleaned/H2_cleaned2.csv`

This file contained 79,162 bookings in the clustering run.

| K | Silhouette | Davies-Bouldin | Inertia | Smallest cluster | Largest cluster |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.1413 | 1.9945 | 474,212.04 | 6.45% | 49.40% |
| 4 | 0.2074 | 1.2971 | 413,210.80 | 0.47% | 73.61% |
| 5 | 0.1978 | 1.3129 | 361,076.50 | 0.47% | 56.85% |
| 6 | 0.2006 | 1.3897 | 322,415.97 | 0.47% | 38.07% |
| 7 | 0.2000 | 1.3473 | 296,196.77 | 0.47% | 30.36% |
| 8 | 0.2034 | 1.3147 | 280,176.25 | 0.47% | 27.65% |
| 9 | 0.2026 | 1.1814 | 260,265.66 | 0.003% (2 bookings) | 25.30% |
| 10 | 0.1892 | 1.3254 | 259,274.38 | 0.47% | 19.47% |
| 11 | 0.1897 | 1.1870 | 239,034.76 | 0.003% (2 bookings) | 20.63% |
| 12 | 0.1819 | 1.3104 | 231,147.43 | 0.003% (2 bookings) | 17.63% |
| 13 | 0.1855 | 1.4453 | 223,695.21 | 0.003% (2 bookings) | 16.05% |
| 14 | 0.1698 | 1.4816 | 214,924.22 | 0.003% (2 bookings) | 17.68% |

### Notes

- K=3 is the required simple baseline.
- K=4 had the highest silhouette, but one cluster contained 73.61% of bookings.
- K=8 was the most balanced candidate in this dataset version.
- K=9 and above created a two-booking cluster, which is likely over-segmentation.

## Test set B: `data/cleaned/H2_cleaned3.csv`

This is the current input file used by `main.py`.

| K | Silhouette | Davies-Bouldin | Inertia | Smallest cluster | Largest cluster |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.1418 | 2.4151 | 361,169.17 | 45.35% | 54.65% |
| 3 | 0.1534 | 1.9447 | 316,779.58 | 9.20% | 47.33% |
| 4 | 0.1627 | 1.5449 | 278,249.75 | 0.69% | 46.96% |
| 5 | 0.1811 | 1.5062 | 249,918.91 | 0.69% | 50.19% |
| 6 | 0.1552 | 1.6491 | 248,713.08 | 0.69% | 30.70% |
| 7 | 0.1731 | 1.4315 | 210,999.79 | 0.69% | 29.69% |
| 8 | 0.1800 | 1.4102 | 200,461.25 | 0.69% | 27.69% |
| 9 | 0.1800 | 1.2609 | 186,912.35 | 0.004% (2 bookings) | 27.69% |
| 10 | 0.1787 | 1.3244 | 177,691.46 | 0.004% (2 bookings) | 26.24% |
| 11 | 0.1765 | 1.2575 | 170,704.90 | 0.004% (2 bookings) | 20.91% |
| 12 | 0.1688 | 1.3303 | 166,312.84 | 0.004% (2 bookings) | 17.99% |
| 13 | 0.1841 | 1.3020 | 159,002.67 | 0.004% (2 bookings) | 21.54% |
| 14 | 0.1867 | 1.3490 | 149,015.23 | 0.004% (2 bookings) | 16.92% |

### Current decision

- Keep K=3 as the baseline for the report.
- Treat K=8 as the current tuned candidate: its silhouette is close to the
  best practical result, Davies-Bouldin is much better than K=3, and its large
  clusters are more balanced than K=5.
- Do not select K=9 through K=14 yet. Each produces a two-booking cluster.
- The 0.69% cluster that appears at K=4 through K=8 needs profiling before a
  final choice. It may be a meaningful rare segment or an outlier group.

## Profiling check: K=3, K=5, and K=8 on `H2_cleaned3.csv`

The same data and preprocessing were used. K=3 is the baseline, K=5 has the
strongest silhouette before two-booking clusters appear, and K=8 is the
balanced candidate.

| K | Key finding from profiles |
|---:|---|
| 3 | Separates a family group (9.20%, median 1 child), short-stay bookings (47.33%), and earlier long-stay bookings (43.47%). |
| 5 | Separates advance planners (22.84%, 170-day median lead time), short-stay one-adult bookings (17.59%), a large general group (50.19%), a baby group (0.69%), and a family group (8.69%). |
| 8 | Splits the patterns into clearer segments: short-stay one-adult/repeated guests, advance planners, general bookings, baby bookings, two-child families, short-stay couples, three-adult bookings, and long-stay bookings. |

### Result

- K=8 has interpretable patterns that justify examining it further.
- The 369-booking cluster persists at K=5 and K=8. It has a median of one
  baby, so it is a plausible rare booking segment rather than a random split.
- Cluster names remain provisional until the full category-distribution tables
  are reviewed by the team.

## Future test template

Add a section for every new experiment:

```text
### Test: [short name]

- Date:
- Dataset file:
- What changed:
- What stayed the same:
- K value(s):
- Silhouette:
- Davies-Bouldin:
- Inertia:
- Smallest/largest cluster share:
- Decision and reason:
```

Do not compare scores across `H2_cleaned2.csv` and `H2_cleaned3.csv` as though
they came from the same data, because their row sets differ.
