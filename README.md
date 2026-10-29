# City-Hotel-ML-Project-Team-23

University CSE382 machine learning project for City Hotel booking data.
This repository is a skeleton only: no data loading, cleaning, preprocessing,
training, evaluation, or plotting has been implemented.

```text
City-Hotel-ML-Project-Team-23/
|-- data/
|   |-- precleaned/
|   `-- cleaned/
|-- clustering/
|   |-- __init__.py
|   |-- features.py
|   |-- preprocessing.py
|   |-- models.py
|   |-- evaluation.py
|   |-- profiling.py
|   |-- tuning.py
|   `-- visualization.py
|-- classification/__init__.py
|-- regression/__init__.py
|-- artifacts/clustering/
|-- outputs/clustering/
|-- tests/test_clustering.py
|-- clean_data.py
|-- main.py
|-- config.yaml
|-- requirements.txt
|-- .gitignore
`-- CLUSTERING_PLAN.md
```

See [CLUSTERING_PLAN.md](CLUSTERING_PLAN.md) for the architecture, booking-time
feature decisions, proposed functions, and milestones. Classification and regression
are empty packages reserved for the other team members.

`clean_data.py` is the one-time shared cleaning script placeholder, separate from
clustering preprocessing. Its intended input is `data/precleaned/H2.csv` and output
is `data/cleaned/H2_cleaned.csv`. No datasets are included in this restored scaffold.

The layout follows the plan's precleaned, artifact, output, and test directories.
Empty directories use `.gitkeep` so they survive cloning. Generated cleaned data,
fitted artifacts, and outputs are ignored. Fitted artifacts must also be excluded
from the submitted ZIP.

Dependencies are listed but have not been installed. A documented parameter grid
is planned, so Optuna is not required. All functions, including tests, are stubs;
placeholder tests do not establish correctness. Run commands and the submitted
commit SHA will be documented when implementation and submission exist.
