# Task A: Regression - Average Daily Rate (ADR) Prediction

This directory contains the complete implementation for **Task A (Regression)** of the City Hotel Booking Demand project. The goal is to predict the continuous target variable `ADR` (Average Daily Rate) at the time of booking creation to assist with dynamic pricing and revenue management.

---

## 1. Overview of Task A Implementation

The regression module handles end-to-end data processing, feature engineering, data leakage prevention, baseline pipeline creation, and model evaluation.

### Key Components:
* **Feature Engineering (`features.py`):** Derived domain-specific features including `TotalNights` (`StaysInWeekendNights` + `StaysInWeekNights`), `TotalGuests` (`Adults` + `Children` + `Babies`), and `IsFamilyBooking` (`(Children + Babies) > 0`).
* **Data Preprocessing & Leakage Filtering (`preprocessing.py`):** Filtered out post-booking and checkout features to isolate strict booking-time inputs.
* **Target Profiling (`profiling.py`):** Evaluated `ADR` summary statistics (Mean: \$111.17, Median: \$105.30, Std: \$41.95 on deduplicated data).
* **Pipeline Architecture (`models.py`):** Built a modular `scikit-learn` `Pipeline` utilizing `ColumnTransformer` for numerical preprocessing (median imputation + standard scaling) and categorical preprocessing (most-frequent imputation + one-hot encoding).
* **Automated Evaluation (`evaluation.py`):** Implemented automated scoring across standard regression metrics (MAE, RMSE, $R^2$).
* **Baseline Script (`train_baseline.py`):** Automated data loading, 80/20 train/validation splitting, pipeline fitting on `X_train`, and validation reporting.

---

## 2. Data Leakage Audit & Prediction Point

### Defined Prediction Point
Predictions are strictly defined to take place at the **exact moment a booking enters the Property Management System (PMS)**.

### Leakage Prevention Audit
1. **Post-Booking / Lifecycle Feature Exclusions:**
   To prevent target leakage, features populated or modified after booking creation or during check-out were excluded from model input:
   * `ReservationStatus` & `ReservationStatusDate`: Updated upon checkout or cancellation.
   * `AssignedRoomType`: Determined at check-in (room upgrades occur post-booking).
   * `BookingChanges` & `DaysInWaitingList`: Post-booking operational amendments.
   * `IsCanceled`: Strictly reserved as the target variable for Task B (Classification).

2. **Distribution Leakage Prevention:**
   All statistical transformers (scaling parameters, median values, one-hot categories) are encapsulated inside `scikit-learn` pipelines and **fitted strictly on the training split (`X_train`)** to prevent distribution parameters from validation/test sets leaking into training.

---

## 3. Evaluation Metrics & Baseline Results

### Evaluation Metrics Overview
* **MAE (Mean Absolute Error):** Measures average absolute difference between actual and predicted `ADR` in dollars.
* **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger price deviations more heavily.
* **$R^2$ (Coefficient of Determination):** Measures the proportion of `ADR` variance explained by the model compared to guessing the mean `ADR`.

### Milestone 1 Baseline Performance
The baseline experiment was conducted using **Linear Regression** on an 80/20 train-validation split (`random_state=42`) using `data/cleaned/H2_cleaned3.csv`.

| Model | MAE | RMSE | $R^2$ Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | **\$15.19** | **\$22.91** | **0.7070** | Completed |

*Interpretation:* The baseline model predicts room rates within an average error of \$15.19 and explains **~70.7%** of total pricing variance.

---

## 4. File Structure & Running Instructions

```
src/regression/
├── __init__.py           # Package marker
├── features.py           # Domain feature engineering
├── preprocessing.py      # Feature cleaning & leakage drop
├── profiling.py          # Summary statistics & ADR analysis
├── models.py             # Pipeline builder (ColumnTransformer + Estimator)
├── evaluation.py         # Metrics calculator (MAE, RMSE, R²)
├── train_baseline.py     # Main baseline execution script
├── tuning.py             # Model tuning & optimization (Milestone 2)
└── README.md             # Regression module documentation
```

### Execution
To run the baseline training script from the repository root:
```bash
python src/regression/train_baseline.py
```