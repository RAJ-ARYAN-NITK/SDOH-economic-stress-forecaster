# MA SDOH Economic Stress Forecaster

> Forecasting county-level unemployment as a public health stress indicator across 11 Massachusetts counties using 10 years of Social Determinants of Health (SDOH) data — built with XGBoost, LSTM, and SHAP explainability.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-MAE%3D0.0325-brightgreen)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)](https://sdoh-er-predictorgit-mhgx4s6sibltmvtcmbwuw3.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What This Project Does

Unemployment is more than an economic indicator — it functions as an early-warning signal for regional economic stress. When unemployment rises in a county, financial strain compounds quickly: mental health crises increase and preventive care gets skipped as households cut back. This project treats unemployment rate as a **SDOH stress indicator** and forecasts it 1 month ahead using economic features, giving public health planners early warning of demand surges.

The model ingests 10 years of monthly data across 11 Massachusetts counties, combining **labor market conditions** (unemployment rate, labor force participation rate, employment counts) with **inflation dynamics** (CPI across all-items, housing, energy, and medical care), and outputs county-specific unemployment forecasts with SHAP-based explanations of which economic factors are driving stress in each region.

---

## Raw Data Schema

Before feature engineering, the merged dataset (`data/processed/with_unemployment_processed.csv`) has the following 10 columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Month of observation (e.g. `2015-01-01`) |
| `region` | string | County identifier (e.g. "Barnstable County") |
| `unemployment_rate` | float | Monthly unemployment rate (%) — **prediction target** |
| `labor_force_participation_rate` | float | % of working-age population in the labor force (BLS LAUS) |
| `employment_count` | int | Total covered employment for the county (BLS QCEW) |
| `cpi_all_items` | float | CPI-U, All Items — Boston-Cambridge-Newton, MA-NH metro |
| `cpi_housing` | float | CPI-U, Housing sub-index |
| `cpi_energy` | float | CPI-U, Energy sub-index |
| `cpi_medical_care` | float | CPI-U, Medical Care sub-index |
| `year` | int | Calendar year extracted from `date` |

These 10 raw columns feed into the feature engineering step (see [Feature Engineering Details](#feature-engineering-details)) which expands them into 27 model-ready features per timestep.

---

## Results

> **Target scale:** unemployment rate stored as a percentage (e.g. `4.5` = 4.5%). All errors are in **percentage points (pp)**.

| Model | MAE (pp) | RMSE (pp) | vs. Naive baseline |
|-------|----------|-----------|---------------------|
| Naive lag-1 (persistence) | 0.5083 | 0.8573 | — |
| LSTM | 0.0961 | 0.1351 | 5.3× better |
| **XGBoost** | **0.0325** | **0.0402** | **15.6× better** |

All metrics are on the **held-out chronological test set** (last 15% of each county's timeline). The naive lag-1 baseline predicts "next month = this month" — the standard persistence benchmark for time-series. XGBoost achieves **15.6× lower MAE** than this baseline (0.0325 pp vs. 0.5083 pp), confirming the model has learned genuine patterns beyond simple autocorrelation.

Both models successfully capture the COVID-19 unemployment spike (2020) and recovery. XGBoost outperforms LSTM by **3× on MAE**, confirming gradient boosting's edge over sequence models on low-volume, high-feature tabular data.

**Key SHAP finding:** `medical_care_affordability` (ratio of medical CPI to overall CPI) is the strongest leading indicator of unemployment stress — ahead of energy costs and labor force participation rate. SHAP values computed on the held-out test set (first 200 samples of test split).

---

## Pipeline Overview

```mermaid
flowchart TD
    A[BLS Website\nbls.gov] -->|Download Excel files\nper county + CPI series| B[Raw Excel Files\n11 counties × employment\n4 CPI series]
    B -->|Manual cleaning\nstandardize columns| C[data/raw/\nsdoh_dataset_employment_inflation.csv\nsdoh_dataset_with_unemployment.csv]
    C -->|notebooks/analysis.ipynb\nmerge + IQR cap + save| D[data/processed/\nwith_unemployment_processed.csv]
    D -->|data_preprocessing.py| E[Feature Engineering\nlag + rolling + cyclical + ratios]
    E -->|36-month sequences\nunscaled| F[X_sequences.npy\ny_target.npy\nregions.npy]
    F -->|model_training.py\nper-county 70/15/15\nscaler fit on train only| G[Training Data\n~6098 samples]
    G -->|Naive persistence| N[Naive lag-1 Baseline\npredict next = current\nMAE=0.5083 RMSE=0.8573]
    G -->|Block A| H[XGBoost Model\nxgb_model.pkl\nMAE=0.0325 pp 15.6x vs baseline]
    G -->|Block B| I[LSTM Model\ner_lstm_model.keras\nMAE=0.0961 pp 5.3x vs baseline]
    N --> J
    H --> J[analysis.py\nSHAP on held-out test set\n+ per-county plots]
    I --> J
    J --> K[shap_summary.png\nmodel_comparison_bars.png\ncounty_*.png]
    H --> L[app.py\nStreamlit Dashboard]
    I --> L
    K --> L
    L -->|streamlit run app.py| M[Live Dashboard\nshare.streamlit.io]
```

---

## Data Sources

All data was manually downloaded from the **U.S. Bureau of Labor Statistics (BLS)** at `bls.gov`.

### How the data was collected

#### Employment Data
1. Go to `bls.gov` → **Data Tools** → **County Employment and Wages (QCEW)**
2. Select **Massachusetts** → choose each county individually
3. Select **All Industries**, **All Establishment Sizes**, **Total Covered**
4. Download as Excel (.xlsx) for years 2014–2024
5. Repeat for all 11 counties:
   - Barnstable, Berkshire, Bristol, Dukes, Essex
   - Middlesex, Nantucket, Norfolk, Plymouth, Suffolk, Worcester

#### Unemployment Rate & Labor Force Participation
1. Go to `bls.gov` → **Data Tools** → **Local Area Unemployment Statistics (LAUS)**
2. Select **Massachusetts** → choose each county individually
3. Download monthly **unemployment rate** and **labor force participation rate** series for 2015–2024

#### Inflation / CPI Data
1. Go to `bls.gov` → **Data Tools** → **CPI Databases**
2. Select **All Urban Consumers (CPI-U)** for **Boston-Cambridge-Newton, MA-NH**
3. Select series for:
   - `CUURA103SA0` — All Items
   - `CUURA103SAH` — Housing
   - `CUURA103SAE` — Energy
   - `CUURA103SAM` — Medical Care
4. Download as Excel for 2015–2024

#### Manual Merging Process
- Opened each county Excel file and standardized column names
- Added a `region` column with the full county identifier
- Merged all county files into `sdoh_dataset_employment_inflation.csv`
- Separately downloaded unemployment rate and labor force participation rate series and merged into `sdoh_dataset_with_unemployment.csv`
- Final merge and cleaning was done in `notebooks/analysis.ipynb` → saved as `with_unemployment_processed.csv`

### Data Collection Flow

```mermaid
flowchart TD
    A[bls.gov] --> B{Data Type}
    B -->|Employment| C[QCEW — County Employment\nand Wages]
    B -->|Unemployment & Labor Force| D2[LAUS — Local Area\nUnemployment Statistics]
    B -->|Inflation| D[CPI-U — Boston Metro\nAll Urban Consumers]
    C --> E[Select Massachusetts\nAll Industries\nAll Establishment Sizes\nTotal Covered]
    E --> F[Download per county\nBarnstable, Berkshire\nBristol, Dukes, Essex\nMiddlesex, Nantucket\nNorfolk, Plymouth\nSuffolk, Worcester]
    D2 --> F2[unemployment_rate\nlabor_force_participation_rate\nper county, monthly]
    D --> G[Select 4 series\nSA0 — All Items\nSAH — Housing\nSAE — Energy\nSAM — Medical Care]
    F --> H[11 Excel files\n840 rows each\n2015-01-01 to 2024-12-01]
    F2 --> H2[11 Excel files\nUnemployment + LFPR\n2015-2024]
    G --> I[4 CPI Excel files\nMonthly values\n2015-2024]
    H --> J[Manual merge in Excel\nAdd region column\nStandardize headers]
    H2 --> J
    I --> J
    J --> K[sdoh_dataset_employment_inflation.csv\nsdoh_dataset_with_unemployment.csv]
```

---

## Project Structure

```
SDoH-ER-Predictor/
├── app.py                          # Streamlit dashboard (project root)
├── requirements.txt                # Dependencies for Streamlit Cloud
├── backend/
│   └── ml/
│       ├── data_preprocessing.py   # Feature engineering + sequence building
│       ├── model_training.py       # XGBoost + LSTM training + comparison
│       ├── analysis.py             # SHAP + per-county plots + metrics
│       └── predictions.py          # Inference on new data
├── data/
│   ├── raw/
│   │   ├── sdoh_dataset_employment_inflation.csv
│   │   └── sdoh_dataset_with_unemployment.csv
│   └── processed/
│       ├── with_unemployment_processed.csv
│       ├── X_sequences.npy
│       ├── y_target.npy
│       ├── regions.npy
│       ├── X_test.npy / y_test.npy / regions_test.npy
│       ├── xgb_model.pkl
│       ├── er_lstm_model.keras
│       ├── feature_columns.pkl
│       ├── scaler.pkl
│       ├── shap_summary.png
│       └── model_comparison_bars.png
├── notebooks/
│   └── analysis.ipynb              # EDA, merging, cleaning
└── frontend/                       # (optional Next.js frontend)
```

---

## Workflow

### Step 1 — Data Collection
Download Excel files from BLS for each county and each CPI series. Standardize column names manually in Excel. Save as CSVs in `data/raw/`.

### Step 2 — EDA and Merging
Run `notebooks/analysis.ipynb` to:
- Load both raw CSVs
- Standardize column names
- Merge on date + region
- Handle missing values with IQR capping
- Save `with_unemployment_processed.csv`

### Step 3 — Feature Engineering
Run `backend/ml/data_preprocessing.py` to:
- Shorten region names to county name only
- Create time features (month_sin, month_cos, quarter cyclical encoding)
- Create lag features (lag 1, 3, 6, 12 months)
- Create rolling statistics (mean and std for 3, 6, 12 month windows)
- Engineer SDOH ratios (medical_care_affordability, housing_energy_ratio)
- Build 36-month LSTM sequences
- Save **unscaled** `X_sequences.npy`, `y_target.npy`, `regions.npy` — scaling is deferred to Step 4 to prevent leakage

### Step 4 — Model Training
Run `backend/ml/model_training.py` to:
- Split each county **70/15/15 chronologically** (per-county so all 11 counties appear in train, val, and test)
- Fit `StandardScaler` **on the training split only**, then transform val/test — no leakage from future data
- Train XGBoost (500 trees, early stopping on val MAE, patience 20)
- Train LSTM (64→32 units, Dropout 0.2, early stopping on val loss, patience 10)
- Save `xgb_model.pkl`, `er_lstm_model.keras`, `scaler.pkl`, `regions_test.npy`

### Step 5 — Analysis
Run `backend/ml/analysis.py` to:
- Generate per-county true vs predicted plots
- Compute SHAP values on the **held-out test set** (first 200 samples for speed — no training data used)
- Save SHAP summary and waterfall charts
- Save model comparison bar charts

### Step 6 — Dashboard
Run `streamlit run app.py` to launch the interactive dashboard locally, or deploy to Streamlit Cloud.

---

## Feature Engineering Details

Raw inputs (see [Raw Data Schema](#raw-data-schema)) are expanded into the following engineered features:

| Feature | Description |
|---------|-------------|
| `unemployment_lag_1/3/6/12` | Lagged unemployment values |
| `unemployment_rolling_mean_3/6/12` | Rolling average unemployment |
| `unemployment_rolling_std_3/6/12` | Rolling volatility |
| `month_sin / month_cos` | Cyclical month encoding |
| `quarter_sin / quarter_cos` | Cyclical quarter encoding |
| `medical_care_affordability` | cpi_medical_care / cpi_all_items |
| `housing_energy_ratio` | cpi_housing / cpi_energy |
| `employment_unemployment_ratio` | employment_count / unemployment_rate |

### Feature Engineering Pipeline

```mermaid
flowchart LR
    A[with_unemployment_processed.csv\n9240 rows × 10 cols] --> B[Shorten region names\nExtract county name only]
    B --> C[Per-county processing]
    C --> D[create_time_features\nyear, month, quarter\nmonth_sin, month_cos\nquarter_sin, quarter_cos]
    C --> E[create_lag_features\nunemployment_lag_1\nunemployment_lag_3\nunemployment_lag_6\nunemployment_lag_12\nrolling_mean_3,6,12\nrolling_std_3,6,12]
    C --> F[engineer_features\nmedical_care_affordability\nhousing_energy_ratio\nemployment_unemployment_ratio]
    D --> G[prepare_lstm_sequences\nsequence_length=36]
    E --> G
    F --> G
    G --> H[X shape: 8712 × 36 × 27\ny shape: 8712\nregions shape: 8712]
    H --> I[StandardScaler\nfit on train\ntransform all]
    I --> J[X_sequences.npy\ny_target.npy\nregions.npy\nfeature_columns.pkl\nscaler.pkl]
```

---

## Model Architecture

### XGBoost
- Input: flattened 36-timestep sequences → (samples, 36×27=972 features)
- n_estimators: 500, learning_rate: 0.05, max_depth: 4
- Early stopping: 20 rounds on validation MAE
- Subsample: 0.8, colsample_bytree: 0.8

### LSTM
- Input: (samples, 36 timesteps, 27 features)
- Layer 1: LSTM(64, return_sequences=True) + Dropout(0.2)
- Layer 2: LSTM(32) + Dropout(0.2)
- Output: Dense(1)
- Optimizer: Adam, Loss: MSE
- Early stopping: patience=10 on val_loss

### Train / Val / Test Split

```mermaid
flowchart TD
    A[8712 total sequences\n11 counties × ~792 each] --> B{Per-county\nchronological split}
    B -->|70% of each county| C[X_train\n~6098 samples\nall 11 counties]
    B -->|15% of each county| D[X_val\n~1307 samples\nall 11 counties]
    B -->|15% of each county| E[X_test\n~1307 samples\nall 11 counties ✓]
    C --> F[model.fit\nXGBoost + LSTM]
    D --> F
    D --> G[Early stopping\nval_loss monitor]
    E --> H[Evaluate\nMAE RMSE\nper county]
    note1[Previous global split:\nonly 2 counties in test set ✗] -.->|Fixed by| B
```

### Architecture Comparison

```mermaid
flowchart LR
    subgraph Baseline
        A0[Naive lag-1\npredict next = current\nno training needed]
        A0 --> D0[Output\nMAE=0.5083 pp\nRMSE=0.8573 pp]
    end
    subgraph XGBoost
        A1["Input\n~6098 × 1008\nflattened sequences"] --> B1[500 Decision Trees\ndepth=4\nlr=0.05]
        B1 --> C1[Early stopping\n20 rounds\nval MAE]
        C1 --> D1[Output\nMAE=0.0325 pp\nRMSE=0.0402 pp]
    end
    subgraph LSTM
        A2["Input\n~6098 × 36 × 28\n3D sequences"] --> B2[LSTM 64\nreturn_sequences=True\nDropout 0.2]
        B2 --> C2[LSTM 32\nDropout 0.2]
        C2 --> D2[Dense 1]
        D2 --> E2[Output\nMAE=0.0961 pp\nRMSE=0.1351 pp]
    end
    D0 --> F[Winner: XGBoost\n15.6x lower MAE than baseline\n3x lower MAE than LSTM]
    D1 --> F
    E2 --> F
```

---

## SHAP Explainability

```mermaid
flowchart TD
    A[xgb_model.pkl] --> B[shap.Explainer\nTreeExplainer]
    C[X_test_2d\n200 samples subset] --> B
    B --> D[SHAP values\n200 × 972 matrix]
    D --> E[summary_plot\nFeature importance\nranked by mean abs SHAP]
    D --> F[waterfall plot\nSingle prediction\nexplained]
    E --> G[Key finding:\nmedical_care_affordability\nis top predictor\nahead of lag features]
    F --> H[Shows exactly which\nfeatures pushed this\nprediction up or down]
    G --> I[Resume insight:\nHealthcare cost inflation\nprecedes unemployment\nstress by 1-3 quarters]
```

**Key SHAP insight:** `medical_care_affordability` ranks as the strongest leading predictor of unemployment stress — ahead of energy costs and labor force participation rate. Healthcare cost inflation precedes employment deterioration by 1–3 quarters in Massachusetts counties.

---

## Streamlit Dashboard

```mermaid
flowchart TD
    A[User opens dashboard] --> B[Sidebar: select county\nBarstable...Worcester]
    B --> C[Load regions_test.npy\nfilter mask for county]
    C --> D[Plot y_test filtered\ny_pred_xgb filtered\ny_pred_lstm filtered]
    D --> E[Chart: True vs Predicted\nfor selected county only]
    A --> F[Model comparison table\nXGBoost vs LSTM\nMAE RMSE]
    A --> G[SHAP panel\nTop SDOH features\nby importance]
    A --> H[Raw data expander\nlast 24 months\nfor selected county]
    E --> I[Each county looks\ngenuinely different\nSuffolk vs Worcester\nvs Nantucket]
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/RAJ-ARYAN-NITK/SDoH-ER-Predictor.git
cd SDoH-ER-Predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
cd backend/ml
python3 data_preprocessing.py
python3 model_training.py
python3 analysis.py

# Launch dashboard
cd ../..
streamlit run app.py
```
---

## License

MIT — free to use, modify, and distribute.
