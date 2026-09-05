# MA SDOH Economic Stress Forecaster

> Forecasting county-level unemployment as a public health stress indicator across 11 Massachusetts counties using 10 years of Social Determinants of Health (SDOH) data — built with XGBoost, LSTM, and SHAP explainability.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-MAE%3D0.0325-brightgreen)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)](https://sdoh-er-predictorgit-mhgx4s6sibltmvtcmbwuw3.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What This Project Does

Unemployment is not just an economic problem — it is a public health crisis. When unemployment rises in a county, emergency room visits spike, mental health crises increase, and preventive care gets skipped. This project treats unemployment rate as a **SDOH stress indicator** and forecasts it 1 month ahead using economic features, giving public health planners early warning of demand surges.

The model ingests 10 years of monthly employment, inflation, and labor force data across 11 Massachusetts counties and outputs county-specific unemployment forecasts with SHAP-based explanations of which economic factors are driving stress in each region.

---

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| **XGBoost** | **0.0325** | **0.0402** |
| LSTM | 0.0961 | 0.1351 |

XGBoost outperforms LSTM by **3x on MAE** — demonstrating that gradient boosting outperforms deep learning on structured tabular time-series with moderate data size. Both models successfully capture the two major unemployment spikes (COVID-19 2020, subsequent surge) and recovery patterns.

**Key SHAP finding:** `medical_care_affordability` (ratio of medical CPI to overall CPI) is the strongest leading indicator of unemployment stress — ahead of energy costs and labor force participation rate. This suggests that healthcare cost inflation precedes employment deterioration in Massachusetts counties.

---

## Pipeline Overview

```mermaid
flowchart TD
    A[BLS Website\nbls.gov] -->|Download Excel files\nper county + CPI series| B[Raw Excel Files\n11 counties × employment\n4 CPI series]
    B -->|Manual cleaning\nstandardize columns| C[data/raw/\nsdoh_dataset_employment_inflation.csv\nsdoh_dataset_with_unemployment.csv]
    C -->|notebooks/analysis.ipynb\nmerge + IQR cap + save| D[data/processed/\nwith_unemployment_processed.csv]
    D -->|data_preprocessing.py| E[Feature Engineering\nlag + rolling + cyclical + ratios]
    E -->|36-month sequences\nStandardScaler| F[X_sequences.npy\ny_target.npy\nregions.npy]
    F -->|model_training.py\nper-county 70/15/15 split| G[Training Data\n6098 samples]
    G -->|Block A| H[XGBoost Model\nxgb_model.pkl\nMAE=0.0325 RMSE=0.0402]
    G -->|Block B| I[LSTM Model\ner_lstm_model.keras\nMAE=0.0961 RMSE=0.1351]
    H --> J[analysis.py\nSHAP + per-county plots]
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

#### Employment & Unemployment Data
1. Go to `bls.gov` → **Data Tools** → **County Employment and Wages (QCEW)**
2. Select **Massachusetts** → choose each county individually
3. Select **All Industries**, **All Establishment Sizes**, **Total Covered**
4. Download as Excel (.xlsx) for years 2014–2024
5. Repeat for all 11 counties:
   - Barnstable, Berkshire, Bristol, Dukes, Essex
   - Middlesex, Nantucket, Norfolk, Plymouth, Suffolk, Worcester

#### Inflation / CPI Data
1. Go to `bls.gov` → **Data Tools** → **CPI Databases**
2. Select **All Urban Consumers (CPI-U)** for **Boston-Cambridge-Newton, MA-NH**
3. Select series for:
   - `CUURA103SA0` — All Items
   - `CUURA103SAH` — Housing
   - `CUURA103SAE` — Energy
   - `CUURA103SAM` — Medical Care
4. Download as Excel for 2014–2024

#### Manual Merging Process
- Opened each county Excel file and standardized column names
- Added a `region` column with the full county identifier
- Merged all county files into `sdoh_dataset_employment_inflation.csv`
- Separately downloaded unemployment rate series and merged into `sdoh_dataset_with_unemployment.csv`
- Final merge and cleaning was done in `notebooks/analysis.ipynb` → saved as `with_unemployment_processed.csv`

### Data Collection Flow

```mermaid
flowchart TD
    A[bls.gov] --> B{Data Type}
    B -->|Employment| C[QCEW — County Employment\nand Wages]
    B -->|Inflation| D[CPI-U — Boston Metro\nAll Urban Consumers]
    C --> E[Select Massachusetts\nAll Industries\nAll Establishment Sizes\nTotal Covered]
    E --> F[Download per county\nBarnstable, Berkshire\nBristol, Dukes, Essex\nMiddlesex, Nantucket\nNorfolk, Plymouth\nSuffolk, Worcester]
    D --> G[Select 4 series\nSA0 — All Items\nSAH — Housing\nSAE — Energy\nSAM — Medical Care]
    F --> H[11 Excel files\n840 rows each\n2015-01-01 to 2024-12-01]
    G --> I[4 CPI Excel files\nMonthly values\n2015-2024]
    H --> J[Manual merge in Excel\nAdd region column\nStandardize headers]
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
- Scale with StandardScaler
- Save `X_sequences.npy`, `y_target.npy`, `regions.npy`

### Step 4 — Model Training
Run `backend/ml/model_training.py` to:
- Split each county 70/15/15 chronologically (per-county to ensure all counties appear in test set)
- Train XGBoost (500 trees, early stopping, MAE metric)
- Train LSTM (64→32 units, dropout, early stopping)
- Compare both models and save results
- Save `xgb_model.pkl`, `er_lstm_model.keras`, `regions_test.npy`

### Step 5 — Analysis
Run `backend/ml/analysis.py` to:
- Generate per-county true vs predicted plots
- Compute SHAP values (200 sample subset for speed)
- Save SHAP summary and waterfall charts
- Save model comparison bar charts

### Step 6 — Dashboard
Run `streamlit run app.py` to launch the interactive dashboard locally, or deploy to Streamlit Cloud.

---

## Feature Engineering Details

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
    subgraph XGBoost
        A1[Input\n8712 × 972\nflattened sequences] --> B1[500 Decision Trees\ndepth=4\nlr=0.05]
        B1 --> C1[Early stopping\n20 rounds\nval MAE]
        C1 --> D1[Output\nMAE=0.0325\nRMSE=0.0402]
    end
    subgraph LSTM
        A2[Input\n8712 × 36 × 27\n3D sequences] --> B2[LSTM 64\nreturn_sequences=True\nDropout 0.2]
        B2 --> C2[LSTM 32\nDropout 0.2]
        C2 --> D2[Dense 1]
        D2 --> E2[Output\nMAE=0.0961\nRMSE=0.1351]
    end
    D1 --> F[Winner: XGBoost\n3x lower MAE\nfaster training]
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

## Resume Bullets

```
MA SDOH Economic Stress Forecaster | Python, XGBoost, LSTM, SHAP, Streamlit

• Engineered 27+ features from 10 years of BLS employment and CPI data
  across 11 Massachusetts counties including cyclical time encoding,
  multi-lag rolling statistics, and SDOH economic ratios

• Compared XGBoost vs LSTM; XGBoost achieved 3x lower MAE (0.0325 vs 0.0961)
  and 3x lower RMSE, demonstrating gradient boosting outperforms deep learning
  on structured tabular time-series at moderate data scale

• Applied SHAP explainability to identify medical_care_affordability as
  the strongest leading indicator of unemployment stress, ahead of energy
  costs and labor force participation rate

• Deployed interactive per-county forecast dashboard to Streamlit Cloud
  with SHAP waterfall charts explaining individual predictions
```

---

## License

MIT — free to use, modify, and distribute.
