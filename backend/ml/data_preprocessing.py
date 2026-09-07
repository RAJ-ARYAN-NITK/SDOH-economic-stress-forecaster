
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/data/processed/')


class SDoHDataProcessor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = PROCESSED_DIR

    def load_data(self):
        print("Loading dataset...")
        df = pd.read_csv(self.processed_dir / "with_unemployment_processed.csv")
        df['region'] = df['region'].str.extract(r'in (.+)$')
        print("Columns:", list(df.columns))
        print("Shape:", df.shape)
        print("Regions:", sorted(df['region'].unique()))
        return df

    def create_time_features(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        return df

    def create_lag_features(self, df, target_col, lags=[1, 3, 6, 12]):
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        for window in [3, 6, 12]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df

    def engineer_features(self, df):
        df = df.copy()
        if 'employment_count' in df.columns and 'unemployment_rate' in df.columns:
            df['employment_unemployment_ratio'] = (
                df['employment_count'] / (df['unemployment_rate'] + 1e-6)
            )
        if 'cpi_medical_care' in df.columns and 'cpi_all_items' in df.columns:
            df['medical_care_affordability'] = (
                df['cpi_medical_care'] / df['cpi_all_items']
            )
        if 'cpi_housing' in df.columns and 'cpi_energy' in df.columns:
            df['housing_energy_ratio'] = (
                df['cpi_housing'] / (df['cpi_energy'] + 1e-6)
            )
        return df

    def prepare_lstm_sequences(self, df, target_col, sequence_length=36):
        df = df.sort_values('date').reset_index(drop=True)
        feature_cols = [col for col in df.columns if col not in ['date', 'region', target_col]]
        features = df[feature_cols].values
        target = df[target_col].values
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            y.append(target[i])
        X = np.array(X)
        y = np.array(y)
        X = np.nan_to_num(X, nan=0.0)
        return X, y, feature_cols

    def save_processed_data(self, X, y, feature_cols, regions_array, county_map):
        """
        Saves RAW (unscaled) sequences.
        Scaling is deferred to model_training.py (fit on train only — no leakage).
        county_map is saved so downstream scripts can decode IDs back to names.
        """
        np.save(self.processed_dir / "X_sequences.npy", X)
        np.save(self.processed_dir / "y_target.npy", y)
        np.save(self.processed_dir / "regions.npy", regions_array)
        with open(self.processed_dir / "feature_columns.pkl", 'wb') as f:
            pickle.dump(feature_cols, f)
        with open(self.processed_dir / "county_map.pkl", 'wb') as f:
            pickle.dump(county_map, f)
        print(f"\nSaved (unscaled): X={X.shape}  y={y.shape}  regions={regions_array.shape}")
        print(f"Counties: {sorted(np.unique(regions_array))}")
        print(f"County ID map: {county_map}")


if __name__ == "__main__":
    processor = SDoHDataProcessor(project_root="../../")
    df = processor.load_data()

    TARGET_COL = "unemployment_rate"
    SEQUENCE_LENGTH = 36

   
    county_map = {c: i for i, c in enumerate(sorted(df['region'].unique()))}
    print(f"\nCounty ID map: {county_map}")

    all_X, all_y, all_regions = [], [], []
    feature_cols = None

    for region, region_df in df.groupby('region'):
        region_df = processor.create_time_features(region_df)
        region_df = processor.create_lag_features(region_df, TARGET_COL)
        region_df = region_df.dropna(subset=[TARGET_COL] + [
            col for col in region_df.columns if 'lag' in col or 'rolling' in col
        ])
        if region_df.shape[0] < SEQUENCE_LENGTH:
            print(f"Skipping {region} — only {region_df.shape[0]} rows after dropna")
            continue

        region_df = processor.engineer_features(region_df)

        
        region_df['county_id'] = county_map[region]

        X_region, y_region, feature_cols = processor.prepare_lstm_sequences(
            region_df, TARGET_COL, SEQUENCE_LENGTH
        )
        all_X.append(X_region)
        all_y.append(y_region)
        all_regions.extend([region] * len(y_region))
        print(f"  {region} (id={county_map[region]}): {len(y_region)} sequences")

    if not all_X:
        raise Exception("No valid regions found after preprocessing.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    regions_array = np.array(all_regions)

    processor.save_processed_data(X, y, feature_cols, regions_array, county_map)