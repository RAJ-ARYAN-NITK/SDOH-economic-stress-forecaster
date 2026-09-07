import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
from tensorflow.keras.models import load_model

PROCESSED_DIR     = '/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/data/processed/'
X_test_path       = os.path.join(PROCESSED_DIR, 'X_test.npy')
y_test_path       = os.path.join(PROCESSED_DIR, 'y_test.npy')
REGIONS_TEST_PATH = os.path.join(PROCESSED_DIR, 'regions_test.npy')
LSTM_MODEL_PATH   = os.path.join(PROCESSED_DIR, 'er_lstm_model.keras')
XGB_MODEL_PATH    = os.path.join(PROCESSED_DIR, 'xgb_model.pkl')
FEATURES_PATH     = os.path.join(PROCESSED_DIR, 'feature_columns.pkl')
HISTORY_PATH      = os.path.join(PROCESSED_DIR, 'history.pkl')


def plot_true_vs_predicted(y_true, y_pred, label='Model'):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='True', alpha=0.7)
    plt.plot(y_pred, label=f'Predicted ({label})', alpha=0.7)
    plt.title(f"True vs Predicted Unemployment Rate — {label} (Test Set, all counties)")
    plt.xlabel("Sample Index")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f'true_vs_pred_{label.lower()}.png'))
    plt.close()


def plot_residuals(y_true, y_pred, label='Model'):
    residuals = y_true - y_pred

    plt.figure(figsize=(12, 5))
    plt.plot(residuals)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.title(f"Residuals (True − Predicted) — {label}")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f'residuals_line_{label.lower()}.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=0.8)
    plt.title(f"Histogram of Residuals — {label}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f'residuals_hist_{label.lower()}.png'))
    plt.close()


def plot_scatter_true_vs_pred(y_true, y_pred, label='Model'):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=12)
    lo, hi = y_true.min(), y_true.max()
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    plt.title(f"Scatter: True vs Predicted — {label}")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f'scatter_true_pred_{label.lower()}.png'))
    plt.close()


def plot_absolute_error_over_time(y_true, y_pred, label='Model'):
    abs_error = np.abs(y_true - y_pred)
    plt.figure(figsize=(12, 5))
    plt.plot(abs_error, color='red', linewidth=0.8)
    plt.title(f"Absolute Error Over Samples — {label}")
    plt.xlabel("Sample Index")
    plt.ylabel("Absolute Error")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f'abs_error_time_{label.lower()}.png'))
    plt.close()


def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'],     label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("LSTM Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'training_history.png'))
    plt.close()


def print_comparison_table(metrics: dict):
    print("\n" + "=" * 52)
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 52)
    for name, m in metrics.items():
        print(f"{name:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f}")
    print("=" * 52)

    names = list(metrics.keys())
    maes  = [metrics[n]['MAE']  for n in names]
    rmses = [metrics[n]['RMSE'] for n in names]
    r2s   = [metrics[n]['R2']   for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title, color in zip(
        axes,
        [maes, rmses, r2s],
        ['MAE (lower=better)', 'RMSE (lower=better)', 'R² (higher=better)'],
        ['#5DCAA5', '#7F77DD', '#EF9F27'],
    ):
        ax.bar(names, vals, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(bottom=0)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
    plt.suptitle("XGBoost vs LSTM — Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'model_comparison_bars.png'))
    plt.close()


def plot_per_county(y_test, y_pred_xgb, y_pred_lstm, regions_test):
    """
    One plot per county showing True, XGBoost, and LSTM predictions.
    X-axis is 'months into test period' (the last ~15% of each county's timeline).
    """
    counties = sorted(set(regions_test))
    print(f"\nGenerating per-county plots for {len(counties)} counties ...")

    for county in counties:
        mask = regions_test == county
        n = mask.sum()
        if n == 0:
            continue

        x_axis = np.arange(n)  

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x_axis, y_test[mask],      label='True',    linewidth=1.8, color='#2C2C2A')
        ax.plot(x_axis, y_pred_xgb[mask],  label='XGBoost', linestyle='--', alpha=0.85, color='#1D9E75')
        ax.plot(x_axis, y_pred_lstm[mask], label='LSTM',    linestyle=':',  alpha=0.85, color='#7F77DD')
        ax.set_title(f"True vs Predicted — {county}  (last ~15% of timeline)")
        ax.set_xlabel("Months into test period")
        ax.set_ylabel("Unemployment Rate (%)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        safe_name = county.replace(' ', '_').replace(',', '').lower()
        out_path  = os.path.join(PROCESSED_DIR, f'county_{safe_name}.png')
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved: county_{safe_name}.png  ({n} test samples)")


def plot_per_county_metrics(y_test, y_pred_xgb, y_pred_lstm, regions_test):
    """
    Bar chart comparing MAE of XGBoost vs LSTM for every county side-by-side.
    """
    counties = sorted(set(regions_test))
    xgb_maes, lstm_maes = [], []

    for county in counties:
        mask = regions_test == county
        xgb_maes.append(mean_absolute_error(y_test[mask], y_pred_xgb[mask]))
        lstm_maes.append(mean_absolute_error(y_test[mask], y_pred_lstm[mask]))

    short = [c.replace(' County', '') for c in counties]
    x     = np.arange(len(counties))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, xgb_maes,  w, label='XGBoost', color='#1D9E75', alpha=0.85)
    ax.bar(x + w / 2, lstm_maes, w, label='LSTM',    color='#7F77DD', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha='right')
    ax.set_ylabel("MAE")
    ax.set_title("Per-County MAE — XGBoost vs LSTM")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'county_mae_comparison.png'))
    plt.close()
    print("Per-county MAE chart saved → county_mae_comparison.png")


def plot_shap(xgb_model, X_test_flat, feature_cols):
    """SHAP summary + waterfall for the XGBoost model."""
    n_shap = min(200, X_test_flat.shape[0])
    print(f"\nGenerating SHAP values ({n_shap} samples) ...")
    X_shap = X_test_flat[:n_shap]

    explainer   = shap.Explainer(xgb_model)
    shap_values = explainer(X_shap)

    # Build flattened feature names: <feature>_lag<t>  (lag36 … lag1)
    if X_shap.shape[1] > len(feature_cols):
        num_timesteps = X_shap.shape[1] // len(feature_cols)
        flattened_cols = [
            f"{col}_lag{t}"
            for t in range(num_timesteps, 0, -1)
            for col in feature_cols
        ]
        final_feature_names = flattened_cols
    else:
        final_feature_names = feature_cols

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, feature_names=final_feature_names, show=False)
    plt.title("SDoH Feature Impact on Unemployment Rate (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'shap_summary.png'))
    plt.close()
    print("SHAP summary saved → shap_summary.png")

    # Waterfall for a single prediction
    plt.figure()
    shap_obj = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_shap[0],
        feature_names=final_feature_names,
    )
    shap.plots.waterfall(shap_obj, show=False)
    plt.title("SHAP Waterfall — Single Prediction Explained")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'shap_waterfall.png'))
    plt.close()
    print("SHAP waterfall saved → shap_waterfall.png")




def main():
   
    X_test       = np.load(X_test_path)
    y_test       = np.load(y_test_path)
    regions_test = np.load(REGIONS_TEST_PATH, allow_pickle=True)

    print(f"X_test: {X_test.shape}  y_test: {y_test.shape}")
    print(f"Counties in test set ({len(set(regions_test))}): {sorted(set(regions_test))}")

    with open(FEATURES_PATH, 'rb') as f:
        feature_cols = pickle.load(f)


    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred_xgb  = xgb_model.predict(X_test_flat)

   
    lstm_model  = load_model(LSTM_MODEL_PATH, compile=False)
    y_pred_lstm = lstm_model.predict(X_test).flatten()

   
    metrics = {}
    for name, y_pred in [('XGBoost', y_pred_xgb), ('LSTM', y_pred_lstm)]:
        metrics[name] = {
            'MAE':  mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2':   r2_score(y_test, y_pred),
        }
    print_comparison_table(metrics)

    with open(os.path.join(PROCESSED_DIR, 'analysis_results.txt'), 'w') as f:
        for name, m in metrics.items():
            f.write(f"{name}: MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}\n")
    print("Metrics saved → analysis_results.txt")

    
    for name, y_pred in [('XGBoost', y_pred_xgb), ('LSTM', y_pred_lstm)]:
        plot_true_vs_predicted(y_test, y_pred, label=name)
        plot_residuals(y_test, y_pred, label=name)
        plot_scatter_true_vs_pred(y_test, y_pred, label=name)
        plot_absolute_error_over_time(y_test, y_pred, label=name)

   
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        plot_training_history(history)

    
    plot_per_county(y_test, y_pred_xgb, y_pred_lstm, regions_test)
    plot_per_county_metrics(y_test, y_pred_xgb, y_pred_lstm, regions_test)

    
    plot_shap(xgb_model, X_test_flat, feature_cols)

    print("\nAll outputs saved to data/processed/")


if __name__ == "__main__":
    main()