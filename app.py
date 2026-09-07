import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import shap


PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')

st.set_page_config(
    page_title="SDoH Unemployment Predictor",
    page_icon="📊",
    layout="wide",
)
@st.cache_resource
def load_all():
    X_test       = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
    y_test       = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    regions_test = np.load(os.path.join(PROCESSED_DIR, 'regions_test.npy'), allow_pickle=True)

    with open(os.path.join(PROCESSED_DIR, 'xgb_model.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)

    lstm_model = load_model(os.path.join(PROCESSED_DIR, 'er_lstm_model.keras'), compile=False)

    with open(os.path.join(PROCESSED_DIR, 'feature_columns.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)

    history = None
    hist_path = os.path.join(PROCESSED_DIR, 'history.pkl')
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            history = pickle.load(f)

    return X_test, y_test, regions_test, xgb_model, lstm_model, feature_cols, history


@st.cache_data
def get_predictions(_xgb_model, _lstm_model, X_test):
    X_flat      = X_test.reshape(X_test.shape[0], -1)
    y_pred_xgb  = _xgb_model.predict(X_flat)
    y_pred_lstm = _lstm_model.predict(X_test, verbose=0).flatten()
    return y_pred_xgb, y_pred_lstm, X_flat



def metrics(y_true, y_pred):
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred),
    }


def fig_county(y_test, y_pred_xgb, y_pred_lstm, regions_test, county):
    mask   = regions_test == county
    x_axis = np.arange(mask.sum())
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_axis, y_test[mask],      label='True',    linewidth=2,    color='#111111')
    ax.plot(x_axis, y_pred_xgb[mask],  label='XGBoost', linestyle='--', alpha=0.85, color='#1D9E75')
    ax.plot(x_axis, y_pred_lstm[mask], label='LSTM',    linestyle=':',  alpha=0.85, color='#7F77DD')
    ax.set_title(f"{county} — True vs Predicted (last ~15% of timeline)", fontsize=13)
    ax.set_xlabel("Months into test period")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def fig_residuals(y_true, y_pred, label):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, linewidth=0.8)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_title(f"Residuals — {label}")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[1].hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title(f"Residual Distribution — {label}")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def fig_scatter(y_true, y_pred, label):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=10)
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    ax.set_title(f"Scatter: True vs Predicted — {label}")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def fig_training_history(history):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['loss'],     label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_title("LSTM Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def fig_county_mae(y_test, y_pred_xgb, y_pred_lstm, regions_test):
    counties  = sorted(set(regions_test))
    xgb_maes  = [mean_absolute_error(y_test[regions_test == c], y_pred_xgb[regions_test == c])  for c in counties]
    lstm_maes = [mean_absolute_error(y_test[regions_test == c], y_pred_lstm[regions_test == c]) for c in counties]
    short = [c.replace(' County', '') for c in counties]
    x, w  = np.arange(len(counties)), 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, xgb_maes,  w, label='XGBoost', color='#1D9E75', alpha=0.85)
    ax.bar(x + w/2, lstm_maes, w, label='LSTM',    color='#7F77DD', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha='right')
    ax.set_ylabel("MAE")
    ax.set_title("Per-County MAE — XGBoost vs LSTM")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


@st.cache_data
def get_shap_figs(_xgb_model, X_flat, feature_cols):
    n      = min(200, X_flat.shape[0])
    X_shap = X_flat[:n]

    explainer   = shap.Explainer(_xgb_model)
    shap_values = explainer(X_shap)

    n_feat = len(feature_cols)
    if X_shap.shape[1] > n_feat:
        nt    = X_shap.shape[1] // n_feat
        names = [f"{col}_lag{t}" for t in range(nt, 0, -1) for col in feature_cols]
    else:
        names = feature_cols

    fig_sum, _ = plt.subplots()
    shap.summary_plot(shap_values, X_shap, feature_names=names, show=False)
    plt.title("SDoH Feature Impact (SHAP Summary)")
    plt.tight_layout()

    fig_wf, _ = plt.subplots()
    shap_obj = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_shap[0],
        feature_names=names,
    )
    shap.plots.waterfall(shap_obj, show=False)
    plt.title("SHAP Waterfall — Single Prediction")
    plt.tight_layout()

    return fig_sum, fig_wf



with st.spinner("Loading models and data..."):
    X_test, y_test, regions_test, xgb_model, lstm_model, feature_cols, history = load_all()

with st.spinner("Running predictions..."):
    y_pred_xgb, y_pred_lstm, X_flat = get_predictions(xgb_model, lstm_model, X_test)

counties = sorted(set(regions_test))


st.sidebar.title("🗂 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Per-County Analysis", "Residuals & Scatter", "LSTM Training", "SHAP Explainability"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Test samples:** {len(y_test)}")
st.sidebar.markdown(f"**Counties:** {len(counties)}")
st.sidebar.markdown(f"**Features:** {len(feature_cols)}")


if page == "Overview":
    st.title("📊 SDoH Unemployment Rate Predictor")
    st.markdown("XGBoost vs LSTM comparison across **11 Massachusetts counties** (2015–2024)")

    st.subheader("Global Model Performance")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    m_xgb  = metrics(y_test, y_pred_xgb)
    m_lstm = metrics(y_test, y_pred_lstm)
    col1.metric("XGBoost MAE",  f"{m_xgb['MAE']:.4f}")
    col2.metric("XGBoost RMSE", f"{m_xgb['RMSE']:.4f}")
    col3.metric("XGBoost R²",   f"{m_xgb['R2']:.4f}")
    col4.metric("LSTM MAE",     f"{m_lstm['MAE']:.4f}")
    col5.metric("LSTM RMSE",    f"{m_lstm['RMSE']:.4f}")
    col6.metric("LSTM R²",      f"{m_lstm['R2']:.4f}")

    st.markdown("---")
    st.subheader("Metric Comparison")
    fig_bar, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mname, vals, color in zip(
        axes,
        ['MAE (lower=better)', 'RMSE (lower=better)', 'R² (higher=better)'],
        [[m_xgb['MAE'], m_lstm['MAE']], [m_xgb['RMSE'], m_lstm['RMSE']], [m_xgb['R2'], m_lstm['R2']]],
        ['#5DCAA5', '#7F77DD', '#EF9F27'],
    ):
        ax.bar(['XGBoost', 'LSTM'], vals, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(mname, fontsize=11)
        ax.set_ylim(bottom=min(0, min(vals) - 0.05))
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    plt.suptitle("XGBoost vs LSTM — Overall Performance", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

    st.markdown("---")
    st.subheader("Per-County MAE Breakdown")
    st.pyplot(fig_county_mae(y_test, y_pred_xgb, y_pred_lstm, regions_test))
    plt.close()

    st.markdown("---")
    st.subheader("Global True vs Predicted")
    model_choice  = st.selectbox("Select model", ["XGBoost", "LSTM"])
    y_pred_sel    = y_pred_xgb if model_choice == "XGBoost" else y_pred_lstm
    fig_g, ax_g   = plt.subplots(figsize=(14, 4))
    ax_g.plot(y_test,     label='True',                    alpha=0.8, linewidth=1)
    ax_g.plot(y_pred_sel, label=f'Predicted ({model_choice})', alpha=0.7, linestyle='--')
    ax_g.set_xlabel("Sample Index")
    ax_g.set_ylabel("Unemployment Rate (%)")
    ax_g.set_title(f"True vs Predicted — {model_choice} (all counties concatenated)")
    ax_g.legend()
    ax_g.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_g)
    plt.close()


elif page == "Per-County Analysis":
    st.title("🗺 Per-County Analysis")

    selected_county = st.selectbox("Select county", counties)
    mask            = regions_test == selected_county
    m_xgb_c         = metrics(y_test[mask], y_pred_xgb[mask])
    m_lstm_c        = metrics(y_test[mask], y_pred_lstm[mask])

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("XGBoost MAE",  f"{m_xgb_c['MAE']:.4f}")
    col2.metric("XGBoost RMSE", f"{m_xgb_c['RMSE']:.4f}")
    col3.metric("XGBoost R²",   f"{m_xgb_c['R2']:.4f}")
    col4.metric("LSTM MAE",     f"{m_lstm_c['MAE']:.4f}")
    col5.metric("LSTM RMSE",    f"{m_lstm_c['RMSE']:.4f}")
    col6.metric("LSTM R²",      f"{m_lstm_c['R2']:.4f}")

    st.pyplot(fig_county(y_test, y_pred_xgb, y_pred_lstm, regions_test, selected_county))
    plt.close()

    st.subheader("Absolute Error over Test Period")
    fig_ae, ax_ae = plt.subplots(figsize=(12, 3))
    ax_ae.plot(np.abs(y_test[mask] - y_pred_xgb[mask]),  label='XGBoost', color='#1D9E75', alpha=0.85)
    ax_ae.plot(np.abs(y_test[mask] - y_pred_lstm[mask]), label='LSTM',    color='#7F77DD', alpha=0.85)
    ax_ae.set_xlabel("Months into test period")
    ax_ae.set_ylabel("Absolute Error")
    ax_ae.set_title(f"Absolute Error — {selected_county}")
    ax_ae.legend()
    ax_ae.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_ae)
    plt.close()


elif page == "Residuals & Scatter":
    st.title("📉 Residuals & Scatter Plots")

    model_choice = st.radio("Select model", ["XGBoost", "LSTM"], horizontal=True)
    y_pred_sel   = y_pred_xgb if model_choice == "XGBoost" else y_pred_lstm

    st.subheader("Residuals")
    st.pyplot(fig_residuals(y_test, y_pred_sel, model_choice))
    plt.close()

    st.subheader("Scatter: True vs Predicted")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(fig_scatter(y_test, y_pred_sel, model_choice))
        plt.close()
    with c2:
        st.markdown("### Interpretation")
        m = metrics(y_test, y_pred_sel)
        st.markdown(f"- **MAE:** {m['MAE']:.4f} percentage points average error")
        st.markdown(f"- **RMSE:** {m['RMSE']:.4f}")
        st.markdown(f"- **R²:** {m['R2']:.4f} — {'good fit ✅' if m['R2'] > 0.5 else 'poor fit ❌'}")
        if m['R2'] < 0:
            st.warning(
                "R² < 0 means the model predicts worse than the mean. "
                "This is common for LSTM when test samples from different "
                "counties are concatenated without temporal continuity."
            )


elif page == "LSTM Training":
    st.title("🧠 LSTM Training History")

    if history:
        st.pyplot(fig_training_history(history))
        plt.close()
        final_train = history['loss'][-1]
        final_val   = history['val_loss'][-1]
        best_val    = min(history['val_loss'])
        best_epoch  = history['val_loss'].index(best_val) + 1
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Train Loss", f"{final_train:.4f}")
        col2.metric("Final Val Loss",   f"{final_val:.4f}")
        col3.metric("Best Val Loss",    f"{best_val:.4f} (epoch {best_epoch})")
    else:
        st.warning("No training history found at data/processed/history.pkl")


elif page == "SHAP Explainability":
    st.title("🔍 SHAP Feature Explainability (XGBoost)")
    st.info("Computing SHAP values for the first 200 test samples...")

    with st.spinner("Running SHAP — this may take a moment..."):
        fig_sum, fig_wf = get_shap_figs(xgb_model, X_flat, feature_cols)

    st.subheader("SHAP Summary Plot")
    st.pyplot(fig_sum)
    plt.close()

    st.subheader("SHAP Waterfall — Single Prediction Explained")
    st.pyplot(fig_wf)
    plt.close()