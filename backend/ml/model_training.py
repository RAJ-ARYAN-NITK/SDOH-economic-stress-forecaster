import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

PROCESSED_DIR   = '/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/data/processed/'
X_PATH          = os.path.join(PROCESSED_DIR, 'X_sequences.npy')
Y_PATH          = os.path.join(PROCESSED_DIR, 'y_target.npy')
REGIONS_PATH    = os.path.join(PROCESSED_DIR, 'regions.npy')
FEATURES_PATH   = os.path.join(PROCESSED_DIR, 'feature_columns.pkl')
LSTM_MODEL_PATH = os.path.join(PROCESSED_DIR, 'er_lstm_model.keras')
XGB_MODEL_PATH  = os.path.join(PROCESSED_DIR, 'xgb_model.pkl')
SCALER_PATH     = os.path.join(PROCESSED_DIR, 'scaler.pkl')
HISTORY_PATH    = os.path.join(PROCESSED_DIR, 'history.pkl')


X       = np.load(X_PATH)
y       = np.load(Y_PATH)
regions = np.load(REGIONS_PATH, allow_pickle=True)

with open(FEATURES_PATH, 'rb') as f:
    feature_columns = pickle.load(f)

print(f"Loaded: X={X.shape}  y={y.shape}  regions={regions.shape}")
print(f"All counties: {sorted(set(regions))}")


train_idx, val_idx, test_idx = [], [], []

print("\nPer-county split:")
for county in sorted(set(regions)):
    idx = np.where(regions == county)[0]   
    n   = len(idx)
    t   = int(n * 0.70)
    v   = int(n * 0.15)
    train_idx.extend(idx[:t])
    val_idx.extend(idx[t:t + v])
    test_idx.extend(idx[t + v:])
    print(f"  {county}: total={n}  train={t}  val={v}  test={n - t - v}")

train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)
test_idx  = np.array(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]
regions_test     = regions[test_idx]

print(f"\nSplit sizes — train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")
print(f"Counties in test set: {sorted(set(regions_test))}")


n_samples_tr, seq_len, n_feat = X_train.shape

scaler = StandardScaler()

X_train_2d = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
X_val_2d_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
X_test_2d_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)


X_train_flat = X_train_2d.reshape(X_train.shape[0], -1)
X_val_flat   = X_val_2d_s.reshape(X_val.shape[0], -1)
X_test_flat  = X_test_2d_s.reshape(X_test.shape[0], -1)


with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'),       X_test_2d_s)
np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'),       y_test)
np.save(os.path.join(PROCESSED_DIR, 'regions_test.npy'), regions_test)
print("\nScaler and test split saved.")


print("\nChecking for NaNs / Infs ...")
for name, arr in [('X_train', X_train_2d), ('y_train', y_train),
                  ('X_val',   X_val_2d_s), ('y_val',   y_val),
                  ('X_test',  X_test_2d_s),('y_test',  y_test)]:
    print(f"  {name}: NaNs={np.isnan(arr).sum()}  Infs={np.isinf(arr).sum()}")


print("\n" + "=" * 60)
print("BLOCK A: Training XGBoost")
print("=" * 60)

xgb_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=20, eval_metric='mae', random_state=42,
)
xgb_model.fit(
    X_train_flat, y_train,
    eval_set=[(X_val_flat, y_val)],
    verbose=50,
)

y_pred_xgb = xgb_model.predict(X_test_flat)
mae_xgb    = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb   = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb     = r2_score(y_test, y_pred_xgb)
print(f"\nXGBoost  →  MAE: {mae_xgb:.4f}  RMSE: {rmse_xgb:.4f}  R²: {r2_xgb:.4f}")

with open(XGB_MODEL_PATH, 'wb') as f:
    pickle.dump(xgb_model, f)
np.save(os.path.join(PROCESSED_DIR, 'y_pred_xgb.npy'), y_pred_xgb)
print("XGBoost model saved.")


print("\n" + "=" * 60)
print("BLOCK B: Training LSTM")
print("=" * 60)

model = Sequential([
    Input(shape=(seq_len, n_feat)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = ModelCheckpoint(LSTM_MODEL_PATH, save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train_2d, y_train,
    epochs=100, batch_size=32,
    validation_data=(X_val_2d_s, y_val),
    callbacks=[es, mc],
)

with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history.history, f)

if os.path.exists(LSTM_MODEL_PATH):
    model.load_weights(LSTM_MODEL_PATH)

y_pred_lstm = model.predict(X_test_2d_s).flatten()
mae_lstm    = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm   = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
r2_lstm     = r2_score(y_test, y_pred_lstm)
print(f"\nLSTM     →  MAE: {mae_lstm:.4f}  RMSE: {rmse_lstm:.4f}  R²: {r2_lstm:.4f}")

np.save(os.path.join(PROCESSED_DIR, 'y_pred_lstm.npy'), y_pred_lstm)
print("LSTM model saved.")


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 40)
print(f"{'XGBoost':<12} {mae_xgb:>8.4f} {rmse_xgb:>8.4f} {r2_xgb:>8.4f}")
print(f"{'LSTM':<12} {mae_lstm:>8.4f} {rmse_lstm:>8.4f} {r2_lstm:>8.4f}")

winner = "XGBoost" if mae_xgb < mae_lstm else "LSTM"
print(f"\nWinner by MAE: {winner}")

with open(os.path.join(PROCESSED_DIR, 'model_comparison.txt'), 'w') as f:
    f.write(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8}\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'XGBoost':<12} {mae_xgb:>8.4f} {rmse_xgb:>8.4f} {r2_xgb:>8.4f}\n")
    f.write(f"{'LSTM':<12} {mae_lstm:>8.4f} {rmse_lstm:>8.4f} {r2_lstm:>8.4f}\n")
    f.write(f"\nWinner by MAE: {winner}\n")


plt.figure(figsize=(14, 5))
plt.plot(y_test,      label='True',    alpha=0.9, linewidth=1.5)
plt.plot(y_pred_xgb,  label='XGBoost', alpha=0.7, linestyle='--')
plt.plot(y_pred_lstm, label='LSTM',    alpha=0.7, linestyle=':')
plt.title("True vs Predicted — XGBoost vs LSTM (Test Set, all counties)")
plt.xlabel("Sample index")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'model_comparison_plot.png'))
plt.close()
print("\nAll files saved to data/processed/")