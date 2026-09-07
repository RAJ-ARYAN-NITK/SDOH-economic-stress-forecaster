
import numpy as np
import pickle


PROCESSED_DIR = "/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/data/processed/"
xgb_model_path = PROCESSED_DIR + "xgb_model.pkl"         
scaler_path    = PROCESSED_DIR + "scaler.pkl"
x_test_path    = PROCESSED_DIR + "X_test.npy"

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


with open(xgb_model_path, "rb") as f:
    model = pickle.load(f)

X_test = np.load(x_test_path)


X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
X_test_scaled = X_test_scaled.reshape(X_test.shape)

X_test_2d = X_test_scaled.reshape(X_test_scaled.shape[0], -1)


y_pred = model.predict(X_test_2d)


np.save(PROCESSED_DIR + "y_pred.npy", y_pred)
print("Prediction complete.")
print("Shape:", y_pred.shape)
print("Sample predictions:\n", y_pred[:15])