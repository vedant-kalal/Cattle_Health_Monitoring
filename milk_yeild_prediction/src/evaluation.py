from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(y_true, y_pred, dataset="Test"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{dataset} → R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return r2, mae, rmse
