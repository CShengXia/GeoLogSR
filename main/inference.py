import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from config import DEVICE

def predict(model, x_data):
    model.eval()
    with torch.no_grad():
        x_tensor = x_data.to(DEVICE)
        y_pred_tensor = model(x_tensor)
    y_pred = y_pred_tensor.cpu().numpy().squeeze()
    return y_pred

def calculate_metrics(y_true, y_pred):

    min_len = min(len(y_true), len(y_pred))
    y_true = np.asarray(y_true[:min_len])
    y_pred = np.asarray(y_pred[:min_len])
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pcc = pearsonr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "PCC": pcc, "R2": r2}

def save_predictions(x_values, y_values, output_path):
    df = pd.DataFrame({0: x_values, 1: y_values})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, header=False, index=False)
    return output_path

def save_metrics(metrics, output_path, curve_name):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df = pd.DataFrame({
        "Curve": [curve_name],
        "MSE": [metrics["MSE"]],
        "MAE": [metrics["MAE"]],
        "RMSE": [metrics["RMSE"]],
        "PCC": [metrics["PCC"]],
        "R2": [metrics["R2"]]
    })
    metrics_df.to_csv(output_path, index=False)