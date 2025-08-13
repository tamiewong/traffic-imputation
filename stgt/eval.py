"""
Evaluation metrics and simple reports.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_metrics(yhat: np.ndarray, ytrue: np.ndarray, mask: np.ndarray):
    yh = yhat[mask]; yt = ytrue[mask]
    mae = np.mean(np.abs(yh - yt)) if len(yh) else np.nan
    rmse = np.sqrt(np.mean((yh - yt)**2)) if len(yh) else np.nan
    # R^2 with mean over observed
    if len(yh):
        ybar = np.mean(yt)
        ss_res = np.sum((yt - yh)**2)
        ss_tot = np.sum((yt - ybar)**2) + 1e-9
        r2 = 1.0 - ss_res/ss_tot
    else:
        r2 = np.nan
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def metrics_by_group(yhat: np.ndarray, ytrue: np.ndarray, mask: np.ndarray, edges_df: pd.DataFrame, group_col: str):
    df = edges_df[[group_col]].copy()
    df['edge_idx'] = np.arange(len(edges_df))
    rows = []
    for g, sub in df.groupby(group_col):
        idx = sub['edge_idx'].values[:, None]
        m = mask[idx, :]
        if m.sum() == 0:
            rows.append({group_col: g, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
            continue
        yh = yhat[idx, :][m]
        yt = ytrue[idx, :][m]
        rows.append({"MAE": float(np.mean(np.abs(yh - yt))),
                     "RMSE": float(np.sqrt(np.mean((yh - yt)**2))),
                     "R2": float(1 - np.sum((yt-yh)**2)/(np.sum((yt-yt.mean())**2)+1e-9)),
                     group_col: g})
    return pd.DataFrame(rows)
