"""
Evaluation metrics and simple reports.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse import csr_matrix, issparse

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
import torch

@torch.no_grad()
def evaluate_period(
    model,
    window_generator,
    X: sparse.csr_matrix, #np.ndarray,           # [E, T] original flows (NaNs where missing)
    M_obs: sparse.csr_matrix, #np.ndarray,       # [E, T] bool mask of observed entries in this period
    times,                   # aligned times for this period
    window: int,
    step: int,
    Ulap=None,               # pass your LapPE or None
    use_mask_channel: bool = True,   # set False if your model expects 1 input channel
    use_log1p: bool = True,          # match your training transform
    hide_frac: float | None = None,  # if set, randomly hide this frac of observed entries for scoring
    rng: np.random.Generator | None = None,
):
    """
    Returns: metrics dict, yhat_full np.ndarray[E,T]
    - If hide_frac is None: scores on observed entries (reconstruction).
    - If hide_frac > 0: scores only on the hidden subset (true imputation).
    """
    model.eval()
    device = next(model.parameters()).device

    if issparse(X):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(X, dtype=np.float32)

    if issparse(M_obs):
        M_dense = M_obs.toarray().astype(bool)
    else:
        M_dense = np.asarray(M_obs, dtype=bool)

    E, T = X_dense.shape
    X_in = np.nan_to_num(X_dense, nan=0.0)

    # Build input/train mask for the model & a separate scoring mask
    if hide_frac is not None:
        rng = np.random.default_rng() if rng is None else rng
        # hide = np.zeros_like(M_obs, dtype=bool)
        obs_idx = np.argwhere(M_dense) #M_obs)
        n_hide = int(len(obs_idx) * hide_frac)
        hide_dense = np.zeros_like(M_dense, dtype=bool)
        if n_hide > 0 and len(obs_idx)>0:
            sel = rng.choice(len(obs_idx), size=n_hide, replace=False)
            hide_dense[tuple(obs_idx[sel].T)] = True #hide[tuple(obs_idx[sel].T)] = True
        M_in_dense = M_dense & (~hide_dense) #M_in = M_obs & (~hide)   # model sees observed minus hidden
        M_score_dense = hide_dense #M_score = hide           # we score only on hidden entries
    else:
        M_in_dense = M_dense.copy() #M_in = M_obs.copy()
        M_score_dense = M_dense.copy() #M_score = M_obs.copy()

    # Accumulators to stitch overlapping windows (works for any step)
    yhat_sum  = np.zeros((E, T), dtype=np.float32)
    yhat_count = np.zeros((E, T), dtype=np.int32)

    for Xb, Mb_tr, _Mb_val, hours, dows, s, e in window_generator(
        csr_matrix(X_in), csr_matrix(M_in_dense), None, times, window, step
    ):
        xb = torch.from_numpy(Xb).float().unsqueeze(-1).to(device)     # (E,W,1)
        if use_log1p:
            xb = torch.log1p(xb)
        if use_mask_channel:
            mb_in = torch.from_numpy(Mb_tr).to(device).unsqueeze(-1).float()
            xb = torch.cat([xb, mb_in], dim=-1)                        # (E,W,2)

        # Features, Target
        x = torch.from_numpy(np.nan_to_num(Xb, nan=0.0)).float().unsqueeze(-1).to(device)   # (E,W,1)
        y = torch.from_numpy(np.nan_to_num(Xb, nan=0.0)).float().to(device)                 # (E,W)
        m = torch.from_numpy(Mb_tr).bool().to(device)                                       # (E,W)
        h = torch.from_numpy(hours).float().to(device)
        d = torch.from_numpy(dows).float().to(device)
        print('[EVAL] build features ok', flush=True)

        # Normalise target and input for stability
        y = torch.log1p(y)   # log(1 + y) -> compresses large values
        x = torch.log1p(x)   # optional: do the same to input features
        # x = torch.cat([x, m.unsqueeze(-1).float()], dim=-1)  # (E, W, 2)
        print('[EVAL] normalise x,y ok', flush=True)

        yhat = model(x, h, d, Ulap)                                   # (E,W) on log or linear scale
        yhat = torch.expm1(yhat) if use_log1p else yhat                # back to original scale

        y_np = yhat.detach().cpu().numpy().astype(np.float32)
        yhat_sum[:, s:e]  += y_np
        yhat_count[:, s:e] += 1

    # Average overlaps; leave untouched positions as NaN
    yhat_full = np.full((E, T), np.nan, dtype=np.float32)
    nz = yhat_count > 0
    yhat_full[nz] = yhat_sum[nz] / yhat_count[nz]

    # Compute metrics on the chosen scoring mask
    metrics = compute_metrics(yhat_full, X_dense, M_score_dense & ~np.isnan(X_dense)) #M_score & ~np.isnan(X))
    print('[EVAL] compute metrics ok')
    return metrics, yhat_full

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

