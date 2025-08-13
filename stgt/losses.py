"""
Losses: masked MSE + flow-conservation penalty.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_mse(yhat: torch.Tensor, ytrue: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
    diff = (yhat - ytrue)[mask_bool]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=yhat.device, dtype=yhat.dtype)
    return (diff * diff).mean()

def conservation_penalty(yhat: torch.Tensor, incidence_sparse: torch.Tensor) -> torch.Tensor:
    """
    yhat: (E,T) dense
    incidence_sparse: torch.sparse_coo_tensor (V,E)
    Returns mean squared imbalance across nodes and time.
    """
    # (V,E) @ (E,T) -> (V,T)
    cons = torch.sparse.mm(incidence_sparse, yhat)  # (V,T)
    return (cons * cons).mean()

def impute_loss(yhat, ytrue, mask_bool, incidence_sparse, lam_c: float = 1e-3):
    return masked_mse(yhat, ytrue, mask_bool) + lam_c * conservation_penalty(yhat, incidence_sparse)
