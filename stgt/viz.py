"""
Quick visual helpers you can call from notebooks.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_corridor(times, yhat, ytrue, edge_ids, title="Predictions vs Observations"):
    t = np.asarray(times)
    for e in edge_ids:
        plt.figure(figsize=(10,3))
        plt.plot(t, yhat[e], label="pred")
        obs_mask = ~np.isnan(ytrue[e])
        plt.scatter(t[obs_mask], ytrue[e, obs_mask], s=10, label="obs", alpha=0.7)
        plt.title(f"{title} | edge {e}")
        plt.legend(); plt.tight_layout(); plt.show()
