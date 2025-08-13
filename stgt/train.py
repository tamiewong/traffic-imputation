"""
Training orchestration (CPU-friendly; windowed).
Usage:
    python -m stgt.train --config configs/stgt.yaml
"""
from __future__ import annotations
import argparse, os, json
import yaml
import os, numpy as np
import torch
import torch.nn as nn
from scipy import sparse
import sys

from .data import load_inputs, build_edge_index, build_time_index, build_flow_mask, make_observation_split, window_generator
from .graph import build_edge_adjacency_lists, build_neighbor_index, compute_laplacian_pe, build_incidence
from .model import STGT
from .losses import impute_loss

import time
def tic(msg): 
    print(msg); return time.time()
def toc(t, msg=""): 
    print(f"{msg} took {time.time()-t:.2f}s")
    


def seed_everything(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_torch_sparse_coo(mat_csr: sparse.csr_matrix, device=None, dtype=torch.float32):
    coo = mat_csr.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data).to(dtype)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)

def train_model(config_path: str):
    print('Start training'); sys.stdout.flush()
    cfg = yaml.safe_load(open(config_path, "r"))
    seed_everything(cfg.get("seed", 42))

    # Load inputs
    paths = cfg["paths"]
    roads, utd, node_map = load_inputs(paths["roads"], paths["utd"], paths["node_map"])
    times, time_to_col, T = build_time_index(utd)
    edges_df, edge_idx_of_node, node_tuple_of_edge, E = build_edge_index(roads)

    # --- sanity on shapes & config ---
    print(f"[SHAPE] E={E} edges, T={T} timesteps")
    window = int(cfg.get("window", 96))
    step   = int(cfg.get("step", 96))
    epochs = int(cfg.get("epochs", 20))
    print(f"[CFG] window={window}, step={step}, epochs={epochs}")

    if T == 0:
        print("[FATAL] T == 0 (no timestamps). Check your 'utd' date parsing / filtering.")
        return
    if window <= 0 or step <= 0:
        print("[FATAL] window/step must be positive.")
        return


    cache_dir = paths.get("cache", os.path.join(paths["outputs"], "cache"))
    os.makedirs(cache_dir, exist_ok=True)

    # Graph precomutes with caching

    # Adjacency lists
    adj_path = os.path.join(cache_dir, "adj.npy")
    if os.path.exists(adj_path):
        adj = np.load(adj_path, allow_pickle=True).tolist()
    else:
        adj = build_edge_adjacency_lists(edges_df)
        np.save(adj_path, np.array(adj, dtype=object), allow_pickle=True)

    # Neighbors
    neigh_path = os.path.join(cache_dir, f"neighbors_H{cfg['spd_hops']}_K{cfg['K_neighbors']}.npy")
    if os.path.exists(neigh_path):
        neighbors = np.load(neigh_path)
    else:
        neighbors = build_neighbor_index(adj, H=cfg["spd_hops"], K=cfg["K_neighbors"])
        np.save(neigh_path, neighbors)

    # LapPE (only if enabled)
    if cfg["K_lappe"] > 0:
        pe_path = os.path.join(cache_dir, f"U_lappe_K{cfg['K_lappe']}.npy")
        if os.path.exists(pe_path):
            U_lappe = np.load(pe_path)
        else:
            U_lappe = compute_laplacian_pe(edges_df, adj, K_lappe=cfg["K_lappe"])
            np.save(pe_path, U_lappe)
    else:
        U_lappe = np.zeros((len(edges_df), 0), dtype=np.float32)



    # Build flows/masks (sparse)
    print('Building flows/masks...'); sys.stdout.flush()
    X_sparse, M_sparse = build_flow_mask(utd, node_map, edge_idx_of_node, time_to_col, E, T)

    # Split observed entries into train/val
    M_train, M_val = make_observation_split(M_sparse, val_frac=cfg.get("val_frac", 0.1), seed=cfg.get("seed", 42))

    # # Graph structures
    # print('Building graph structures...')
    # # Sanity check
    # deg = np.array([len(n) for n in adj])
    # print(f"E={len(adj)} | avg_deg={deg.mean():.2f} | max_deg={deg.max()} | total_edges={deg.sum()}")

    # # t0=tic("Building adjacency lists"); adj = build_edge_adjacency_lists(edges_df); toc(t0,"adj")
    # # t0=tic("Building neighbor index"); neighbors = build_neighbor_index(adj, H=cfg.get("spd_hops", 3), K=cfg.get("K_neighbors", 32)); toc(t0,"neighbors")
    # # t0=tic("Computing LapPE"); U_lappe = compute_laplacian_pe(edges_df, adj, K_lappe=cfg.get("K_lappe", 16)); toc(t0,"LapPE")
    # adj = build_edge_adjacency_lists(edges_df)
    # neighbors = build_neighbor_index(adj, H=cfg.get("spd_hops", 3), K=cfg.get("K_neighbors", 32))
    # U_lappe = compute_laplacian_pe(edges_df, adj, K_lappe=cfg.get("K_lappe", 16))

    # Incidence (for conservation)
    print('Building incidence csr...'); sys.stdout.flush()
    incidence_csr, _ = build_incidence(edges_df)

    # Torch tensors that are static across windows
    print('Building torch tensors (static across windows)...'); sys.stdout.flush()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", False) else "cpu")
    neighbor_index = torch.from_numpy(neighbors).long().to(device)
    Ulap = torch.from_numpy(U_lappe).float().to(device)
    incidence_t = to_torch_sparse_coo(incidence_csr, device=device)

    # Model
    print('Building model...'); sys.stdout.flush()
    model = STGT(
        d_in= 1 + int(cfg.get("time2vec_k", 6)),
        d_model=int(cfg.get("d_model", 96)),
        n_heads=int(cfg.get("heads", 4)),
        K_lappe=int(cfg.get("K_lappe", 16)),
        neighbor_index=neighbor_index,
        time2vec_k=int(cfg.get("time2vec_k", 6)),
        n_layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        temporal_edge_batch = int(cfg.get("temporal_edge_batch", 2048)),
        spatial_edge_batch  = int(cfg.get("spatial_edge_batch", 4096))
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 0.002), weight_decay=cfg.get("weight_decay", 0.0001))
    grad_clip = cfg.get("grad_clip", 1.0)

    window = cfg.get("window", 96)
    step   = cfg.get("step", 96)
    epochs = cfg.get("epochs", 20)
    lam_c  = cfg.get("lambda_conserve", 0.001)



    # Training loop
    print('Training...'); sys.stdout.flush()
    out_dir = paths.get("outputs", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Peek one window so we know training will actually have batches
    probe = next(window_generator(X_sparse, M_train, M_val, times, window, step), None)
    if probe is None:
        print("[FATAL] window_generator produced 0 windows. "
            "Reduce 'window', fix 'times', or check M_train/M_val building.")
        return
    else:
        _, Mb_tr_probe, _, _, _, s_probe, e_probe = probe
        print(f"[OK] First window indices: [{s_probe}:{e_probe}), "
            f"observed entries in train mask: {Mb_tr_probe.sum()}")

    try:
        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            win_i = 0

            for Xb, Mb_tr, Mb_val, hours, dows, s, e in window_generator(X_sparse, M_train, M_val, times, window, step):
                any_batch = True
                win_i += 1
                print(f"[DEBUG] epoch {ep} - window {win_i}", flush=True)
                # stop after N windows (debug mode)
                if cfg.get("max_windows_debug") and win_i > cfg["max_windows_debug"]:
                    print("[DEBUG] Stopping early after max_windows_debug windows.", flush=True)
                    break

                # Features, Target
                x = torch.from_numpy(np.nan_to_num(Xb, nan=0.0)).float().unsqueeze(-1).to(device)   # (E,W,1)
                y = torch.from_numpy(np.nan_to_num(Xb, nan=0.0)).float().to(device)                 # (E,W)

                # Normalise target and input for stability
                y = torch.log1p(y)   # log(1 + y) -> compresses large values
                x = torch.log1p(x)   # optional: do the same to input features

                m = torch.from_numpy(Mb_tr).bool().to(device)                                       # (E,W)
                h = torch.from_numpy(hours).float().to(device)
                d = torch.from_numpy(dows).float().to(device)

                opt.zero_grad(set_to_none=True)

                t0 = time.time()
                # print('x, h, d, Ulap = ', x, h, d, Ulap)
                yhat = model(x, h, d, Ulap)
                t1 = time.time()
                print(f"[FORWARD] ok in {t1 - t0:.2f}s", flush=True)

                loss = impute_loss(yhat, y, m, incidence_t, lam_c=lam_c)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

                total_loss += loss.item()
                n_batches += 1

            last_avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {ep+1}/{epochs} - train_loss: {last_avg_loss:.4f}"); sys.stdout.flush()

            # save checkpoint every epoch
            ckpt_path = os.path.join(out_dir, f"stgt_ep{ep+1}.pt")
            torch.save({"epoch": ep+1, "model_state": model.state_dict(), "cfg": cfg}, ckpt_path)
            print(f"[SAVE] checkpoint -> {os.path.abspath(ckpt_path)}"); sys.stdout.flush()

            # (Optional) quick val over one window:
            # You can add a validation loop similar to the train loop with Mb_val.

        if not any_batch:
            print("[WARN] No training windows generated. Check your time range, window, and step."); sys.stdout.flush()

    finally:
        # Save a final snapshot
        print('[SAVE] final snapshot...'); sys.stdout.flush()
        final_path = os.path.join(out_dir, "stgt.pt")
        torch.save({"epoch": epochs, "model_state": model.state_dict(), "cfg": cfg}, final_path)
        summary_path = os.path.join(out_dir, "train_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"epochs": epochs, "train_loss": float(last_avg_loss)}, f, indent=2)
        print(f"[SAVE] final model -> {os.path.abspath(final_path)}")
        print(f"[SAVE] summary     -> {os.path.abspath(summary_path)}")
        sys.stdout.flush()

        # torch.save(model.state_dict(), os.path.join(out_dir, "stgt.pt"))
        # with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        #     json.dump({"epochs": epochs, "train_loss": float(avg_loss)}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stgt.yaml")
    args = ap.parse_args()
    train_model(args.config)
