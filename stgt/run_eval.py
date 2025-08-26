
# stgt/run_eval.py
import os, sys, json, glob, argparse, torch, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from stgt.model import STGT
from stgt.eval import evaluate_period, compute_metrics
from stgt.data import window_generator

def find_latest_ckpt(dirpath="outputs", pattern="stgt_*.pt"):
    paths = glob.glob(os.path.join(dirpath, pattern))
    if not paths:
        return None
    # newest by modification time
    return max(paths, key=os.path.getmtime)

def prompt_ckpt(default_path: str | None) -> str:
    hint = f" [{default_path}]" if default_path else ""
    try:
        user_inp = input(f"Enter path to checkpoint .pt file{hint}: ").strip().strip('"').strip("'")
    except EOFError:
        user_inp = ""
    ckpt = user_inp or default_path
    if not ckpt:
        raise FileNotFoundError("No checkpoint path provided and no default found.")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not ckpt.lower().endswith(".pt"):
        raise ValueError(f"Expected a .pt file, got: {ckpt}")
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default=os.environ.get("STGT_CKPT"),
                        help="Path to checkpoint .pt (overrides prompt).")
    args = parser.parse_args()

    default_ckpt = find_latest_ckpt("outputs", "stgt_*.pt")
    ckpt_path = args.ckpt or prompt_ckpt(default_ckpt)
    print(f"[INFO] using checkpoint -> {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)

    cfg   = state["cfg"]
    sdict = state["model_state"]
    epoch = state["epoch"]
    print(f"[INFO] load checkpoint ok -> {ckpt_path} (epoch {epoch})", flush=True)

    # --- load neighbors + LapPE from cfg paths ---
    neigh_path = os.path.join("outputs\cache", f"neighbors_H{cfg['spd_hops']}_K{cfg['K_neighbors']}.npy")
    neighbors = np.load(neigh_path)
    # neighbors = np.load(cfg["paths"]["neighbors"])
    neighbor_index = torch.from_numpy(neighbors).long()
    E = neighbors.shape[0] # neighbors: (E, K)

    Ulap, K_lappe = None, 0
    if "U_lappe" in cfg["paths"] and cfg["paths"]["U_lappe"]:
        U_lappe = np.load(cfg["paths"]["U_lappe"]).astype("float32")
        Ulap = torch.from_numpy(U_lappe)
        if Ulap is not None: Ulap = Ulap.to(device).float()
        K_lappe = U_lappe.shape[-1]

    print('[INFO] load neighbors + LapPE ok', flush=True)

    # --- build row-normalized sparse adjacency ---
    rows, cols = np.repeat(np.arange(E), neighbors.shape[1]), neighbors.reshape(-1) # cols = neighbors.ravel()
    valid = (cols >= 0) & (cols < E)
    rows, cols = rows[valid], cols[valid]
    vals = np.ones(rows.shape[0], dtype=np.float32) # vals = np.ones_like(rows, dtype=np.float32)
    # row-normalize using filtered rows
    row_deg = np.bincount(rows, minlength=E).astype(np.float32)
    vals = vals / np.maximum(row_deg[rows], 1.0)

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values  = torch.tensor(vals, dtype=torch.float32, device=device)
    sp_adj  = torch.sparse_coo_tensor(indices, values, (E, E), device=device).coalesce()
    print(f"[INFO] build sp_adj ok -> nnz={sp_adj._nnz()}", flush=True)

    # --- rebuild model ---
    model = STGT(
        d_in = 1 + int(cfg["time2vec_k"]),
        d_model=cfg["d_model"],
        n_heads=cfg["heads"],
        K_lappe=K_lappe,
        neighbor_index=neighbor_index,
        time2vec_k=cfg.get("time2vec_k",6),
        n_layers=cfg.get("layers",1),
        dropout=cfg.get("dropout",0.2),
        temporal_edge_batch = int(cfg.get("temporal_edge_batch", 2048)),
        spatial_edge_batch  = int(cfg.get("spatial_edge_batch", 4096)),
        spatial_mode=cfg['spatial_mode'], 
        sp_adj=sp_adj if cfg['spatial_mode']=="sparsemm" else None
    )
    model.load_state_dict(sdict)
    model.to(device)
    model.eval()
    print('[INFO] rebuild model ok', flush=True)


    # --- build X_sparse + times ---
    # # paths = cfg["paths"]
    # # roads, utd, node_map = load_inputs(paths["roads"], paths["utd"], paths["node_map"])
    # # times, time_to_col, T = build_time_index(utd)
    # # edges_df, edge_idx_of_node, node_tuple_of_edge, E = build_edge_index(roads)
    # # X_sparse, M_sparse = build_flow_mask(utd, node_map, edge_idx_of_node, time_to_col, E, T)

    edges_df, utd = pd.read_parquet(cfg["paths"]["roads"]), pd.read_parquet(cfg["paths"]["utd"])
    utd["date"] = pd.to_datetime(utd["date"])
    times = np.array(sorted(utd["date"].unique()))
    edge_order = edges_df.sort_values("road_idx")["road_idx"].to_numpy()

    pt = utd.pivot_table(index="road_idx", columns="date", values="total_flow", aggfunc="mean")
    pt = pt.reindex(index=edge_order, columns=times)
    X_sparse = pt.to_numpy(dtype=np.float32)
    print('[INFO] build X_sparse + time ok', flush=True)


    # # --- test on last 20% of time ---
    # T = X_sparse.shape[1]
    # t0 = int(T * 0.8)
    # X_test = X_sparse[:, t0:] # csr_matrix(X_sparse[:, t0:])
    # times_test = times[t0:]
    # M_test = ~np.isnan(X_test)
    # print('[INFO] test on last 20% of time', flush=True)

    # # --- test on random 20% of time ---
    T = X_sparse.shape[1]
    rng = np.random.default_rng(42)   # fixed seed for reproducibility
    test_idx = rng.choice(T, size=int(T*0.2), replace=False)
    # mask for test times
    test_mask = np.zeros(T, dtype=bool)
    test_mask[test_idx] = True
    X_test = X_sparse[:, test_mask]
    times_test = times[test_mask]
    M_test = ~np.isnan(X_test)
    print('[INFO] test on random 20% of time', flush=True)


    # --- run eval ---
    metrics, _ = evaluate_period(
        model, window_generator,
        csr_matrix(X_test), csr_matrix(M_test), times_test,
        window=cfg["window"], step=cfg["step"],
        Ulap=Ulap,
        use_mask_channel=True,
        use_log1p=True,
    )
    print("[TEST/Recon]", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    main()



# from .eval import evaluate_period, metrics_by_group
# import argparse, os, json
# import yaml
# import os, numpy as np
# import torch
# from datetime import datetime

# from .data import load_inputs, build_edge_index, build_time_index, build_flow_mask, make_observation_split, window_generator
# from .model import STGT

# # # load config
# # ap = argparse.ArgumentParser()
# # ap.add_argument("--config", default="configs/stgt.yaml")
# # args = ap.parse_args()
# # cfg = yaml.safe_load(open(args.config, "r"))
# # # cfg = yaml.safe_load(open("output/config.yaml"))
# # print('[TEST] load config ok', flush=True)

# # # rebuild model
# # model = STGT(cfg)   # replace with your model class + init
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)
# # print('[TEST] rebuild model ok', flush=True)

# # load weights
# ckpt_path = "outputs/stgt_20250825-172800.pt"
# state = torch.load(ckpt_path, map_location=device)
# cfg = state["cfg"]               # exact training config
# epoch = state["epoch"]           # last epoch trained
# sdict = state["model_state"]     # the trained weights
# print("[INFO] checkpoint from epoch", epoch)
# print("[INFO] config keys:", list(cfg.keys()))

# # model.load_state_dict(state["model_state_dict"])   # if you saved with optimizer, etc.
# # model.eval()
# # print('[TEST] load weights ok', flush=True)

# # # load inputs
# # paths = cfg["paths"]
# # roads, utd, node_map = load_inputs(paths["roads"], paths["utd"], paths["node_map"])
# # times, time_to_col, T = build_time_index(utd)
# # edges_df, edge_idx_of_node, node_tuple_of_edge, E = build_edge_index(roads)
# # X_sparse, M_sparse = build_flow_mask(utd, node_map, edge_idx_of_node, time_to_col, E, T)

# # cache_dir = paths.get("cache", os.path.join(paths["outputs"], "cache"))
# # out_dir = paths.get("outputs", "outputs")
# # ulap_path = os.path.join(cache_dir, "U_lappe.npy")   # adjust if different
# # if os.path.exists(ulap_path):
# #     U_lappe = np.load(ulap_path).astype(np.float32)   # e.g., [N_nodes, k] or [E, k] depending on your pipeline
# #     Ulap = torch.from_numpy(U_lappe).to(device)
# # else:
# #     Ulap = None
# # print('[TEST] load inputs ok', flush=True)

# # Build model using cfg values
# neighbor_index = torch.from_numpy(np.load(cfg["paths"]["neighbors"])).long()
# Ulap = None
# K_lappe = 0
# if "U_lappe" in cfg["paths"] and cfg["paths"]["U_lappe"]:
#     import numpy as np
#     U_lappe = np.load(cfg["paths"]["U_lappe"]).astype("float32")
#     Ulap = torch.from_numpy(U_lappe)
#     K_lappe = U_lappe.shape[-1]

# model = STGT(
#     d_model=cfg["d_model"],
#     n_heads=cfg["heads"],
#     K_lappe=K_lappe,
#     neighbor_index=neighbor_index,
#     layers=cfg.get("layers", 1),
#     dropout=cfg.get("dropout", 0.2),
#     time2vec_k=cfg.get("time2vec_k", 6),
#     # add in_dim here if you trained with mask channel
# )
# model.load_state_dict(sdict)
# model.eval()

# # # choose a test slice (example: last 20% of time)
# # E, T = X_sparse.shape
# # t0 = int(T * 0.8)
# # X_test = X_sparse[:, t0:]
# # M_test = ~np.isnan(X_test)
# # times_test = times[t0:]

# # A) Reconstruction on observed entries
# metrics_rec, yhat_rec = evaluate_period(
#     model, window_generator,
#     X_test, M_test, times_test,
#     window=cfg["window"], step=cfg["step"],
#     Ulap=Ulap,
#     use_mask_channel=True,   # set False if you did NOT concat the mask during training
#     use_log1p=True,
#     hide_frac=None
# )
# print("[TEST/Recon]", metrics_rec)

# # # B) True imputation: hide 20% of observed entries and score only on them
# # metrics_imp, yhat_imp = evaluate_period(
# #     model, window_generator,
# #     X_test, M_test, times_test,
# #     window=cfg["window"], step=cfg["step"],
# #     Ulap=Ulap,
# #     use_mask_channel=True,
# #     use_log1p=True,
# #     hide_frac=0.20, rng=np.random.default_rng(42)
# # )
# # print("[TEST/Impute@20%]", metrics_imp)

# # (optional) by-group breakdown
# # edges_df must align with E (edge order)
# # print(metrics_by_group(yhat_rec, X_test, M_test & ~np.isnan(X_test), edges_df, "highway"))
