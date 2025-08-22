"""
Data shaping & batching utilities.
- Sparse master storage (E x T) for flows/mask.
- Time features.
- Observation splits.
- Windowed densification to keep RAM small.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Iterator, Optional
import numpy as np
import pandas as pd
from scipy import sparse

def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".feather", ".ft"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table format: {ext}")

def load_inputs(roads_path: str, utd_path: str, node_map_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roads = _read_table(roads_path)
    utd = _read_table(utd_path)
    node_map = _read_table(node_map_path)
    return roads, utd, node_map

def build_edge_index(roads: pd.DataFrame):
    """Create canonical edge index and lookup maps."""
    cols = ['u','v','key','road_idx']
    assert all(c in roads.columns for c in cols), f"roads must contain {cols}"
    edges_df = roads[cols].drop_duplicates().reset_index(drop=True)
    # sort to ensure stability
    edges_df = edges_df.sort_values(['u','v','key','road_idx']).reset_index(drop=True)
    edges_df['node_tuple'] = list(zip(edges_df['u'], edges_df['v'], edges_df['key']))
    edge_idx_of_node = {t:i for i,t in enumerate(edges_df['node_tuple'])}
    node_tuple_of_edge = edges_df['node_tuple'].tolist()
    E = len(edges_df)
    return edges_df, edge_idx_of_node, node_tuple_of_edge, E

def build_time_index(utd_filtered: pd.DataFrame):
    """Exact sorted time vector to train on (after any filtering to bins)."""
    utd_filtered['date'] = utd_filtered['date'].astype(str)
    times = np.sort(pd.to_datetime(utd_filtered['date'].dropna().unique()).values)
    time_to_col = {t:i for i,t in enumerate(times)}
    T = len(times)
    return times, time_to_col, T

def build_flow_mask(
    utd: pd.DataFrame,
    node_map: pd.DataFrame,
    edge_idx_of_node: Dict[tuple, int],
    time_to_col: Dict[np.datetime64, int],
    E: int,
    T: int,
    dtype: str = "float32"
):
    """
    Convert long table to sparse X (flows) and M (mask) of shape (E,T), CSR.
    """
    assert {'road_idx','date','total_flow'}.issubset(utd.columns)
    assert {'road_idx','u','v','key'}.issubset(node_map.columns)

    # check for graph mismatch
    known = set(edge_idx_of_node.keys())
    nm = node_map[['u','v','key']].drop_duplicates()
    nm['node_tuple'] = list(zip(nm['u'], nm['v'], nm['key']))
    missing = nm.loc[~nm['node_tuple'].isin(known)]
    if not missing.empty:
        print(f"[WARN] {len(missing)} node_map edges not in roads edge index; they will be ignored.")

    df = utd[['road_idx','date','total_flow']].dropna().copy()
    df['date'] = pd.to_datetime(df['date']).values
    df = df.merge(node_map[['road_idx','u','v','key']], on='road_idx', how='left')
    df['node_tuple'] = list(zip(df['u'], df['v'], df['key']))
    df = df[df['node_tuple'].isin(edge_idx_of_node)]

    rows = np.fromiter((edge_idx_of_node[t] for t in df['node_tuple'].values), int, count=len(df))
    cols = np.fromiter((time_to_col[d] for d in df['date'].values), int, count=len(df))
    vals = df['total_flow'].astype(dtype).values

    X = sparse.coo_matrix((vals, (rows, cols)), shape=(E, T)).tocsr()
    M = X.copy()
    M.data[:] = 1.0
    return X, M

def make_temporal_features(times: np.ndarray, E: int):
    """Return hours,dows as (E,T,1) and cyclic features as (E,T,4), float32."""
    dt = pd.to_datetime(times)
    hours = dt.hour.values.astype(np.float32)[None, :]
    dows  = dt.dayofweek.values.astype(np.float32)[None, :]
    hours = np.repeat(hours, E, axis=0)[..., None]
    dows  = np.repeat(dows,  E, axis=0)[..., None]

    h_sin = np.sin(2*np.pi*hours/24); h_cos = np.cos(2*np.pi*hours/24)
    d_sin = np.sin(2*np.pi*dows/7);   d_cos = np.cos(2*np.pi*dows/7)
    temporal_feats = np.concatenate([h_sin, h_cos, d_sin, d_cos], axis=-1).astype(np.float32)
    return hours.astype(np.float32), dows.astype(np.float32), temporal_feats

def make_observation_split(M: sparse.csr_matrix, val_frac: float = 0.1, seed: int = 42):
    """Randomly split observed indices into train/val masks (same sparsity pattern as M)."""
    rng = np.random.default_rng(seed)
    r, c = M.nonzero()
    n = len(r)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _mask_from_indices(rr, cc):
        data = np.ones(len(rr), dtype=np.float32)
        return sparse.coo_matrix((data, (rr, cc)), shape=M.shape).tocsr()

    M_train = _mask_from_indices(r[tr_idx], c[tr_idx])
    M_val   = _mask_from_indices(r[val_idx], c[val_idx])
    return M_train, M_val

def window_generator(
    X: sparse.csr_matrix,
    M_train: sparse.csr_matrix,
    M_val: Optional[sparse.csr_matrix],
    times: np.ndarray,
    window: int,
    step: int
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, int, int]]:
    """
    Yield dense windows (float32) to keep memory small.
    Yields: Xb, Mb_train, Mb_val (or None), hours_b, dows_b, s, e
    """
    E, T = X.shape
    for s in range(0, T, step):
        e = min(s + window, T)
        Xb = X[:, s:e].toarray().astype('float32')
        Mb_train = M_train[:, s:e].toarray().astype('bool')
        Mb_val = M_val[:, s:e].toarray().astype('bool') if M_val is not None else None

        # temporal features for this window
        E_ = E
        dt = pd.to_datetime(times[s:e])
        hours = dt.hour.values.astype(np.float32)[None, :]
        dows  = dt.dayofweek.values.astype(np.float32)[None, :]
        hours = np.repeat(hours, E_, axis=0)[..., None]
        dows  = np.repeat(dows,  E_, axis=0)[..., None]

        yield Xb, Mb_train, Mb_val, hours, dows, s, e
