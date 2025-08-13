"""
Graph features & constraints.
- Build edge adjacency from edges_df (line-graph behavior).
- Laplacian positional encodings (optional, K small).
- Neighbor index (E x K) via BFS up to H hops (no giant E x E tensors).
- Incidence matrix for flow conservation on original nodes.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from scipy import sparse
from scipy.sparse.linalg import eigsh

def build_edge_adjacency_lists(edges_df: pd.DataFrame):
    """
    Treat each directed edge e=(u->v). Its 1-hop neighbors are all edges that start at v.
    Returns: adj_list: Dict[int, List[int]] for edges 0..E-1
    """
    E = len(edges_df)
    out_by_u = defaultdict(list)
    for ei, (u, v, k) in enumerate(edges_df[['u','v','key']].itertuples(index=False, name=None)):
        out_by_u[u].append(ei)
    adj = [[] for _ in range(E)]
    for ei, (u, v, k) in enumerate(edges_df[['u','v','key']].itertuples(index=False, name=None)):
        adj[ei] = out_by_u.get(v, [])
    return adj  # List[List[int]] length E

def build_neighbor_index(adj: List[List[int]], H: int = 3, K: int = 32):
    """
    From 1-hop adjacency (edge->list of neighbor edges), compute up-to-H-hop neighbor
    sets per edge using BFS, keep up to K per edge. Always include self at index 0.
    Returns: neighbor_index: np.ndarray[int64] shape (E, K) padded with self.
    """
    E = len(adj)
    neighbors = np.full((E, K), -1, dtype=np.int64)
    for s in range(E):
        seen = {s}
        q = deque([(s,0)])
        layer = []
        while q and len(layer) < (K-1):
            u, d = q.popleft()
            if d == 0:
                pass  # skip adding self here; we'll set it later
            else:
                layer.append(u)
            if d >= H:
                continue
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append((v, d+1))
        # assemble with self at position 0
        row = [s] + layer[:K-1]
        if len(row) < K:
            row = row + [s] * (K - len(row))
        neighbors[s, :] = np.array(row, dtype=np.int64)
    return neighbors  # (E,K)

def compute_laplacian_pe(edges_df: pd.DataFrame, adj: List[List[int]], K_lappe: int = 16):
    """
    Compute normalized Laplacian eigenvectors on the (undirected) edge adjacency graph.
    Output: U_lappe (E, K_lappe) float32 with sign-stabilized columns.
    """
    E = len(edges_df)
    rows, cols = [], []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
    if len(rows) == 0:
        # disconnected graph edge case
        U = np.zeros((E, K_lappe), dtype=np.float32)
        return U

    A = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(E, E)).tocsr()
    deg = np.array(A.sum(1)).ravel()
    D_inv_sqrt = sparse.diags(1.0/np.sqrt(np.clip(deg, 1e-6, None)))
    Lap = sparse.eye(E, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt

    k = min(K_lappe, max(E-2, 1))
    try:
        evals, evecs = eigsh(Lap.asfptype(), k=k, which='SM', tol=1e-3)
        U = evecs.astype(np.float32)
        signs = np.sign(U.sum(axis=0, keepdims=True)); signs[signs==0] = 1
        U *= signs
    except Exception:
        # fallback: random small PE if eigsh fails
        rng = np.random.default_rng(0)
        U = rng.normal(0, 1, size=(E, k)).astype(np.float32)
    if U.shape[1] < K_lappe:
        pad = np.zeros((E, K_lappe - U.shape[1]), dtype=np.float32)
        U = np.concatenate([U, pad], axis=1)
    return U[:, :K_lappe]

def build_incidence(edges_df: pd.DataFrame):
    """
    Node–edge incidence on original graph nodes.
    +1 on tail u, -1 on head v.
    Returns: incidence csr (V x E), node index mapping.
    """
    nodes = pd.unique(edges_df[['u','v']].values.ravel())
    node_to_idx = {n:i for i,n in enumerate(nodes)}
    V = len(nodes)

    rows, cols, vals = [], [], []
    for ei, (u, v, k) in enumerate(edges_df[['u','v','key']].itertuples(index=False, name=None)):
        rows += [node_to_idx[u], node_to_idx[v]]
        cols += [ei, ei]
        vals += [ +1.0, -1.0 ]
    incidence = sparse.coo_matrix((vals, (rows, cols)), shape=(V, len(edges_df))).tocsr()
    return incidence, node_to_idx
