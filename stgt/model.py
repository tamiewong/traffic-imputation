"""
Spatio-Temporal Graph Transformer (memory-safe).
- TemporalBlock: MHA along time (per edge).
- SpatialNeighborAttentionBlock: multihead attention over a fixed K neighbor set per edge (no E x E).
- STGT wrapper combining temporal + spatial blocks, LapPE, Time2Vec, and softplus head.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        assert k >= 2
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, k-1)
    def forward(self, t):  # (E,T,1)
        return torch.cat([self.lin(t), torch.sin(self.per(t))], dim=-1)  # (E,T,k)

class TemporalBlock(nn.Module):
    """Temporal self-attention along T, processed in edge chunks to avoid E×T×d blow-ups."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 256, dropout: float = 0.2, edge_chunk: int = 2048):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.edge_chunk = edge_chunk

    def forward(self, x):  # x: (E,T,d)
        E, T, d = x.shape
        out = torch.empty_like(x)
        chunk = self.edge_chunk
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            xi = x[s:e]                             # (e-s, T, d)
            yi,_ = self.mha(xi, xi, xi, need_weights=False)
            yi = self.ln1(xi + self.drop(yi))
            zi = self.ffn(yi)
            out[s:e] = self.ln2(yi + self.drop(zi))
        return out

class SpatialNeighborAttentionBlock(nn.Module):
    """
    Attention over a fixed K neighbor set per edge, processed in edge chunks per time step.
    neighbor_index: (E,K) LongTensor with self at index 0.
    """
    def __init__(self, d_model: int, n_heads: int, neighbor_index: torch.LongTensor,
                 dropout: float = 0.1, edge_chunk: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)
        self.neighbor_index = neighbor_index  # (E,K)
        self.drop = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.edge_chunk = edge_chunk

    def _attend_t_chunk(self, Q, Kx, Vx, idx_chunk):
        """
        Q: (E,H,dh), Kx/Vx: (E,H,dh), idx_chunk: (Ec,K)
        returns: (Ec,d)
        """
        Ec, K = idx_chunk.size()
        # slice the queries for this chunk
        # gather neighbor keys/values for this chunk: (Ec, K, H, dh)
        Knbr = Kx[idx_chunk]  # advanced indexing over dim0
        Vnbr = Vx[idx_chunk]
        Qc   = Q[:Ec]         # caller passes chunked Q
        # attention: (Ec,H,K)
        attn = torch.einsum("ehd,ekhd->ehk", Qc, Knbr) / (self.d_head ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        Z = torch.einsum("ehk,ekhd->ehd", attn, Vnbr).contiguous().view(Ec, self.d_model)  # (Ec,d)
        return self.o(Z)

    def forward(self, x):  # x: (E,T,d)
        E, T, d = x.shape
        K = self.neighbor_index.size(1)
        out = torch.empty_like(x)

        # project once per pass
        Q = self.q(x).view(E, T, self.n_heads, self.d_head)
        Kx = self.k(x[:,0,:]).view(E, self.n_heads, self.d_head)  # keys/values are time-local; reuse per t
        Vx = self.v(x[:,0,:]).view(E, self.n_heads, self.d_head)

        # for each time step, process edges in chunks
        for t in range(T):
            Qt = Q[:, t, :, :]        # (E,H,dh)
            start = 0
            while start < E:
                end = min(start + self.edge_chunk, E)
                idx_chunk = self.neighbor_index[start:end]  # (Ec,K)
                Qtc = Qt[start:end]                         # (Ec,H,dh)
                # gather neighbors for this chunk using global Kx/Vx
                Knbr = Kx[idx_chunk]                        # (Ec,K,H,dh)
                Vnbr = Vx[idx_chunk]
                attn = torch.einsum("ehd,ekhd->ehk", Qtc, Knbr) / (self.d_head ** 0.5)
                attn = torch.softmax(attn, dim=-1)
                attn = self.drop(attn)
                Z = torch.einsum("ehk,ekhd->ehd", attn, Vnbr).contiguous().view(end-start, self.d_model)
                out[start:end, t, :] = self.o(Z)
                start = end

        # residual + FFN
        h = self.ln1(x + self.drop(out))
        y = self.ffn(h)
        h = self.ln2(h + self.drop(y))
        return h

class STGT(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        K_lappe: int,
        neighbor_index: torch.LongTensor,
        time2vec_k: int = 6,
        n_layers: int = 2,
        dropout: float = 0.2,
        temporal_edge_batch: int = 2048,
        spatial_edge_batch: int = 4096
    ):
        super().__init__()
        self.time2vec = Time2Vec(time2vec_k)
        self.proj_in = nn.Linear(d_in, d_model)
        # LapPE optional
        self.use_pe = (K_lappe is not None) and (K_lappe > 0)
        if self.use_pe:
            self.pe_spatial = nn.Linear(K_lappe, d_model, bias=False)
        else:
            self.register_parameter("pe_spatial", None)

        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(d_model, n_heads, dropout=dropout, edge_chunk=temporal_edge_batch)
            for _ in range(n_layers)
        ])
        self.spatial_blocks  = nn.ModuleList([
            SpatialNeighborAttentionBlock(d_model, n_heads, neighbor_index, dropout=dropout, edge_chunk=spatial_edge_batch)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, hours, dows, U_lappe):
        """
        x: (E,T,1); hours,dows: (E,T,1); U_lappe: (E,K) or empty if not used
        """
        t2v = self.time2vec(hours)                 # (E,T,k)
        h = torch.cat([x, t2v], dim=-1)            # (E,T,1+k)
        h = self.proj_in(h)                        # (E,T,d)
        if self.use_pe and U_lappe.numel() > 0:
            h = h + self.pe_spatial(U_lappe).unsqueeze(1)

        for tblk, sblk in zip(self.temporal_blocks, self.spatial_blocks):
            h = tblk(h)
            h = sblk(h)

        return F.softplus(self.head(h)).squeeze(-1)  # (E,T)