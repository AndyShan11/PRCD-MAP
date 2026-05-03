"""
model_trust_propagation.py — Structure-Aware Trust Propagation Module.

Key idea: upgrade the per-group scalar temperature tau to structure-aware trust propagation.
If an edge's neighbors are confirmed trustworthy by the data, the edge's prior trust should
be raised (local consistency).

Implementation: a lightweight GAT (1-2 layers) does message passing on the prior graph:
  tau_ij = MLP(aggregate({P_prior_kl, |W_kl|} for (k,l) in neighborhood of (i,j)))

Theoretical properties:
  - Trust propagation converges to a tighter epsilon-safety bound under an accurate prior
  - Neighborhood consistency provides an additional signal, reducing variance of per-edge trust
  - Compatible with the existing EB framework: a structured extension of the tau parametrization
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureExtractor(nn.Module):
    """
    Edge feature extractor: encodes (P_prior_ij, |W_ij|, group_idx) into an edge embedding.
    """

    def __init__(self, feat_dim: int = 16):
        super().__init__()
        # Input: [P_prior_ij, |W_ij|_normalized, tau_group_embed]
        self.proj = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        # Group embedding (up to 8 groups)
        self.group_embed = nn.Embedding(8, 1)
        nn.init.zeros_(self.group_embed.weight)

    def forward(self, P_prior: torch.Tensor, W_strength: torch.Tensor,
                group_indices: torch.Tensor, off_diag_mask: torch.Tensor):
        """
        P_prior: (d, d), W_strength: (d, d), group_indices: (d, d) LongTensor
        Returns: edge_features (d, d, feat_dim)
        """
        d = P_prior.shape[0]
        # Normalize W_strength
        w_max = W_strength.max().clamp(min=1e-6)
        W_norm = (W_strength / w_max) * off_diag_mask

        # Group embedding
        g_idx = group_indices.clamp(0, 7)
        g_feat = self.group_embed(g_idx).squeeze(-1)  # (d, d)

        # Stack features: (d, d, 3)
        feat = torch.stack([P_prior, W_norm, g_feat], dim=-1)
        return self.proj(feat) * off_diag_mask.unsqueeze(-1)


class EdgeGATLayer(nn.Module):
    """
    Edge-level Graph Attention on the prior graph.

    "Nodes" = edges (i, j); "neighbors" = edges sharing an endpoint.
    That is, N(i, j) = {(k, j) : k != i} U {(i, l) : l != j}  (in-edge and out-edge neighbors).

    Attention weights:
      α_{(i,j),(k,l)} = softmax( LeakyReLU( a^T [h_ij || h_kl] ) )
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert self.head_dim * n_heads == out_dim

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_scale = math.sqrt(self.head_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, edge_feat: torch.Tensor, off_diag_mask: torch.Tensor):
        """
        edge_feat: (d, d, in_dim) -- edge features
        off_diag_mask: (d, d) -- off-diagonal mask

        Returns: (d, d, out_dim) -- updated edge features
        """
        d = edge_feat.shape[0]
        dev = edge_feat.device

        # Flatten valid edges: (d*d, in_dim)
        flat_feat = edge_feat.view(d * d, -1)  # (d*d, in_dim)

        Q = self.W_q(flat_feat).view(d, d, self.n_heads, self.head_dim)
        K = self.W_k(flat_feat).view(d, d, self.n_heads, self.head_dim)
        V = self.W_v(flat_feat).view(d, d, self.n_heads, self.head_dim)

        # Build neighborhoods: neighbors of edge (i, j) are {(k, j): k != i, k != j} U {(i, l): l != j, l != i}
        # Efficient implementation: for each (i, j), aggregate features of the same column (in-edges) and the same row (out-edges)
        # Column aggregation: for column j, all (k, j) with k != j contribute to (i, j)
        # Row aggregation: for row i, all (i, l) with l != i contribute to (i, j)

        # Column attention: Q[i,j] attends to K[k,j] for all k
        # shape: Q[:, :, h, :] is (d, d, head_dim)
        # For edge (i,j), query=Q[i,j], keys=K[:,j]
        # Efficient: (d, d, n_heads, head_dim) x (d, d, n_heads, head_dim) -> attention

        # Reshape for batched attention over columns
        # Q_col[j, i, h, :] = Q[i, j, h, :]  (transpose first two dims)
        Q_col = Q.permute(1, 0, 2, 3)  # (d, d, n_heads, head_dim)
        K_col = K.permute(1, 0, 2, 3)  # (d, d, n_heads, head_dim)
        V_col = V.permute(1, 0, 2, 3)  # (d, d, n_heads, head_dim)

        # Attention scores: (d_col, d_query, n_heads) @ (d_col, d_key, n_heads) -> (d_col, d_query, d_key, n_heads)
        # Batched: for each column j, attention among d edges
        # (d, d, n_heads, head_dim) -> bmm -> (d, d, d, n_heads)
        attn_col = torch.einsum('cqhd,ckhd->cqkh', Q_col, K_col) / self.attn_scale
        # Mask: only off-diagonal edges contribute
        col_mask = off_diag_mask.T.unsqueeze(1).unsqueeze(-1)  # (d, 1, d, 1)
        attn_col = attn_col.masked_fill(col_mask == 0, float('-inf'))
        attn_col = F.softmax(attn_col, dim=2)
        attn_col = self.dropout(attn_col)
        # Aggregate
        out_col = torch.einsum('cqkh,ckhd->cqhd', attn_col, V_col)  # (d, d, n_heads, head_dim)
        out_col = out_col.permute(1, 0, 2, 3)  # back to (d, d, n_heads, head_dim)

        # Row attention: for edge (i,j), attend to (i,l) for all l
        Q_row = Q  # (d, d, n_heads, head_dim) -- row i, col j
        K_row = K
        V_row = V
        attn_row = torch.einsum('rqhd,rkhd->rqkh', Q_row, K_row) / self.attn_scale
        row_mask = off_diag_mask.unsqueeze(1).unsqueeze(-1)  # (d, 1, d, 1)
        attn_row = attn_row.masked_fill(row_mask == 0, float('-inf'))
        attn_row = F.softmax(attn_row, dim=2)
        attn_row = self.dropout(attn_row)
        out_row = torch.einsum('rqkh,rkhd->rqhd', attn_row, V_row)

        # Combine column and row aggregations
        out_combined = (out_col + out_row) / 2.0  # (d, d, n_heads, head_dim)
        out_combined = out_combined.reshape(d, d, self.out_dim)

        # Output projection + residual + LayerNorm
        out = self.out_proj(out_combined)

        # Residual connection (with projection if dims differ)
        if self.in_dim == self.out_dim:
            out = self.layer_norm(out + edge_feat)
        else:
            out = self.layer_norm(out)

        return out * off_diag_mask.unsqueeze(-1)


class TrustPropagationModule(nn.Module):
    """
    Structure-Aware Trust Propagation.

    Inputs: P_prior (d, d), W_strength (d, d), group_indices (d, d)
    Output: tau_matrix (d, d) -- per-edge trust

    Architecture:
      1. EdgeFeatureExtractor: extracts edge features
      2. 1-2 EdgeGATLayer: structure-aware aggregation
      3. tau prediction head: MLP -> sigmoid -> [tau_min, tau_max]

    Interface with the existing code:
      - Replaces the (d, d) tau_matrix returned by _expand_tau()
      - Downstream calls such as calibrated_prior(tau_matrix) are unchanged
    """

    def __init__(self, feat_dim: int = 16, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.0,
                 tau_min: float = 0.05, tau_max: float = 3.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Edge feature extraction
        self.edge_extractor = EdgeFeatureExtractor(feat_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(
                EdgeGATLayer(feat_dim, feat_dim, n_heads=n_heads, dropout=dropout)
            )

        # τ prediction head
        self.tau_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
        )

        # Global bias (analogous to the old tau_groups; provides a global baseline)
        self.tau_bias = nn.Parameter(torch.tensor(0.0))

        self._init_tau_head()

    def _init_tau_head(self):
        """Initialize so the initial tau output is near tau_min (conservative)."""
        # Make the initial output near 0 (sigmoid(0)=0.5); tau_bias then controls the starting point
        for m in self.tau_head:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        # Init tau_bias so that sigmoid(bias) corresponds to tau_min
        # sigmoid(x) * (tau_max - tau_min) + tau_min
        # To get output ~ tau_min we need sigmoid(x) ~ 0, i.e. x << 0
        self.tau_bias.data.fill_(-3.0)

    def forward(self, P_prior: torch.Tensor, W_strength: torch.Tensor,
                group_indices: torch.Tensor, off_diag_mask: torch.Tensor):
        """
        Returns: tau_matrix (d, d) in [tau_min, tau_max]
        """
        # 1. Extract edge features
        edge_feat = self.edge_extractor(P_prior, W_strength, group_indices, off_diag_mask)

        # 2. Message passing
        for gat in self.gat_layers:
            edge_feat = gat(edge_feat, off_diag_mask)

        # 3. Predict per-edge tau
        tau_raw = self.tau_head(edge_feat).squeeze(-1)  # (d, d)
        tau_raw = tau_raw + self.tau_bias

        # 4. Scale to [tau_min, tau_max]
        tau_matrix = torch.sigmoid(tau_raw) * (self.tau_max - self.tau_min) + self.tau_min
        tau_matrix = tau_matrix * off_diag_mask

        return tau_matrix

    def get_tau_mean(self, P_prior, W_strength, group_indices, off_diag_mask):
        """Return mean tau (compatible with the logging interface)."""
        with torch.no_grad():
            tau_mat = self.forward(P_prior, W_strength, group_indices, off_diag_mask)
            mask = off_diag_mask.bool()
            return tau_mat[mask].mean()


class TrustPropagationLite(nn.Module):
    """
    Lightweight trust propagation (no GAT, only local statistics).
    Used in place of the full GAT when d > 50 to save GPU memory.

    τ_ij = MLP( P_prior_ij, mean(P_prior_Nij), std(P_prior_Nij),
                |W_ij|_norm, mean(|W_Nij|), agreement_ij )

    where N(i, j) = (row-i neighborhood) U (column-j neighborhood).
    """

    def __init__(self, tau_min: float = 0.05, tau_max: float = 3.0,
                 hidden: int = 16):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        # 6 features: P_ij, mean_P_N, std_P_N, W_ij_norm, mean_W_N, agreement
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.tau_bias = nn.Parameter(torch.tensor(-3.0))
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, P_prior: torch.Tensor, W_strength: torch.Tensor,
                group_indices: torch.Tensor, off_diag_mask: torch.Tensor):
        d = P_prior.shape[0]
        dev = P_prior.device

        w_max = W_strength.max().clamp(min=1e-6)
        W_norm = (W_strength / w_max) * off_diag_mask

        # Neighborhood statistics
        # Row-wise: for edge (i,j), row neighbors are {(i,l): l≠j, l≠i}
        # Col-wise: for edge (i,j), col neighbors are {(k,j): k≠i, k≠j}
        # Simplified: use row mean and col mean as proxies

        P_row_mean = (P_prior * off_diag_mask).sum(dim=1, keepdim=True) / off_diag_mask.sum(dim=1, keepdim=True).clamp(min=1)
        P_col_mean = (P_prior * off_diag_mask).sum(dim=0, keepdim=True) / off_diag_mask.sum(dim=0, keepdim=True).clamp(min=1)
        P_neigh_mean = (P_row_mean.expand_as(P_prior) + P_col_mean.expand_as(P_prior)) / 2

        P_row_std = ((P_prior - P_row_mean).pow(2) * off_diag_mask).sum(dim=1, keepdim=True) / off_diag_mask.sum(dim=1, keepdim=True).clamp(min=1)
        P_col_std = ((P_prior - P_col_mean).pow(2) * off_diag_mask).sum(dim=0, keepdim=True) / off_diag_mask.sum(dim=0, keepdim=True).clamp(min=1)
        P_neigh_std = ((P_row_std.expand_as(P_prior) + P_col_std.expand_as(P_prior)) / 2).sqrt()

        W_row_mean = (W_norm * off_diag_mask).sum(dim=1, keepdim=True) / off_diag_mask.sum(dim=1, keepdim=True).clamp(min=1)
        W_col_mean = (W_norm * off_diag_mask).sum(dim=0, keepdim=True) / off_diag_mask.sum(dim=0, keepdim=True).clamp(min=1)
        W_neigh_mean = (W_row_mean.expand_as(W_norm) + W_col_mean.expand_as(W_norm)) / 2

        # Agreement: |P_ij - 0.5| * sign_agreement(P vs W)
        # High P + high W → positive, Low P + low W → positive
        agreement = (P_prior - 0.5) * (W_norm - 0.5) * 4  # scale to [-1, 1]

        # Stack features: (d, d, 6)
        feat = torch.stack([
            P_prior, P_neigh_mean, P_neigh_std, W_norm, W_neigh_mean, agreement
        ], dim=-1)

        tau_raw = self.mlp(feat).squeeze(-1) + self.tau_bias
        tau_matrix = torch.sigmoid(tau_raw) * (self.tau_max - self.tau_min) + self.tau_min
        return tau_matrix * off_diag_mask

    def get_tau_mean(self, P_prior, W_strength, group_indices, off_diag_mask):
        with torch.no_grad():
            tau_mat = self.forward(P_prior, W_strength, group_indices, off_diag_mask)
            mask = off_diag_mask.bool()
            return tau_mat[mask].mean()
