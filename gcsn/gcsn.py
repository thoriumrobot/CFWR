# gcsn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# --------- Utilities ---------
def cosine_anneal(start, end, t, T):
    if T <= 0: return end
    alpha = 0.5 * (1 + math.cos(math.pi * min(t, T) / T))
    return end + (start - end) * alpha

# --------- L0 (Concrete) feature gates ---------
class L0Mask(nn.Module):
    def __init__(self, D, tau=2./3., lmbda=1e-4, init_logit=0.5):
        super().__init__()
        self.logit = nn.Parameter(torch.full((D,), float(init_logit)))
        self._tau0 = tau
        self.tau = tau
        self.lmbda = lmbda

    def set_tau(self, tau: float):
        self.tau = max(1e-3, float(tau))

    def forward(self, x, training: bool = True):
        if training:
            u = torch.rand_like(self.logit)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logit) / self.tau)
            z = s  # in (0,1), reparameterized
        else:
            z = (self.logit > 0).float()  # hard at test time
        return x * z, z

    def penalty(self):
        # Use sigmoid(logit) as a smooth proxy for E[L0]
        return self.lmbda * torch.sigmoid(self.logit).sum()

# --------- Straight-through top-k gate converted to continuous edge feature ---------
def st_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.numel() == 0 or k <= 0:
        return torch.zeros_like(scores)
    k = min(k, scores.numel())
    idx = torch.topk(scores, k).indices
    hard = torch.zeros_like(scores)
    hard[idx] = 1.0
    soft = torch.sigmoid(scores)  # smooth scores
    return hard + soft - soft.detach()  # straight-through

class EdgeGate(nn.Module):
    def __init__(self, in_dim_for_scoring, hidden=64, k_ratio=0.2, lmbda=1e-5, init_bias=0.5):
        super().__init__()
        self.k_ratio = float(k_ratio)
        self.lmbda = lmbda
        self.scorer = nn.Sequential(
            nn.Linear(in_dim_for_scoring, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        nn.init.constant_(self.scorer[-1].bias, init_bias)

    def forward(self, node_x, edge_index, edge_attr=None):
        # Build edge features for scoring:
        # If edge_attr is provided, concatenate it with src/dst node embeddings.
        src, dst = edge_index
        if edge_attr is None:
            feat = torch.cat([node_x[src], node_x[dst]], dim=-1)  # (E, 2D)
        else:
            feat = torch.cat([node_x[src], node_x[dst], edge_attr], dim=-1)
        scores = self.scorer(feat).squeeze(-1)  # (E,)
        k = max(1, int(self.k_ratio * scores.numel())) if scores.numel() > 0 else 0
        mask = st_topk_mask(scores, k)  # (E,), in [0,1] with straight-through gradient
        return mask, scores

    def penalty(self, mask):
        return self.lmbda * mask.sum()

# --------- GCSN Core ---------
class GCSN(nn.Module):
    """
    Feature L0 gates + edge top-k gate passed as edge_attr to GATv2,
    so attention can learn to zero/scale messages.
    """
    def __init__(self, in_dim, num_classes, hidden=128, heads=4, layers=2,
                 l0_tau=2./3., l0_lambda=1e-4, l0_init_logit=0.5,
                 edge_k_ratio=0.2, edge_lambda=1e-5, edge_hidden=64):
        super().__init__()
        self.l0 = L0Mask(in_dim, tau=l0_tau, lmbda=l0_lambda, init_logit=l0_init_logit)
        # edge scorer uses [x_src || x_dst] (+ optional attr)
        self.edge_gate = EdgeGate(2*in_dim, hidden=edge_hidden, k_ratio=edge_k_ratio, lmbda=edge_lambda)

        convs = []
        for i in range(layers):
            in_ch = in_dim if i == 0 else hidden
            # edge_dim=1 to inject the gate mask as a scalar edge feature
            convs.append(GATv2Conv(in_channels=in_ch, out_channels=hidden // heads,
                                   heads=heads, concat=True, edge_dim=1))
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

        # Schedulers’ state
        self._tau0 = l0_tau

    def set_temperature(self, tau: float):
        self.l0.set_tau(tau)

    def forward(self, x, edge_index, y=None, train_mask=None, edge_attr=None):
        # 1) Feature L0 gate
        x_masked, z = self.l0(x, self.training)  # z in [0,1]
        # 2) Edge gate: returns mask in [0,1] with STE
        mask, raw_scores = self.edge_gate(x_masked, edge_index, edge_attr=None)
        # 3) Message passing with mask injected as edge_attr (shape [E,1])
        ea = mask.unsqueeze(-1)

        h = x_masked
        for i, (conv, ln) in enumerate(zip(self.convs, self.norms)):
            if i == 0:
                # First layer: no residual connection due to dimension mismatch
                h = conv(h, edge_index, ea)
                h = ln(F.elu(h))
            else:
                # Subsequent layers: residual connection
                h_res = h
                h = conv(h, edge_index, ea)
                h = ln(F.elu(h) + h_res)  # pre-norm residual

        logits = self.head(h)
        # Penalties for sparsity
        penalties = self.l0.penalty() + self.edge_gate.penalty(mask)

        out = {
            "logits": logits,
            "penalty": penalties,
            "feat_mask": torch.sigmoid(self.l0.logit).detach(),  # expected keep per feature
            "edge_mask": mask.detach(),                          # per-edge gate (after STE)
            "edge_scores": raw_scores.detach()
        }
        return out
