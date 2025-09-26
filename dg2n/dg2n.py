import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch_scatter import scatter_add

class L0Mask(nn.Module):
    """
    Hard-Concrete (Louizos et al., 2018) gate.
    - One learnable log_alpha per unit (feature dim or edge type scalar).
    - During training: noisy, differentiable gates in [0,1].
    - During eval: hard gates in {0,1}.
    """
    def __init__(self, size, init_log_alpha=0.0, temperature=2.0/3.0, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((size,), float(init_log_alpha)))
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta

    def _stretch(self, s):
        return s * (self.zeta - self.gamma) + self.gamma

    def _hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def sample_gate(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha).clamp_(1e-6, 1-1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + self.log_alpha) / self.temperature)
            s_bar = self._stretch(s)
            z = self._hard_sigmoid(s_bar)
        else:
            z = (self.log_alpha >= 0).float()
        return z

    def expected_L0(self):
        k = -self.gamma / self.zeta
        return torch.sigmoid(self.log_alpha - self.temperature * torch.log(torch.tensor(k, device=self.log_alpha.device)))

    def forward(self, x):
        z = self.sample_gate(self.training)
        return x * z, z

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden]*(num_layers-1) + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DG2N(nn.Module):
    """
    Deterministic-Gate Graph Network.
    - Feature-level gates (per input channel).
    - Edge-type gates (per relation type).
    - Message passing with per-type MLPs and sum aggregation.
    - Pluggable rule head for neuro-symbolic scoring.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        edge_types: Dict[str, int],
        num_classes: int,
        rule_head: nn.Module = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.edge_types = list(edge_types)
        self.num_classes = num_classes
        self.dropout = dropout

        self.feat_gate = L0Mask(in_dim, init_log_alpha=0.0)
        self.edge_gates = nn.ModuleDict({et: L0Mask(1, init_log_alpha=0.0) for et in self.edge_types})

        self.in_proj = nn.Linear(in_dim, hidden)
        self.msg_mlps = nn.ModuleDict({et: MLP(hidden, hidden, hidden, num_layers=2, dropout=dropout) for et in self.edge_types})
        self.up_mlps = nn.ModuleList([MLP(2*hidden, hidden, hidden, num_layers=2, dropout=dropout) for _ in range(num_layers)])
        self.cls = MLP(hidden, hidden, num_classes, num_layers=2, dropout=dropout)
        self.rule_head = rule_head

    def message_pass(self, h, edge_index_dict):
        N, H = h.shape
        device = h.device
        agg = torch.zeros_like(h)
        for et, ei in edge_index_dict.items():
            if et not in self.msg_mlps:
                continue
            gate = self.edge_gates[et].sample_gate(self.training)
            if gate.item() == 0.0:
                continue
            src, dst = ei.to(device)
            m = self.msg_mlps[et](h[src])
            agg = agg.index_add(0, dst, m)
        return agg

    def forward(self, x, edge_index_dict):
        x_gated, z_feat = self.feat_gate(x)
        h = torch.relu(self.in_proj(x_gated))
        for l in range(self.num_layers):
            agg = self.message_pass(h, edge_index_dict)
            h = self.up_mlps[l](torch.cat([h, agg], dim=-1))
        logits = self.cls(h)
        aux = {
            "z_feat": z_feat,
            "z_edges": {et: self.edge_gates[et].sample_gate(self.training) for et in self.edge_types},
            "h": h
        }
        if self.rule_head is not None:
            rule_logits, rule_proofs = self.rule_head(h, x)
            logits = logits + rule_logits
            aux["rule_proofs"] = rule_proofs
        return logits, aux

    def l0_penalty(self, lambda_feat=1e-4, lambda_edge=1e-3):
        p_feat = self.feat_gate.expected_L0().sum()
        p_edge = sum(m.expected_L0().sum() for m in self.edge_gates.values())
        return lambda_feat * p_feat + lambda_edge * p_edge
