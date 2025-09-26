import torch
import torch.nn as nn
from typing import Tuple

class DifferentiableHornRules(nn.Module):
    """
    A tiny, generic rule scorer (placeholder—extend with domain rules).
    Each rule r computes a logit contribution: A_r(h,x) * W_r->class
    where A_r is a sigmoid MLP over [h|x].
    """
    def __init__(self, h_dim: int, x_dim: int, num_classes: int, num_rules: int = 32, hidden=64, dropout=0.1):
        super().__init__()
        self.num_rules = num_rules
        self.num_classes = num_classes
        self.ante_mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim + x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        ) for _ in range(num_rules)])
        self.rule_to_class = nn.Parameter(torch.zeros(num_rules, num_classes))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        hx = torch.cat([h, x], dim=-1)
        strengths = []
        for mlp in self.ante_mlps:
            strengths.append(torch.sigmoid(mlp(hx)))  # [N,1]
        A = torch.cat(strengths, dim=1)  # [N,R]
        rule_logits = A @ self.rule_to_class  # [N,C]
        proofs = {"rule_strength": A}
        return rule_logits, proofs
