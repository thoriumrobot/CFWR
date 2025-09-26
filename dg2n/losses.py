import torch
import torch.nn.functional as F

def prediction_loss(logits, y, mask=None, class_weights=None):
    if mask is None:
        mask = y >= 0
    if mask.sum() == 0:
        return logits.new_tensor(0.0, requires_grad=True)
    return F.cross_entropy(logits[mask], y[mask], weight=class_weights)

def stability_loss(model, x, edge_index_dict, logits_ref, drop_prob=0.2):
    with torch.no_grad():
        z_exp = model.feat_gate.expected_L0().detach()
        irrelevant = (z_exp < 0.5).float()
        noise = torch.randn_like(x) * drop_prob
        x_pert = x + noise * irrelevant
    logits_pert, _ = model(x_pert, edge_index_dict)
    return F.mse_loss(torch.log_softmax(logits_pert, dim=-1), torch.log_softmax(logits_ref, dim=-1))
