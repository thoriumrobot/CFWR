import os, argparse, json
import torch
from dg2n import DG2N
from rules import DifferentiableHornRules

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    edge_types = {et:1 for et in ckpt["edge_types"]}
    rule_head = None if ckpt["args"].get("no_rules", False) else DifferentiableHornRules(
        h_dim=ckpt["hidden"], x_dim=ckpt["in_dim"], num_classes=ckpt["num_classes"], num_rules=ckpt["args"].get("num_rules", 32)
    )
    model = DG2N(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"], num_layers=ckpt["layers"], edge_types=edge_types, num_classes=ckpt["num_classes"], rule_head=rule_head, dropout=ckpt["args"].get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, ckpt

def predict_one(model, sample, device):
    x = sample['x'].float().to(device)
    edge_index_dict = {k: v.to(device) for k,v in sample['edge_index_dict'].items()}
    logits, aux = model(x, edge_index_dict)
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1).cpu()
    out = {
        "pred": pred.tolist(),
        "probs": probs.detach().cpu().tolist(),
        "feat_gate": aux["z_feat"].detach().cpu().tolist(),
        "edge_gates": {k: float(v.item()) for k,v in aux["z_edges"].items()},
    }
    if "rule_proofs" in aux:
        out["rule_strength"] = aux["rule_proofs"]["rule_strength"].detach().cpu().tolist()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--graph_pt", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="prediction.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    sample = torch.load(args.graph_pt, map_location="cpu")
    result = predict_one(model, sample, device)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote", args.out_json)
