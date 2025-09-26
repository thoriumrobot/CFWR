# predict_gcsn.py
import os, json, argparse, torch
from torch_geometric.loader import DataLoader
from gcsn import GCSN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True, help="Path to a .pt file with a list of PyG Data or a single Data.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="pred_out")
    ap.add_argument("--edge_threshold", type=float, default=0.5, help="For exporting a hard subgraph.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    obj = torch.load(args.data, map_location="cpu")
    graphs = obj if isinstance(obj, list) else [obj]
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.ckpt, map_location=device)
    in_dim = ckpt["in_dim"]; num_classes = ckpt["num_classes"]
    model = GCSN(in_dim=in_dim, num_classes=num_classes, hidden=ckpt["args"]["hidden"],
                 heads=ckpt["args"]["heads"], layers=ckpt["args"]["layers"],
                 l0_tau=ckpt["args"]["l0_tau_start"], l0_lambda=ckpt["args"]["l0_lambda"],
                 edge_k_ratio=ckpt["args"]["edge_k_ratio"], edge_lambda=ckpt["args"]["edge_lambda"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            logits = out["logits"]
            preds = logits.argmax(-1).cpu()

            # Export per-graph hard subgraph (thresholded from gate mask)
            edge_mask = out["edge_mask"].cpu()  # (E,)
            hard = (edge_mask >= args.edge_threshold).float()
            # Save indices of kept edges
            kept = hard.nonzero(as_tuple=False).view(-1).tolist()
            torch.save({"preds": preds, "kept_edge_indices": kept}, os.path.join(args.out_dir, f"graph_{i:05d}.pt"))

            all_preds.append(preds)

    # Also export the feature keep probabilities (global)
    feat_keep = torch.sigmoid(model.l0.logit).detach().cpu().tolist()
    with open(os.path.join(args.out_dir, "feature_keep_probs.json"), "w") as f:
        json.dump({"keep_prob": feat_keep}, f, indent=2)

    print(f"Wrote predictions and evidence to {args.out_dir}")

if __name__ == "__main__":
    main()
