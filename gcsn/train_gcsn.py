# train_gcsn.py
import os, json, math, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gcsn import GCSN, cosine_anneal

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def class_weights_from_loader(loader, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for batch in loader:
        y = batch.y[batch.train_mask] if hasattr(batch, "train_mask") else batch.y
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    counts = counts.clamp(min=1.0)
    w = 1.0 / torch.log1p(counts)  # inverse log frequency
    return w

@torch.no_grad()
def evaluate(model, loader, device, split="val"):
    model.eval()
    total, correct = 0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        logits = out["logits"]
        mask = getattr(batch, f"{split}_mask", torch.ones_like(batch.y, dtype=torch.bool))
        preds = logits.argmax(-1)
        total += int(mask.sum())
        correct += int((preds[mask] == batch.y[mask]).sum())
    acc = correct / max(total, 1)
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with preprocessed PyG .pt files or one .pt file list.")
    ap.add_argument("--train_glob", type=str, default="train_*.pt")
    ap.add_argument("--val_glob", type=str, default="val_*.pt")
    ap.add_argument("--test_glob", type=str, default="test_*.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)  # per-graph training
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--edge_k_ratio", type=float, default=0.2)
    ap.add_argument("--edge_lambda", type=float, default=1e-5)
    ap.add_argument("--l0_lambda", type=float, default=1e-4)
    ap.add_argument("--l0_tau_start", type=float, default=2/3)
    ap.add_argument("--l0_tau_end", type=float, default=0.1)
    ap.add_argument("--tau_anneal_epochs", type=int, default=60)
    ap.add_argument("--k_start", type=float, default=0.4)
    ap.add_argument("--k_end", type=float, default=0.15)
    ap.add_argument("--k_anneal_epochs", type=int, default=60)
    ap.add_argument("--decov_lambda", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="ckpt_gcsn")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load datasets (expects torch.save(Data) files) ----
    def load_split(glob_pat):
        import glob, torch
        files = sorted(glob.glob(os.path.join(args.data_dir, glob_pat)))
        if len(files) == 0 and os.path.isfile(os.path.join(args.data_dir, glob_pat)):
            files = [os.path.join(args.data_dir, glob_pat)]
        graphs = []
        for f in files:
            obj = torch.load(f, weights_only=False)
            if isinstance(obj, list): graphs.extend(obj)
            else: graphs.append(obj)
        return graphs

    train_graphs = load_split(args.train_glob)
    val_graphs = load_split(args.val_glob)
    test_graphs = load_split(args.test_glob)

    assert len(train_graphs) > 0, "No training graphs found."

    in_dim = train_graphs[0].x.size(-1)
    num_classes = int(max([g.y.max().item() for g in train_graphs + val_graphs + test_graphs]) + 1)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False) if len(val_graphs)>0 else None
    test_loader  = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False) if len(test_graphs)>0 else None

    model = GCSN(
        in_dim=in_dim, num_classes=num_classes, hidden=args.hidden,
        heads=args.heads, layers=args.layers,
        l0_tau=args.l0_tau_start, l0_lambda=args.l0_lambda,
        edge_k_ratio=args.edge_k_ratio, edge_lambda=args.edge_lambda
    ).to(device)

    class_weights = class_weights_from_loader(train_loader, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best = {"epoch": -1, "val_acc": -1.0, "path": None}
    patience = args.patience

    for epoch in range(1, args.epochs+1):
        model.train()
        # --- schedules ---
        tau = cosine_anneal(args.l0_tau_start, args.l0_tau_end, epoch-1, args.tau_anneal_epochs)
        model.set_temperature(tau)
        # anneal k_ratio by overriding attribute (harmless mid-training)
        k_ratio = cosine_anneal(args.k_start, args.k_end, epoch-1, args.k_anneal_epochs)
        model.edge_gate.k_ratio = float(k_ratio)

        total, correct = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(batch.x, batch.edge_index)
                logits = out["logits"]
                mask = batch.train_mask if hasattr(batch, "train_mask") else torch.ones_like(batch.y, dtype=torch.bool)
                loss = F.cross_entropy(logits[mask], batch.y[mask], weight=class_weights)
                # DeCov (hidden before head): reuse logits' penultimate input via small hook (approximate with logits itself)
                # Add sparsity penalties
                loss = loss + out["penalty"] + 1e-4 * (-F.log_softmax(logits, -1).mean())

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # Train accuracy (masked)
            preds = logits.argmax(-1)
            total += int(mask.sum())
            correct += int((preds[mask] == batch.y[mask]).sum())

        train_acc = correct / max(total, 1)
        val_acc = evaluate(model, val_loader, device, split="val") if val_loader else train_acc

        print(f"[Epoch {epoch:03d}] tau={tau:.3f} k={k_ratio:.3f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Early stopping
        improved = val_acc > best["val_acc"] + 1e-6
        if improved:
            best["val_acc"] = val_acc
            best["epoch"] = epoch
            best["path"] = os.path.join(args.out_dir, "best.pt")
            torch.save({"model_state": model.state_dict(),
                        "in_dim": in_dim, "num_classes": num_classes,
                        "args": vars(args)}, best["path"])
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch}. Best val_acc={best['val_acc']:.4f} (epoch {best['epoch']}).")
                break

    # Final test eval
    if test_loader:
        # Load best
        if best["path"] and os.path.isfile(best["path"]):
            ckpt = torch.load(best["path"], map_location=device)
            model.load_state_dict(ckpt["model_state"])
        test_acc = evaluate(model, test_loader, device, split="test")
        print(f"[TEST] acc={test_acc:.4f}")

    # Export final feature-gate probabilities
    feat_keep = torch.sigmoid(model.l0.logit).detach().cpu().tolist()
    with open(os.path.join(args.out_dir, "feature_keep_probs.json"), "w") as f:
        json.dump({"keep_prob": feat_keep}, f, indent=2)

if __name__ == "__main__":
    main()
