import os, argparse, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataio import GraphDirDataset, collate_graphs
from dg2n import DG2N
from rules import DifferentiableHornRules
from losses import prediction_loss, stability_loss

def infer_edge_types(sample):
    return {et: 1 for et in sample['edge_index_dict'].keys()}

def train(args):
    device = torch.device("cpu")
    ds = GraphDirDataset(args.data_dir)
    n_total = len(ds)
    n_val = max(1, int(n_total * args.val_frac))
    n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Infer feature dim and global class count across dataset
    sample = ds[0]
    x, y, edge_index_dict = sample['x'].float(), sample['y'].long(), sample['edge_index_dict']
    in_dim = x.size(-1)

    max_label = -1
    for i in range(len(ds)):
        yi = ds[i]['y']
        if (yi>=0).any():
            max_label = max(max_label, int(yi[yi>=0].max().item()))
    num_classes = max(2, (max_label + 1) if max_label >= 0 else args.num_classes)
    edge_types = infer_edge_types(sample)

    rule_head = None if args.no_rules else DifferentiableHornRules(h_dim=args.hidden, x_dim=in_dim, num_classes=num_classes, num_rules=args.num_rules)

    model = DG2N(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, edge_types=edge_types, num_classes=num_classes, rule_head=rule_head, dropout=args.dropout).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
        total_loss = 0.0
        for batch in train_loader:
            for sample in batch:
                x = sample['x'].float().to(device)
                y = sample['y'].long().to(device)
                edge_index_dict = {k: v.to(device) for k,v in sample['edge_index_dict'].items()}
                mask = sample.get('mask', (y>=0))

                logits, aux = model(x, edge_index_dict)
                loss_pred = prediction_loss(logits, y, mask=mask)
                loss_l0 = model.l0_penalty(lambda_feat=args.lmbd_feat, lambda_edge=args.lmbd_edge)
                loss_stab = stability_loss(model, x, edge_index_dict, logits, drop_prob=args.stab_noise)
                loss = loss_pred + loss_l0 + args.stab_coef * loss_stab

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                total_loss += float(loss.item())

        sched.step()

        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_graphs)
            val_loss = 0.0
            val_acc = 0.0
            val_count = 0
            for batch in val_loader:
                for sample in batch:
                    x = sample['x'].float().to(device)
                    y = sample['y'].long().to(device)
                    edge_index_dict = {k: v.to(device) for k,v in sample['edge_index_dict'].items()}
                    mask = sample.get('mask', (y>=0))
                    logits, _ = model(x, edge_index_dict)
                    loss_pred = prediction_loss(logits, y, mask=mask)
                    val_loss += float(loss_pred.item())
                    if mask.sum() > 0:
                        pred = logits.argmax(dim=-1)
                        val_acc += float((pred[mask] == y[mask]).sum().item())
                        val_count += int(mask.sum().item())
            val_acc = val_acc / max(1, val_count)
        print(f"Epoch {epoch:03d} | train_loss={total_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "in_dim": in_dim,
                "hidden": args.hidden,
                "layers": args.layers,
                "edge_types": list(edge_types.keys()),
                "num_classes": num_classes,
                "args": vars(args)
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_dg2n.pt"))

    print("Training complete. Best val_loss:", best_val)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lmbd_feat", type=float, default=1e-4)
    ap.add_argument("--lmbd_edge", type=float, default=5e-4)
    ap.add_argument("--stab_coef", type=float, default=1.0)
    ap.add_argument("--stab_noise", type=float, default=0.2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_rules", action="store_true")
    ap.add_argument("--num_rules", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--num_classes", type=int, default=8)
    args = ap.parse_args()
    train(args)
