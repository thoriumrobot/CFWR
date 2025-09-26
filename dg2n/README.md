# DG2N: Deterministic-Gate Graph Network (reference)

This package implements DG²N for node-level type inference on CFG/DFG graphs with 100%/0% feature relevance.

## Failure modes & fixes
- **Mask collapse (underfit):** Start with small sparsity penalties and anneal upward. Cosine LR + grad clipping.
- **Spurious edge dominance:** Learn edge-type gates; prefer shallow message depth (2–3).
- **Rule overfit:** Keep few rules, use dropout, early stop by val loss.
- **Instability to irrelevant perturbations:** Stability loss perturbs low-gate features only.
- **Imbalance/sparse labels:** Provide class weights; oversample labeled nodes.
- **Graph blow-up:** Train on slices; periodically harden/prune.

## Data format (.pt per graph)
```
{
  "x": FloatTensor [N,F],
  "edge_index_dict": { "cfg": LongTensor [2,E_cfg], "dfg": LongTensor [2,E_dfg], ... },
  "y": LongTensor [N],    # -1 for unlabeled
  "mask": BoolTensor [N]  # optional
}
```

## Train
```
python train_dg2n.py --data_dir /path/to/graphs --out_dir ckpts --epochs 60 --layers 3 --hidden 192
```

## Predict
```
python predict_dg2n.py --ckpt ckpts/best_dg2n.pt --graph_pt /path/to/sample.pt --out_json preds.json
```

## Extend rule head
Edit `rules.py` to encode typed Horn-like templates over (h, x). For Lower Bounds, include features like index-vs-length, array construction size, loop bounds, etc.
