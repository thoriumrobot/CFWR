#!/usr/bin/env python3
"""
Build NullGTN training data (edges.pkl, node_features.pkl, labels.pkl) from CFWR CFG JSONs.

Output directory defaults to GTN_alltypes/data/Null inside this repo.
This is a minimal integration to unblock training; feel free to refine features later.
"""

import os
import re
import json
import argparse
import numpy as np
import pickle
from pathlib import Path


def safe_tokenize(label: str) -> list:
    if not label:
        return []
    parts = re.split(r"[^A-Za-z0-9_]+", label)
    return [p.lower() for p in parts if p]


def collect_graph(cfg_dir: str):
    nodes_all = []
    names_vocab = {}
    name_index = 0
    control_edges = []

    for root, _, files in os.walk(cfg_dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, 'r') as jf:
                    data = json.load(jf)
            except Exception:
                continue
            base_offset = len(nodes_all)
            nodes = data.get('nodes', [])
            nodes_all.extend(nodes)
            for e in data.get('control_edges', []):
                s = int(e.get('source', 0)) + base_offset
                t = int(e.get('target', 0)) + base_offset
                control_edges.append((s, t))
            for n in nodes:
                for tok in safe_tokenize(n.get('label', '')):
                    if tok not in names_vocab:
                        names_vocab[tok] = name_index
                        name_index += 1

    num_nodes = len(nodes_all)
    num_terms = len(names_vocab)

    A_n = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for s, t in control_edges:
        if 0 <= s < num_nodes and 0 <= t < num_nodes:
            A_n[s, t] = 1.0

    A_t = np.zeros((num_nodes, num_terms), dtype=np.float32)
    for i, n in enumerate(nodes_all):
        for tok in safe_tokenize(n.get('label', '')):
            j = names_vocab.get(tok)
            if j is not None:
                A_t[i, j] = 1.0

    feats = np.zeros((num_nodes, 2), dtype=np.float32)
    for i, n in enumerate(nodes_all):
        feats[i, 0] = float(len(n.get('label', '')))
        feats[i, 1] = float(n.get('line') or 0)

    def is_target(n):
        lab = (n.get('label') or '').lower()
        return int(any(k in lab for k in ['methoddeclaration', 'fielddeclaration', 'formalparameter', 'variabledeclarator']))

    targets = np.array([is_target(n) for n in nodes_all], dtype=np.int64)
    idx_all = np.arange(num_nodes)
    if num_nodes:
        np.random.shuffle(idx_all)
    n_train = int(0.7 * num_nodes)
    n_val = int(0.15 * num_nodes)
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:n_train + n_val]
    test_idx = idx_all[n_train + n_val:]

    labels = [
        np.vstack((train_idx, targets[train_idx])).T if len(train_idx) else np.zeros((0, 2), dtype=np.int64),
        np.vstack((val_idx, targets[val_idx])).T if len(val_idx) else np.zeros((0, 2), dtype=np.int64),
        np.vstack((test_idx, targets[test_idx])).T if len(test_idx) else np.zeros((0, 2), dtype=np.int64),
    ]

    edges = [A_n, A_n.T, A_t, A_t.T]
    return edges, feats, labels


def main():
    ap = argparse.ArgumentParser(description='Build NullGTN data from CFWR CFGs')
    ap.add_argument('--cfg_dir', required=True, help='Directory of CFG JSONs (recursively)')
    ap.add_argument('--out_root', default='GTN_alltypes/data/Null', help='Output dataset directory')
    args = ap.parse_args()

    edges, feats, labels = collect_graph(args.cfg_dir)
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'edges.pkl', 'wb') as f:
        pickle.dump(edges, f)
    with open(out_dir / 'node_features.pkl', 'wb') as f:
        pickle.dump(feats, f)
    with open(out_dir / 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    print(f"Wrote NullGTN data to {out_dir}")


if __name__ == '__main__':
    main()


