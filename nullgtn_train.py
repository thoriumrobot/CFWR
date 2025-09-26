#!/usr/bin/env python3
"""
Train NullGTN using GTN_alltypes on data built from CFWR CFGs.

This script assumes GTN_alltypes/data/Null contains edges.pkl, node_features.pkl, labels.pkl.
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Train NullGTN with GTN_alltypes')
    ap.add_argument('--dataset', default='Null')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--node_dim', type=int, default=64)
    ap.add_argument('--num_channels', type=int, default=2)
    ap.add_argument('--lr', type=float, default=0.04)
    ap.add_argument('--weight_decay', type=float, default=0.001)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--model', type=str, default='FastGTN')
    ap.add_argument('--cluster', type=str, default='data0.json')
    ap.add_argument('--pre_train', action='store_true')
    ap.add_argument('--num_FastGTN_layers', type=int, default=1)
    args = ap.parse_args()

    # Defer import to preserve module paths
    from GTN_alltypes.fgtn_main import trainTheModel

    class A:
        pass
    a = A()
    for k, v in vars(args).items():
        setattr(a, k, v)
    trainTheModel(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())


