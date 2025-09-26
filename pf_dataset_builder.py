#!/usr/bin/env python3
"""
PF Dataset Builder
Generates a balanced parameter-free dataset (CFG-style JSON) for
@Positive, @NonNegative, @GTENegativeOne, @SearchIndexBottom, NO_ANNOTATION,
using realistic code contexts and node attributes so annotations align with
where they'd appear in real code.
"""

import os
import json
from typing import List, Dict

OUTPUT_DIR = "test_results/pf_dataset"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train", "cfg_output")
TEST_DIR = os.path.join(OUTPUT_DIR, "test", "cfg_output")

PF_CLASSES = [
    "@Positive",
    "@NonNegative",
    "@GTENegativeOne",
    "@SearchIndexBottom",
    "NO_ANNOTATION",
]


def ensure_dirs():
    for d in [TRAIN_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)


def make_nodes_for(label: str, method_idx: int) -> Dict:
    nodes: List[Dict] = []
    control_edges: List[Dict] = []
    dataflow_edges: List[Dict] = []
    node_id = 0

    def add(label_text: str, ntype: str = 'control', extras: Dict = None):
        nonlocal node_id
        node = {
            'id': node_id,
            'label': label_text,
            'type': ntype,
            'line_number': node_id + 1
        }
        if extras:
            node.update(extras)
        nodes.append(node)
        if node_id > 0:
            control_edges.append({'source': node_id - 1, 'target': node_id})
        node_id += 1

    # Entry
    add('Entry')

    # Realistic contexts for each PF class
    if label == "@Positive":
        # Method parameter that determines capacity/size
        # Annotation would be on the parameter in real code
        add('MethodDecl: createBuffer(int size)', ntype='method', extras={'is_parameter': True})
        add('Statement: byte[] buf = new byte[size]', ntype='statement')
        # Dataflow from parameter to allocation
        dataflow_edges.append({'source': 1, 'target': 2, 'variable': 'size'})
    elif label == "@NonNegative":
        # Loop start index; annotation belongs on index/start parameter
        add('MethodDecl: process(List items, int startIndex)', ntype='method', extras={'is_parameter': True})
        add('ForLoop: for (int i = startIndex; i < items.size(); i++)', ntype='statement')
        add('Statement: items.get(i)', ntype='statement')
        dataflow_edges.append({'source': 1, 'target': 2, 'variable': 'startIndex'})
        dataflow_edges.append({'source': 2, 'target': 3, 'variable': 'i'})
    elif label == "@GTENegativeOne":
        # Search index variable can be -1, so annotation belongs on variable/return
        add('MethodDecl: find(List items, String t)', ntype='method')
        add('Statement: int index = -1', ntype='statement')
        add('ForLoop: for (int i = 0; i < items.size(); i++)', ntype='statement')
        add('If: if (items.get(i).equals(t)) index = i', ntype='statement')
        add('Return: return index', ntype='return')
        dataflow_edges.append({'source': 2, 'target': 5, 'variable': 'index'})
        dataflow_edges.append({'source': 3, 'target': 4, 'variable': 'i'})
        dataflow_edges.append({'source': 4, 'target': 5, 'variable': 'index'})
    elif label == "@SearchIndexBottom":
        # indexOf result; annotation belongs on search index result
        add('MethodDecl: getOrNull(List items, String t)', ntype='method')
        add('MethodCall: int index = items.indexOf(t)', ntype='statement', extras={'has_method_call': True})
        add('If: if (index >= 0) return items.get(index)', ntype='statement')
        add('Return: return null', ntype='return')
        dataflow_edges.append({'source': 2, 'target': 3, 'variable': 'index'})
    else:  # NO_ANNOTATION
        # Computation that can be any integer; no annotation expected
        add('MethodDecl: diff(int a, int b)', ntype='method')
        add('Statement: int result = a - b', ntype='statement')
        add('Return: return result', ntype='return')
        dataflow_edges.append({'source': 2, 'target': 3, 'variable': 'result'})

    # Exit
    add('Exit')

    return {
        'method_name': f'method_{label}_{method_idx}',
        'nodes': nodes,
        'control_edges': control_edges,
        'dataflow_edges': dataflow_edges
    }


def write_split(split_dir: str, per_class: int):
    idx = 0
    for cls in PF_CLASSES:
        for k in range(per_class):
            cfg = make_nodes_for(cls, idx)
            path = os.path.join(split_dir, f'cfg_{cls.strip("@").lower()}_{idx}.json')
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            idx += 1


def main():
    ensure_dirs()
    # 50 per class for train (total 250), 10 per class for test (50)
    write_split(TRAIN_DIR, per_class=50)
    write_split(TEST_DIR, per_class=10)
    meta = {
        'train_total': 50 * len(PF_CLASSES),
        'test_total': 10 * len(PF_CLASSES),
        'classes': PF_CLASSES,
        'path': OUTPUT_DIR
    }
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print("Parameter-free dataset generated:", meta)

if __name__ == '__main__':
    main()
