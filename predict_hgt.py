#!/usr/bin/env python3
"""
HGT Prediction Script with Best Practices Defaults

This script runs HGT predictions on Java files or slices using:
- Dataflow-augmented CFGs by default
- Augmented slices when available
- Consistent pipeline integration

Best Practices:
- Uses dataflow information in CFGs for better predictions
- Prefers augmented slices over original slices
- Integrates with the same pipeline as training
"""

import os
import json
import argparse
import torch
from torch_geometric.data import DataLoader
from hgt import HGTModel, create_heterodata, load_cfgs, label_nodes

def load_graphs_for_file(java_file, cfg_output_dir):
    graphs = []
    cfgs = load_cfgs(java_file, cfg_output_dir)
    for cfg_data in cfgs:
        data = create_heterodata(cfg_data)
        if data is not None:
            graphs.append((cfg_data, data))
    return graphs

def load_graphs_for_directory(slices_dir, cfg_output_dir):
    """Load graphs for all Java files in a directory (for project-based prediction)."""
    graphs = []
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_file = os.path.join(root, file)
                file_graphs = load_graphs_for_file(java_file, cfg_output_dir)
                graphs.extend(file_graphs)
    return graphs

def main():
    parser = argparse.ArgumentParser(description='Run HGT predictions on Java files or slices with best practices defaults')
    parser.add_argument('--java_file', help='Path to a Java slice to predict on')
    parser.add_argument('--slices_dir', help='Path to directory containing Java slices (prefers augmented slices by default)')
    parser.add_argument('--model_path', required=True, help='Path to trained HGT model .pth')
    parser.add_argument('--out_path', required=True, help='Path to write predictions JSON')
    parser.add_argument('--cfg_output_dir', default=os.environ.get('CFG_OUTPUT_DIR', 'cfg_output'),
                       help='Directory containing dataflow-augmented CFGs (default behavior)')
    parser.add_argument('--use_original_slices', action='store_true', default=False,
                       help='Use original slices instead of augmented slices (default: prefers augmented)')
    args = parser.parse_args()

    # Validate arguments
    if not args.java_file and not args.slices_dir:
        parser.error("Either --java_file or --slices_dir must be specified")
    if args.java_file and args.slices_dir:
        parser.error("Cannot specify both --java_file and --slices_dir")

    # Load graphs
    if args.java_file:
        graphs = load_graphs_for_file(args.java_file, args.cfg_output_dir)
        print(f"Loaded {len(graphs)} graphs from {args.java_file}")
    else:
        graphs = load_graphs_for_directory(args.slices_dir, args.cfg_output_dir)
        print(f"Loaded {len(graphs)} graphs from {args.slices_dir}")

    if not graphs:
        print('No graphs found')
        return

    # Build model metadata from first graph
    sample_graph = graphs[0][1]
    metadata = sample_graph.metadata()
    in_channels = sample_graph['node'].x.size(-1)
    model = HGTModel(in_channels=in_channels, hidden_channels=64, out_channels=2, num_heads=2, num_layers=2, metadata=metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for cfg_data, data in graphs:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            # Map back to line numbers
            lines = [node.get('line') for node in cfg_data['nodes']]
            results.append({
                'method': cfg_data.get('method_name'),
                'file': cfg_data.get('java_file'),
                'pred_lines': [int(l) for i, l in enumerate(lines) if l is not None and int(pred[i]) == 1],
                'scores': [{ 'line': int(l) if l is not None else None, 'score': float(prob[i]) } for i, l in enumerate(lines)]
            })

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Predictions written to {args.out_path}')

if __name__ == '__main__':
    main()


