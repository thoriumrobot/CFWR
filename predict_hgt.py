import os
import json
import argparse
import torch
from torch_geometric.data import DataLoader
from hgt import HGTModel, create_heterodata, load_cfgs, label_nodes

def load_graphs_for_file(java_file, cfg_output_dir):
    graphs = []
    cfgs = load_cfgs(java_file)
    for cfg_data in cfgs:
        data = create_heterodata(cfg_data)
        if data is not None:
            graphs.append((cfg_data, data))
    return graphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--java_file', required=True, help='Path to a Java slice to predict on')
    parser.add_argument('--model_path', required=True, help='Path to trained HGT model .pth')
    parser.add_argument('--out_path', required=True, help='Path to write predictions JSON')
    parser.add_argument('--cfg_output_dir', default=os.environ.get('CFG_OUTPUT_DIR', 'cfg_output'))
    args = parser.parse_args()

    graphs = load_graphs_for_file(args.java_file, args.cfg_output_dir)
    if not graphs:
        print('No graphs found for file')
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


