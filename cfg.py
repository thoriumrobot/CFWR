import networkx as nx
import javalang
import json
import os
from javalang.parse import parse
from javalang.tree import *

def parse_java_file(java_file_path):
    """
    Parses a Java file and returns the Abstract Syntax Tree (AST).
    """
    with open(java_file_path, 'r') as file:
        java_code = file.read()
    tree = parse(java_code)
    return tree

def extract_method_declarations(parsed_java_code):
    """
    Extracts method declarations from the parsed Java code.
    """
    method_declarations = []
    for _, node in parsed_java_code.filter(MethodDeclaration):
        method_declarations.append(node)
    return method_declarations

def create_cfg(method):
    """
    Creates a Control Flow Graph (CFG) for a given method.
    """
    cfg = nx.DiGraph()
    entry_node = 'Entry'
    exit_node = 'Exit'
    cfg.add_node(entry_node, label='Entry')
    cfg.add_node(exit_node, label='Exit')

    if method.body:
        # Start processing the method body from the entry node
        last_nodes = process_block_statements(method.body, cfg, entry_node)
    else:
        last_nodes = [entry_node]

    # Connect the last nodes to the exit node
    for node in last_nodes:
        cfg.add_edge(node, exit_node)

    return cfg

def process_block_statements(block, cfg, current_node):
    last_nodes = [current_node]
    for statement in block.statements:
        new_last_nodes = []
        for node in last_nodes:
            nodes = process_statement(statement, cfg, node)
            new_last_nodes.extend(nodes)
        last_nodes = new_last_nodes
    return last_nodes

def process_statement(statement, cfg, current_node):
    # (Implementation from your snippet, unchanged)
    # ...
    # (Truncated here for brevity, but copy from your snippet)
    # ...
    if isinstance(statement, StatementExpression):
        expr_node = f'{statement.expression}'
        cfg.add_node(expr_node, label=expr_node)
        cfg.add_edge(current_node, expr_node)
        return [expr_node]
    elif isinstance(statement, LocalVariableDeclaration):
        var_node = f'{statement}'
        cfg.add_node(var_node, label=var_node)
        cfg.add_edge(current_node, var_node)
        return [var_node]
    elif isinstance(statement, EmptyStatement):
        return [current_node]
    else:
        stmt_node = f'Unknown({type(statement).__name__})'
        cfg.add_node(stmt_node, label=stmt_node)
        cfg.add_edge(current_node, stmt_node)
        return [stmt_node]

def generate_control_flow_graphs(java_file_path):
    parsed_java_code = parse_java_file(java_file_path)
    method_declarations = extract_method_declarations(parsed_java_code)

    cfgs = []
    for method in method_declarations:
        cfg = create_cfg(method)
        cfgs.append((method.name, cfg))
    return cfgs

def save_cfgs(cfgs, output_dir='cfg_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for method_name, cfg in cfgs:
        data = {
            'nodes': [],
            'edges': []
        }
        node_id_map = {}

        for idx, (node, attr) in enumerate(cfg.nodes(data=True)):
            node_id_map[node] = idx
            data['nodes'].append({
                'id': idx,
                'label': attr.get('label', ''),
                # Optionally store line info if you’d like
                # "line": ...
            })

        for source, target in cfg.edges():
            data['edges'].append({
                'source': node_id_map[source],
                'target': node_id_map[target]
            })

        file_name = f'{method_name}.json'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'CFG saved for method "{method_name}" at {file_path}')


if __name__ == '__main__':
    java_file_path = "path/to/your/java/file.java"
    cfgs = generate_control_flow_graphs(java_file_path)
    save_cfgs(cfgs)
