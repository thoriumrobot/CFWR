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
    try:
        with open(java_file_path, 'r') as file:
            java_code = file.read()
        tree = parse(java_code)
        return tree
    except Exception as e:
        print(f"Warning: Could not parse {java_file_path}: {e}")
        return None

def extract_method_declarations(parsed_java_code):
    """
    Extracts method declarations from the parsed Java code.
    """
    if parsed_java_code is None:
        return []
    method_declarations = []
    for _, node in parsed_java_code.filter(MethodDeclaration):
        method_declarations.append(node)
    return method_declarations

def create_cfg(method, source_lines):
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
        last_nodes = process_block_statements(method.body, cfg, entry_node, source_lines)
    else:
        last_nodes = [entry_node]

    # Connect the last nodes to the exit node
    for node in last_nodes:
        cfg.add_edge(node, exit_node)

    return cfg

def process_block_statements(block, cfg, current_node, source_lines):
    """
    Processes a block of statements and adds them to the CFG starting from current_node.
    Returns a list of last nodes.
    """
    last_nodes = [current_node]
    # javalang can represent a block as a BlockStatement with .statements or directly as a list
    stmts = None
    if hasattr(block, 'statements'):
        stmts = block.statements
    elif isinstance(block, list):
        stmts = block
    else:
        stmts = []
    for statement in stmts:
        new_last_nodes = []
        for node in last_nodes:
            nodes = process_statement(statement, cfg, node, source_lines)
            new_last_nodes.extend(nodes)
        last_nodes = new_last_nodes
    return last_nodes

def process_statement(statement, cfg, current_node, source_lines):
    """
    Processes a single statement and adds it to the CFG starting from current_node.
    Returns a list of last nodes.
    """
    if isinstance(statement, BlockStatement):
        # Process nested block statements
        return process_block_statements(statement, cfg, current_node, source_lines)

    elif isinstance(statement, IfStatement):
        # Handle if statement
        condition_node = f'If({statement.condition})'
        cfg.add_node(condition_node, label=f'If({statement.condition})', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, condition_node)

        # Process the 'then' block
        then_last_nodes = process_statement(statement.then_statement, cfg, condition_node, source_lines)

        # Process the 'else' block if present
        if statement.else_statement:
            else_last_nodes = process_statement(statement.else_statement, cfg, condition_node, source_lines)
        else:
            else_last_nodes = [condition_node]

        # Merge the paths
        return then_last_nodes + else_last_nodes

    elif isinstance(statement, WhileStatement):
        # Handle while loop
        condition_node = f'While({statement.condition})'
        cfg.add_node(condition_node, label=f'While({statement.condition})', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, condition_node)

        # Process the body of the while loop
        body_last_nodes = process_statement(statement.body, cfg, condition_node, source_lines)

        # Add edge from body last nodes back to condition node
        for node in body_last_nodes:
            cfg.add_edge(node, condition_node)

        # The loop can exit after checking the condition
        return [condition_node]

    elif isinstance(statement, ForStatement):
        # Handle for loop
        condition_str = str(statement.control.condition) if statement.control and statement.control.condition else ""
        condition_node = f'For({condition_str})'
        cfg.add_node(condition_node, label=f'For({condition_str})', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, condition_node)

        # Process the body of the for loop
        body_last_nodes = process_statement(statement.body, cfg, condition_node, source_lines)

        # Handle loop updates
        if statement.control and statement.control.update:
            update_str = ", ".join(map(str, statement.control.update)) if isinstance(statement.control.update, list) else str(statement.control.update)
            update_node = f'Update({update_str})'
            cfg.add_node(update_node, label=update_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
            for node in body_last_nodes:
                cfg.add_edge(node, update_node)
            cfg.add_edge(update_node, condition_node)
        else:
            for node in body_last_nodes:
                cfg.add_edge(node, condition_node)

        # The loop can exit after checking the condition
        return [condition_node]

    elif isinstance(statement, DoStatement):
        # Handle do-while loop
        body_node = f'DoWhileBody'
        cfg.add_node(body_node, label='DoWhileBody', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, body_node)

        # Process the body
        body_last_nodes = process_statement(statement.body, cfg, body_node, source_lines)

        # Process the condition
        condition_node = f'DoWhile({statement.condition})'
        cfg.add_node(condition_node, label=f'DoWhile({statement.condition})', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        for node in body_last_nodes:
            cfg.add_edge(node, condition_node)

        # Loop back to body
        cfg.add_edge(condition_node, body_node)

        # Exit after condition
        return [condition_node]

    elif isinstance(statement, ReturnStatement):
        # Return statement
        return_node = f'Return({statement.expression})'
        cfg.add_node(return_node, label=return_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, return_node)
        # Control flow ends here
        return []

    elif isinstance(statement, ThrowStatement):
        # Throw statement
        throw_node = f'Throw({statement.expression})'
        cfg.add_node(throw_node, label=throw_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, throw_node)
        # Control flow ends here
        return []

    elif isinstance(statement, BreakStatement):
        # Break statement
        break_node = 'Break'
        cfg.add_node(break_node, label=break_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, break_node)
        # Break statements terminate the loop, handled in higher context
        return []

    elif isinstance(statement, ContinueStatement):
        # Continue statement
        continue_node = 'Continue'
        cfg.add_node(continue_node, label=continue_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, continue_node)
        # Continue statements loop back, handled in higher context
        return []

    elif isinstance(statement, SynchronizedStatement):
        # Synchronized statement
        sync_node = f'Synchronized({statement.expression})'
        cfg.add_node(sync_node, label=sync_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, sync_node)
        # Process the block inside synchronized
        return process_block_statements(statement.block, cfg, sync_node, source_lines)

    elif isinstance(statement, TryStatement):
        # Try statement
        try_node = 'Try'
        cfg.add_node(try_node, label='Try', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, try_node)

        # Process try block
        try_last_nodes = process_block_statements(statement.block, cfg, try_node, source_lines)

        # Process catches
        catch_last_nodes = []
        for catch_clause in statement.catches:
            catch_node = f'Catch({catch_clause.parameter.name})'
            cfg.add_node(catch_node, label=catch_node, line=getattr(catch_clause, 'position', None).line if getattr(catch_clause, 'position', None) else None)
            cfg.add_edge(try_node, catch_node)
            catch_last_nodes.extend(process_block_statements(catch_clause.block, cfg, catch_node, source_lines))

        # Process finally block if present
        if statement.finally_block:
            finally_node = 'Finally'
            cfg.add_node(finally_node, label='Finally', line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
            for node in try_last_nodes + catch_last_nodes:
                cfg.add_edge(node, finally_node)
            finally_last_nodes = process_block_statements(statement.finally_block, cfg, finally_node, source_lines)
            return finally_last_nodes
        else:
            # Merge paths from try and catches
            return try_last_nodes + catch_last_nodes

    elif isinstance(statement, SwitchStatement):
        # Handle switch statements
        switch_node = f'Switch({statement.expression})'
        cfg.add_node(switch_node, label=switch_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, switch_node)

        # Process each case
        case_last_nodes = []
        for case in statement.cases:
            case_node = f'Case({case.case})'
            cfg.add_node(case_node, label=case_node, line=getattr(case, 'position', None).line if getattr(case, 'position', None) else None)
            cfg.add_edge(switch_node, case_node)
            
            # Process statements in this case
            case_body_nodes = process_block_statements(case.statements, cfg, case_node, source_lines)
            case_last_nodes.extend(case_body_nodes)

        return case_last_nodes

    elif isinstance(statement, StatementExpression):
        # Expression statement
        expr_node = f'{statement.expression}'
        cfg.add_node(expr_node, label=expr_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, expr_node)
        return [expr_node]

    elif isinstance(statement, LocalVariableDeclaration):
        # Variable declaration
        var_node = f'{statement}'
        cfg.add_node(var_node, label=var_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, var_node)
        return [var_node]

    elif statement is None or str(statement).strip() == '':
        # Empty statement
        return [current_node]

    else:
        # Other statements
        print(f"Warning: Unknown statement type: {type(statement).__name__} - {statement}")
        stmt_node = f'Unknown({type(statement).__name__})'
        cfg.add_node(stmt_node, label=stmt_node, line=getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None)
        cfg.add_edge(current_node, stmt_node)
        return [stmt_node]

def generate_control_flow_graphs(java_file_path, output_base_dir='cfg_output'):
    """
    Generates control flow graphs for a given Java file.
    """
    parsed_java_code = parse_java_file(java_file_path)
    if parsed_java_code is None:
        return []
    with open(java_file_path, 'r') as f:
        source_lines = f.readlines()
    method_declarations = extract_method_declarations(parsed_java_code)

    cfgs = []
    for method in method_declarations:
        cfg = create_cfg(method, source_lines)
        cfgs.append((method.name, cfg))

    return cfgs

def save_cfgs(cfgs, output_dir='cfg_output'):
    """
    Saves the CFGs for each method into JSON files for machine learning models.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for method_name, cfg in cfgs:
        # Prepare data for JSON serialization
        data = {
            'nodes': [],
            'edges': []
        }
        node_id_map = {}  # Map node names to unique IDs

        for idx, (node, attr) in enumerate(cfg.nodes(data=True)):
            node_id_map[node] = idx
            data['nodes'].append({
                'id': idx,
                'label': attr.get('label', ''),
                'line': attr.get('line', None)
            })

        for source, target in cfg.edges():
            data['edges'].append({
                'source': node_id_map[source],
                'target': node_id_map[target]
            })

        # Save to JSON file
        file_name = f'{method_name}.json'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'CFG saved for method "{method_name}" at {file_path}')

# Example usage
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--java_file', required=True)
    parser.add_argument('--out_dir', default='cfg_output')
    args = parser.parse_args()
    cfgs = generate_control_flow_graphs(args.java_file, args.out_dir)
    # Save under a per-file subdirectory using the base name
    base = os.path.splitext(os.path.basename(args.java_file))[0]
    out_dir = os.path.join(args.out_dir, base)
    save_cfgs(cfgs, out_dir)
