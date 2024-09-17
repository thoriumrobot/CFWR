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
    """
    Processes a block of statements and adds them to the CFG starting from current_node.
    Returns a list of last nodes.
    """
    last_nodes = [current_node]
    for statement in block.statements:
        new_last_nodes = []
        for node in last_nodes:
            nodes = process_statement(statement, cfg, node)
            new_last_nodes.extend(nodes)
        last_nodes = new_last_nodes
    return last_nodes

def process_statement(statement, cfg, current_node):
    """
    Processes a single statement and adds it to the CFG starting from current_node.
    Returns a list of last nodes.
    """
    if isinstance(statement, BlockStatement):
        # Process nested block statements
        return process_block_statements(statement, cfg, current_node)

    elif isinstance(statement, IfStatement):
        # Handle if statement
        condition_node = f'If({statement.condition})'
        cfg.add_node(condition_node, label=f'If({statement.condition})')
        cfg.add_edge(current_node, condition_node)

        # Process the 'then' block
        then_last_nodes = process_statement(statement.then_statement, cfg, condition_node)

        # Process the 'else' block if present
        if statement.else_statement:
            else_last_nodes = process_statement(statement.else_statement, cfg, condition_node)
        else:
            else_last_nodes = [condition_node]

        # Merge the paths
        return then_last_nodes + else_last_nodes

    elif isinstance(statement, WhileStatement):
        # Handle while loop
        condition_node = f'While({statement.condition})'
        cfg.add_node(condition_node, label=f'While({statement.condition})')
        cfg.add_edge(current_node, condition_node)

        # Process the body of the while loop
        body_last_nodes = process_statement(statement.body, cfg, condition_node)

        # Add edge from body last nodes back to condition node
        for node in body_last_nodes:
            cfg.add_edge(node, condition_node)

        # The loop can exit after checking the condition
        return [condition_node]

    elif isinstance(statement, ForStatement):
        # Handle for loop
        condition_node = f'For({statement.condition or ""})'
        cfg.add_node(condition_node, label=f'For({statement.condition or ""})')
        cfg.add_edge(current_node, condition_node)

        # Process the body of the for loop
        body_last_nodes = process_statement(statement.body, cfg, condition_node)

        # Handle loop updates
        if statement.update:
            update_node = f'Update({", ".join(map(str, statement.update))})'
            cfg.add_node(update_node, label=update_node)
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
        cfg.add_node(body_node, label='DoWhileBody')
        cfg.add_edge(current_node, body_node)

        # Process the body
        body_last_nodes = process_statement(statement.body, cfg, body_node)

        # Process the condition
        condition_node = f'DoWhile({statement.condition})'
        cfg.add_node(condition_node, label=f'DoWhile({statement.condition})')
        for node in body_last_nodes:
            cfg.add_edge(node, condition_node)

        # Loop back to body
        cfg.add_edge(condition_node, body_node)

        # Exit after condition
        return [condition_node]

    elif isinstance(statement, ReturnStatement):
        # Return statement
        return_node = f'Return({statement.expression})'
        cfg.add_node(return_node, label=return_node)
        cfg.add_edge(current_node, return_node)
        # Control flow ends here
        return []

    elif isinstance(statement, ThrowStatement):
        # Throw statement
        throw_node = f'Throw({statement.expression})'
        cfg.add_node(throw_node, label=throw_node)
        cfg.add_edge(current_node, throw_node)
        # Control flow ends here
        return []

    elif isinstance(statement, BreakStatement):
        # Break statement
        break_node = 'Break'
        cfg.add_node(break_node, label=break_node)
        cfg.add_edge(current_node, break_node)
        # Break statements terminate the loop, handled in higher context
        return []

    elif isinstance(statement, ContinueStatement):
        # Continue statement
        continue_node = 'Continue'
        cfg.add_node(continue_node, label=continue_node)
        cfg.add_edge(current_node, continue_node)
        # Continue statements loop back, handled in higher context
        return []

    elif isinstance(statement, SynchronizedStatement):
        # Synchronized statement
        sync_node = f'Synchronized({statement.expression})'
        cfg.add_node(sync_node, label=sync_node)
        cfg.add_edge(current_node, sync_node)
        # Process the block inside synchronized
        return process_block_statements(statement.block, cfg, sync_node)

    elif isinstance(statement, TryStatement):
        # Try statement
        try_node = 'Try'
        cfg.add_node(try_node, label='Try')
        cfg.add_edge(current_node, try_node)

        # Process try block
        try_last_nodes = process_block_statements(statement.block, cfg, try_node)

        # Process catches
        catch_last_nodes = []
        for catch_clause in statement.catches:
            catch_node = f'Catch({catch_clause.parameter.name})'
            cfg.add_node(catch_node, label=catch_node)
            cfg.add_edge(try_node, catch_node)
            catch_last_nodes.extend(process_block_statements(catch_clause.block, cfg, catch_node))

        # Process finally block if present
        if statement.finally_block:
            finally_node = 'Finally'
            cfg.add_node(finally_node, label='Finally')
            for node in try_last_nodes + catch_last_nodes:
                cfg.add_edge(node, finally_node)
            finally_last_nodes = process_block_statements(statement.finally_block, cfg, finally_node)
            return finally_last_nodes
        else:
            # Merge paths from try and catches
            return try_last_nodes + catch_last_nodes

    elif isinstance(statement, StatementExpression):
        # Expression statement
        expr_node = f'{statement.expression}'
        cfg.add_node(expr_node, label=expr_node)
        cfg.add_edge(current_node, expr_node)
        return [expr_node]

    elif isinstance(statement, LocalVariableDeclaration):
        # Variable declaration
        var_node = f'{statement}'
        cfg.add_node(var_node, label=var_node)
        cfg.add_edge(current_node, var_node)
        return [var_node]

    elif isinstance(statement, EmptyStatement):
        # Empty statement
        return [current_node]

    else:
        # Other statements
        stmt_node = f'Unknown({type(statement).__name__})'
        cfg.add_node(stmt_node, label=stmt_node)
        cfg.add_edge(current_node, stmt_node)
        return [stmt_node]

def generate_control_flow_graphs(java_file_path):
    """
    Generates control flow graphs for a given Java file.
    """
    parsed_java_code = parse_java_file(java_file_path)
    method_declarations = extract_method_declarations(parsed_java_code)

    cfgs = []
    for method in method_declarations:
        cfg = create_cfg(method)
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
                'label': attr.get('label', '')
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
    java_file_path = "path/to/your/java/file.java"  # Replace with your actual Java file path
    cfgs = generate_control_flow_graphs(java_file_path)
    save_cfgs(cfgs)
