# cfg.py
import networkx as nx
from javalang.parse import parse
from javalang.tree import MethodDeclaration, BlockStatement, LocalVariableDeclaration

def parse_java_file(java_file_path):
    with open(java_file_path, 'r') as file:
        java_code = file.read()
    return parse(java_code)

def extract_method_blocks(parsed_java_code):
    method_blocks = []
    for _, node in parsed_java_code:
        if isinstance(node, MethodDeclaration):
            method_blocks.append(node)
    return method_blocks

def create_cfg(method_block):
    cfg = nx.DiGraph()
    previous_node = None

    def add_statement_to_cfg(statement, parent_node):
        nonlocal cfg, previous_node
        statement_node = str(statement) + str(parent_node)
        cfg.add_node(statement_node, label=statement)
        if parent_node:
            cfg.add_edge(parent_node, statement_node)
        previous_node = statement_node
        if isinstance(statement, BlockStatement):
            for child_statement in statement.statements:
                add_statement_to_cfg(child_statement, statement_node)
        elif isinstance(statement, LocalVariableDeclaration):
            for declarator in statement.declarators:
                identifier_node = declarator.name
                cfg.add_node(identifier_node, label="Identifier")
                cfg.add_edge(statement_node, identifier_node)

    add_statement_to_cfg(method_block.body, None)
    return cfg

def generate_control_flow_graphs(java_file_path):
    parsed_java_code = parse_java_file(java_file_path)
    method_blocks = extract_method_blocks(parsed_java_code)

    cfgs = []
    for method_block in method_blocks:
        cfg = create_cfg(method_block)
        cfgs.append(cfg)
    
    return cfgs
