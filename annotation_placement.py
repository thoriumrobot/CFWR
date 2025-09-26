#!/usr/bin/env python3
"""
Advanced Annotation Placement Module

This module provides sophisticated annotation placement capabilities for Java code,
including support for different annotation types and intelligent placement strategies.
"""

import os
import re
import ast
import javalang
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class AnnotationType(Enum):
    """Types of Checker Framework annotations"""
    # Nullness Checker
    NON_NULL = "@NonNull"
    NULLABLE = "@Nullable"
    NON_NULL_BY_DEFAULT = "@NonNullByDefault"
    POLY_NULL = "@PolyNull"
    MONOTONIC_NON_NULL = "@MonotonicNonNull"
    
    # Index Checker
    INDEX_FOR = "@IndexFor"
    INDEX_OR_LOW = "@IndexOrLow"
    INDEX_OR_HIGH = "@IndexOrHigh"
    LOWER_BOUND = "@LowerBound"
    UPPER_BOUND = "@UpperBound"
    
    # Lower Bound Checker - Enhanced Support
    MIN_LEN = "@MinLen"
    ARRAY_LEN = "@ArrayLen"
    LT_EQ_LENGTH_OF = "@LTEqLengthOf"
    GT_LENGTH_OF = "@GTLengthOf"
    LENGTH_OF = "@LengthOf"
    POSITIVE = "@Positive"
    NON_NEGATIVE = "@NonNegative"
    GT_NEG_ONE = "@GTENegativeOne"
    LT_LENGTH_OF = "@LTLengthOf"
    SEARCH_INDEX_FOR = "@SearchIndexFor"
    SEARCH_INDEX_BOTTOM = "@SearchIndexBottom"
    SEARCH_INDEX_UNKNOWN = "@SearchIndexUnknown"
    
    # Additional useful annotations
    SAME_LEN = "@SameLen"
    CAPACITY_FOR = "@CapacityFor"
    HAS_SUBSEQUENCE = "@HasSubsequence"

@dataclass
class AnnotationPlacement:
    """Represents a placement for an annotation"""
    line_number: int
    annotation_type: AnnotationType
    target_element: str  # variable name, method name, etc.
    placement_strategy: str  # 'before_line', 'before_method', 'before_class', etc.

class JavaCodeAnalyzer:
    """Analyzes Java code to understand structure and find placement opportunities"""
    
    def __init__(self, java_file_path: str):
        self.java_file_path = java_file_path
        self.content = self._read_file()
        self.lines = self.content.split('\n')
        self.ast = self._parse_java()
        
    def _read_file(self) -> str:
        """Read the Java file content"""
        with open(self.java_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_java(self) -> Optional[javalang.tree.CompilationUnit]:
        """Parse Java code using javalang"""
        try:
            return javalang.parse.parse(self.content)
        except Exception as e:
            print(f"Error parsing Java file: {e}")
            return None
    
    def find_variable_declarations(self) -> List[Dict]:
        """Find all variable declarations in the code"""
        variables = []
        if not self.ast:
            return variables
            
        for path, node in self.ast:
            if isinstance(node, javalang.tree.VariableDeclarator):
                # Find the line number
                line_num = self._get_line_number(node.position)
                variables.append({
                    'name': node.name,
                    'line': line_num,
                    'type': self._get_variable_type(path, node),
                    'is_field': self._is_field_declaration(path),
                    'is_parameter': self._is_parameter(path)
                })
        
        return variables
    
    def find_method_declarations(self) -> List[Dict]:
        """Find all method declarations in the code"""
        methods = []
        if not self.ast:
            return methods
            
        for path, node in self.ast:
            if isinstance(node, javalang.tree.MethodDeclaration):
                line_num = self._get_line_number(node.position)
                methods.append({
                    'name': node.name,
                    'line': line_num,
                    'return_type': node.return_type,
                    'parameters': self._get_method_parameters(node),
                    'is_public': 'public' in node.modifiers if node.modifiers else False,
                    'is_static': 'static' in node.modifiers if node.modifiers else False
                })
        
        return methods
    
    def find_class_declarations(self) -> List[Dict]:
        """Find all class declarations in the code"""
        classes = []
        if not self.ast:
            return classes
            
        for path, node in self.ast:
            if isinstance(node, javalang.tree.ClassDeclaration):
                line_num = self._get_line_number(node.position)
                classes.append({
                    'name': node.name,
                    'line': line_num,
                    'is_public': 'public' in node.modifiers if node.modifiers else False,
                    'is_abstract': 'abstract' in node.modifiers if node.modifiers else False
                })
        
        return classes
    
    def _get_line_number(self, position) -> int:
        """Get line number from position"""
        if position and hasattr(position, 'line'):
            return position.line
        return 1
    
    def _get_variable_type(self, path, node) -> str:
        """Get the type of a variable"""
        # Walk up the path to find the type declaration
        for parent_node in reversed(path):
            if isinstance(parent_node, javalang.tree.VariableDeclaration):
                return str(parent_node.type)
        return "Object"
    
    def _is_field_declaration(self, path) -> bool:
        """Check if variable is a field declaration"""
        for parent_node in path:
            if isinstance(parent_node, javalang.tree.FieldDeclaration):
                return True
        return False
    
    def _is_parameter(self, path) -> bool:
        """Check if variable is a method parameter"""
        for parent_node in path:
            if isinstance(parent_node, javalang.tree.FormalParameter):
                return True
        return False
    
    def _get_method_parameters(self, method_node) -> List[Dict]:
        """Get method parameters"""
        parameters = []
        if method_node.parameters:
            for param in method_node.parameters:
                parameters.append({
                    'name': param.name,
                    'type': str(param.type)
                })
        return parameters

class AnnotationPlacementStrategy:
    """Strategies for placing annotations in Java code"""
    
    def __init__(self, analyzer: JavaCodeAnalyzer):
        self.analyzer = analyzer
    
    def place_nullness_annotations(self, predicted_lines: List[int]) -> List[AnnotationPlacement]:
        """Place nullness annotations based on predicted lines"""
        placements = []
        
        # Get code structure
        variables = self.analyzer.find_variable_declarations()
        methods = self.analyzer.find_method_declarations()
        
        for line_num in predicted_lines:
            # Find the best annotation type and placement for this line
            placement = self._determine_annotation_placement(line_num, variables, methods)
            if placement:
                placements.append(placement)
        
        return placements
    
    def _determine_annotation_placement(self, line_num: int, variables: List[Dict], methods: List[Dict]) -> Optional[AnnotationPlacement]:
        """Determine the best annotation placement for a given line"""
        
        # Check if line contains a variable declaration
        for var in variables:
            if var['line'] == line_num:
                # Determine annotation type based on context
                annotation_type = self._choose_nullness_annotation(var)
                return AnnotationPlacement(
                    line_number=line_num,
                    annotation_type=annotation_type,
                    target_element=var['name'],
                    placement_strategy='before_line'
                )
        
        # Check if line contains a method declaration
        for method in methods:
            if method['line'] == line_num:
                # Place annotation on method return type or parameters
                annotation_type = AnnotationType.NON_NULL
                return AnnotationPlacement(
                    line_number=line_num,
                    annotation_type=annotation_type,
                    target_element=method['name'],
                    placement_strategy='before_method'
                )
        
        return None
    
    def _choose_nullness_annotation(self, var: Dict) -> AnnotationType:
        """Choose the appropriate nullness annotation for a variable"""
        var_type = var.get('type', '').lower()
        
        # Heuristics for choosing annotation type
        if 'string' in var_type:
            return AnnotationType.NON_NULL
        elif 'list' in var_type or 'array' in var_type:
            return AnnotationType.NON_NULL
        elif 'int' in var_type or 'long' in var_type or 'double' in var_type:
            return AnnotationType.NON_NULL  # Primitives are non-null
        else:
            return AnnotationType.NON_NULL  # Default to non-null
    
    def place_index_annotations(self, predicted_lines: List[int]) -> List[AnnotationPlacement]:
        """Place index annotations based on predicted lines"""
        placements = []
        
        variables = self.analyzer.find_variable_declarations()
        
        for line_num in predicted_lines:
            for var in variables:
                if var['line'] == line_num:
                    var_type = var.get('type', '').lower()
                    
                    # Choose index annotation based on type
                    if 'array' in var_type or 'list' in var_type:
                        annotation_type = AnnotationType.INDEX_FOR
                    elif 'int' in var_type:
                        annotation_type = AnnotationType.LOWER_BOUND
                    else:
                        continue
                    
                    placements.append(AnnotationPlacement(
                        line_number=line_num,
                        annotation_type=annotation_type,
                        target_element=var['name'],
                        placement_strategy='before_line'
                    ))
        
        return placements

class AnnotationInserter:
    """Handles the actual insertion of annotations into Java code"""
    
    def __init__(self, java_file_path: str):
        self.java_file_path = java_file_path
        self.content = self._read_file()
        self.lines = self.content.split('\n')
    
    def _read_file(self) -> str:
        """Read the Java file content"""
        with open(self.java_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def insert_annotations(self, placements: List[AnnotationPlacement]) -> bool:
        """Insert annotations at the specified placements"""
        try:
            # Sort placements by line number in descending order to avoid line shifts
            sorted_placements = sorted(placements, key=lambda p: p.line_number, reverse=True)
            
            for placement in sorted_placements:
                self._insert_single_annotation(placement)
            
            # Write the modified content back to file
            self._write_file()
            return True
            
        except Exception as e:
            print(f"Error inserting annotations: {e}")
            return False
    
    def _insert_single_annotation(self, placement: AnnotationPlacement):
        """Insert a single annotation at the specified placement"""
        line_idx = placement.line_number - 1
        
        if line_idx < 0 or line_idx >= len(self.lines):
            return
        
        # Determine the annotation string
        annotation_str = self._format_annotation(placement)
        
        # Insert the annotation based on strategy
        if placement.placement_strategy == 'before_line':
            self._insert_before_line(line_idx, annotation_str)
        elif placement.placement_strategy == 'before_method':
            self._insert_before_method(line_idx, annotation_str)
        elif placement.placement_strategy == 'before_class':
            self._insert_before_class(line_idx, annotation_str)
    
    def _format_annotation(self, placement: AnnotationPlacement) -> str:
        """Format the annotation string"""
        annotation_type = placement.annotation_type.value
        
        # Add appropriate indentation
        target_line = self.lines[placement.line_number - 1]
        indentation = self._get_indentation(target_line)
        
        return f"{indentation}{annotation_type}"
    
    def _get_indentation(self, line: str) -> str:
        """Get the indentation of a line"""
        return re.match(r'^(\s*)', line).group(1) if line.strip() else ""
    
    def _insert_before_line(self, line_idx: int, annotation_str: str):
        """Insert annotation before a specific line"""
        self.lines.insert(line_idx, annotation_str)
    
    def _insert_before_method(self, line_idx: int, annotation_str: str):
        """Insert annotation before a method declaration"""
        # Find the method declaration line
        method_line = self.lines[line_idx]
        
        # If the method has modifiers, insert after them
        if any(modifier in method_line for modifier in ['public', 'private', 'protected', 'static']):
            # Find where the method signature starts
            parts = method_line.split()
            insert_pos = 0
            for i, part in enumerate(parts):
                if part in ['public', 'private', 'protected', 'static', 'final', 'abstract']:
                    insert_pos = i + 1
                else:
                    break
            
            # Insert annotation
            annotation_str = self._get_indentation(method_line) + annotation_str
            self.lines.insert(line_idx, annotation_str)
        else:
            # Simple method, insert before
            self.lines.insert(line_idx, annotation_str)
    
    def _insert_before_class(self, line_idx: int, annotation_str: str):
        """Insert annotation before a class declaration"""
        self.lines.insert(line_idx, annotation_str)
    
    def _write_file(self):
        """Write the modified content back to the file"""
        with open(self.java_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))

class AnnotationPlacementManager:
    """Main manager for annotation placement operations"""
    
    def __init__(self, java_file_path: str):
        self.java_file_path = java_file_path
        self.analyzer = JavaCodeAnalyzer(java_file_path)
        self.strategy = AnnotationPlacementStrategy(self.analyzer)
        self.inserter = AnnotationInserter(java_file_path)
    
    def place_annotations(self, predicted_lines: List[int], annotation_category: str = 'nullness') -> bool:
        """Place annotations based on predicted lines and category"""
        try:
            # Determine placement strategy based on category
            if annotation_category == 'nullness':
                placements = self.strategy.place_nullness_annotations(predicted_lines)
            elif annotation_category == 'index':
                placements = self.strategy.place_index_annotations(predicted_lines)
            else:
                print(f"Unknown annotation category: {annotation_category}")
                return False
            
            if not placements:
                print("No valid placements found")
                return False
            
            # Insert the annotations
            success = self.inserter.insert_annotations(placements)
            
            if success:
                print(f"Successfully placed {len(placements)} annotations")
                for placement in placements:
                    print(f"  - {placement.annotation_type.value} at line {placement.line_number} "
                          f"for {placement.target_element}")
            
            return success
            
        except Exception as e:
            print(f"Error in annotation placement: {e}")
            return False
    
    def get_code_structure(self) -> Dict:
        """Get the structure of the Java code"""
        return {
            'variables': self.analyzer.find_variable_declarations(),
            'methods': self.analyzer.find_method_declarations(),
            'classes': self.analyzer.find_class_declarations()
        }

def main():
    """Test the annotation placement functionality"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python annotation_placement.py <java_file>")
        sys.exit(1)
    
    java_file = sys.argv[1]
    
    if not os.path.exists(java_file):
        print(f"File not found: {java_file}")
        sys.exit(1)
    
    # Test annotation placement
    manager = AnnotationPlacementManager(java_file)
    
    # Get code structure
    structure = manager.get_code_structure()
    print("Code Structure:")
    print(f"  Variables: {len(structure['variables'])}")
    print(f"  Methods: {len(structure['methods'])}")
    print(f"  Classes: {len(structure['classes'])}")
    
    # Test with some predicted lines (example)
    predicted_lines = [3, 5, 7]  # Example line numbers
    success = manager.place_annotations(predicted_lines, 'nullness')
    
    if success:
        print("Annotation placement test completed successfully!")
    else:
        print("Annotation placement test failed!")

if __name__ == '__main__':
    main()
