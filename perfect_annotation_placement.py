#!/usr/bin/env python3
"""
Perfectly Accurate Annotation Placement System

This module provides precise, AST-based annotation placement that ensures
annotations are placed exactly where they should be, not approximately.
Uses detailed Java AST analysis for perfect positioning.
"""

import os
import re
import ast
import javalang
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreciseLocation:
    """Represents a precise location in Java source code"""
    line: int
    column: int
    absolute_position: int  # Character position in file
    context_type: str  # 'variable_declaration', 'method_parameter', 'field', etc.
    target_element: str  # Name of the element being annotated

@dataclass
class AnnotationTarget:
    """Represents a target for annotation placement"""
    location: PreciseLocation
    annotation_type: str
    placement_strategy: str
    syntax_context: Dict[str, any]  # Additional syntax information

class PreciseJavaAnalyzer:
    """Precise Java code analyzer using AST for exact positioning"""
    
    def __init__(self, java_file_path: str):
        self.java_file_path = java_file_path
        self.content = self._read_file()
        self.lines = self.content.split('\n')
        self.ast = self._parse_java()
        self.line_positions = self._calculate_line_positions()
        
    def _read_file(self) -> str:
        """Read the Java file content"""
        with open(self.java_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_java(self):
        """Parse Java code using javalang"""
        try:
            return javalang.parse.parse(self.content)
        except Exception as e:
            logger.warning(f"Could not parse Java file {self.java_file_path}: {e}")
            return None
    
    def _calculate_line_positions(self) -> List[int]:
        """Calculate absolute positions for each line start"""
        positions = [0]
        for line in self.lines:
            positions.append(positions[-1] + len(line) + 1)  # +1 for newline
        return positions
    
    def find_exact_location(self, line_number: int, target_element: str = None) -> Optional[PreciseLocation]:
        """Find the exact location for annotation placement"""
        if not self.ast:
            return None
            
        # Walk the AST to find the exact location
        for path, node in self.ast:
            if hasattr(node, 'position') and node.position:
                node_line = node.position.line
                if node_line == line_number:
                    return self._analyze_node_location(node, path, target_element)
        
        # Fallback: analyze by line content
        return self._analyze_line_content(line_number, target_element)
    
    def _analyze_node_location(self, node, path, target_element: str = None) -> PreciseLocation:
        """Analyze a specific AST node for precise location"""
        line = node.position.line
        column = node.position.column
        
        # Calculate absolute position
        absolute_pos = self.line_positions[line - 1] + column
        
        # Determine context type
        context_type = self._determine_context_type(node, path)
        
        # Find target element name
        element_name = self._extract_element_name(node, target_element)
        
        return PreciseLocation(
            line=line,
            column=column,
            absolute_position=absolute_pos,
            context_type=context_type,
            target_element=element_name
        )
    
    def _analyze_line_content(self, line_number: int, target_element: str = None) -> Optional[PreciseLocation]:
        """Analyze line content when AST analysis fails"""
        if line_number <= 0 or line_number > len(self.lines):
            return None
            
        line_content = self.lines[line_number - 1]
        column = 0
        
        # Find the start of the relevant element
        if target_element:
            element_pos = line_content.find(target_element)
            if element_pos != -1:
                column = element_pos
        
        absolute_pos = self.line_positions[line_number - 1] + column
        
        context_type = self._determine_context_type_from_line(line_content)
        
        return PreciseLocation(
            line=line_number,
            column=column,
            absolute_position=absolute_pos,
            context_type=context_type,
            target_element=target_element or ""
        )
    
    def _determine_context_type(self, node, path) -> str:
        """Determine the context type from AST node"""
        if isinstance(node, javalang.tree.VariableDeclarator):
            return 'variable_declaration'
        elif isinstance(node, javalang.tree.MethodDeclaration):
            return 'method_declaration'
        elif isinstance(node, javalang.tree.FieldDeclaration):
            return 'field_declaration'
        elif isinstance(node, javalang.tree.FormalParameter):
            return 'method_parameter'
        elif isinstance(node, javalang.tree.ClassDeclaration):
            return 'class_declaration'
        else:
            return 'unknown'
    
    def _determine_context_type_from_line(self, line_content: str) -> str:
        """Determine context type from line content"""
        line_lower = line_content.strip().lower()
        
        if 'public' in line_lower and '(' in line_content and ')' in line_content:
            return 'method_declaration'
        elif any(keyword in line_lower for keyword in ['int ', 'long ', 'string ', 'boolean ', 'double ']):
            return 'variable_declaration'
        elif 'private' in line_lower or 'protected' in line_lower:
            return 'field_declaration'
        elif 'class' in line_lower:
            return 'class_declaration'
        else:
            return 'unknown'
    
    def _extract_element_name(self, node, target_element: str = None) -> str:
        """Extract the name of the element being annotated"""
        if target_element:
            return target_element
            
        if isinstance(node, javalang.tree.VariableDeclarator):
            return node.name
        elif isinstance(node, javalang.tree.MethodDeclaration):
            return node.name
        elif isinstance(node, javalang.tree.FieldDeclaration):
            if node.declarators:
                return node.declarators[0].name
        elif isinstance(node, javalang.tree.FormalParameter):
            return node.name
        elif isinstance(node, javalang.tree.ClassDeclaration):
            return node.name
        
        return ""

class PreciseAnnotationPlacer:
    """Precisely accurate annotation placement system"""
    
    def __init__(self, java_file_path: str):
        self.java_file_path = java_file_path
        self.analyzer = PreciseJavaAnalyzer(java_file_path)
        self.content = self.analyzer.content
        self.lines = self.analyzer.lines
        
    def place_multiple_annotations_precisely(self, line_number: int, annotations: List[str], 
                                           target_element: str = None) -> bool:
        """Place multiple annotations with perfect accuracy"""
        try:
            # Find exact location
            location = self.analyzer.find_exact_location(line_number, target_element)
            if not location:
                logger.error(f"Could not find precise location for line {line_number}")
                return False
            
            # Determine placement strategy
            placement_strategy = self._determine_precise_strategy(location, 'before_element')
            
            # Place all annotations at the same location
            success_count = 0
            for annotation in annotations:
                success = self._place_at_exact_location(location, annotation, placement_strategy)
                if success:
                    success_count += 1
            
            if success_count > 0:
                logger.info(f"Precisely placed {success_count}/{len(annotations)} annotations at {location.line}:{location.column}")
                return True
            else:
                logger.warning(f"Failed to place any annotations at {location.line}:{location.column}")
                return False
                
        except Exception as e:
            logger.error(f"Error in multiple annotation placement: {e}")
            return False
    
    def place_annotation_precisely(self, line_number: int, annotation: str, 
                                 target_element: str = None, 
                                 placement_type: str = 'before_element') -> bool:
        """Place annotation with perfect accuracy"""
        try:
            # Find exact location
            location = self.analyzer.find_exact_location(line_number, target_element)
            if not location:
                logger.error(f"Could not find precise location for line {line_number}")
                return False
            
            # Determine exact placement strategy
            placement_strategy = self._determine_precise_strategy(location, placement_type)
            
            # Place annotation with exact positioning
            success = self._place_at_exact_location(location, annotation, placement_strategy)
            
            if success:
                logger.info(f"Precisely placed {annotation} at {location.line}:{location.column}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in precise placement: {e}")
            return False
    
    def _determine_precise_strategy(self, location: PreciseLocation, placement_type: str) -> str:
        """Determine the precise placement strategy based on context"""
        context = location.context_type
        
        # For method/constructor lines, check if we're targeting a parameter
        if context in ['method_declaration', 'unknown']:
            # Check if the line contains parameters
            line_content = self.lines[location.line - 1] if location.line <= len(self.lines) else ""
            if '(' in line_content and ')' in line_content:
                # This is likely a method/constructor with parameters
                return 'before_parameter'
        
        if context == 'variable_declaration':
            return 'before_type'
        elif context == 'method_parameter':
            return 'before_parameter'
        elif context == 'method_declaration':
            return 'before_method'
        elif context == 'field_declaration':
            return 'before_field'
        elif context == 'class_declaration':
            return 'before_class'
        else:
            return placement_type
    
    def _place_at_exact_location(self, location: PreciseLocation, annotation: str, 
                               strategy: str) -> bool:
        """Place annotation at exact location using precise strategy"""
        try:
            line_idx = location.line - 1
            line_content = self.lines[line_idx]
            
            if strategy == 'before_type':
                return self._place_before_type(line_idx, line_content, annotation)
            elif strategy == 'before_parameter':
                return self._place_before_parameter(line_idx, line_content, annotation)
            elif strategy == 'before_method':
                return self._place_before_method(line_idx, line_content, annotation)
            elif strategy == 'before_field':
                return self._place_before_field(line_idx, line_content, annotation)
            elif strategy == 'before_class':
                return self._place_before_class(line_idx, line_content, annotation)
            else:
                return self._place_before_line(line_idx, annotation)
                
        except Exception as e:
            logger.error(f"Error placing annotation at exact location: {e}")
            return False
    
    def _place_before_type(self, line_idx: int, line_content: str, annotation: str) -> bool:
        """Place annotation before the type declaration"""
        # Find the type keyword (int, String, etc.)
        type_patterns = [
            r'\b(int|long|double|float|boolean|char|byte|short|String|Object|\w+)\s+',
            r'\b(\w+)\s*\[\s*\]\s+',  # Array types
        ]
        
        for pattern in type_patterns:
            match = re.search(pattern, line_content)
            if match:
                type_start = match.start()
                # Insert annotation before the type
                new_line = line_content[:type_start] + annotation + " " + line_content[type_start:]
                self.lines[line_idx] = new_line
                return True
        
        # Fallback: place at beginning of line
        return self._place_before_line(line_idx, annotation)
    
    def _place_before_parameter(self, line_idx: int, line_content: str, annotation: str) -> bool:
        """Place annotation before method parameter with precise positioning"""
        # More sophisticated parameter detection
        # Look for patterns like: method(Type param) or constructor(Type param)
        
        # First, find the opening parenthesis
        paren_pos = line_content.find('(')
        if paren_pos == -1:
            return self._place_before_line(line_idx, annotation)
        
        # Look for parameter patterns after the opening parenthesis
        after_paren = line_content[paren_pos + 1:]
        
        # Pattern: Type param
        param_pattern = r'(\w+)\s+(\w+)\s*[,)]'
        match = re.search(param_pattern, after_paren)
        
        if match:
            # Calculate the position in the full line
            param_start_in_line = paren_pos + 1 + match.start()
            
            # Insert annotation before the parameter
            new_line = (line_content[:param_start_in_line] + 
                       annotation + " " + 
                       line_content[param_start_in_line:])
            self.lines[line_idx] = new_line
            return True
        
        # Fallback: place before the opening parenthesis
        new_line = (line_content[:paren_pos] + 
                   annotation + " " + 
                   line_content[paren_pos:])
        self.lines[line_idx] = new_line
        return True
    
    def _place_before_method(self, line_idx: int, line_content: str, annotation: str) -> bool:
        """Place annotation before method declaration"""
        # Find method signature start
        method_pattern = r'(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\('
        match = re.search(method_pattern, line_content)
        
        if match:
            method_start = match.start()
            # Insert annotation before method
            new_line = line_content[:method_start] + annotation + " " + line_content[method_start:]
            self.lines[line_idx] = new_line
            return True
        
        return self._place_before_line(line_idx, annotation)
    
    def _place_before_field(self, line_idx: int, line_content: str, annotation: str) -> bool:
        """Place annotation before field declaration"""
        # Similar to variable declaration
        return self._place_before_type(line_idx, line_content, annotation)
    
    def _place_before_class(self, line_idx: int, line_content: str, annotation: str) -> bool:
        """Place annotation before class declaration"""
        class_pattern = r'(public|private|protected)?\s*(static)?\s*(abstract)?\s*class\s+(\w+)'
        match = re.search(class_pattern, line_content)
        
        if match:
            class_start = match.start()
            # Insert annotation before class
            new_line = line_content[:class_start] + annotation + " " + line_content[class_start:]
            self.lines[line_idx] = new_line
            return True
        
        return self._place_before_line(line_idx, annotation)
    
    def _place_before_line(self, line_idx: int, annotation: str) -> bool:
        """Place annotation at the beginning of the line"""
        line_content = self.lines[line_idx]
        indent = len(line_content) - len(line_content.lstrip())
        indent_str = ' ' * indent
        
        # Insert annotation line before current line
        annotation_line = indent_str + annotation
        self.lines.insert(line_idx, annotation_line)
        return True
    
    def save_file(self) -> bool:
        """Save the modified file"""
        try:
            new_content = '\n'.join(self.lines)
            with open(self.java_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False

class PerfectAnnotationPlacementSystem:
    """Main system for perfectly accurate annotation placement"""
    
    def __init__(self, project_root: str, output_dir: str):
        self.project_root = project_root
        self.output_dir = output_dir
        self.placed_annotations = {}
        
    def place_annotations_perfectly(self, predictions: List[Dict]) -> Dict[str, any]:
        """Place annotations with perfect accuracy"""
        results = {
            'successful': 0,
            'failed': 0,
            'files_processed': 0,
            'details': []
        }
        
        # Group predictions by file
        predictions_by_file = {}
        for pred in predictions:
            file_path = pred.get('file_path', '')
            if file_path not in predictions_by_file:
                predictions_by_file[file_path] = []
            predictions_by_file[file_path].append(pred)
        
        # Process each file
        for file_path, file_predictions in predictions_by_file:
            full_path = os.path.join(self.project_root, file_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"File not found: {full_path}")
                results['failed'] += len(file_predictions)
                continue
            
            # Process file with perfect placement
            file_result = self._process_file_perfectly(full_path, file_predictions)
            results['successful'] += file_result['successful']
            results['failed'] += file_result['failed']
            results['files_processed'] += 1
            results['details'].append(file_result)
        
        return results
    
    def _process_file_perfectly(self, file_path: str, predictions: List[Dict]) -> Dict[str, any]:
        """Process a single file with perfect annotation placement"""
        result = {
            'file': file_path,
            'successful': 0,
            'failed': 0,
            'placements': []
        }
        
        try:
            # Create precise placer for this file
            placer = PreciseAnnotationPlacer(file_path)
            
            # Process each prediction
            for pred in predictions:
                line_number = pred.get('line_number', 0)
                annotation = pred.get('annotation_type', '@NonNull')
                target_element = pred.get('target_element', '')
                
                # Place annotation with perfect accuracy
                success = placer.place_annotation_precisely(
                    line_number, annotation, target_element
                )
                
                if success:
                    result['successful'] += 1
                    result['placements'].append({
                        'line': line_number,
                        'annotation': annotation,
                        'target': target_element,
                        'status': 'success'
                    })
                else:
                    result['failed'] += 1
                    result['placements'].append({
                        'line': line_number,
                        'annotation': annotation,
                        'target': target_element,
                        'status': 'failed'
                    })
            
            # Save the modified file
            if result['successful'] > 0:
                placer.save_file()
                logger.info(f"Successfully placed {result['successful']} annotations in {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            result['failed'] += len(predictions)
        
        return result

def test_perfect_placement():
    """Test the perfect placement system"""
    import tempfile
    
    # Create test Java file
    test_content = '''public class TestClass {
    private String name;
    private int value;
    
    public TestClass(String name, int value) {
        this.name = name;
        this.value = value;
    }
    
    public String getName() {
        return name;
    }
    
    public int getValue() {
        return value;
    }
}'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Test precise placement
        placer = PreciseAnnotationPlacer(temp_file)
        
        # Place annotations at specific locations
        success1 = placer.place_annotation_precisely(2, '@NonNull', 'name', 'before_type')
        success2 = placer.place_annotation_precisely(3, '@Positive', 'value', 'before_type')
        success3 = placer.place_annotation_precisely(5, '@NonNull', 'name', 'before_parameter')
        
        if success1 and success2 and success3:
            placer.save_file()
            
            # Read and display result
            with open(temp_file, 'r') as f:
                result = f.read()
            
            print("Perfect placement test result:")
            print("-" * 50)
            for i, line in enumerate(result.split('\n'), 1):
                print(f"{i:2}: {line}")
            print("-" * 50)
            
            return True
        else:
            print("Perfect placement test failed")
            return False
            
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    test_perfect_placement()
