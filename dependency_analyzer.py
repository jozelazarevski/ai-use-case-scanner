# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:51:33 2025

@author: joze_
"""

#!/usr/bin/env python3
"""
Code Dependency Analyzer

This script analyzes Python code files to identify dependencies and find unused files.
It scans all Python files in a directory, builds a dependency graph, and identifies
files that are not imported by any other file (potential unused files).

Usage:
    python dependency_analyzer.py [directory_to_scan] [--move]

Options:
    directory_to_scan    Directory to scan for Python files (default: current directory)
    --move               Move unused files to 'unused_scripts' directory
"""

import os
import sys
import re
import shutil
import importlib.util
import ast
from pathlib import Path
from collections import defaultdict, deque
import argparse


def extract_imports(file_path):
    """Extract all imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        try:
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                # Handle import statements
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                # Handle from ... import ... statements
                elif isinstance(node, ast.ImportFrom):
                    if node.module is not None:
                        imports.append(node.module)
            
            # Also look for relative imports in strings that might not be parsed properly
            # This handles cases like dynamic imports or string-based imports
            import_patterns = [
                r'from\s+([\w.]+)\s+import',  # from x import y
                r'import\s+([\w.]+)',         # import x
                r'importlib\.import_module\([\'"](.+?)[\'"]\)',  # importlib.import_module('x')
                r'__import__\([\'"](.+?)[\'"]\)'  # __import__('x')
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)
            
            return imports
        except SyntaxError:
            print(f"Syntax error in {file_path}, skipping AST parsing")
            # Fallback to regex only for syntax error files
            import_patterns = [
                r'from\s+([\w.]+)\s+import',
                r'import\s+([\w.]+)',
                r'importlib\.import_module\([\'"](.+?)[\'"]\)',
                r'__import__\([\'"](.+?)[\'"]\)'
            ]
            
            imports = []
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)
            
            return imports
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def find_all_python_files(directory):
    """Find all Python files in the specified directory and its subdirectories."""
    python_files = []
    
    for root, _, files in os.walk(directory):
        # Skip the unused_scripts directory if it exists
        if os.path.basename(root) == 'unused_scripts':
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def normalize_import(import_name, file_path, all_files):
    """Normalize import names to match file paths."""
    # Handle standard library and third-party imports
    if import_name.split('.')[0] in sys.modules or importlib.util.find_spec(import_name.split('.')[0]) is not None:
        return None
    
    # Convert import format to file path format
    potential_file = import_name.replace('.', os.sep) + '.py'
    
    # Check if this is a local import
    for file in all_files:
        # Check if the file path ends with our potential file path
        if file.endswith(potential_file):
            return file
        
        # Also check if it's a module (directory with __init__.py)
        module_dir = os.path.dirname(file)
        if os.path.basename(module_dir) == import_name and os.path.exists(os.path.join(module_dir, '__init__.py')):
            return module_dir
        
        # For packages like 'from .submodule import x'
        if '.' in potential_file and os.path.dirname(file) == os.path.dirname(os.path.join(os.path.dirname(file_path), potential_file)):
            return os.path.join(os.path.dirname(file_path), potential_file)
    
    # Check for relative imports like '.module'
    if import_name.startswith('.'):
        relative_path = os.path.dirname(file_path)
        for _ in range(import_name.count('.')):
            relative_path = os.path.dirname(relative_path)
        
        module_name = import_name.lstrip('.')
        if module_name:
            potential_file = os.path.join(relative_path, module_name.replace('.', os.sep) + '.py')
            if os.path.exists(potential_file) and potential_file in all_files:
                return potential_file
    
    return None


def build_dependency_graph(directory):
    """Build a dependency graph from all Python files in the specified directory."""
    all_files = find_all_python_files(directory)
    dependency_graph = defaultdict(set)
    reverse_dependency_graph = defaultdict(set)
    
    # Process Python script files
    for file_path in all_files:
        imports = extract_imports(file_path)
        
        for import_name in imports:
            imported_file = normalize_import(import_name, file_path, all_files)
            if imported_file is not None:
                dependency_graph[file_path].add(imported_file)
                reverse_dependency_graph[imported_file].add(file_path)
    
    return all_files, dependency_graph, reverse_dependency_graph


def find_entry_points(all_files):
    """Find potential entry point scripts (scripts that would be run directly)."""
    entry_points = []
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Check for indicators of an entry point
                if '__main__' in content and ('if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content):
                    entry_points.append(file_path)
                    
                # Also check for shebang lines
                if content.startswith('#!') and ('python' in content.splitlines()[0]):
                    entry_points.append(file_path)
        except Exception as e:
            print(f"Error checking entry point for {file_path}: {e}")
    
    return entry_points


def find_unused_files(all_files, dependency_graph, reverse_dependency_graph, entry_points):
    """Find unused Python files (files not imported by any other file)."""
    used_files = set()
    
    # Files imported by other files
    for imports in dependency_graph.values():
        used_files.update(imports)
    
    # Add entry points to used files
    used_files.update(entry_points)
    
    # Mark files called directly by entry points as used
    for entry_point in entry_points:
        queue = deque([entry_point])
        while queue:
            current_file = queue.popleft()
            used_files.add(current_file)
            for import_file in dependency_graph[current_file]:
                if import_file not in used_files:
                    queue.append(import_file)
                    used_files.add(import_file)
    
    # Find unused files
    unused_files = [file for file in all_files if file not in used_files]
    
    return unused_files


def move_unused_files(unused_files, directory):
    """Move unused files to 'unused_scripts' directory."""
    unused_dir = os.path.join(directory, 'unused_scripts')
    os.makedirs(unused_dir, exist_ok=True)
    
    moved_files = []
    
    for file_path in unused_files:
        # Create relative path structure in unused_scripts
        rel_path = os.path.relpath(file_path, directory)
        target_path = os.path.join(unused_dir, rel_path)
        
        # Create parent directories
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Move the file
        try:
            shutil.move(file_path, target_path)
            moved_files.append((file_path, target_path))
            print(f"Moved: {file_path} -> {target_path}")
        except Exception as e:
            print(f"Error moving file {file_path}: {e}")
    
    return moved_files


def print_summary(all_files, dependency_graph, reverse_dependency_graph, entry_points, unused_files):
    """Print a summary of the dependency analysis."""
    print("\n--- DEPENDENCY ANALYSIS SUMMARY ---")
    print(f"Total Python files found: {len(all_files)}")
    print(f"Entry points (runnable scripts): {len(entry_points)}")
    
    # Most imported files
    if reverse_dependency_graph:
        most_imported = sorted(reverse_dependency_graph.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nMost imported files:")
        for file, importers in most_imported[:5]:
            print(f"  {os.path.basename(file)}: imported by {len(importers)} files")
    
    # Files with most dependencies
    if dependency_graph:
        most_dependencies = sorted(dependency_graph.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nFiles with most dependencies:")
        for file, dependencies in most_dependencies[:5]:
            print(f"  {os.path.basename(file)}: imports {len(dependencies)} files")
    
    # Print unused files
    print(f"\nUnused files found: {len(unused_files)}")
    for file in unused_files[:10]:  # Show only first 10 to avoid long output
        print(f"  {file}")
    
    if len(unused_files) > 10:
        print(f"  ...and {len(unused_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(description='Analyze Python code dependencies and find unused files')
    parser.add_argument('directory', nargs='?', default=os.getcwd(),
                        help='Directory to scan for Python files (default: current directory)')
    parser.add_argument('--move', action='store_true',
                        help='Move unused files to unused_scripts directory')
    
    args = parser.parse_args()
    directory = args.directory
    
    print(f"Scanning directory: {directory}")
    
    # Build dependency graph
    all_files, dependency_graph, reverse_dependency_graph = build_dependency_graph(directory)
    
    # Find entry points
    entry_points = find_entry_points(all_files)
    
    # Find unused files
    unused_files = find_unused_files(all_files, dependency_graph, reverse_dependency_graph, entry_points)
    
    # Print summary
    print_summary(all_files, dependency_graph, reverse_dependency_graph, entry_points, unused_files)
    
    # Move unused files if requested
    if args.move and unused_files:
        print("\nMoving unused files to 'unused_scripts' directory...")
        moved_files = move_unused_files(unused_files, directory)
        print(f"Moved {len(moved_files)} unused files.")
    elif args.move:
        print("\nNo unused files to move.")
    else:
        if unused_files:
            print("\nUse --move to move unused files to 'unused_scripts' directory")


if __name__ == "__main__":
    main()