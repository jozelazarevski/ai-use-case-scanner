#!/usr/bin/env python3
"""
Enhanced App Dependency Tree Analyzer

This script analyzes the dependency tree of app.py, showing all direct and indirect
imports (recursive dependencies). It also identifies all files and folders that
are NOT used by app.py and provides a comprehensive list of unused resources.

Usage:
    python app_dependency_analyzer.py [path_to_app.py] [--max-depth MAX_DEPTH] [--move]

Options:
    path_to_app.py   Path to app.py or other Python file to analyze (default: app.py)
    --max-depth      Maximum depth to display in the dependency tree
    --move           Move unused files to 'unused_scripts' directory
"""

import os
import sys
import re
import ast
import shutil
from pathlib import Path
from collections import defaultdict, deque
import argparse
import importlib.util


class DependencyTreeAnalyzer:
    def __init__(self, app_file_path):
        self.app_file_path = os.path.abspath(app_file_path)
        self.root_dir = os.path.dirname(self.app_file_path)
        self.dependency_graph = defaultdict(set)
        self.file_sizes = {}  # Initialize file sizes dictionary
        self.processed_modules = set()
        self.all_python_files = self.find_all_python_files(self.root_dir)
        self.used_files = set()  # Will store all files used directly or indirectly
    
    def find_all_python_files(self, directory):
        """Find all Python files in the specified directory and its subdirectories."""
        python_files = []
        
        for root, _, files in os.walk(directory):
            # Skip the unused_scripts directory if it exists
            if 'unused_scripts' in root.split(os.sep):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.abspath(os.path.join(root, file))
                    python_files.append(full_path)
                    # Store file size for reporting
                    self.file_sizes[full_path] = os.path.getsize(full_path)
        
        return python_files
    
    def extract_imports(self, file_path):
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
                
                # Also look for relative imports and special patterns
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
    
    def normalize_import(self, import_name, file_path):
        """Normalize import names to match file paths."""
        # Handle standard library and third-party imports
        if (import_name.split('.')[0] in sys.modules or 
            importlib.util.find_spec(import_name.split('.')[0]) is not None):
            # Check if it's not a local module with the same name
            found_local = False
            for py_file in self.all_python_files:
                rel_path = os.path.relpath(py_file, self.root_dir)
                if rel_path == f"{import_name.replace('.', os.sep)}.py" or \
                   os.path.basename(py_file) == f"{import_name}.py":
                    found_local = True
                    return py_file
            
            if not found_local:
                return None
        
        # Convert import format to file path format
        potential_file = os.path.join(self.root_dir, import_name.replace('.', os.sep) + '.py')
        
        # Check for exact matches
        if potential_file in self.all_python_files:
            return potential_file
        
        # Check for matches based on file name
        for py_file in self.all_python_files:
            file_name = os.path.basename(py_file)
            module_name = os.path.splitext(file_name)[0]
            
            # Check if import is just the file name
            if module_name == import_name:
                return py_file
            
            # Check if import is a module with subdirectories
            rel_path = os.path.relpath(py_file, self.root_dir)
            if rel_path.replace(os.sep, '.').endswith(f"{import_name}.py"):
                return py_file
        
        # Check for packages (directories with __init__.py)
        potential_package = os.path.join(self.root_dir, import_name.replace('.', os.sep))
        init_file = os.path.join(potential_package, '__init__.py')
        if os.path.exists(init_file) and init_file in self.all_python_files:
            return init_file
        
        # Handle relative imports
        if import_name.startswith('.'):
            file_dir = os.path.dirname(file_path)
            for _ in range(import_name.count('.')):
                file_dir = os.path.dirname(file_dir)
            
            # Remove the leading dots
            module_name = import_name.lstrip('.')
            if module_name:
                rel_potential_file = os.path.join(file_dir, module_name.replace('.', os.sep) + '.py')
                if os.path.exists(rel_potential_file) and rel_potential_file in self.all_python_files:
                    return rel_potential_file
            # For cases like 'from . import module'
            else:
                init_file = os.path.join(file_dir, '__init__.py')
                if os.path.exists(init_file) and init_file in self.all_python_files:
                    return init_file
        
        # One more attempt with the file directory as context
        file_dir = os.path.dirname(file_path)
        context_file = os.path.join(file_dir, import_name.replace('.', os.sep) + '.py')
        if os.path.exists(context_file) and context_file in self.all_python_files:
            return context_file
        
        return None
    
    def build_dependency_graph(self):
        """Build the dependency graph for all Python files."""
        for file_path in self.all_python_files:
            imports = self.extract_imports(file_path)
            
            for import_name in imports:
                imported_file = self.normalize_import(import_name, file_path)
                if imported_file is not None:
                    self.dependency_graph[file_path].add(imported_file)
    
    def analyze_recursive_dependencies(self, start_file=None):
        """
        Analyze dependencies recursively starting from app.py or specified file.
        Returns a tree structure of dependencies.
        """
        if start_file is None:
            start_file = self.app_file_path
        
        self.build_dependency_graph()
        
        # Build a tree-like structure for visualization
        tree = {}
        visited = set()
        
        def build_tree(file, depth=0, path=None):
            if path is None:
                path = []
            
            # Mark this file as used
            self.used_files.add(file)
            
            # Prevent infinite recursion due to circular dependencies
            if file in path:
                return {"name": os.path.basename(file) + " (circular ref)", "children": [], "full_path": file}
            
            # Check if we've processed this file before
            if file in visited and depth > 0:
                return {"name": os.path.basename(file) + " (see above)", "children": [], "full_path": file}
            
            visited.add(file)
            
            node = {
                "name": os.path.basename(file),
                "full_path": file,
                "size": self.file_sizes.get(file, 0),
                "children": []
            }
            
            # Process dependencies
            for dep in sorted(self.dependency_graph.get(file, [])):
                child = build_tree(dep, depth + 1, path + [file])
                node["children"].append(child)
            
            return node
        
        # Start building the tree from app.py
        dependency_tree = build_tree(start_file)
        
        return dependency_tree
    
    def find_unused_files(self):
        """
        Find all Python files that are not used by app.py directly or indirectly.
        """
        # If used_files is empty, run the analysis first
        if not self.used_files:
            self.analyze_recursive_dependencies()
        
        # Find files that are not in used_files
        unused_files = [f for f in self.all_python_files if f not in self.used_files]
        
        # Group by directory for better organization
        unused_by_directory = defaultdict(list)
        for file in unused_files:
            directory = os.path.dirname(file)
            unused_by_directory[directory].append(file)
        
        return unused_by_directory
    
    def find_unused_directories(self):
        """
        Find directories that contain only unused Python files.
        """
        unused_dirs = set()
        unused_by_directory = self.find_unused_files()
        
        for directory, files in unused_by_directory.items():
            # Check if all Python files in this directory are unused
            all_py_files_in_dir = [f for f in self.all_python_files if os.path.dirname(f) == directory]
            
            if len(all_py_files_in_dir) == len(files):
                unused_dirs.add(directory)
                
                # Also check parent directories recursively
                parent = os.path.dirname(directory)
                while parent and parent != self.root_dir:
                    # Check if all subdirectories of this parent are unused
                    all_child_dirs = [d for d in unused_dirs if os.path.dirname(d) == parent]
                    all_subdirs = [d for d in [os.path.dirname(f) for f in self.all_python_files] 
                                  if os.path.dirname(d) == parent]
                    
                    if len(set(all_subdirs)) == len(all_child_dirs):
                        unused_dirs.add(parent)
                    
                    parent = os.path.dirname(parent)
        
        return unused_dirs
    
    def print_dependency_tree(self, tree, indent=0, max_depth=None):
        """Print the dependency tree in a visual format."""
        if max_depth is not None and indent > max_depth:
            return
        
        # Calculate file size in KB
        size_kb = tree.get("size", 0) / 1024
        
        # Print the current node
        prefix = "│   " * (indent - 1) + "├── " if indent > 0 else ""
        size_info = f" ({size_kb:.1f} KB)" if size_kb > 0 else ""
        print(f"{prefix}{tree['name']}{size_info}")
        
        # Print children
        for i, child in enumerate(tree["children"]):
            self.print_dependency_tree(child, indent + 1, max_depth)
    
    def print_unused_resources(self):
        """Print a list of unused files and directories."""
        unused_files_by_dir = self.find_unused_files()
        unused_dirs = self.find_unused_directories()
        
        total_unused_files = sum(len(files) for files in unused_files_by_dir.values())
        total_size = sum(self.file_sizes.get(f, 0) for files in unused_files_by_dir.values() for f in files)
        
        print("\n=== UNUSED RESOURCES ===")
        
        # Print unused directories
        if unused_dirs:
            print(f"\nUnused Directories ({len(unused_dirs)}):")
            for directory in sorted(unused_dirs):
                rel_path = os.path.relpath(directory, self.root_dir)
                print(f"  {rel_path}")
        
        # Print unused files by directory
        print(f"\nUnused Files ({total_unused_files}, {total_size/1024:.1f} KB total):")
        
        for directory, files in sorted(unused_files_by_dir.items()):
            rel_dir = os.path.relpath(directory, self.root_dir)
            print(f"\n  {rel_dir}/")
            
            for file in sorted(files):
                file_name = os.path.basename(file)
                size_kb = self.file_sizes.get(file, 0) / 1024
                print(f"    {file_name} ({size_kb:.1f} KB)")
    
    def move_unused_files(self, output_dir):
        """Move unused files to specified directory."""
        unused_files_by_dir = self.find_unused_files()
        moved_files = []
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten the list of unused files
        all_unused_files = [file for files in unused_files_by_dir.values() for file in files]
        
        for file in all_unused_files:
            # Create relative path structure in unused_scripts
            rel_path = os.path.relpath(file, self.root_dir)
            target_path = os.path.join(output_dir, rel_path)
            
            # Create parent directories
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Move the file
            try:
                shutil.move(file, target_path)
                moved_files.append((file, target_path))
                print(f"Moved: {rel_path} -> {target_path}")
            except Exception as e:
                print(f"Error moving file {file}: {e}")
        
        return moved_files
    
    def count_unique_dependencies(self, tree, visited=None):
        """Count the number of unique dependencies in the tree."""
        if visited is None:
            visited = set()
        
        # Add current file
        visited.add(tree["full_path"])
        
        # Process children
        for child in tree["children"]:
            if child["full_path"] not in visited:
                visited.add(child["full_path"])
                self.count_unique_dependencies(child, visited)
        
        return len(visited)
    
    def calculate_tree_stats(self, tree):
        """Calculate statistics about the dependency tree."""
        stats = {
            "total_size": 0,
            "max_depth": 0,
            "unique_dependencies": 0
        }
        
        def process_node(node, depth=0):
            stats["total_size"] += node.get("size", 0)
            stats["max_depth"] = max(stats["max_depth"], depth)
            
            for child in node["children"]:
                process_node(child, depth + 1)
        
        process_node(tree)
        stats["unique_dependencies"] = self.count_unique_dependencies(tree) - 1  # Exclude root
        
        return stats
    
    def find_standard_and_third_party_imports(self):
        """Find standard library and third-party package imports in app.py."""
        imports = self.extract_imports(self.app_file_path)
        
        standard_libs = []
        third_party = []
        
        for import_name in imports:
            base_module = import_name.split('.')[0]
            
            # Try to determine if it's a standard library
            if base_module in sys.modules:
                try:
                    module_path = sys.modules[base_module].__file__
                    if module_path and 'site-packages' not in module_path:
                        standard_libs.append(import_name)
                    else:
                        third_party.append(import_name)
                except AttributeError:
                    # Built-in modules don't have a __file__ attribute
                    standard_libs.append(import_name)
            # Check if it's a third-party package
            elif importlib.util.find_spec(base_module) is not None:
                third_party.append(import_name)
        
        return standard_libs, third_party


def main():
    parser = argparse.ArgumentParser(description='Analyze app.py dependencies recursively')
    parser.add_argument('app_file', nargs='?', default='app.py',
                        help='Path to app.py or other Python file to analyze (default: app.py)')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum depth to display in the dependency tree')
    parser.add_argument('--move', action='store_true',
                        help='Move unused files to unused_scripts directory')
    
    args = parser.parse_args()
    app_file = args.app_file
    
    # Validate input file
    if not os.path.exists(app_file):
        print(f"Error: File {app_file} does not exist")
        return 1
    
    analyzer = DependencyTreeAnalyzer(app_file)
    print(f"Analyzing dependencies for: {os.path.basename(app_file)}")
    
    # Build and print dependency tree
    tree = analyzer.analyze_recursive_dependencies()
    
    # Calculate and print statistics
    stats = analyzer.calculate_tree_stats(tree)
    print(f"\nDependency Statistics:")
    print(f"- Total unique dependencies: {stats['unique_dependencies']}")
    print(f"- Maximum dependency depth: {stats['max_depth']}")
    print(f"- Total size of all dependencies: {stats['total_size']/1024:.1f} KB")
    
    # Print standard and third-party imports
    std_libs, third_party = analyzer.find_standard_and_third_party_imports()
    print(f"\nStandard Library Imports ({len(std_libs)}):")
    for lib in sorted(std_libs):
        print(f"- {lib}")
    
    print(f"\nThird-Party Package Imports ({len(third_party)}):")
    for pkg in sorted(third_party):
        print(f"- {pkg}")
    
    # Print the dependency tree
    print(f"\nDependency Tree for {os.path.basename(app_file)}:")
    analyzer.print_dependency_tree(tree, max_depth=args.max_depth)
    
    # Print unused resources
    analyzer.print_unused_resources()
    
    # Move unused files if requested
    if args.move:
        print("\nMoving unused files to 'unused_scripts' directory...")
        unused_scripts_dir = os.path.join(os.path.dirname(app_file), 'unused_scripts')
        moved_files = analyzer.move_unused_files(unused_scripts_dir)
        print(f"Moved {len(moved_files)} unused files to {unused_scripts_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())