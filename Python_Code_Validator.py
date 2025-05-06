#!/usr/bin/env python3
"""
Python Code Validator

This script checks all Python files in a directory (and its subdirectories) for syntax errors,
import errors, and other common issues. It provides a detailed report of problems found.

Usage:
    python code_validator.py [directory_to_scan] [--verbose]

Options:
    directory_to_scan    Directory to scan for Python files (default: current directory)
    --verbose            Show more detailed output about each file
    --fix                Attempt to fix common issues automatically
    --output FILE        Write report to file instead of console
"""

import os
import sys
import ast
import importlib
import traceback
import argparse
import time
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


class ErrorInfo:
    """Class to store information about an error in a Python file."""
    def __init__(self, file_path, error_type, error_message, line_number=None, context=None):
        self.file_path = file_path
        self.error_type = error_type
        self.error_message = error_message
        self.line_number = line_number
        self.context = context
        self.timestamp = time.time()


def check_syntax(file_path):
    """Check if a Python file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Try to parse the file
        ast.parse(content)
        return None
    except SyntaxError as e:
        # Extract line number and error message
        line_number = e.lineno
        error_message = str(e)
        
        # Try to get context (the problematic line and surrounding lines)
        context = None
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                start = max(0, line_number - 3)
                end = min(len(lines), line_number + 2)
                context = ''.join(lines[start:end])
        except Exception:
            pass
        
        return ErrorInfo(file_path, "SyntaxError", error_message, line_number, context)
    except Exception as e:
        # Handle other reading or parsing errors
        return ErrorInfo(file_path, "ParseError", str(e))


def check_imports(file_path):
    """Check if imports in a Python file can be resolved."""
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse the file to get imports
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Check regular imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_name = name.name
                    line_number = node.lineno
                    
                    # Skip relative imports (starting with .)
                    if module_name.startswith('.'):
                        continue
                    
                    # Skip standard library and installed packages
                    if module_name in sys.modules or importlib.util.find_spec(module_name.split('.')[0]) is not None:
                        continue
                    
                    # Try to find the module in the same directory
                    module_found = False
                    module_path = os.path.join(os.path.dirname(file_path), module_name.replace('.', os.sep) + '.py')
                    if os.path.exists(module_path):
                        module_found = True
                    
                    # Check for package with __init__.py
                    package_path = os.path.join(os.path.dirname(file_path), module_name.replace('.', os.sep))
                    if os.path.exists(os.path.join(package_path, '__init__.py')):
                        module_found = True
                    
                    if not module_found:
                        errors.append(ErrorInfo(
                            file_path, 
                            "ImportError", 
                            f"Cannot find module '{module_name}'", 
                            line_number
                        ))
            
            # Check from ... import ...
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:  # Skip imports like "from . import x"
                    module_name = node.module
                    line_number = node.lineno
                    
                    # Skip relative imports for simplicity
                    if module_name.startswith('.'):
                        continue
                    
                    # Skip standard library and installed packages
                    if module_name in sys.modules or importlib.util.find_spec(module_name.split('.')[0]) is not None:
                        continue
                    
                    # Try to find the module in the same directory
                    module_found = False
                    module_path = os.path.join(os.path.dirname(file_path), module_name.replace('.', os.sep) + '.py')
                    if os.path.exists(module_path):
                        module_found = True
                    
                    # Check for package with __init__.py
                    package_path = os.path.join(os.path.dirname(file_path), module_name.replace('.', os.sep))
                    if os.path.exists(os.path.join(package_path, '__init__.py')):
                        module_found = True
                    
                    if not module_found:
                        errors.append(ErrorInfo(
                            file_path, 
                            "ImportError", 
                            f"Cannot find module '{module_name}'", 
                            line_number
                        ))
    
    except SyntaxError:
        # Skip import checking if there's a syntax error
        pass
    except Exception as e:
        errors.append(ErrorInfo(file_path, "Error", f"Error checking imports: {str(e)}"))
    
    return errors


def check_common_issues(file_path):
    """Check for common issues and best practices violations."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            lines = content.split('\n')
        
        # Check for very long lines
        for i, line in enumerate(lines):
            if len(line) > 120:
                issues.append(ErrorInfo(
                    file_path,
                    "StyleWarning",
                    f"Line too long ({len(line)} > 120 characters)",
                    i + 1,
                    line
                ))
        
        # Check for TODO comments without assignee or ticket number
        for i, line in enumerate(lines):
            if 'TODO' in line and not any(word in line for word in ['ticket', '#', 'issue', '@']):
                issues.append(ErrorInfo(
                    file_path,
                    "TodoWarning",
                    "TODO comment without assignee or ticket reference",
                    i + 1,
                    line
                ))
        
        # Check for unused imports (simple version)
        try:
            tree = ast.parse(content)
            imported_names = set()
            used_names = set()
            
            for node in ast.walk(tree):
                # Collect imported names
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imported_names.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        if name.name != '*':
                            if node.module:
                                imported_name = f"{name.name}"
                                imported_names.add(imported_name)
                
                # Collect name usage
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            # Check for unused imports
            for name in imported_names:
                base_name = name.split('.')[0]
                if base_name not in used_names:
                    issues.append(ErrorInfo(
                        file_path,
                        "UnusedImport",
                        f"Potentially unused import: '{name}'",
                        None,
                        None
                    ))
        except:
            # Skip AST-based checks if parsing fails
            pass
        
    except Exception as e:
        issues.append(ErrorInfo(file_path, "Error", f"Error checking common issues: {str(e)}"))
    
    return issues


def process_file(file_path):
    """Process a single Python file and return any errors found."""
    errors = []
    
    # Check syntax first
    syntax_error = check_syntax(file_path)
    if syntax_error:
        errors.append(syntax_error)
        # If there's a syntax error, don't check for other issues
        return errors
    
    # Check imports
    import_errors = check_imports(file_path)
    errors.extend(import_errors)
    
    # Check for common issues
    common_issues = check_common_issues(file_path)
    errors.extend(common_issues)
    
    return errors


def find_all_python_files(directory):
    """Find all Python files in the specified directory and its subdirectories."""
    python_files = []
    
    for root, _, files in os.walk(directory):
        # Skip virtual environments and hidden directories
        if any(part.startswith('.') for part in Path(root).parts) or \
           any(part in ['venv', '__pycache__', 'env', '.env', '.venv'] for part in Path(root).parts):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def process_files_parallel(files, max_workers=None):
    """Process files in parallel and collect errors."""
    all_errors = []
    
    # Determine a reasonable number of workers based on CPU count
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(process_file, file): file for file in files}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                errors = future.result()
                if errors:
                    all_errors.extend(errors)
            except Exception as e:
                # Handle any exceptions that occurred during processing
                error_info = ErrorInfo(
                    file_path,
                    "ProcessingError",
                    f"Error processing file: {str(e)}"
                )
                all_errors.append(error_info)
    
    return all_errors


def generate_error_report(errors, verbose=False):
    """Generate a formatted error report."""
    if not errors:
        return "No errors found! All Python files are valid."
    
    # Group errors by file
    errors_by_file = {}
    for error in errors:
        if error.file_path not in errors_by_file:
            errors_by_file[error.file_path] = []
        errors_by_file[error.file_path].append(error)
    
    # Sort files by number of errors (most errors first)
    sorted_files = sorted(
        errors_by_file.keys(),
        key=lambda f: len(errors_by_file[f]),
        reverse=True
    )
    
    report = [f"Found {len(errors)} errors/warnings in {len(errors_by_file)} files\n"]
    
    # Summary of error types
    error_types = {}
    for error in errors:
        if error.error_type not in error_types:
            error_types[error.error_type] = 0
        error_types[error.error_type] += 1
    
    report.append("Error Type Summary:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  {error_type}: {count}")
    
    report.append("\nDetailed Error Report:")
    
    # Generate detailed report
    for file_path in sorted_files:
        rel_path = os.path.relpath(file_path)
        file_errors = errors_by_file[file_path]
        report.append(f"\n{rel_path} ({len(file_errors)} {'error' if len(file_errors) == 1 else 'errors'}):")
        
        # Sort errors by line number
        file_errors.sort(key=lambda e: e.line_number if e.line_number is not None else 0)
        
        for error in file_errors:
            line_info = f"line {error.line_number}" if error.line_number is not None else "unknown line"
            report.append(f"  [{error.error_type}] {line_info}: {error.error_message}")
            
            # Show context if available and in verbose mode
            if verbose and error.context:
                context_lines = error.context.split('\n')
                for i, ctx_line in enumerate(context_lines):
                    if i == len(context_lines) - 1 and not ctx_line:
                        continue  # Skip empty last line
                    report.append(f"    | {ctx_line}")
                report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Validate Python code files in a directory')
    parser.add_argument('directory', nargs='?', default=os.getcwd(),
                        help='Directory to scan for Python files (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show more detailed output including code context')
    parser.add_argument('--output', '-o', type=str,
                        help='Write report to this file instead of stdout')
    parser.add_argument('--fix', '-f', action='store_true',
                        help='Attempt to fix common issues automatically (not implemented yet)')
    
    args = parser.parse_args()
    directory = args.directory
    
    print(f"Scanning Python files in {directory}...")
    
    # Find all Python files
    python_files = find_all_python_files(directory)
    print(f"Found {len(python_files)} Python files to check")
    
    if not python_files:
        print("No Python files found in the specified directory")
        return 0
    
    # Process files and collect errors
    start_time = time.time()
    all_errors = process_files_parallel(python_files)
    end_time = time.time()
    
    # Generate error report
    report = generate_error_report(all_errors, args.verbose)
    
    # Add processing time and summary
    error_file_count = len(set(error.file_path for error in all_errors))
    success_rate = (len(python_files) - error_file_count) / len(python_files) * 100
    summary = (
        f"\nProcessing complete in {end_time - start_time:.2f} seconds\n"
        f"Checked {len(python_files)} files\n"
        f"Found issues in {error_file_count} files\n"
        f"Success rate: {success_rate:.1f}%\n"
    )
    
    report = report + "\n" + summary
    
    # Output the report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    
    # Return non-zero exit code if errors were found
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())