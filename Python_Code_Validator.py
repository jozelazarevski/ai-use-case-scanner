#!/usr/bin/env python3
"""
Python Build Error Checker

This script checks all Python files in a directory (and its subdirectories)
for errors that would break a build or prevent execution:
- Syntax errors
- Critical import errors
- Name errors in global scope

It ignores style warnings, minor issues, and anything that wouldn't
prevent the code from running.

Usage:
    python build_error_checker.py [directory_to_scan] [--output FILE]

Options:
    directory_to_scan    Directory to scan for Python files (default: current directory)
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
from importlib.machinery import SourceFileLoader
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


class BuildError:
    """Class to store information about a build-breaking error in a Python file."""
    def __init__(self, file_path, error_type, error_message, line_number=None, context=None):
        self.file_path = file_path
        self.error_type = error_type
        self.error_message = error_message
        self.line_number = line_number
        self.context = context
        self.timestamp = time.time()


def check_syntax(file_path):
    """Check if a Python file has syntax errors that would break a build."""
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
        
        return BuildError(file_path, "SyntaxError", error_message, line_number, context)
    except Exception as e:
        # Handle other reading or parsing errors
        return BuildError(file_path, "ParseError", str(e))


def check_critical_imports(file_path):
    """
    Check for import errors that would break a build.
    Focuses only on imports that would prevent the script from running.
    """
    errors = []
    
    try:
        # Get the directory of the file to add it to sys.path temporarily
        file_dir = os.path.dirname(os.path.abspath(file_path))
        original_sys_path = sys.path.copy()
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)
            
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Check top-level imports that would break the build
            if isinstance(node, ast.Import) and node.col_offset == 0:
                for name in node.names:
                    try:
                        # Skip relative imports that AST can't verify
                        if name.name.startswith('.'):
                            continue
                        
                        # Skip multi-part imports that may be resolved within the project
                        if '.' in name.name:
                            continue
                            
                        # If this doesn't throw an exception, the import exists
                        importlib.util.find_spec(name.name)
                    except (ImportError, ModuleNotFoundError) as e:
                        # Only report missing modules that would break the build
                        module_name = name.name
                        
                        # Check if the module exists in the local project structure
                        is_local_module = False
                        for search_path in sys.path:
                            module_path = os.path.join(search_path, module_name + '.py')
                            package_path = os.path.join(search_path, module_name, '__init__.py')
                            if os.path.isfile(module_path) or os.path.isfile(package_path):
                                is_local_module = True
                                break
                                
                        if not is_local_module:
                            errors.append(BuildError(
                                file_path, 
                                "ImportError", 
                                f"Missing required module '{module_name}'", 
                                node.lineno
                            ))
                            
            # Check top-level from-imports that would break the build
            elif isinstance(node, ast.ImportFrom) and node.col_offset == 0:
                if node.module and not node.module.startswith('.'):
                    try:
                        # If this doesn't throw an exception, the import exists
                        importlib.util.find_spec(node.module)
                    except (ImportError, ModuleNotFoundError) as e:
                        # Check if the module exists in the local project structure
                        module_name = node.module
                        is_local_module = False
                        for search_path in sys.path:
                            module_path = os.path.join(search_path, module_name.replace('.', os.sep) + '.py')
                            package_path = os.path.join(search_path, module_name.replace('.', os.sep), '__init__.py')
                            if os.path.isfile(module_path) or os.path.isfile(package_path):
                                is_local_module = True
                                break
                                
                        if not is_local_module:
                            errors.append(BuildError(
                                file_path, 
                                "ImportError", 
                                f"Missing required module '{module_name}'", 
                                node.lineno
                            ))
                  
        # Restore original sys.path
        sys.path = original_sys_path
    
    except SyntaxError:
        # Skip import checking if there's a syntax error
        pass
    except Exception as e:
        # Only report critical errors
        errors.append(BuildError(file_path, "CriticalError", f"Error checking imports: {str(e)}"))
    
    return errors


def check_execution(file_path):
    """Try to execute the file in isolation to check for runtime errors."""
    errors = []
    
    # Skip checking execution for specific modules known to require special environments
    skip_patterns = [
        'setup.py',
        'test_',
        '__init__.py',
        'conftest.py'
    ]
    
    if any(pattern in os.path.basename(file_path) for pattern in skip_patterns):
        return errors
        
    try:
        # Create a new module name based on the file path
        module_name = f"_checker_module_{abs(hash(file_path))}"
        
        # Try to compile the file to detect early execution errors
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Compile the code object
        try:
            compiled = compile(content, file_path, 'exec')
            
            # This is a simple check for name errors at the module level
            # It doesn't catch all runtime errors, but finds the most common ones
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.col_offset == 0:
                    # This is a top-level name being referenced
                    name = node.id
                    if (name not in __builtins__ and 
                        name not in globals() and
                        name not in ['__name__', '__file__', '__doc__']):
                        errors.append(BuildError(
                            file_path,
                            "NameError",
                            f"Name '{name}' is not defined at module level",
                            node.lineno
                        ))
                        
        except SyntaxError:
            # Already reported by syntax check
            pass
            
    except Exception as e:
        # If there's an error preparing the execution check, it's likely critical
        errors.append(BuildError(
            file_path,
            "ExecutionError",
            f"Error preparing execution check: {str(e)}"
        ))
    
    return errors


def process_file(file_path):
    """Process a single Python file and return any build-breaking errors found."""
    errors = []
    
    # Check syntax first
    syntax_error = check_syntax(file_path)
    if syntax_error:
        errors.append(syntax_error)
        # If there's a syntax error, don't check for other issues
        return errors
    
    # Check critical imports
    import_errors = check_critical_imports(file_path)
    errors.extend(import_errors)
    
    # Check execution (if there are no import errors)
    if not import_errors:
        execution_errors = check_execution(file_path)
        errors.extend(execution_errors)
    
    return errors


def find_all_python_files(directory):
    """Find all Python files in the specified directory and its subdirectories."""
    python_files = []
    
    for root, _, files in os.walk(directory):
        # Skip virtual environments, hidden directories, and other non-source directories
        if any(part.startswith('.') for part in Path(root).parts) or \
           any(part in ['venv', '__pycache__', 'env', '.env', '.venv', 'build', 'dist'] 
               for part in Path(root).parts):
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
                error_info = BuildError(
                    file_path,
                    "ProcessingError",
                    f"Error processing file: {str(e)}"
                )
                all_errors.append(error_info)
    
    return all_errors


def generate_error_report(errors):
    """Generate a formatted error report."""
    if not errors:
        return "No build-breaking errors found! All Python files should execute without problems."
    
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
    
    report = [f"Found {len(errors)} build-breaking errors in {len(errors_by_file)} files\n"]
    
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
            
            # Always show context for build-breaking errors
            if error.context:
                context_lines = error.context.split('\n')
                for i, ctx_line in enumerate(context_lines):
                    if i == len(context_lines) - 1 and not ctx_line:
                        continue  # Skip empty last line
                    report.append(f"    | {ctx_line}")
                report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Check Python files for build-breaking errors')
    parser.add_argument('directory', nargs='?', default=os.getcwd(),
                        help='Directory to scan for Python files (default: current directory)')
    parser.add_argument('--output', '-o', type=str,
                        help='Write report to this file instead of stdout')
    
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
    report = generate_error_report(all_errors)
    
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