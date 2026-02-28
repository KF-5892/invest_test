#!/usr/bin/env python3
"""
Script to automatically generate source_code.json for the Vercel frontend.
Extracts source code from all alpha strategies in the alpha directory.
"""

import os
import json
import ast
import inspect
import sys
from typing import Dict, List, Any

# Add alpha module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_class_source(file_path: str, class_name: str) -> str:
    """Extract source code for a specific class from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Find the class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Get the source lines
                lines = content.split('\n')
                start_line = node.lineno - 1
                
                # Find the end of the class
                end_line = len(lines)
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    end_line = node.end_lineno
                else:
                    # Fallback: find next class or end of file
                    for next_node in ast.walk(tree):
                        if (isinstance(next_node, ast.ClassDef) and 
                            next_node.name != class_name and 
                            next_node.lineno > node.lineno):
                            end_line = next_node.lineno - 1
                            break
                
                # Extract class source
                class_lines = lines[start_line:end_line]
                return '\n'.join(class_lines)
        
        return f"# Class {class_name} not found in {file_path}"
    
    except Exception as e:
        return f"# Error reading {class_name} from {file_path}: {str(e)}"

def get_classes_from_file(file_path: str) -> List[str]:
    """Extract all class names from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Find all class definitions
        class_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
        
        return class_names
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def get_all_alpha_classes() -> Dict[str, Dict[str, str]]:
    """Get all alpha classes and their source code."""
    alpha_dir = 'alpha'
    source_code_map = {}
    
    # Get all Python files in the alpha directory
    if not os.path.exists(alpha_dir):
        print(f"Warning: {alpha_dir} directory not found!")
        return source_code_map
    
    python_files = [f for f in os.listdir(alpha_dir) if f.endswith('.py') and not f.startswith('__')]
    
    # Process each file
    for file_name in python_files:
        file_path = os.path.join(alpha_dir, file_name)
        
        print(f"Processing {file_name}...")
        
        # Get all class names from this file
        class_names = get_classes_from_file(file_path)
        
        if not class_names:
            print(f"  - No classes found in {file_name}")
            continue
        
        for class_name in class_names:
            try:
                source_code = extract_class_source(file_path, class_name)
                
                # Create entry for this class
                source_code_map[class_name] = {
                    'file': file_name,
                    'class_name': class_name,
                    'source_code': source_code,
                    'language': 'python'
                }
                
                print(f"  - Extracted {class_name}")
                
            except Exception as e:
                print(f"  - Error extracting {class_name}: {e}")
                source_code_map[class_name] = {
                    'file': file_name,
                    'class_name': class_name,
                    'source_code': f"# Error extracting source code: {str(e)}",
                    'language': 'python'
                }
    
    return source_code_map

def generate_source_code_json():
    """Generate the source_code.json file for the Vercel frontend."""
    print("Generating source_code.json for Vercel frontend...")
    
    # Get all alpha source code
    source_code_map = get_all_alpha_classes()
    
    # Create output directory if it doesn't exist
    output_dir = 'vercel-frontend/public/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, 'source_code.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(source_code_map, f, indent=2, ensure_ascii=False)
    
    print(f"Source code JSON saved to {output_file}")
    print(f"Total classes extracted: {len(source_code_map)}")
    
    # Print summary by file
    file_counts = {}
    for class_info in source_code_map.values():
        file_name = class_info['file']
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
    
    print("\nSummary by file:")
    for file_name, count in sorted(file_counts.items()):
        print(f"  {file_name}: {count} classes")
    
    return source_code_map

def main():
    """Main function."""
    try:
        source_code_map = generate_source_code_json()
        
        # Validate the output
        if not source_code_map:
            print("Warning: No source code extracted!")
            return False
        
        # Check for errors
        error_count = 0
        for class_name, info in source_code_map.items():
            if info['source_code'].startswith('# Error') or info['source_code'].startswith('# Class'):
                error_count += 1
                print(f"Warning: Issue with {class_name}: {info['source_code'][:100]}...")
        
        if error_count > 0:
            print(f"\nWarning: {error_count} classes had extraction issues")
        else:
            print("\nAll classes extracted successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error generating source code JSON: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)