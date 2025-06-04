#!/usr/bin/env python3
"""
Script to inspect attention allocation ratio results from attention_analysis_sft.json
and display them in a formatted table.
"""

import json
import sys
from pathlib import Path

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Warning: tabulate library not available. Install with: pip install tabulate")
    print("Using simple table format instead.\n")

def load_attention_data(json_path):
    """Load attention analysis data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}")
        sys.exit(1)

def extract_table_data(data):
    """Extract data for the table format."""
    table_data = []
    
    # For clean data, we can use any noise type since they're identical
    # Let's use 'inf' as the reference
    inf_data = data['inf']
    
    # Extract clean data (same for all noise types)
    llama_clean = inf_data['llama']['clean']
    diffllama_clean = inf_data['diffllama']['clean']
    
    # Add clean rows
    table_data.append([
        'llama-clean',
        f"{llama_clean['kmi_mean']:.4f}",
        f"{llama_clean['kmi_std']:.4f}",
        f"{llama_clean['ni_mean']:.4f}",
        f"{llama_clean['ni_std']:.4f}",
        f"{llama_clean['oc_mean']:.4f}",
        f"{llama_clean['oc_std']:.4f}"
    ])
    
    table_data.append([
        'diffllama-clean',
        f"{diffllama_clean['kmi_mean']:.4f}",
        f"{diffllama_clean['kmi_std']:.4f}",
        f"{diffllama_clean['ni_mean']:.4f}",
        f"{diffllama_clean['ni_std']:.4f}",
        f"{diffllama_clean['oc_mean']:.4f}",
        f"{diffllama_clean['oc_std']:.4f}"
    ])
    
    # Add noisy data for each noise type
    noise_types = ['inf', 'rcs', 'sd']
    for noise_type in noise_types:
        noise_data = data[noise_type]
        
        # llama noisy
        llama_noisy = noise_data['llama']['noisy']
        table_data.append([
            f'llama-{noise_type}',
            f"{llama_noisy['kmi_mean']:.4f}",
            f"{llama_noisy['kmi_std']:.4f}",
            f"{llama_noisy['ni_mean']:.4f}",
            f"{llama_noisy['ni_std']:.4f}",
            f"{llama_noisy['oc_mean']:.4f}",
            f"{llama_noisy['oc_std']:.4f}"
        ])
        
        # diffllama noisy
        diffllama_noisy = noise_data['diffllama']['noisy']
        table_data.append([
            f'diffllama-{noise_type}',
            f"{diffllama_noisy['kmi_mean']:.4f}",
            f"{diffllama_noisy['kmi_std']:.4f}",
            f"{diffllama_noisy['ni_mean']:.4f}",
            f"{diffllama_noisy['ni_std']:.4f}",
            f"{diffllama_noisy['oc_mean']:.4f}",
            f"{diffllama_noisy['oc_std']:.4f}"
        ])
    
    return table_data

def print_simple_table(table_data, headers):
    """Print a simple table without tabulate."""
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in table_data:
            max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)
    
    # Print header
    header_row = "|".join([f" {header:<{col_widths[i]-1}}" for i, header in enumerate(headers)])
    print(f"|{header_row}|")
    
    # Print separator
    separator = "|".join(["-" * col_widths[i] for i in range(len(headers))])
    print(f"|{separator}|")
    
    # Print data rows
    for row in table_data:
        data_row = "|".join([f" {str(row[i]):<{col_widths[i]-1}}" for i in range(len(row))])
        print(f"|{data_row}|")

def main():
    """Main function to run the script."""
    # Path to the JSON file
    json_path = Path('results/attention_analysis_sft.json')
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {json_path}...")
    data = load_attention_data(json_path)
    
    # Extract table data
    table_data = extract_table_data(data)
    
    # Define headers
    headers = ['Models', 'kmi_mean', 'kmi_std', 'ni_mean', 'ni_std', 'oc_mean', 'oc_std']
    
    # Print table
    print("\nAttention Allocation Ratio Results:")
    print("=" * 80)
    
    if TABULATE_AVAILABLE:
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        print_simple_table(table_data, headers)
    
    print("=" * 80)

if __name__ == "__main__":
    main() 