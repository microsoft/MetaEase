#!/usr/bin/env python3
"""
Automatically generate experiment tracking files from logs directory.

This script scans a logs directory and automatically creates CSV files
that can be used with plot_methods.py, eliminating the need to manually
create tracking files.

Usage examples:
    # Generate experiment file for TE experiments
    python generate_experiment_file.py --logs-dir ../logs_final  
    
    # Generate experiment file for VBP experiments with custom output name
    python generate_experiment_file.py --logs-dir ../logs_final_vbp_ffd --output-name vbp_ffd

The script automatically:
- Parses folder names following the pattern: {timestamp}__{method}__{problem_type}__{params}
- Groups experiments by problem type and heuristic name
- Extracts appropriate identifiers (topology for TE, num_items for knapsack/vbp, etc.)
- Keeps all experiments (including duplicates) since they are processed individually
- Writes CSV files compatible with plot_methods.py
"""

import os
import re
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_folder_name(folder_name: str) -> Optional[Dict[str, str]]:
    """
    Parse a folder name following the pattern:
    {timestamp}__{method}__{problem_type}__{parameters}

    Returns a dict with method, problem_type, and parsed parameters.
    """
    # Pattern: timestamp__method__problem_type__param1-value1_param2-value2...
    pattern = r"(\d{8}_\d{6})__(.+?)__(.+?)__(.+)"
    match = re.match(pattern, folder_name)

    if not match:
        return None

    timestamp, method, problem_type, params_str = match.groups()

    # Parse parameters (format: KEY-value_KEY-value...)
    params = {}
    param_pattern = r"([A-Z]+(?:[A-Z][a-z]*)*)-([^_]+)"
    for param_match in re.finditer(param_pattern, params_str):
        key, value = param_match.groups()
        params[key] = value

    return {
        "timestamp": timestamp,
        "method": method,
        "problem_type": problem_type,
        "params": params,
        "folder_name": folder_name,
    }


def extract_identifier(problem_type: str, params: Dict[str, str]) -> str:
    """
    Extract the appropriate identifier for grouping experiments.

    For different problem types:
    - TE: topology (T-{value})
    - vbp: num_items (NI-{value}) or num_dimensions_num_items
    - knapsack: num_items (NI-{value})
    - mwm: topology (T-{value})
    - arrow: topology_path (but simplified)
    - tsp: num_cities
    """
    problem_type_lower = problem_type.lower()

    if problem_type_lower == "te":
        # Extract topology (T-{value})
        return params.get("T", "unknown")

    elif problem_type_lower == "vbp":
        # Use num_items, or num_dimensions_num_items if num_dimensions > 1
        num_items = params.get("NI", "unknown")
        num_dimensions = params.get("ND", None)
        if num_dimensions and num_dimensions != "unknown" and num_dimensions != "1":
            # Only include dimensions if it's not 1 (multi-dimensional case)
            return f"{num_dimensions}_{num_items}"
        return num_items

    elif problem_type_lower == "knapsack":
        # Extract num_items (NI-{value})
        return params.get("NI", "unknown")

    elif problem_type_lower == "mwm":
        # Extract topology (T-{value})
        return params.get("T", "unknown")

    elif problem_type_lower == "arrow":
        # Extract topology_path (TP-{value}) or use a default
        return params.get("TP", params.get("T", "unknown"))

    elif problem_type_lower == "tsp":
        # Extract num_cities (NC-{value} or similar)
        return params.get("NC", params.get("NI", "unknown"))

    else:
        # Default: try to find num_items or topology
        return params.get("NI", params.get("T", "unknown"))


def scan_directory(logs_dir: str) -> Dict[str, List[Dict]]:
    """
    Scan a logs directory and group experiments by problem type and identifier.

    Returns a dict: {problem_key: [experiments]}
    where problem_key is like "TE_DemandPinning" or "vbp_FFD"
    """
    experiments = defaultdict(list)

    # Convert to absolute path
    logs_dir = os.path.abspath(logs_dir)

    if not os.path.isdir(logs_dir):
        print(f"Error: {logs_dir} is not a directory")
        return experiments

    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)

        # Skip non-directories
        if not os.path.isdir(item_path):
            continue

        # Parse folder name
        parsed = parse_folder_name(item)
        if not parsed:
            print(f"Warning: Could not parse folder name: {item}")
            continue

        # Extract identifier
        identifier = extract_identifier(parsed["problem_type"], parsed["params"])

        # Create problem key (problem_type_heuristic_name)
        heuristic_name = parsed["params"].get("HN", "unknown")
        problem_key = f"{parsed['problem_type']}_{heuristic_name}"

        # Store experiment info
        experiment_info = {
            "method": parsed["method"],
            "identifier": identifier,
            "path": os.path.abspath(item_path),  # Use absolute path
            "timestamp": parsed["timestamp"],
            "problem_type": parsed["problem_type"],
            "heuristic_name": heuristic_name,
            "all_params": parsed["params"],
        }

        experiments[problem_key].append(experiment_info)

    return experiments


def write_experiment_file(
    experiments: List[Dict],
    output_path: str,
    problem_type: str,
    identifier_name: str = None,
):
    """
    Write experiments to a CSV file.

    Args:
        experiments: List of experiment dicts
        output_path: Path to output CSV file
        problem_type: Type of problem (for column naming)
        identifier_name: Name for the identifier column (e.g., "Topology", "NumItems")
    """
    if not experiments:
        print(f"Warning: No experiments to write for {output_path}")
        return

    # Determine identifier column name based on problem type
    if identifier_name is None:
        problem_type_lower = problem_type.lower()
        if problem_type_lower == "te" or problem_type_lower == "mwm":
            identifier_name = "Topology"
        elif problem_type_lower == "vbp" or problem_type_lower == "knapsack":
            identifier_name = "NumItems"
        elif problem_type_lower == "tsp":
            identifier_name = "NumCities"
        else:
            identifier_name = "Identifier"

    # Sort experiments by method (consistent order) and then by timestamp
    method_order = [
        "MetaEase",
        "SampleBasedGradient",
        "HillClimbing",
        "SimulatedAnnealing",
        "Random",
    ]

    def sort_key(exp):
        method_idx = (
            method_order.index(exp["method"]) if exp["method"] in method_order else 999
        )
        return (method_idx, exp["timestamp"])

    experiments_sorted = sorted(experiments, key=sort_key)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", identifier_name, "Path"])

        for exp in experiments_sorted:
            writer.writerow([exp["method"], exp["identifier"], exp["path"]])

    print(f"Created {output_path} with {len(experiments_sorted)} experiments")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate experiment tracking files from logs directory"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        required=True,
        help="Directory containing experiment results (e.g., logs_final, logs_final_vbp_ffd)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (without .txt extension). If not provided, auto-generates from problem type and heuristic name",
    )

    args = parser.parse_args()

    # Scan directory
    print(f"Scanning {args.logs_dir}...")
    experiments_by_problem = scan_directory(args.logs_dir)

    if not experiments_by_problem:
        print("No experiments found!")
        return

    # Write a file for each problem type
    for problem_key, experiments in experiments_by_problem.items():
        # Determine output filename
        if args.output_name:
            output_filename = f"{args.output_name}.txt"
        else:
            output_filename = f"{problem_key}.txt"
        output_path = output_filename

        # Get problem type for identifier naming
        problem_type = experiments[0]["problem_type"]

        # Write file
        write_experiment_file(experiments, output_path, problem_type)

    print(
        f"\nGenerated {len(experiments_by_problem)} experiment file(s) at {output_path}"
    )


if __name__ == "__main__":
    main()
