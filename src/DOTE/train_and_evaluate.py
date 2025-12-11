#!/usr/bin/env python3
"""
Simple script to train DOTE on a topology and compare DOTE flow vs optimal flow on test set.
Usage: python train_and_evaluate.py <topology_name>
"""

import sys
import subprocess
import os
import re
RETRAIN = True

def run_dote_command(args):
    """Run dote.py with given arguments and capture output"""
    cmd = ['python', '../dote.py'] + args
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None
    
    return result.stdout

def extract_served_flow(output):
    """Extract served flow value from DOTE output"""
    # Look for "Served flow value: X" in the output
    match = re.search(r'Served flow value: ([\d.]+)', output)
    if match:
        return float(match.group(1))
    return None

def extract_avg_loss(output):
    """Extract average loss from DOTE output"""
    # Look for "Avg loss: X" in the output
    match = re.search(r'Avg loss: ([\d.e+-]+)', output)
    if match:
        return float(match.group(1))
    return None

def get_optimal_flow_from_test_data(topology_name):
    """Read optimal flow values from test .opt files and sum them"""
    opt_files = []
    test_dir = f"data/{topology_name}/test/"
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return None
    
    # Find all .opt files in test directory
    for file in os.listdir(test_dir):
        if file.endswith('.opt'):
            opt_files.append(os.path.join(test_dir, file))
    
    if not opt_files:
        print(f"No .opt files found in {test_dir}")
        return None
    
    total_optimal_flow = 0.0
    total_samples = 0
    
    for opt_file in opt_files:
        with open(opt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # Use the raw optimal values as they are
                        # The optimal values should be in the same units as the DOTE flow computation
                        opt_value = float(line)
                        total_optimal_flow += opt_value
                        total_samples += 1
                    except ValueError:
                        continue
    
    print(f"Found {total_samples} optimal flow values in test set")
    return total_optimal_flow

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_and_evaluate.py <topology_name>")
        print("Example: python train_and_evaluate.py Abilene")
        sys.exit(1)
    
    topology_name = sys.argv[1]
    
    print(f"Training and evaluating DOTE on {topology_name} topology...")
    print("=" * 60)
    
    if RETRAIN:
        # Step 1: Train the model
        print("\n1. Training DOTE model...")
        train_args = [
            '--ecmp_topo', topology_name,
            '--paths_from', 'sp',
            '--so_mode', 'train',
            '--so_epochs', '500',
            '--so_batch_size', '16',
            '--opt_function', 'MAXFLOW',
            '--hist_len', '0'
        ]
        
        train_output = run_dote_command(train_args)
        if train_output is None:
            print("Training failed!")
            sys.exit(1)
    
    print("Training completed successfully!")
    
    # Step 2: Test the model
    print("\n2. Testing DOTE model...")
    test_args = [
        '--ecmp_topo', topology_name,
        '--paths_from', 'sp',
        '--so_mode', 'test',
        '--opt_function', 'MAXFLOW',
        '--hist_len', '0'
    ]
    
    test_output = run_dote_command(test_args)
    if test_output is None:
        print("Testing failed!")
        sys.exit(1)
    
    # Step 3: Extract results
    dote_flow = extract_served_flow(test_output)
    avg_loss = extract_avg_loss(test_output)
    
    if dote_flow is None:
        print("Could not extract DOTE flow from test output")
        sys.exit(1)
    
    # Step 4: Get optimal flow from test data
    print("\n3. Reading optimal flow values from test data...")
    optimal_flow = get_optimal_flow_from_test_data(topology_name)
    
    if optimal_flow is None:
        print("Could not read optimal flow values")
        sys.exit(1)
    
    # Step 5: Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON:")
    print("=" * 60)
    print(f"Topology: {topology_name}")
    print(f"DOTE Total Flow:     {dote_flow:.2f}")
    print(f"Optimal Total Flow:  {optimal_flow:.2f}")
    print(f"Flow Ratio:          {dote_flow/optimal_flow:.4f}")
    print(f"Performance:         {(dote_flow/optimal_flow)*100:.2f}% of optimal")
    if avg_loss:
        print(f"Average Loss:        {avg_loss:.2e}")
    print("=" * 60)

if __name__ == "__main__":
    main()