import os
import json
import random
import time
import sys
import os
# Add parent directory to path for utils and common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from ortools.linear_solver import pywraplp
from collections import defaultdict
from .problem import Problem
from common import LAMBDA_MAX_VALUE
from ortools.linear_solver import pywraplp
import os
import numpy as np
import subprocess
import re
import shutil
import time
import random
# TODO: many parameters are the same across problem types, should have defaults and only override what is different.
# TODO: rename get_heuristic_program to get_heuristic_C_program. and explain in a comment why the existance of these functions is necessary.
# TODO: there are a bunch of duplicate imports in the code.
# TODO: relative paths you have strewn about everywhere are a bit risky, better to fix with respect to base directory of MetaEase.
# TODO: remove all the print statements you had added for debugging/remove all the comments of that too.
# TODO: line 176-181 -- you can cache the edge to path mapping once to avoid recommputing it often.
ENABLE_PRINT = False
DISABLE_CUSTOMIZATION = False

# TODO: put a comment as to what this function is doing and what its inputs are and where it is used.
def find_all_paths(problem_config, source, destination, max_num_paths=float("inf")):
    # Build adjacency list from the graph dictionary
    adjacency_list = defaultdict(list)
    graph = {key: value for key, value in problem_config.items() if "edge_" in key}

    for edge in graph:
        _, from_node, to_node = edge.split("_")
        adjacency_list[from_node].append(to_node)

    def dfs(current, path):
        # If current node is the destination, add the path to the result
        if current == destination:
            paths.append(path[:])
            # Stop if we've reached the max_num_paths
            if max_num_paths is not None and len(paths) >= max_num_paths:
                return True  # Signal to stop
            return False

        # Explore neighbors
        for neighbor in adjacency_list[current]:
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                stop = dfs(neighbor, path)
                path.pop()  # Backtrack
                if stop:
                    return True
        return False

    paths = []
    dfs(source, [source])
    return paths


def get_min_capacity_on_path(path, edges):
    min_capacity = float("inf")
    for i in range(len(path) - 1):
        edge = f"edge_{path[i]}_{path[i + 1]}"
        min_capacity = min(min_capacity, edges[edge])
    return min_capacity

# TODO: describe the function, what it does and how it is used and where.
def find_possible_demands(problem_config):
    import concurrent.futures
    import os
    import time
    if ENABLE_PRINT:
        print(f"Finding possible demands for {problem_config['num_nodes']} nodes")
    start_time = time.time()
    possible_demands = []
    all_paths = {}
    num_nodes = problem_config["num_nodes"]
    max_num_paths = problem_config.get("max_num_paths", float("inf"))
    pairs = [(from_, to_) for from_ in range(num_nodes) for to_ in range(num_nodes) if from_ != to_]

    def process_pair(pair):
        from_, to_ = pair
        paths = find_all_paths(problem_config, str(from_), str(to_), max_num_paths)
        return (pair, paths)

    max_workers = min(32, int(os.cpu_count()* 0.8))
    # TODO: such hard coded parameters either should be part of the parameter settings for the function
    # OR they should be set somewhere in a master file. Strewn about configs make it hard to track.
    batch_size = 100  # You can adjust this batch size as needed
    results = []
    for i in range(0, len(pairs), batch_size):
        if ENABLE_PRINT:
            print(f"Processed {i+batch_size} pairs out of {len(pairs)}")
        batch = pairs[i:i+batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_pair, batch))
        results.extend(batch_results)

    for (from_, to_), paths in results:
        all_paths[(from_, to_)] = paths
        if len(paths) > 0:
            possible_demands.append((from_, to_))
    end_time = time.time()
    # print in red
    if ENABLE_PRINT:
        print(f"\033[91mTime taken: {end_time - start_time} seconds to get all paths\033[0m")
    return possible_demands, all_paths

# TODO: can use comments throughout it for what each constraint is encoding.
def optimal_TE(num_nodes, edges, demands, possible_demands=None, given_all_paths=None):
    # Create the solver
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise Exception("Solver could not be created!")
    all_vars = {}
    flow_on_path_vars = {}
    all_paths = given_all_paths if given_all_paths is not None else {}
    nodes = range(num_nodes)
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                if (from_, to_) in all_paths:
                    paths = all_paths[(from_, to_)]
                else:
                    paths = find_all_paths(
                        {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                    )
                    all_paths[(from_, to_)] = paths
                for path in paths:
                    path_string = "_".join(path)
                    var_name = f"aux_flowpath_{path_string}"
                    min_capacity_on_path = get_min_capacity_on_path(path, edges)
                    flow_on_path_vars[var_name] = solver.NumVar(
                        0, min_capacity_on_path, var_name
                    )

    # Create flow variables for
    if possible_demands is None:
        possible_demands, _ = find_possible_demands({"num_nodes": num_nodes, **edges})
    for from_, to_ in demands:
        if (from_, to_) in possible_demands:
            all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]
    flow_pair_vars = {}
    flowpath_constraints = {}
    flow_demand_constraints = {}

    for from_, to_ in possible_demands:
        if (from_, to_) in demands:
            flow_pair_vars[f"aux_flow_{from_}_{to_}"] = solver.NumVar(
                0, 2 * demands[(from_, to_)], f"aux_flow_{from_}_{to_}"
            )
            flow_demand_constraints[(from_, to_)] = solver.Add(
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] <= demands[(from_, to_)]
            )
            # the aux_flow_{from_}_{to_} is equal to the sum of the flow on all paths from from_ to to_
            paths = all_paths[(from_, to_)]
            path_names = ["_".join(path) for path in paths]
            sum_flow_on_paths = solver.Sum(
                flow_on_path_vars[f"aux_flowpath_{path_name}"]
                for path_name in path_names
            )
            constraint = solver.Add(
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] == sum_flow_on_paths
            )
            flowpath_constraints[(from_, to_)] = constraint

    capacity_constraints = {}
    # Add capacity constraints for each edge
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                edge = f"edge_{from_}_{to_}"
                if edge in edges:
                    capacity = edges[edge]
                    # find all the paths that use this edge
                    every_path = []
                    for key, value in all_paths.items():
                        every_path.extend("_".join(path) for path in value)
                    paths_with_this_edge = [
                        path for path in every_path if f"{from_}_{to_}" in path
                    ]
                    constraint = solver.Add(
                        solver.Sum(
                            flow_on_path_vars[f"aux_flowpath_{path}"]
                            for path in paths_with_this_edge
                        )
                        <= capacity
                    )
                    capacity_constraints[f"{from_}_{to_}"] = constraint

    # Objective: Maximize the total flow
    total_flow = solver.Sum(flow_pair_vars.values())
    solver.Maximize(total_flow)

    # Solve the problem
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("Solver did not find an optimal solution!")

    # Extract results
    flow_values = {
        edge: flow_pair_vars[edge].solution_value() for edge in flow_pair_vars
    }
    flowpath_values = {
        path: flow_on_path_vars[path].solution_value() for path in flow_on_path_vars
    }
    for key in flow_pair_vars:
        all_vars[key] = flow_pair_vars[key].solution_value()

    for key in flow_on_path_vars:
        all_vars[key] = flow_on_path_vars[key].solution_value()

    for key in flow_demand_constraints:
        all_vars[f"lambda_flow_demand_{key[0]}_{key[1]}"] = flow_demand_constraints[key].dual_value()

    for key in flowpath_constraints:
        all_vars[f"lambda_flowpath_{key[0]}_{key[1]}"] = flowpath_constraints[
            key
        ].dual_value()

    for key in capacity_constraints:
        all_vars[f"lambda_capacity_{key}"] = capacity_constraints[key].dual_value()

    total_flow_value = total_flow.solution_value()
    return {
        "flow_values": flow_values,
        "flowpath_values": flowpath_values,
        "optimal_total_flow": total_flow_value,
        "all_vars": all_vars,
    }

# TODO: can use a comment as to what this function is doing and how/where it should be used.
def DOTE_wrapper(topology_name, demands):
    # remove DOTE from topology_name
    topology_name = topology_name.replace("DOTE", "")
    capital_topology_name = topology_name.capitalize()
    # Handle empty demands
    if not demands:
        return {"heuristic_value": 0.0, "code_path_num": "", "all_vars": {}}

    # Get the number of nodes from the topology
    max_node = max(max(pair) for pair in demands.keys()) + 1
    num_nodes = max_node

    # Create flattened demand matrix
    demand_matrix = np.zeros(num_nodes * num_nodes)
    for (from_node, to_node), demand_value in demands.items():
        if from_node != to_node:  # Skip diagonal elements
            demand_matrix[from_node * num_nodes + to_node] = demand_value

    # Create the flattened vector without diagonal elements (as DOTE expects)
    # TODO: would be good to explain a bit more what this means.
    flattened_demands = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                flattened_demands.append("0")
            if i != j:
                flattened_demands.append(str(demand_matrix[i * num_nodes + j] * 1000000000))

    DOTE_DIR = os.path.join(os.path.dirname(__file__), "DOTE")
    temp_dir = os.path.join(DOTE_DIR, f"temp_dote_data_{topology_name}_{time.time()}_{random.randint(0, 1000000)}")

    # Create temporary test directory structure
    temp_test_dir = os.path.join(temp_dir, "test")
    os.makedirs(temp_test_dir, exist_ok=True)

    # Write the demand matrix to a temporary file in the format DOTE expects
    temp_demand_file = os.path.join(temp_test_dir, "1.hist")
    with open(temp_demand_file, 'w') as f:
        f.write(' '.join(flattened_demands) + '\n')

    # Create a temporary .opt file (DOTE expects this even for inference)
    temp_opt_file = os.path.join(temp_test_dir, "1.opt")
    with open(temp_opt_file, 'w') as f:
        f.write("0.0\n")  # Dummy optimal value

    # Create a temporary topology directory with the required files
    temp_topo_dir = os.path.join(temp_dir, capital_topology_name)
    os.makedirs(temp_topo_dir, exist_ok=True)

    # Copy the test directory to the topology directory
    shutil.copytree(temp_test_dir, os.path.join(temp_topo_dir, "test"))

    # Copy the original topology files (excluding test directory)
    original_data_dir = os.path.join(DOTE_DIR, "networking_envs", "data")
    original_topo_dir = os.path.join(original_data_dir, capital_topology_name)

    if os.path.exists(original_topo_dir):
        # Copy the original topology files
        for file in os.listdir(original_topo_dir):
            if file != "test":  # Skip the test directory, we'll use our custom one
                src = os.path.join(original_topo_dir, file)
                dst = os.path.join(temp_topo_dir, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src) and file != "test":
                    shutil.copytree(src, dst)

        # Ensure our custom test directory is used
        test_dst = os.path.join(temp_topo_dir, "test")
        if os.path.exists(test_dst):
            shutil.rmtree(test_dst)
        shutil.copytree(temp_test_dir, test_dst)
    else:
        raise FileNotFoundError(f"Topology directory {original_topo_dir} not found. Please ensure DOTE data is properly set up.")

    # Create a temporary data directory with our custom topology
    temp_data_dir = os.path.join(temp_dir, "data")
    os.makedirs(temp_data_dir, exist_ok=True)
    shutil.copytree(temp_topo_dir, os.path.join(temp_data_dir, capital_topology_name))

    # Run DOTE using the modified script with custom data directory
    networking_envs_dir = os.path.join(DOTE_DIR, "networking_envs")

    # Construct the command to run from networking_envs directory
    cmd_str = f"cd {networking_envs_dir} && python ../dote.py --ecmp_topo {capital_topology_name} --paths_from sp --so_mode test --so_batch_size 1 --opt_function MAXFLOW --hist_len 0"
    # print(f"export DOTE_CUSTOM_DATA_DIR={temp_data_dir}")
    # print(f"Running command: {cmd_str}")

    # Set environment variables to use our custom data directory
    env = os.environ.copy()
    # print(f"temp_data_dir: {temp_data_dir}")
    env['DOTE_CUSTOM_DATA_DIR'] = temp_data_dir
    env['PYTHONPATH'] = f"{networking_envs_dir}:{os.path.join(DOTE_DIR, 'openai_baselines')}:{env.get('PYTHONPATH', '')}"

    # Run the command from the networking_envs directory (as DOTE expects)
    try:
        # print(f"Running command string: {cmd_str}")
        # print(f"Working directory: {os.getcwd()}")
        # print(f"Target working directory: {networking_envs_dir}")
        # print(f"Environment DOTE_CUSTOM_DATA_DIR: {env.get('DOTE_CUSTOM_DATA_DIR')}")

        proc = subprocess.run(cmd_str, capture_output=True, text=True, env=env, shell=True, timeout=100, cwd=networking_envs_dir)
        stdout = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        print("DOTE command timed out - likely missing trained model")
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                # print(f"Cleaned up temporary directory after timeout: {temp_dir}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temporary directory {temp_dir} after timeout: {cleanup_error}")
        # Return a dummy value for testing purposes
        return {
            "heuristic_value": 0.0,  # Dummy value for testing
            "code_path_num": "",
            "all_vars": {}
        }
    except Exception as e:
        print(f"Error running command: {e}")
        # Clean up before returning
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory after error: {temp_dir}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temporary directory {temp_dir} after error: {cleanup_error}")
        return {
            "heuristic_value": 0.0,
            "code_path_num": "",
            "all_vars": {}
        }

    # Clean up temporary directories
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            # print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

    # Extract the served flow value
    m = re.search(r"Served flow value:\s*([0-9eE+\-.]+)", stdout)
    if not m:
        # If no served flow value found, try to extract from other patterns
        print(f"DOTE output: {stdout}")
        # For debugging, let's see what we got
        if "Test Error" in stdout:
            # Try to extract from test error output
            avg_loss_match = re.search(r"Avg loss:\s*([0-9eE+\-.]+)", stdout)
            if avg_loss_match:
                # If we only have loss, we can't get flow value
                print("Warning: Could not extract served flow value, only loss available")
                served = 0.0
            else:
                served = 0.0
        else:
            served = 0.0
    else:
        served = float(m.group(1))

    return {
        "heuristic_value": served,
        "code_path_num": "",
        "all_vars": {}
    }

def DOTE_C_wrapper(topology_name, demands):
    """
    Simplified C-based DOTE implementation that can work with KLEE.
    This provides a DOTE-like heuristic without requiring the full neural network.
    """
    # Handle empty demands
    if not demands:
        return {"heuristic_value": 0.0, "code_path_num": "dote_c_empty", "all_vars": {}}

    # Get the number of nodes from the topology
    max_node = max(max(pair) for pair in demands.keys()) + 1
    num_nodes = max_node

    # Simple DOTE-inspired heuristic: route demands optimally on available paths
    total_served_flow = 0.0
    
    # Create a simple capacity matrix (assuming full mesh topology)
    # TODO: what if the user inputs a topology with a given capacity?
    capacity_matrix = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                capacity_matrix[(i, j)] = 1000.0  # Default capacity
    
    # Route each demand
    for (from_node, to_node), demand_value in demands.items():
        if demand_value <= 0:
            continue
            
        # Try direct path first
        direct_capacity = capacity_matrix.get((from_node, to_node), 0.0)
        served_on_direct = min(demand_value, direct_capacity)
        total_served_flow += served_on_direct
        
        remaining_demand = demand_value - served_on_direct
        
        # If direct path insufficient, try alternative paths
        if remaining_demand > 0:
            # Try routing through intermediate nodes
            for intermediate in range(num_nodes):
                if intermediate == from_node or intermediate == to_node:
                    continue
                    
                # Check capacity on path from->intermediate->to
                cap1 = capacity_matrix.get((from_node, intermediate), 0.0)
                cap2 = capacity_matrix.get((intermediate, to_node), 0.0)
                path_capacity = min(cap1, cap2)
                
                additional_served = min(remaining_demand, path_capacity)
                total_served_flow += additional_served
                remaining_demand -= additional_served
                
                if remaining_demand <= 0:
                    break

    return {
        "heuristic_value": total_served_flow,
        "code_path_num": f"dote_c_{topology_name}",
        "all_vars": {}
    }

def LLM_TE(
    num_nodes,
    edges,
    demands,
    possible_demands=None,
    given_all_paths=None,
):
    # we sort the demands by the demand value, high to low, get the top 20% as critical demands, first route those optimally without considering non critical demands,
    # then we adjust the remaining capacity of the edges to route the non critical demands optimally
    all_paths = given_all_paths if given_all_paths is not None else {}
    flow_on_path_vars = {}
    flow_pair_vars = {}
    all_vars = {}
    remaining_demands = demands.copy()
    remaining_edges = edges.copy()
    critical_demands_threshold = 0.2
    critical_demands_processed = []

    # sort the demands by the demand value, high to low
    sorted_demands = sorted(demands.items(), key=lambda x: x[1], reverse=True)
    all_demands_keys = [key for key, _ in sorted_demands]
    critical_demands_keys = [key for key, _ in sorted_demands[:int(len(sorted_demands) * critical_demands_threshold)]]
    critical_demands = {key: demands[key] if key in critical_demands_keys else 0 for key in all_demands_keys}
    non_critical_demands = {key: demands[key] if key not in critical_demands_keys else 0 for key in all_demands_keys}

    # First, store all demand values in all_vars
    for from_, to_ in demands:
        all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]

    # Critical demands phase - route them optimally first
    if len(critical_demands_keys) > 0:
        # Solve optimization for critical demands only
        critical_optimal = optimal_TE(
            num_nodes,
            edges,
            critical_demands,
            possible_demands=possible_demands,
            given_all_paths=all_paths,
        )

        # Store the critical demand flow solutions
        critical_flow_values = critical_optimal["flow_values"]
        critical_flowpath_values = critical_optimal["flowpath_values"]

        # Update flow variables with critical demand solutions
        for key, value in critical_flow_values.items():
            flow_pair_vars[key] = value
            all_vars[key] = value

        for key, value in critical_flowpath_values.items():
            flow_on_path_vars[key] = value
            all_vars[key] = value

        # Update all_vars with other variables from critical optimal solution
        for key, value in critical_optimal["all_vars"].items():
            if key not in all_vars:
                all_vars[key] = value

        # Reduce remaining edge capacities based on critical demand flows
        # We need to look at flowpath values, not flow values, because flowpath represents actual paths
        for key, value in critical_flowpath_values.items():
            if value > 0:
                # Extract path from variable name: aux_flowpath_0_1_2 -> [0, 1, 2]
                path_parts = key.split("_")[2:]  # Remove "aux_flowpath_" prefix
                path = path_parts

                # Reduce capacity on each edge in the path
                for i in range(len(path) - 1):
                    edge_key = f"edge_{path[i]}_{path[i + 1]}"
                    if edge_key in remaining_edges:
                        remaining_edges[edge_key] = max(0, remaining_edges[edge_key] - value)

        # Remove critical demands from remaining demands
        for (from_, to_), _ in critical_demands.items():
            critical_demands_processed.append((from_, to_))

    # Now solve the optimization problem for the remaining (non-critical) demands and adjusted edges
    if len(critical_demands_keys) < len(all_demands_keys):
        partial_optimal = optimal_TE(
            num_nodes,
            remaining_edges,
            non_critical_demands,
            possible_demands=possible_demands,
            given_all_paths=all_paths,
        )

        partial_flow_values = partial_optimal["flow_values"]
        partial_flowpath_values = partial_optimal["flowpath_values"]

        # Combine results: add non-critical demand flows to existing critical flows
        for key, value in partial_flow_values.items():
            if key in flow_pair_vars:
                flow_pair_vars[key] += value
                all_vars[key] = flow_pair_vars[key]
            else:
                flow_pair_vars[key] = value
                all_vars[key] = value

        for key, value in partial_flowpath_values.items():
            if key in flow_on_path_vars:
                flow_on_path_vars[key] += value
                all_vars[key] = flow_on_path_vars[key]
            else:
                flow_on_path_vars[key] = value
                all_vars[key] = value

        # Add any remaining variables from partial optimal solution
        for key, value in partial_optimal["all_vars"].items():
            if key not in all_vars:
                all_vars[key] = value

    # Prepare final flow values for return
    flow_values = flow_pair_vars.copy()
    total_flow_value = sum(flow_values.values())

    # Generate code path number based on processed critical demands
    code_path_num = ""
    critical_demands_sorted = sorted(critical_demands_processed, key=lambda x: (x[0], x[1]))
    for from_, to_ in critical_demands_sorted:
        code_path_num += f"{from_}-{to_}_"

    return {
        "code_path_num": code_path_num,
        "flow_values": flow_values,
        "flowpath_values": flow_on_path_vars,
        "heuristic_value": total_flow_value,
        "all_vars": all_vars,
    }

def demand_pinning_TE(
    num_nodes,
    edges,
    demands,
    pinning_threshold,
    possible_demands=None,
    given_all_paths=None,
):
    # print(f"Starting demand_pinning_TE with {num_nodes} nodes, {len(demands)} demands, threshold {pinning_threshold}")

    # for each demands that is less than the pinning threshold, we pin the flow on its shortest path, for the rest we optimize
    all_paths = given_all_paths if given_all_paths is not None else {}
    flow_on_path_vars = {}
    flow_pair_vars = {}
    all_vars = {}
    remaining_demands = demands.copy()
    remaining_edges = edges.copy()
    pinned_demands = []
    non_pinned_demands = []
    # First, store all demand values in all_vars
    for from_, to_ in demands:
        all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]

    # Pinning phase
    pinning_start = time.time()
    for from_, to_ in demands:
        if demands[(from_, to_)] <= pinning_threshold:
            if (from_, to_) in all_paths:
                paths = all_paths[(from_, to_)]
            else:
                paths = find_all_paths(
                    {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                )
                all_paths[(from_, to_)] = paths
            # sort the paths by length, and then name
            # first sort by length, then sort by string to break ties
            paths.sort(key=lambda x: (len(x), "_".join(x)))
            shortest_path = paths[0]
            min_capacity_on_path = get_min_capacity_on_path(
                shortest_path, remaining_edges
            )
            if len(shortest_path) > 2:
                # there is actually interference on the shortest path, so we pin the demand
                # for direct demands and path, we don't consider them as pinned
                pinned_demands.append((from_, to_))
            value_to_pin = min(min_capacity_on_path, demands[(from_, to_)])
            if value_to_pin == 0:
                continue
            path_string = "_".join(shortest_path)
            var_name = f"aux_flowpath_{path_string}"
            flow_on_path_vars[var_name] = value_to_pin
            flow_pair_vars[f"aux_flow_{from_}_{to_}"] = value_to_pin

            # Store the decision variables in all_vars
            all_vars[var_name] = value_to_pin
            all_vars[f"aux_flow_{from_}_{to_}"] = value_to_pin
            remaining_demands[(from_, to_)] -= value_to_pin
            for node in range(len(shortest_path) - 1):
                edge = f"edge_{shortest_path[node]}_{shortest_path[node + 1]}"
                remaining_edges[edge] -= value_to_pin
        else:
            non_pinned_demands.append((from_, to_))
    pinning_end = time.time()
    if ENABLE_PRINT:
        print(f"Demand pinning phase took {pinning_end - pinning_start:.4f} seconds")

    # Now solve the optimization problem for the remaining demands and edges
    partial_optimal = optimal_TE(
        num_nodes,
        remaining_edges,
        remaining_demands,
        possible_demands=possible_demands,
        given_all_paths=all_paths,
    )

    partial_flow_values = partial_optimal["flow_values"]
    partial_flowpath_values = partial_optimal["flowpath_values"]

    # Assemble the final flow values and update all_vars
    flow_values = partial_flow_values.copy()
    for key in flow_pair_vars:
        if key not in partial_flow_values:
            flow_values[key] = flow_pair_vars[key]
            all_vars[key] = flow_pair_vars[key]
        else:
            flow_values[key] += flow_pair_vars[key]
            all_vars[key] = flow_values[key]

    for key in partial_flowpath_values:
        if key not in flow_on_path_vars:
            flow_on_path_vars[key] = partial_flowpath_values[key]
            all_vars[key] = partial_flowpath_values[key]
        else:
            flow_on_path_vars[key] += partial_flowpath_values[key]
            all_vars[key] = flow_on_path_vars[key]

    # Add any remaining variables from partial optimal solution
    for key, value in partial_optimal["all_vars"].items():
        if key not in all_vars:
            all_vars[key] = value

    total_flow_value = sum(flow_values.values())

    code_path_num = ""

    pinned_demands_sorted = sorted(pinned_demands, key=lambda x: (x[0], x[1]))
    for from_, to_ in pinned_demands_sorted:
        code_path_num += f"{from_}-{to_}_"

    return {
        "code_path_num": code_path_num,
        "flow_values": flow_values,
        "flowpath_values": flow_on_path_vars,
        "heuristic_value": total_flow_value,
        "all_vars": all_vars,
    }

def pop_TE_wrapper(num_nodes, edges, demands, partition_lists, possible_demands=None, given_all_paths=None):
    # create n random partitions and return the average of the solutions
    solutions = []
    for partitions in partition_lists:
        solution = pop_TE(num_nodes, edges, demands, partitions, possible_demands, given_all_paths)
        solutions.append(solution)
    # average the solutions
    num_random = len(partition_lists)
    heuristic_value = sum(solution["heuristic_value"] for solution in solutions) / num_random
    code_path_num = ""
    # any different path, a different code path
    for solution in solutions:
        code_path_num += solution["code_path_num"]
    return {
        "heuristic_value": heuristic_value,
        "code_path_num": code_path_num,
        "all_vars": solutions[0]["all_vars"],  # Use the first solution's all_vars for consistency
    }

def pop_TE(
    num_nodes, edges, demands, partitions, possible_demands=None, given_all_paths=None
):
    start_time = time.time()
    # print(f"Starting pop_TE with {num_nodes} nodes, {len(demands)} demands, {len(partitions)} partitions")

    """
    Partitioned Optimization Problems (PoP) heuristic for Traffic Engineering
    with provided partitions.

    :param num_nodes: Number of nodes in the topology.
    :param edges: Dictionary representing edge capacities.
    :param demands: Dictionary representing demands between node pairs.
    :param partitions: A list of dictionaries, where each dictionary represents
                       a partition's demands.
    :return: Total flow value and flow allocations.
    """
    # Step 1: Scale edge capacities for each partition
    num_partitions = len(partitions)
    partition_edges = [
        {edge: capacity / num_partitions for edge, capacity in edges.items()}
        for _ in range(num_partitions)
    ]

    # Step 2: Solve the optimization problem for each partition independently
    flow_values_per_partition = []
    flowpath_values_per_partition = []
    all_vars_per_partition = []
    for partition_index, partition_demand_names in enumerate(partitions):
        partition_demands = {key: demands[key] for key in partition_demand_names}
        # Get the edge capacities for the current partition
        partition_edge_capacities = partition_edges[partition_index]
        # Solve the optimization problem for this partition
        partial_optimal = optimal_TE(
            num_nodes,
            partition_edge_capacities,
            partition_demands,
            possible_demands=possible_demands,
            given_all_paths=given_all_paths,
        )
        # Collect the results
        flow_values_per_partition.append(partial_optimal["flow_values"])
        flowpath_values_per_partition.append(partial_optimal["flowpath_values"])
        all_vars_per_partition.append(partial_optimal["all_vars"])

    # Step 3: Combine results across partitions
    combination_start = time.time()
    final_flow_values = {}
    final_flowpath_values = {}
    final_all_vars = {}

    for partition_flow_values in flow_values_per_partition:
        for key, value in partition_flow_values.items():
            if key not in final_flow_values:
                final_flow_values[key] = value
            else:
                final_flow_values[key] += value

    for partition_flowpath_values in flowpath_values_per_partition:
        for key, value in partition_flowpath_values.items():
            if key not in final_flowpath_values:
                final_flowpath_values[key] = value
            else:
                final_flowpath_values[key] += value

    for partition_all_vars in all_vars_per_partition:
        for key, value in partition_all_vars.items():
            if key not in final_all_vars:
                final_all_vars[key] = value
            else:
                final_all_vars[key] += value
    # Compute the total flow value
    total_flow_value = sum(final_flow_values.values())
    combination_end = time.time()
    # print(f"Result combination took {combination_end - combination_start:.4f} seconds")

    code_path_num = ""
    # sort the demands by name
    # sort the final_flow_values by the demand name
    # final_flow_values_sorted = sorted(final_flow_values.items(), key=lambda x: (x[0].split("_")[-2], x[0].split("_")[-1]))
    # for key, value in final_flow_values_sorted:
    #     from_, to_ = key.split("_")[-2], key.split("_")[-1]
    #     value = final_flow_values[key]
    #     if value > 0 or final_all_vars[f"demand_{from_}_{to_}"] == 0:
    #         code_path_num += f"{from_}{to_}"

    end_time = time.time()
    # print(f"Total pop_TE execution took {end_time - start_time:.4f} seconds")

    return {
        "heuristic_value": total_flow_value,
        "code_path_num": code_path_num,
        "flow_values": final_flow_values,
        "flowpath_values": final_flowpath_values,
        "all_vars": final_all_vars,
    }

# TODO: so for each problem you have to manually define what the lagrangian is instead of allowing the code to automatically derive it? That would be against what we claim metaease does.
# We need a similar structure to metaopt here, where you define polynomials/terms and tehn allow the code to automatically derive the lagrangian, if not, you need to figure out how to tell the user that they need to implement this and how, this
# increases the burden on them though.
def get_TE_lagrangian(
    num_nodes, edges, input_dict, give_relaxed_gap=False, given_all_paths=None
):
    start_time = time.time()
    # print(f"Starting get_TE_lagrangian with {num_nodes} nodes")

    # give_relaxed_gap is not used in this problem
    lagrange = 0
    constraints = {}

    # Add contributions from possible demand pairs
    possible_demand_pairs = [
        (pair.split("_")[-2], pair.split("_")[-1])
        for pair in input_dict
        if "aux_flow_" in pair
    ]

    for pair in possible_demand_pairs:
        lagrange += input_dict[f"aux_flow_{pair[0]}_{pair[1]}"]

    if give_relaxed_gap:
        end_time = time.time()
        # print(f"get_TE_lagrangian (relaxed) took {end_time - start_time:.4f} seconds")
        return {"lagrange": lagrange, "constraints": constraints}

    for pair in possible_demand_pairs:
        constraint = (
            input_dict[f"demand_{pair[0]}_{pair[1]}"]
            - input_dict[f"aux_flow_{pair[0]}_{pair[1]}"]
        )
        constraints[f"lambda_flow_demand_{pair[0]}_{pair[1]}"] = constraint
        lagrange += input_dict[f"lambda_flow_demand_{pair[0]}_{pair[1]}"] * constraint

    all_paths = given_all_paths if given_all_paths is not None else {}
    nodes = range(num_nodes)
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                if (from_, to_) in all_paths:
                    paths = all_paths[(from_, to_)]
                else:
                    paths = find_all_paths(
                        {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                    )
                    all_paths[(from_, to_)] = paths

    for pair, path_list in all_paths.items():
        if pair not in possible_demand_pairs:
            continue

        compute_flow_diff = input_dict[f"aux_flow_{pair[0]}_{pair[1]}"]
        for path in path_list:
            compute_flow_diff -= input_dict[f"aux_flowpath_{'_'.join(map(str, path))}"]

        constraints[f"lambda_flowpath_{pair[0]}_{pair[1]}"] = compute_flow_diff
        lagrange += (
            input_dict[f"lambda_flowpath_{pair[0]}_{pair[1]}"] * compute_flow_diff
        )

    # Sum contributions per edge
    sum_per_edge = {}
    for pair, path_list in all_paths.items():
        if pair not in possible_demand_pairs:
            continue

        for path in path_list:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]

                edge = (source, target, edges[f"edge_{source}_{target}"])
                if edge is None:
                    continue

                if edge not in sum_per_edge:
                    sum_per_edge[edge] = 0

                sum_per_edge[edge] += input_dict[
                    f"aux_flowpath_{'_'.join(map(str, path))}"
                ]

    for edge, total in sum_per_edge.items():
        constraints[f"lambda_capacity_{edge[0]}_{edge[1]}"] = edge[2] - total
        lagrange += input_dict[f"lambda_capacity_{edge[0]}_{edge[1]}"] * (
            edge[2] - total
        )

    end_time = time.time()
    print(f"get_TE_lagrangian took {end_time - start_time:.4f} seconds")

    return {"lagrange": lagrange, "constraints": constraints}


def get_TE_lagrangian_gradient_optimized(
    num_nodes, edges, input_dict, given_all_paths=None
):
    """
    Optimized version of get_TE_lagrangian_gradient with significant performance improvements.
    """
    start_time = time.time()
    # print(f"Starting get_TE_lagrangian_gradient_optimized with {num_nodes} nodes")

    gradient = {}
    all_paths = {}
    nodes = range(num_nodes)

    # Precompute all paths if not provided
    if given_all_paths is not None:
        for key in given_all_paths:
            new_key = (str(key[0]), str(key[1]))
            all_paths[new_key] = given_all_paths[key]
    else:
        # Precompute all paths in one pass to avoid repeated calls
        for from_ in nodes:
            for to_ in nodes:
                if from_ != to_:
                    paths = find_all_paths(
                        {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                    )
                    all_paths[(str(from_), str(to_))] = paths

    # Precompute edge-to-paths mapping for capacity constraints
    edge_to_paths = defaultdict(list)
    for pair, path_list in all_paths.items():
        for path in path_list:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                edge_key = f"{source}_{target}"
                path_key = f"aux_flowpath_{'_'.join(path)}"
                edge_to_paths[edge_key].append(path_key)

    # Process gradient computation in optimized batches
    for key in input_dict:
        if key.startswith("demand_"):
            pair = key.split("_")[1:]
            lambda_key = f"lambda_flow_demand_{pair[0]}_{pair[1]}"
            gradient[key] = input_dict.get(lambda_key, 0)

        elif key.startswith("aux_flow_"):
            pair = key.split("_")[2:]
            lambda_flowpath_key = f"lambda_flowpath_{pair[0]}_{pair[1]}"
            lambda_flow_demand_key = f"lambda_flow_demand_{pair[0]}_{pair[1]}"
            gradient[key] = 1
            if lambda_flowpath_key in input_dict:
                gradient[key] += input_dict[lambda_flowpath_key]
            if lambda_flow_demand_key in input_dict:
                gradient[key] -= input_dict[lambda_flow_demand_key]

        elif key.startswith("aux_flowpath_"):
            path = key.split("_")[2:]
            from_node, to_node = path[0], path[-1]
            lambda_flowpath_key = f"lambda_flowpath_{from_node}_{to_node}"
            gradient[key] = 0
            if lambda_flowpath_key in input_dict:
                gradient[key] = -input_dict[lambda_flowpath_key]

            # Use precomputed edge information
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                lambda_capacity_key = f"lambda_capacity_{source}_{target}"
                if lambda_capacity_key in input_dict:
                    gradient[key] -= input_dict[lambda_capacity_key]

        elif key.startswith("lambda_flow_demand_"):
            pair = key.split("_")[3:]
            demand_key = f"demand_{pair[0]}_{pair[1]}"
            aux_flow_key = f"aux_flow_{pair[0]}_{pair[1]}"
            gradient[key] = 0
            if demand_key in input_dict and aux_flow_key in input_dict:
                gradient[key] = input_dict[demand_key] - input_dict[aux_flow_key]

        elif key.startswith("lambda_flowpath_"):
            pair = key.split("_")[2:]
            from_node, to_node = pair[0], pair[1]
            aux_flow_key = f"aux_flow_{from_node}_{to_node}"
            gradient[key] = 0
            if aux_flow_key in input_dict:
                gradient[key] = input_dict[aux_flow_key]

            # Use precomputed paths
            pair_key = (from_node, to_node)
            if pair_key in all_paths:
                for path in all_paths[pair_key]:
                    path_key = f"aux_flowpath_{'_'.join(path)}"
                    if path_key in input_dict:
                        gradient[key] -= input_dict[path_key]

        elif key.startswith("lambda_capacity_"):
            edge = key.split("_")[2:]
            source, target = edge[0], edge[1]
            edge_key = f"{source}_{target}"

            # Use precomputed edge-to-paths mapping
            total_flow = 0
            if edge_key in edge_to_paths:
                for path_key in edge_to_paths[edge_key]:
                    if path_key in input_dict:
                        total_flow += input_dict[path_key]

            capacity = edges[f"edge_{source}_{target}"]
            gradient[key] = capacity - total_flow

    # Apply sign correction for lambda variables
    for key in gradient:
        if "lambda" in key:
            gradient[key] = -gradient[key]

    end_time = time.time()
    if ENABLE_PRINT:
        print(
            f"get_TE_lagrangian_gradient_optimized took {end_time - start_time:.4f} seconds"
        )

    return gradient


def get_TE_lagrangian_gradient(num_nodes, edges, input_dict, given_all_paths=None):
    start_time = time.time()
    # print(f"Starting get_TE_lagrangian_gradient with {num_nodes} nodes")

    gradient = {}
    all_paths = {}
    nodes = range(num_nodes)
    if given_all_paths is not None:
        for key in given_all_paths:
            new_key = (str(key[0]), str(key[1]))
            all_paths[new_key] = given_all_paths[key]
    else:
        for from_ in nodes:
            for to_ in nodes:
                if from_ != to_:
                    if (from_, to_) in all_paths:
                        paths = all_paths[(from_, to_)]
                    else:
                        paths = find_all_paths(
                            {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                        )
                        all_paths[(str(from_), str(to_))] = paths

    for key in input_dict:
        if key.startswith("demand_"):
            pair = key.split("_")[1:]
            lambda_key = f"lambda_flow_demand_{pair[0]}_{pair[1]}"
            if lambda_key in input_dict:
                gradient[key] = input_dict[lambda_key]
            else:
                gradient[key] = 0
        elif key.startswith("aux_flow_"):
            pair = key.split("_")[2:]
            lambda_flowpath_key = f"lambda_flowpath_{pair[0]}_{pair[1]}"
            lambda_flow_demand_key = f"lambda_flow_demand_{pair[0]}_{pair[1]}"
            gradient[key] = 1
            if lambda_flowpath_key in input_dict:
                gradient[key] += input_dict[lambda_flowpath_key]
            if lambda_flow_demand_key in input_dict:
                gradient[key] -= input_dict[lambda_flow_demand_key]
        elif key.startswith("aux_flowpath_"):
            path = key.split("_")[2:]
            from_node, to_node = path[0], path[-1]
            lambda_flowpath_key = f"lambda_flowpath_{from_node}_{to_node}"
            gradient[key] = 0
            if lambda_flowpath_key in input_dict:
                gradient[key] = -input_dict[lambda_flowpath_key]
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                lambda_capacity_key = f"lambda_capacity_{source}_{target}"
                if lambda_capacity_key in input_dict:
                    gradient[key] -= input_dict[lambda_capacity_key]
        elif key.startswith("lambda_flow_demand_"):
            pair = key.split("_")[3:]
            demand_key = f"demand_{pair[0]}_{pair[1]}"
            aux_flow_key = f"aux_flow_{pair[0]}_{pair[1]}"
            gradient[key] = 0
            if demand_key in input_dict and aux_flow_key in input_dict:
                gradient[key] = input_dict[demand_key] - input_dict[aux_flow_key]
        elif key.startswith("lambda_flowpath_"):
            pair = key.split("_")[2:]
            from_node, to_node = pair[0], pair[1]
            aux_flow_key = f"aux_flow_{from_node}_{to_node}"
            gradient[key] = 0
            if aux_flow_key in input_dict:
                gradient[key] = input_dict[aux_flow_key]
            if (from_node, to_node) in all_paths:
                for path in all_paths[(from_node, to_node)]:
                    path_key = f"aux_flowpath_{'_'.join(path)}"
                    if path_key in input_dict:
                        gradient[key] -= input_dict[path_key]
        elif key.startswith("lambda_capacity_"):
            edge = key.split("_")[2:]
            source, target = edge[0], edge[1]
            total_flow = 0
            for pair, path_list in all_paths.items():
                for path in path_list:
                    for i in range(len(path) - 1):
                        if path[i] == source and path[i + 1] == target:
                            path_key = f"aux_flowpath_{'_'.join(path)}"
                            if path_key in input_dict:
                                total_flow += input_dict[path_key]

            capacity = edges[f"edge_{source}_{target}"]
            gradient[key] = capacity - total_flow

    for key in gradient:
        if "lambda" in key:
            gradient[key] = -gradient[key]

    end_time = time.time()
    print(f"get_TE_lagrangian_gradient took {end_time - start_time:.4f} seconds")

    return gradient


def get_all_demands_for_edge(all_paths, edge):
    # get all the demands that could possibly go through the edge
    demands = []
    for from_, to_ in all_paths:
        for path in all_paths[(from_, to_)]:
            for i in range(len(path) - 1):
                if str(path[i]) == str(edge[0]) and str(path[i + 1]) == str(edge[1]):
                    demands.append((from_, to_))
    return demands


def get_all_demands_for_edge_when_shortest_path(all_paths, edge):
    # get all the demands that could possibly go through the edge when the edge is on the shortest path of that demand
    demands = []
    for from_, to_ in all_paths:
        paths = all_paths[(from_, to_)]
        if len(paths) > 0:
            paths.sort(key=lambda x: (len(x), "_".join(x)))
            shortest_path = paths[0]
            for i in range(len(shortest_path) - 1):
                if str(shortest_path[i]) == str(edge[0]) and str(
                    shortest_path[i + 1]
                ) == str(edge[1]):
                    demands.append((from_, to_))
    return demands


class TEProblem(Problem):
    def __init__(self, problem_config_path):
        start_time = time.time()
        # print(f"Starting TEProblem initialization")

        super().__init__(problem_config_path)
        topology_name = self.problem_config["topology"]
        self.max_num_paths = self.problem_config.get("max_num_paths", 4)
        possible_demands_path = f"possible_demands_{topology_name}_{self.max_num_paths}paths.json"
        all_paths_path = f"all_paths_{topology_name}_{self.max_num_paths}paths.json"

        loading_start = time.time()
        if os.path.exists(possible_demands_path) and os.path.exists(all_paths_path):
            print(
                f"Loading possible demands and all paths from {possible_demands_path} and {all_paths_path}"
            )
            with open(possible_demands_path, "r") as f:
                possible_demands_str = json.load(f)
            # possible_demands is a list of list of int
            self.possible_demands = [
                tuple(map(int, demand)) for demand in possible_demands_str
            ]
            with open(all_paths_path, "r") as f:
                all_paths_str_keys = json.load(f)
            self.all_paths = {
                tuple(map(int, k.split("_"))): v for k, v in all_paths_str_keys.items()
            }
        else:
            print(f"Computing possible demands and all paths (this may take a while)")
            self.possible_demands, self.all_paths = self.find_possible_demands()
            all_paths_str_keys = {
                f"{k[0]}_{k[1]}": v for k, v in self.all_paths.items()
            }
            with open(possible_demands_path, "w") as f:
                json.dump(self.possible_demands, f)
            with open(all_paths_path, "w") as f:
                json.dump(all_paths_str_keys, f)
        loading_end = time.time()
        print(f"Loading/computing paths took {loading_end - loading_start:.4f} seconds")

        self.num_total_klee_inputs = len(self.possible_demands)
        self.all_klee_var_names = []
        for from_, to_ in self.possible_demands:
            self.all_klee_var_names.append(f"demand_{from_}_{to_}")
        self.num_random = self.problem_config.get("num_random", 10)
        self.num_partitions = self.problem_config.get("num_partitions", 10)
        self.partitions_file = self.problem_config.get("partitions_file", None)
        self.partition_lists = []
        if self.partitions_file is not None:
            with open(self.partitions_file, "r") as f:
                partitions_data = json.load(f)
            # The partitions are stored in the "partitions" key as an array of dictionaries
            partitions_str = partitions_data.get("partitions", [])
            partition_lists = [
                {tuple(map(int, k.split("_"))): v for k, v in partition.items()}
                for partition in partitions_str
            ]
            # each partition in partition_lists is a dictionary of (from, to) -> partition_num
            # I want to convert it to a list of lists of (from, to) tuples, where the outer list is the partition number and the inner list is the list of (from, to) tuples
            # Filter to only include demand pairs that exist in self.possible_demands
            self.partition_lists = []
            for partition in partition_lists:
                # Filter the partition to only include valid demand pairs
                new_partition = []
                filtered_partition = {k: v for k, v in partition.items() if k in self.possible_demands}
                unique_partition_nums = list(set(filtered_partition.values()))
                for partition_num in unique_partition_nums:
                    partition_demands = [(from_, to_) for (from_, to_), p_num in filtered_partition.items() if p_num == partition_num]
                    if partition_demands:  # Only add non-empty partitions
                        new_partition.append(partition_demands)
                self.partition_lists.append(new_partition)
            print(f"Loaded {len(self.partition_lists)} partitions from {self.partitions_file}")
            print(f"Number of possible demands: {len(self.possible_demands)}")
            print(f"Sample of possible demands: {list(self.possible_demands)[:5]}")
            if self.partition_lists:
                print(f"Sample of partition demands: {self.partition_lists[0][:5] if self.partition_lists[0] else 'Empty partition'}")
        else:
            self.seeds = range(self.num_random)
            for seed in self.seeds:
                partitions = self.create_random_partition(self.possible_demands, seed, self.num_partitions)
                self.partition_lists.append(partitions)
        print(f"Number of possible demands: {len(self.possible_demands)}")

        # Precompute static solver components for optimization
        precompute_start = time.time()
        self._precompute_static_data()
        precompute_end = time.time()
        print(
            f"Static data precomputation took {precompute_end - precompute_start:.4f} seconds"
        )

        end_time = time.time()
        print(
            f"Total TEProblem initialization took {end_time - start_time:.4f} seconds"
        )

    def _precompute_static_data(self):
        start_time = time.time()
        # print(f"Starting _precompute_static_data")

        # Get static data
        num_nodes = self.problem_config["num_nodes"]
        edges = {
            key: value
            for key, value in self.problem_config.items()
            if "edge_" in key and value > 0
        }
        all_paths = self.all_paths
        nodes = range(num_nodes)

        # Precompute path mappings and capacity constraints data (static - don't depend on demands)
        var_creation_start = time.time()

        # Store path information and variable bounds instead of actual solver variables
        self.static_path_info = {}
        self.static_capacity_constraints_info = {}

        for from_ in nodes:
            for to_ in nodes:
                if from_ != to_:
                    if (from_, to_) in all_paths:
                        paths = all_paths[(from_, to_)]
                    else:
                        paths = find_all_paths(
                            {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                        )
                        all_paths[(from_, to_)] = paths

                    # Store path information for each source-destination pair
                    path_info = []
                    for path in paths:
                        path_string = "_".join(path)
                        var_name = f"aux_flowpath_{path_string}"
                        min_capacity_on_path = get_min_capacity_on_path(path, edges)
                        path_info.append(
                            {
                                "var_name": var_name,
                                "path": path,
                                "max_capacity": min_capacity_on_path,
                            }
                        )
                    self.static_path_info[(from_, to_)] = path_info

        var_creation_end = time.time()
        print(
            f"Path info creation took {var_creation_end - var_creation_start:.4f} seconds"
        )

        # Precompute capacity constraints information
        constraint_creation_start = time.time()
        for from_ in nodes:
            for to_ in nodes:
                if from_ != to_:
                    edge = f"edge_{from_}_{to_}"
                    if edge in edges:
                        capacity = edges[edge]
                        # find all the paths that use this edge
                        paths_with_this_edge = []
                        for key, value in all_paths.items():
                            for path in value:
                                path_string = "_".join(path)
                                for i in range(len(path) - 1):
                                    if str(path[i]) == str(from_) and str(
                                        path[i + 1]
                                    ) == str(to_):
                                        paths_with_this_edge.append(
                                            f"aux_flowpath_{path_string}"
                                        )
                                        break

                        self.static_capacity_constraints_info[f"{from_}_{to_}"] = {
                            "capacity": capacity,
                            "paths": paths_with_this_edge,
                        }
        constraint_creation_end = time.time()
        print(
            f"Capacity constraint info creation took {constraint_creation_end - constraint_creation_start:.4f} seconds"
        )

        # Store static all_paths for reuse
        self.static_all_paths = all_paths

        end_time = time.time()
        print(f"Total _precompute_static_data took {end_time - start_time:.4f} seconds")

    def _optimal_TE_original(self, demands):
        start_time = time.time()
        # print(f"Starting _optimal_TE_original with {len(demands)} demands")

        if not hasattr(self, "static_path_info"):
            raise Exception(
                "Static data not precomputed. Call _precompute_static_data first!"
            )

        # Create a new solver for this optimization
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            raise Exception("Solver could not be created!")

        all_paths = self.static_all_paths

        # Create demand-dependent variables and constraints
        demand_vars_start = time.time()
        all_vars = {}
        flow_pair_vars = {}
        flowpath_constraints = {}
        flow_demand_constraints = {}
        flow_on_path_vars = {}

        # Add demand values to all_vars
        for from_, to_ in demands:
            all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]

        # Create flow path variables using precomputed info
        for from_, to_ in all_paths:
            if (from_, to_) in self.static_path_info:
                for path_info in self.static_path_info[(from_, to_)]:
                    var_name = path_info["var_name"]
                    max_capacity = path_info["max_capacity"]
                    flow_on_path_vars[var_name] = solver.NumVar(
                        0, max_capacity, var_name
                    )

        # Create flow variables and constraints for each demand
        for from_, to_ in demands:
            if (from_, to_) in self.possible_demands:
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] = solver.NumVar(
                    0, 2 * demands[(from_, to_)], f"aux_flow_{from_}_{to_}"
                )
                flow_demand_constraints[(from_, to_)] = solver.Add(
                    flow_pair_vars[f"aux_flow_{from_}_{to_}"] <= demands[(from_, to_)]
                )

                # Connect flow to path variables
                if (from_, to_) in self.static_path_info:
                    path_vars = []
                    for path_info in self.static_path_info[(from_, to_)]:
                        var_name = path_info["var_name"]
                        path_vars.append(flow_on_path_vars[var_name])

                    if path_vars:
                        sum_flow_on_paths = solver.Sum(path_vars)
                        constraint = solver.Add(
                            flow_pair_vars[f"aux_flow_{from_}_{to_}"]
                            == sum_flow_on_paths
                        )
                        flowpath_constraints[(from_, to_)] = constraint

        # Create capacity constraints using precomputed info
        capacity_constraints = {}
        for edge_key, constraint_info in self.static_capacity_constraints_info.items():
            capacity = constraint_info["capacity"]
            path_vars = [
                flow_on_path_vars[path]
                for path in constraint_info["paths"]
                if path in flow_on_path_vars
            ]
            if path_vars:
                constraint = solver.Add(solver.Sum(path_vars) <= capacity)
                capacity_constraints[edge_key] = constraint

        demand_vars_end = time.time()
        if ENABLE_PRINT:
            print(
                f"Demand-dependent variable creation took {demand_vars_end - demand_vars_start:.4f} seconds"
            )

        # Set up objective (maximize total flow)
        objective_start = time.time()
        total_flow = solver.Sum(flow_pair_vars.values())
        solver.Maximize(total_flow)
        objective_end = time.time()
        if ENABLE_PRINT:
            print(f"Objective setup took {objective_end - objective_start:.4f} seconds")

        # Solve the problem
        solve_start = time.time()
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise Exception("Solver did not find an optimal solution!")
        solve_end = time.time()
        if ENABLE_PRINT:
            print(f"Solver execution took {solve_end - solve_start:.4f} seconds")

        # Extract results
        extraction_start = time.time()
        flow_values = {
            edge: flow_pair_vars[edge].solution_value() for edge in flow_pair_vars
        }
        flowpath_values = {
            path: flow_on_path_vars[path].solution_value() for path in flow_on_path_vars
        }

        # Store all variable values
        for key in flow_pair_vars:
            all_vars[key] = flow_pair_vars[key].solution_value()

        for key in flow_on_path_vars:
            all_vars[key] = flow_on_path_vars[key].solution_value()

        # Extract dual values for constraints
        for key in flow_demand_constraints:
            all_vars[f"lambda_flow_demand_{key[0]}_{key[1]}"] = flow_demand_constraints[
                key
            ].dual_value()

        for key in flowpath_constraints:
            all_vars[f"lambda_flowpath_{key[0]}_{key[1]}"] = flowpath_constraints[
                key
            ].dual_value()

        for key in capacity_constraints:
            all_vars[f"lambda_capacity_{key}"] = capacity_constraints[key].dual_value()

        total_flow_value = total_flow.solution_value()
        extraction_end = time.time()
        if ENABLE_PRINT:
            print(f"Result extraction took {extraction_end - extraction_start:.4f} seconds")

        end_time = time.time()
        if ENABLE_PRINT:
            print(
                f"Total _optimal_TE_original execution took {end_time - start_time:.4f} seconds"
            )

        return {
            "flow_values": flow_values,
            "flowpath_values": flowpath_values,
            "optimal_total_flow": total_flow_value,
            "all_vars": all_vars,
        }
   # TODO: I see multiple instances of optimal TE, what is each one, why are there multiple? If there is a good reason to have many, add comments to differentiate them.
   # On a similar note, I saw that you have both old version and new version of different functions, always keep the new version and delete all old code in public releases: old code is liability especially since you wont be maintaining it.
    def optimal_TE(
        self, num_nodes, edges, demands, possible_demands=None, given_all_paths=None
    ):
        # Check if we can use the optimized version with precomputed static data
        if hasattr(self, "static_path_info") and self.static_path_info is not None:
            if ENABLE_PRINT:
                print(f"Using optimized _optimal_TE_original with precomputed static data")
            return self._optimal_TE_original(demands)

        # Fall back to original implementation if static data not available
        start_time = time.time()
        # print(f"Starting optimal_TE with {num_nodes} nodes, {len(demands)} demands")

        # Create the solver
        solver_start = time.time()
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            raise Exception("Solver could not be created!")
        solver_end = time.time()
        if ENABLE_PRINT:
            print(f"Solver creation took {solver_end - solver_start:.4f} seconds")

        all_vars = {}
        flow_on_path_vars = {}
        all_paths = given_all_paths if given_all_paths is not None else {}
        nodes = range(num_nodes)

        # Variable creation phase
        var_creation_start = time.time()
        for from_ in nodes:
            for to_ in nodes:
                if from_ != to_:
                    if (from_, to_) in all_paths:
                        paths = all_paths[(from_, to_)]
                    else:
                        paths = find_all_paths(
                            {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                        )
                        all_paths[(from_, to_)] = paths
                    for path in paths:
                        path_string = "_".join(path)
                        var_name = f"aux_flowpath_{path_string}"
                        min_capacity_on_path = get_min_capacity_on_path(path, edges)
                        flow_on_path_vars[var_name] = solver.NumVar(
                            0, min_capacity_on_path, var_name
                        )
        var_creation_end = time.time()
        if ENABLE_PRINT:
            print(
                f"Variable creation took {var_creation_end - var_creation_start:.4f} seconds"
            )

        # Create flow variables for
        if possible_demands is None:
            possible_demands, _ = find_possible_demands(
                {"num_nodes": num_nodes, **edges}
            )
        for from_, to_ in demands:
            if (from_, to_) in possible_demands:
                all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]
        flow_pair_vars = {}
        flowpath_constraints = {}
        flow_demand_constraints = {}

        # Constraint creation phase
        constraint_creation_start = time.time()
        for from_, to_ in possible_demands:
            if (from_, to_) in demands:
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] = solver.NumVar(
                    0, 2 * demands[(from_, to_)], f"aux_flow_{from_}_{to_}"
                )
                flow_demand_constraints[(from_, to_)] = solver.Add(
                    flow_pair_vars[f"aux_flow_{from_}_{to_}"] <= demands[(from_, to_)]
                )
                # the aux_flow_{from_}_{to_} is equal to the sum of the flow on all paths from from_ to to_
                paths = all_paths[(from_, to_)]
                path_names = ["_".join(path) for path in paths]
                sum_flow_on_paths = solver.Sum(
                    flow_on_path_vars[f"aux_flowpath_{path_name}"]
                    for path_name in path_names
                )
                constraint = solver.Add(
                    flow_pair_vars[f"aux_flow_{from_}_{to_}"] == sum_flow_on_paths
                )
                flowpath_constraints[(from_, to_)] = constraint

        capacity_constraints = {}
        # Add capacity constraints for each edge
        for from_ in nodes:
            for to_ in nodes:
                if from_ != to_:
                    edge = f"edge_{from_}_{to_}"
                    if edge in edges:
                        capacity = edges[edge]
                        # find all the paths that use this edge
                        every_path = []
                        for key, value in all_paths.items():
                            every_path.extend("_".join(path) for path in value)
                        paths_with_this_edge = [
                            path for path in every_path if f"{from_}_{to_}" in path
                        ]
                        constraint = solver.Add(
                            solver.Sum(
                                flow_on_path_vars[f"aux_flowpath_{path}"]
                                for path in paths_with_this_edge
                            )
                            <= capacity
                        )
                        capacity_constraints[f"{from_}_{to_}"] = constraint
        constraint_creation_end = time.time()
        if ENABLE_PRINT:
            print(
                f"Constraint creation took {constraint_creation_end - constraint_creation_start:.4f} seconds"
            )

        # Objective: Maximize the total flow
        objective_start = time.time()
        total_flow = solver.Sum(flow_pair_vars.values())
        solver.Maximize(total_flow)
        objective_end = time.time()
        if ENABLE_PRINT:
            print(f"Objective setup took {objective_end - objective_start:.4f} seconds")

        # Solve the problem
        solve_start = time.time()
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise Exception("Solver did not find an optimal solution!")
        solve_end = time.time()
        if ENABLE_PRINT:
            print(f"Solver execution took {solve_end - solve_start:.4f} seconds")

        # Extract results
        extraction_start = time.time()
        flow_values = {
            edge: flow_pair_vars[edge].solution_value() for edge in flow_pair_vars
        }
        flowpath_values = {
            path: flow_on_path_vars[path].solution_value() for path in flow_on_path_vars
        }
        for key in flow_pair_vars:
            all_vars[key] = flow_pair_vars[key].solution_value()

        for key in flow_on_path_vars:
            all_vars[key] = flow_on_path_vars[key].solution_value()

        for key in flow_demand_constraints:
            all_vars[f"lambda_flow_demand_{key[0]}_{key[1]}"] = flow_demand_constraints[
                key
            ].dual_value()

        for key in flowpath_constraints:
            all_vars[f"lambda_flowpath_{key[0]}_{key[1]}"] = flowpath_constraints[
                key
            ].dual_value()

        for key in capacity_constraints:
            all_vars[f"lambda_capacity_{key}"] = capacity_constraints[key].dual_value()

        total_flow_value = total_flow.solution_value()
        extraction_end = time.time()
        if ENABLE_PRINT:
            print(f"Result extraction took {extraction_end - extraction_start:.4f} seconds")

        end_time = time.time()
        if ENABLE_PRINT:
            print(f"Total optimal_TE execution took {end_time - start_time:.4f} seconds")

        return {
            "flow_values": flow_values,
            "flowpath_values": flowpath_values,
            "optimal_total_flow": total_flow_value,
            "all_vars": all_vars,
        }

    def reset_solver(self):
        """Reset the solver state by recreating the static solver components"""
        if ENABLE_PRINT:
            print("Resetting solver state...")
        self._precompute_static_data()
        if ENABLE_PRINT:
            print("Solver reset complete")

    def get_thresholds(self, relaxed_all_vars):
        thresholds = {
            f"demand_{key[0]}_{key[1]}": (0, self.problem_config["max_flow"])
            for key in self.possible_demands
        }
        for key in relaxed_all_vars:
            if key.startswith("lambda_"):
                thresholds[key] = (0, LAMBDA_MAX_VALUE)
            elif key.startswith("aux_flowpath_"):
                # each flowpath goes through an edge, so the max value is the capacity of the edge
                edge_key = key.split("_")[2:]
                thresholds[key] = (
                    0,
                    self.problem_config[f"edge_{edge_key[0]}_{edge_key[1]}"],
                )
            elif key.startswith("aux_flow_"):
                # the flow between a pair of nodes can be at most the demand between them
                thresholds[key] = (0, self.problem_config["max_flow"])
        return thresholds

    def get_decision_to_input_map(self, all_vars):
        # Create a mapping of decision variables to their corresponding input variables
        decision_to_input_map = {}
        # Map aux_flow and aux_flowpath variables to their corresponding demand variables
        for var_name in all_vars.keys():
            if var_name.startswith("aux_flow_"):
                # For aux_flow_X_Y, the corresponding input is demand_X_Y
                _, _, src, dst = var_name.split("_")
                input_var = f"demand_{src}_{dst}"
                decision_to_input_map[var_name] = input_var
            elif var_name.startswith("aux_flowpath_"):
                # For aux_flowpath_X_Y_Z..., the corresponding input is demand_X_Z (first and last nodes)
                parts = var_name.split("_")
                src = parts[2]  # First node in path
                dst = parts[-1]  # Last node in path
                input_var = f"demand_{src}_{dst}"
                decision_to_input_map[var_name] = input_var
        return decision_to_input_map

    def find_possible_demands(self):
        return find_possible_demands(self.problem_config)

    def convert_input_dict_to_args(self, input_dict):
        num_nodes = self.problem_config["num_nodes"]
        edges = {
            key: value
            for key, value in self.problem_config.items()
            if "edge_" in key and value > 0
        }
        demands = {
            (int(key.split("_")[1]), int(key.split("_")[2])): value
            for key, value in input_dict.items()
            if "demand_" in key and value >= 0 and "flow" not in key
        }

        return {
            "num_nodes": num_nodes,
            "edges": edges,
            "demands": demands,
            "input_dict": input_dict
        }

    def compute_optimal_value(self, args_dict):
        self.num_compute_optimal_value_called += 1

        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        demands = args_dict["demands"]

        optimal_start = time.time()
        optimal_sol = optimal_TE(
            num_nodes, edges, demands, self.possible_demands, self.all_paths
        )
        optimal_end = time.time()
        if ENABLE_PRINT:
            print(f"optimal_TE call took {optimal_end - optimal_start:.4f} seconds")

        gradient_start = time.time()
        gradient = get_TE_lagrangian_gradient_wrapper(
            num_nodes,
            edges,
            optimal_sol["all_vars"],
            self.all_paths,
            use_optimized=True,
        )
        gradient_end = time.time()
        if ENABLE_PRINT:
            print(
                f"get_TE_lagrangian_gradient_wrapper call took {gradient_end - gradient_start:.4f} seconds"
            )

        return {
            "optimal_value": optimal_sol["optimal_total_flow"],
            "flow_values": optimal_sol["flow_values"],
            "flowpath_values": optimal_sol["flowpath_values"],
            "gradient": gradient,
            "all_vars": optimal_sol["all_vars"],
        }

    def compute_heuristic_value(self, args_dict):
        # print(f"compute_heuristic_value called")
        self.num_compute_heuristic_value_called += 1

        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        demands = args_dict["demands"]
        num_random = self.problem_config.get("num_random", 10)
        heuristic_name = self.problem_config["heuristic_name"]

        heuristic_start = time.time()
        if heuristic_name == "DemandPinning":
            heuristic_sol = demand_pinning_TE(
                num_nodes,
                edges,
                demands,
                self.problem_config["small_flow_cutoff"],
                possible_demands=self.possible_demands,
                given_all_paths=self.all_paths,
            )
        elif heuristic_name == "PoP":
            heuristic_sol = pop_TE_wrapper(
                num_nodes,
                edges,
                demands,
                self.partition_lists,
                possible_demands=self.possible_demands,
                given_all_paths=self.all_paths
            )
        elif heuristic_name == "LLM":
            heuristic_sol = LLM_TE(
                num_nodes,
                edges,
                demands,
                possible_demands=self.possible_demands,
                given_all_paths=self.all_paths,
            )
        elif heuristic_name == "DOTE":
            heuristic_sol = DOTE_wrapper(
                topology_name=self.problem_config["topology"],
                demands=demands)
        elif heuristic_name == "DOTE_C":
            heuristic_sol = DOTE_C_wrapper(
                topology_name=self.problem_config["topology"],
                demands=demands)

        heuristic_end = time.time()
        if ENABLE_PRINT:
            print(
                f"{heuristic_name} execution took {heuristic_end - heuristic_start:.4f} seconds"
            )

        return {
            "code_path_num": heuristic_sol["code_path_num"],
            "heuristic_value": heuristic_sol["heuristic_value"],
            "all_vars": heuristic_sol["all_vars"],
        }

    def compute_lagrangian_gradient(self, args_dict):
        # print(f"🎯 Starting compute_lagrangian_gradient")

        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        input_dict = args_dict["input_dict"]

        gradient_start = time.time()
        demand_gradient = get_TE_lagrangian_gradient_wrapper(
            num_nodes, edges, input_dict, self.all_paths, use_optimized=True
        )
        gradient_end = time.time()
        if ENABLE_PRINT:
            print(
                f"get_TE_lagrangian_gradient_wrapper call took {gradient_end - gradient_start:.4f} seconds"
            )

        # rest_gradient = compute_all_vars_derivative_over_demands(num_nodes, edges, args_dict["demands"])
        # print("Demand gradient: ", demand_gradient)
        # print("Rest gradient: ", rest_gradient)
        # for key in rest_gradient:
        #     demand_gradient[key] = rest_gradient[key]

        return demand_gradient

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        # print(f"🎯 Starting compute_lagrangian_value")

        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        input_dict = args_dict["input_dict"]

        lagrangian_start = time.time()
        result = get_TE_lagrangian(num_nodes, edges, input_dict, give_relaxed_gap)
        lagrangian_end = time.time()
        if ENABLE_PRINT:
            print(
                f"get_TE_lagrangian call took {lagrangian_end - lagrangian_start:.4f} seconds"
            )

        return result

    def compute_relaxed_optimal_value(self, args_dict):
        # print(f"🎯 Starting compute_relaxed_optimal_value")

        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        demands = args_dict["demands"]

        optimal_start = time.time()
        optimal_sol = optimal_TE(
            num_nodes, edges, demands, self.possible_demands, self.all_paths
        )
        optimal_end = time.time()
        if ENABLE_PRINT:
            print(f"optimal_TE call took {optimal_end - optimal_start:.4f} seconds")

        relaxed_all_vars = optimal_sol["all_vars"]

        return {
            "relaxed_optimal_value": optimal_sol["optimal_total_flow"],
            "relaxed_all_vars": relaxed_all_vars,
        }

    def create_random_partition(self, demands, seed, num_partitions):
        # create n random partitions and return the average of the solutions
        random.seed(seed)
        demands_list = demands.copy()
        # Shuffle the demands list randomly
        random.shuffle(demands_list)
        # Split the shuffled list into num_partitions random disjoint partitions
        partition_size = len(demands_list) // num_partitions
        remainder = len(demands_list) % num_partitions
        
        partitions = []
        start_idx = 0
        for i in range(num_partitions):
            # Add one extra element to the first 'remainder' partitions
            current_size = partition_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            partitions.append(demands_list[start_idx:end_idx])
            start_idx = end_idx
        
        return partitions

    def get_common_header(self, args_dict):
        num_nodes = args_dict["num_nodes"]

        program = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <klee/klee.h>

        #define MAX_NODES {num_nodes}
        #define MAX_EDGES MAX_NODES * MAX_NODES
        #define INF 1000000
        """

        program += """
        typedef struct {
            int from;
            int to;
            int capacity;
        } Edge;

        typedef struct {
            int id;
        } Node;

        typedef struct {
            Node nodes[MAX_NODES];
            Edge edges[MAX_EDGES];
            int node_count;
            int edge_count;
        } Topology;

        typedef struct {
            int from;
            int to;
            int demand;
        } Demand;

        void add_node(Topology* topology, int id) {
            topology->nodes[topology->node_count].id = id;
            topology->node_count++;
        }

        void add_edge(Topology* topology, int from, int to, int capacity) {
            topology->edges[topology->edge_count].from = from;
            topology->edges[topology->edge_count].to = to;
            topology->edges[topology->edge_count].capacity = capacity;
            topology->edge_count++;
        }

        int find_node_index(Topology* topology, int id) {
            for (int i = 0; i < topology->node_count; i++) {
                if (topology->nodes[i].id == id) {
                    return i;
                }
            }
            return -1;
        }

        void floyd_warshall(int dist[MAX_NODES][MAX_NODES], int next[MAX_NODES][MAX_NODES], Topology* topology) {
            for (int i = 0; i < topology->node_count; i++) {
                for (int j = 0; j < topology->node_count; j++) {
                    dist[i][j] = (i == j) ? 0 : INF;
                    next[i][j] = -1;
                }
            }
            for (int i = 0; i < topology->edge_count; i++) {
                int u = find_node_index(topology, topology->edges[i].from);
                int v = find_node_index(topology, topology->edges[i].to);
                dist[u][v] = 1; // Assuming unit weights for shortest path
                next[u][v] = v;
            }
            for (int k = 0; k < topology->node_count; k++) {
                for (int i = 0; i < topology->node_count; i++) {
                    for (int j = 0; j < topology->node_count; j++) {
                        if (dist[i][k] + dist[k][j] < dist[i][j]) {
                            dist[i][j] = dist[i][k] + dist[k][j];
                            next[i][j] = next[i][k];
                        }
                    }
                }
            }
        }

        void get_path_edges(int next[MAX_NODES][MAX_NODES], int u, int v, Topology* topology, Edge path[MAX_NODES]) {
            int i = 0;
            while (u != v) {
                int next_node = next[u][v];
                for (int j = 0; j < topology->edge_count; j++) {
                    if (topology->edges[j].from == u && topology->edges[j].to == next_node) {
                        path[i] = topology->edges[j];
                        i++;
                        break;
                    }
                }
                u = next_node;
            }
        }

        int get_edge_capacity(Topology* topology, int u, int v) {
            for (int i = 0; i < topology->edge_count; i++) {
                if (topology->edges[i].from == u && topology->edges[i].to == v) {
                    return topology->edges[i].capacity;
                }
            }
            return 0;
        }

        int get_path_min_capacity(Topology* topology, Edge path[MAX_NODES], int flow[MAX_NODES][MAX_NODES]) {
            int min_capacity = INF;
            for (int i = 0; i < MAX_NODES; i++) {
                if (path[i].from == -1) {
                    break;
                }
                int node_index_1 = path[i].from;
                int node_index_2 = path[i].to;
                int capacity = get_edge_capacity(topology, node_index_1, node_index_2);
                int consumed_flow = flow[node_index_1][node_index_2];
                int remaining_capacity = capacity - consumed_flow;
                if (remaining_capacity < min_capacity) {
                    min_capacity = remaining_capacity;
                }
            }
            return min_capacity;
        }
        """
        if self.problem_config["heuristic_name"] == "PoP":
            program += """
                // Function to scale edge capacities for each partition
                void scale_edge_capacities(Topology* topology, int num_partitions, Topology partition_topologies[]) {
                    for (int p = 0; p < num_partitions; p++) {
                        partition_topologies[p] = *topology;
                        for (int i = 0; i < partition_topologies[p].edge_count; i++) {
                            partition_topologies[p].edges[i].capacity /= num_partitions;
                        }
                    }
                }

                // Function to calculate flow for a single partition
                void calculate_partition_flow(Topology* partition_topology, Demand* partition_demands, int partition_size, int flow[MAX_NODES][MAX_NODES]) {
                    // Allocate large arrays on heap to avoid stack overflow
                    int (*dist)[MAX_NODES] = malloc(MAX_NODES * sizeof(*dist));
                    int (*next)[MAX_NODES] = malloc(MAX_NODES * sizeof(*next));
                    if (!dist || !next) {
                        printf("Memory allocation failed in calculate_partition_flow\\n");
                        return;
                    }

                    floyd_warshall(dist, next, partition_topology);
                    // Reset the flow matrix
                    memset(flow, 0, sizeof(int) * MAX_NODES * MAX_NODES);

                    for (int i = 0; i < partition_size; i++) {
                        int from = partition_demands[i].from;
                        int to = partition_demands[i].to;
                        int demand = partition_demands[i].demand;

                        // Use the demand to find a feasible path (simple path selection example)
                        Edge *path = calloc(MAX_NODES, sizeof(Edge));
                        if (!path) {
                            printf("Memory allocation failed for path in calculate_partition_flow\\n");
                            free(dist);
                            free(next);
                            return;
                        }
                        // Initialize path with sentinel values
                        for (int j = 0; j < MAX_NODES; j++) {
                            path[j] = (Edge){-1, -1, 0};
                        }

                        get_path_edges(next, from, to, partition_topology, path);
                        int min_capacity = get_path_min_capacity(partition_topology, path, flow);

                        // Clean up path array
                        free(path);
                    }

                    // Clean up allocated memory
                    free(dist);
                    free(next);
                }
            """
        return program

    def get_demand_pinning_program(self, demand_count):
        program = f"""
            // Allocate large arrays on heap instead of stack to avoid stack overflow
            int (*dist)[MAX_NODES] = malloc(MAX_NODES * sizeof(*dist));
            int (*next)[MAX_NODES] = malloc(MAX_NODES * sizeof(*next));
            int (*remaining_demand)[MAX_NODES] = calloc(MAX_NODES, sizeof(*remaining_demand));
            int (*original_demand)[MAX_NODES] = calloc(MAX_NODES, sizeof(*original_demand));

            if (!dist || !next || !remaining_demand || !original_demand) {{
                printf("Memory allocation failed\\n");
                return 1;
            }}

            floyd_warshall(dist, next, &topology);
            int demand_count = {demand_count};
            for (int i = 0; i < demand_count; i++) {{
                int u = demands[i].from;
                int v = demands[i].to;
                remaining_demand[u][v] = demands[i].demand;
                original_demand[u][v] = demands[i].demand;
            }}
            int T_d = {int(self.problem_config["small_flow_cutoff"])};
            // Allocate flow array on heap to avoid stack overflow
            int (*flow)[MAX_NODES] = calloc(MAX_NODES, sizeof(*flow));
            if (!flow) {{
                printf("Memory allocation failed for flow array\\n");
                // Clean up previously allocated memory
                free(dist);
                free(next);
                free(remaining_demand);
                free(original_demand);
                return 1;
            }}

            for (int i = 0; i < demand_count; i++) {{
                int u = demands[i].from;
                int v = demands[i].to;
                if (demands[i].demand > 0) {{
                    if (demands[i].demand <= T_d) {{
                        // Allocate shortest_path array on heap to avoid stack overflow
                        Edge *shortest_path = calloc(MAX_NODES, sizeof(Edge));
                        if (!shortest_path) {{
                            printf("Memory allocation failed for shortest_path\\n");
                            // Clean up previously allocated memory
                            free(dist);
                            free(next);
                            free(remaining_demand);
                            free(original_demand);
                            free(flow);
                            return 1;
                        }}
                        get_path_edges(next, u, v, &topology, shortest_path);
                        int min_remaining_capacity = get_path_min_capacity(&topology, shortest_path, flow);
                        if (min_remaining_capacity >= demands[i].demand) {{
                            flow[u][v] = demands[i].demand;
                            remaining_demand[u][v] -= demands[i].demand;
                        }} else {{
                            flow[u][v] = min_remaining_capacity;
                            remaining_demand[u][v] -= min_remaining_capacity;
                        }}
                        // Clean up shortest_path array
                        free(shortest_path);
                    }}
                }}
            }}

            // Clean up allocated memory
            free(dist);
            free(next);
            free(remaining_demand);
            free(original_demand);
            free(flow);

            return 0;
        }}
        """
        return program

    def get_PoP_program(self, demand_count_dict):
        program = f"int num_partitions = {self.num_partitions};"
        
        # Declare all partition arrays first
        for partition_list_index, partition_list in enumerate(self.partition_lists):
            for partition_index, partition_pairs in enumerate(partition_list):
                # Filter out demand pairs that don't exist in demand_count_dict
                valid_pairs = [pair for pair in partition_pairs if pair in demand_count_dict]
                if valid_pairs:  # Only create partition if it has valid demand pairs
                    demands_string = ", ".join(
                        [f"demands[{demand_count_dict[pair]}]" for pair in valid_pairs]
                    )
                    program += f"""
                    Demand partition_{partition_index}_{partition_list_index}[{len(valid_pairs)}] = {{{demands_string}}};
                    """
        
        # Declare all partition sizes arrays
        for partition_list_index, partition_list in enumerate(self.partition_lists):
            partition_sizes = {}
            for partition_index, partition_pairs in enumerate(partition_list):
                # Count only valid demand pairs
                valid_pairs = [pair for pair in partition_pairs if pair in demand_count_dict]
                partition_sizes[partition_index] = len(valid_pairs)
            
            partition_sizes_string = ", ".join(
                [
                    str(partition_sizes[partition_index])
                    for partition_index in range(self.num_partitions)
                ]
            )
            program += f"""
            int partition_sizes_{partition_list_index}[] = {{{partition_sizes_string}}};
            """
        
        # Declare all topology and flow arrays
        for partition_list_index in range(len(self.partition_lists)):
            program += f"""
            // Allocate topology and flow arrays on heap to avoid stack overflow
            Topology *partition_topologies_{partition_list_index} = malloc(num_partitions * sizeof(Topology));
            if (!partition_topologies_{partition_list_index}) {{
                printf("Memory allocation failed for partition_topologies_{partition_list_index}\\n");
                return 1;
            }}
            scale_edge_capacities(&topology, num_partitions, partition_topologies_{partition_list_index});
            int (*flow_{partition_list_index})[MAX_NODES] = calloc(MAX_NODES, sizeof(*flow_{partition_list_index}));
            if (!flow_{partition_list_index}) {{
                printf("Memory allocation failed for flow_{partition_list_index}\\n");
                free(partition_topologies_{partition_list_index});
                return 1;
            }}
            """
        
        # Flatten the loop - process each partition list sequentially
        for partition_list_index in range(len(self.partition_lists)):
            program += f"""
            // Process partition list {partition_list_index}
            for (int p = 0; p < num_partitions; p++) {{
                Demand* current_partition_demands = NULL;
                switch (p) {{"""
            
            # Add switch cases for each partition
            for partition_index in range(self.num_partitions):
                program += f"""
                    case {partition_index}:
                        current_partition_demands = partition_{partition_index}_{partition_list_index};
                        break;"""
            
            program += f"""
                }}
                calculate_partition_flow(&partition_topologies_{partition_list_index}[p], current_partition_demands, partition_sizes_{partition_list_index}[p], flow_{partition_list_index});
            }}
            """

        # Add cleanup code for all allocated memory
        for partition_list_index in range(len(self.partition_lists)):
            program += f"""
            // Clean up allocated memory for partition list {partition_list_index}
            free(partition_topologies_{partition_list_index});
            free(flow_{partition_list_index});
            """

        program += """
        return 0;
        }
        """

        return program

    def get_LLM_program(self, demand_count_dict):
        demand_count = len(demand_count_dict)
        critical_threshold = int(demand_count * 0.2)  # Top 20% as critical

        program = f"""
            // LLM Traffic Engineering Algorithm Implementation
            // Phase 1: Sort demands and identify critical demands (top 20%)
            // Phase 2: Route critical demands first on full topology
            // Phase 3: Route non-critical demands on adjusted topology

            // Allocate large arrays on heap to avoid stack overflow
            int (*dist)[MAX_NODES] = malloc(MAX_NODES * sizeof(*dist));
            int (*next)[MAX_NODES] = malloc(MAX_NODES * sizeof(*next));
            int (*remaining_capacity)[MAX_NODES] = malloc(MAX_NODES * sizeof(*remaining_capacity));
            int (*critical_flow)[MAX_NODES] = calloc(MAX_NODES, sizeof(*critical_flow));
            int (*non_critical_flow)[MAX_NODES] = calloc(MAX_NODES, sizeof(*non_critical_flow));

            if (!dist || !next || !remaining_capacity || !critical_flow || !non_critical_flow) {{
                printf("Memory allocation failed\\n");
                return 1;
            }}

            floyd_warshall(dist, next, &topology);

            // Initialize remaining capacity with original edge capacities
            for (int i = 0; i < topology.node_count; i++) {{
                for (int j = 0; j < topology.node_count; j++) {{
                    remaining_capacity[i][j] = get_edge_capacity(&topology, i, j);
                }}
            }}

            int demand_count = {demand_count};
            int critical_threshold = {critical_threshold};

            // Create sorted indices array for sorting demands by value
            int *demand_indices = malloc(demand_count * sizeof(int));
            if (!demand_indices) {{
                printf("Memory allocation failed for demand_indices\\n");
                free(dist);
                free(next);
                free(remaining_capacity);
                free(critical_flow);
                free(non_critical_flow);
                return 1;
            }}

            // Initialize indices
            for (int i = 0; i < demand_count; i++) {{
                demand_indices[i] = i;
            }}

            // Sort demands by value (descending) using bubble sort for simplicity
            for (int i = 0; i < demand_count - 1; i++) {{
                for (int j = 0; j < demand_count - i - 1; j++) {{
                    if (demands[demand_indices[j]].demand < demands[demand_indices[j + 1]].demand) {{
                        int temp = demand_indices[j];
                        demand_indices[j] = demand_indices[j + 1];
                        demand_indices[j + 1] = temp;
                    }}
                }}
            }}

            // Phase 1: Route critical demands (top 20%) first
            for (int i = 0; i < critical_threshold && i < demand_count; i++) {{
                int demand_idx = demand_indices[i];
                int from = demands[demand_idx].from;
                int to = demands[demand_idx].to;
                int demand_value = demands[demand_idx].demand;

                if (demand_value > 0) {{
                    // Allocate path array for this critical demand
                    Edge *critical_path = calloc(MAX_NODES, sizeof(Edge));
                    if (!critical_path) {{
                        printf("Memory allocation failed for critical_path\\n");
                        free(demand_indices);
                        free(dist);
                        free(next);
                        free(remaining_capacity);
                        free(critical_flow);
                        free(non_critical_flow);
                        return 1;
                    }}

                    // Get shortest path for critical demand using original topology
                    get_path_edges(next, from, to, &topology, critical_path);
                    int min_capacity = get_path_min_capacity(&topology, critical_path, critical_flow);

                    // Route as much flow as possible for this critical demand
                    int flow_to_route = (min_capacity < demand_value) ? min_capacity : demand_value;
                    if (flow_to_route > 0) {{
                        critical_flow[from][to] += flow_to_route;

                        // Update remaining capacities along the path
                        for (int edge_idx = 0; edge_idx < MAX_NODES; edge_idx++) {{
                            if (critical_path[edge_idx].from == -1) break;
                            int u = critical_path[edge_idx].from;
                            int v = critical_path[edge_idx].to;
                            remaining_capacity[u][v] -= flow_to_route;
                            if (remaining_capacity[u][v] < 0) remaining_capacity[u][v] = 0;
                        }}
                    }}

                    free(critical_path);
                }}
            }}

            // Phase 2: Route non-critical demands on adjusted topology
            for (int i = critical_threshold; i < demand_count; i++) {{
                int demand_idx = demand_indices[i];
                int from = demands[demand_idx].from;
                int to = demands[demand_idx].to;
                int demand_value = demands[demand_idx].demand;

                if (demand_value > 0) {{
                    // Allocate path array for this non-critical demand
                    Edge *non_critical_path = calloc(MAX_NODES, sizeof(Edge));
                    if (!non_critical_path) {{
                        printf("Memory allocation failed for non_critical_path\\n");
                        free(demand_indices);
                        free(dist);
                        free(next);
                        free(remaining_capacity);
                        free(critical_flow);
                        free(non_critical_flow);
                        return 1;
                    }}

                    // Create adjusted topology with remaining capacities
                    Topology adjusted_topology = topology;
                    for (int edge_idx = 0; edge_idx < adjusted_topology.edge_count; edge_idx++) {{
                        int u = adjusted_topology.edges[edge_idx].from;
                        int v = adjusted_topology.edges[edge_idx].to;
                        adjusted_topology.edges[edge_idx].capacity = remaining_capacity[u][v];
                    }}

                    // Get shortest path for non-critical demand using adjusted topology
                    get_path_edges(next, from, to, &adjusted_topology, non_critical_path);
                    int min_remaining_capacity = get_path_min_capacity(&adjusted_topology, non_critical_path, non_critical_flow);

                    // Route as much flow as possible for this non-critical demand
                    int flow_to_route = (min_remaining_capacity < demand_value) ? min_remaining_capacity : demand_value;
                    if (flow_to_route > 0) {{
                        non_critical_flow[from][to] += flow_to_route;

                        // Update remaining capacities along the path
                        for (int edge_idx = 0; edge_idx < MAX_NODES; edge_idx++) {{
                            if (non_critical_path[edge_idx].from == -1) break;
                            int u = non_critical_path[edge_idx].from;
                            int v = non_critical_path[edge_idx].to;
                            remaining_capacity[u][v] -= flow_to_route;
                            if (remaining_capacity[u][v] < 0) remaining_capacity[u][v] = 0;
                        }}
                    }}

                    free(non_critical_path);
                }}
            }}

            // Clean up allocated memory
            free(demand_indices);
            free(dist);
            free(next);
            free(remaining_capacity);
            free(critical_flow);
            free(non_critical_flow);

            return 0;
        }}
        """
        return program

    def get_DOTE_program(self, demand_count_dict):
        """Generate C program for DOTE_C with actual neural network implementation"""
        demand_count = len(demand_count_dict)
        num_nodes = self.problem_config["num_nodes"]
        
        # Try to load trained weights
        weights_data = self._load_dote_weights()
        
        if weights_data is None:
            # Fallback to simplified heuristic if weights not available
            return self._get_dote_simplified_program(demand_count)
        
        program = f"""
            // DOTE_C Traffic Engineering Algorithm Implementation
            // Using actual trained neural network weights from PyTorch model
            
            #include "dote_weights.h"
            
            // Neural network activation functions
            double leaky_relu(double x) {{
                return x > 0 ? x : 0.01 * x;
            }}
            
            double elu(double x, double alpha) {{
                return x > 0 ? x : alpha * (exp(x) - 1);
            }}
            
            // Neural network forward pass
            void dote_neural_network_forward(double* input, double* output, int input_size, int output_size) {{
                double hidden1[DOTE_HIDDEN_SIZE];
                double hidden2[DOTE_HIDDEN_SIZE];
                double hidden3[DOTE_HIDDEN_SIZE];
                double hidden4[DOTE_HIDDEN_SIZE];
                
                // First hidden layer
                for (int i = 0; i < DOTE_HIDDEN_SIZE; i++) {{
                    hidden1[i] = net_0_bias[i];
                    for (int j = 0; j < input_size; j++) {{
                        hidden1[i] += input[j] * net_0_weight[j][i];
                    }}
                    hidden1[i] = leaky_relu(hidden1[i]);
                }}
                
                // Second hidden layer
                for (int i = 0; i < DOTE_HIDDEN_SIZE; i++) {{
                    hidden2[i] = net_2_bias[i];
                    for (int j = 0; j < DOTE_HIDDEN_SIZE; j++) {{
                        hidden2[i] += hidden1[j] * net_2_weight[j][i];
                    }}
                    hidden2[i] = leaky_relu(hidden2[i]);
                }}
                
                // Third hidden layer
                for (int i = 0; i < DOTE_HIDDEN_SIZE; i++) {{
                    hidden3[i] = net_4_bias[i];
                    for (int j = 0; j < DOTE_HIDDEN_SIZE; j++) {{
                        hidden3[i] += hidden2[j] * net_4_weight[j][i];
                    }}
                    hidden3[i] = leaky_relu(hidden3[i]);
                }}
                
                // Fourth hidden layer
                for (int i = 0; i < DOTE_HIDDEN_SIZE; i++) {{
                    hidden4[i] = net_6_bias[i];
                    for (int j = 0; j < DOTE_HIDDEN_SIZE; j++) {{
                        hidden4[i] += hidden3[j] * net_6_weight[j][i];
                    }}
                    hidden4[i] = leaky_relu(hidden4[i]);
                }}
                
                // Output layer
                for (int i = 0; i < output_size; i++) {{
                    output[i] = net_8_bias[i];
                    for (int j = 0; j < DOTE_HIDDEN_SIZE; j++) {{
                        output[i] += hidden4[j] * net_8_weight[j][i];
                    }}
                    output[i] = elu(output[i], 0.1);
                }}
            }}
            
            // DOTE flow computation using neural network
            double compute_dote_flow_with_nn(Topology* topo, Demand* demands, int num_demands) {{
                // Flatten demands to input vector
                double input_vector[DOTE_INPUT_SIZE];
                memset(input_vector, 0, sizeof(input_vector));
                
                for (int i = 0; i < num_demands; i++) {{
                    int from = demands[i].from;
                    int to = demands[i].to;
                    if (from != to) {{
                        int idx = from * {num_nodes} + to;
                        if (idx < DOTE_INPUT_SIZE) {{
                            input_vector[idx] = demands[i].demand;
                        }}
                    }}
                }}
                
                // Neural network forward pass
                double path_weights[DOTE_OUTPUT_SIZE];
                dote_neural_network_forward(input_vector, path_weights, DOTE_INPUT_SIZE, DOTE_OUTPUT_SIZE);
                
                // Add small epsilon to avoid division by zero
                for (int i = 0; i < DOTE_OUTPUT_SIZE; i++) {{
                    path_weights[i] += 0.1;
                }}
                
                // Compute flow allocation using DOTE algorithm
                double total_served_flow = 0.0;
                
                // For each commodity (demand pair)
                for (int c = 0; c < num_demands; c++) {{
                    int from = demands[c].from;
                    int to = demands[c].to;
                    double demand_value = demands[c].demand;
                    
                    if (demand_value <= 0) continue;
                    
                    // Find paths for this commodity (simplified - using direct paths)
                    double commodity_total_weight = 0.0;
                    for (int p = 0; p < DOTE_OUTPUT_SIZE; p++) {{
                        // Simplified: assume each path corresponds to a direct edge
                        if (p == from * {num_nodes} + to) {{
                            commodity_total_weight += path_weights[p];
                        }}
                    }}
                    
                    if (commodity_total_weight > 0) {{
                        // Compute flow on each path
                        double allocated_flow = 0.0;
                        for (int p = 0; p < DOTE_OUTPUT_SIZE; p++) {{
                            if (p == from * {num_nodes} + to) {{
                                double path_flow = (path_weights[p] / commodity_total_weight) * demand_value;
                                
                                // Check capacity constraints
                                double min_capacity = get_edge_capacity(topo, from, to);
                                
                                double actual_flow = (path_flow < min_capacity) ? path_flow : min_capacity;
                                allocated_flow += actual_flow;
                            }}
                        }}
                        total_served_flow += allocated_flow;
                    }}
                }}
                
                return total_served_flow;
            }}
            
            // Main DOTE_C computation
            int demand_count = {demand_count};
            double total_served_flow = compute_dote_flow_with_nn(&topology, demands, demand_count);
            
            return 0;
        }}
        """
        return program
    
    def _load_dote_weights(self):
        """Load DOTE neural network weights from file"""
        weights_file = "dote_weights/dote_weights.json"
        if not os.path.exists(weights_file):
            print(f"DOTE weights file {weights_file} not found. Using simplified heuristic.")
            return None
        
        try:
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
            print(f"Loaded DOTE weights from {weights_file}")
            return weights_data
        except Exception as e:
            print(f"Error loading DOTE weights: {e}")
            return None
    
    def _get_dote_simplified_program(self, demand_count):
        """Fallback simplified DOTE program when weights are not available"""
        return f"""
            // DOTE_C Simplified Heuristic (fallback when trained weights not available)
            
            // Allocate large arrays on heap to avoid stack overflow
            int (*dist)[MAX_NODES] = malloc(MAX_NODES * sizeof(*dist));
            int (*next)[MAX_NODES] = malloc(MAX_NODES * sizeof(*next));
            double (*capacity_matrix)[MAX_NODES] = malloc(MAX_NODES * sizeof(*capacity_matrix));
            double (*remaining_capacity)[MAX_NODES] = malloc(MAX_NODES * sizeof(*remaining_capacity));

            if (!dist || !next || !capacity_matrix || !remaining_capacity) {{
                printf("Memory allocation failed\\n");
                return 1;
            }}

            floyd_warshall(dist, next, &topology);

            // Initialize capacity matrix with edge capacities
            for (int i = 0; i < topology.node_count; i++) {{
                for (int j = 0; j < topology.node_count; j++) {{
                    capacity_matrix[i][j] = get_edge_capacity(&topology, i, j);
                    remaining_capacity[i][j] = capacity_matrix[i][j];
                }}
            }}

            int demand_count = {demand_count};
            double total_served_flow = 0.0;

            // Simplified DOTE heuristic: route each demand optimally
            for (int i = 0; i < demand_count; i++) {{
                int from = demands[i].from;
                int to = demands[i].to;
                double demand_value = demands[i].demand;

                if (demand_value <= 0) continue;

                // Try direct path first
                double direct_capacity = remaining_capacity[from][to];
                double served_on_direct = (demand_value < direct_capacity) ? demand_value : direct_capacity;
                total_served_flow += served_on_direct;
                remaining_capacity[from][to] -= served_on_direct;

                double remaining_demand = demand_value - served_on_direct;

                // If direct path insufficient, try alternative paths
                if (remaining_demand > 0) {{
                    // Try routing through intermediate nodes
                    for (int intermediate = 0; intermediate < topology.node_count; intermediate++) {{
                        if (intermediate == from || intermediate == to) continue;

                        // Check capacity on path from->intermediate->to
                        double cap1 = remaining_capacity[from][intermediate];
                        double cap2 = remaining_capacity[intermediate][to];
                        double path_capacity = (cap1 < cap2) ? cap1 : cap2;

                        if (path_capacity > 0) {{
                            double additional_served = (remaining_demand < path_capacity) ? remaining_demand : path_capacity;
                            total_served_flow += additional_served;
                            remaining_capacity[from][intermediate] -= additional_served;
                            remaining_capacity[intermediate][to] -= additional_served;
                            remaining_demand -= additional_served;

                            if (remaining_demand <= 0) break;
                        }}
                    }}
                }}
            }}

            // Clean up allocated memory
            free(dist);
            free(next);
            free(capacity_matrix);
            free(remaining_capacity);

            return 0;
        }}
        """

    def generate_heuristic_program(
        self,
        program_type,
        list_of_input_paths_to_exclude=[],
        num_klee_inputs=None,
        path_to_assigned_fixed_points=None,
    ):
        num_nodes = self.problem_config["num_nodes"]
        max_flow = int(self.problem_config["max_flow"])
        capacity = int(self.problem_config["capacity"])
        file_fixed_points = None
        if path_to_assigned_fixed_points:
            with open(path_to_assigned_fixed_points, "r") as f:
                file_fixed_points = json.load(f)
            selected_klee_inputs = [
                name
                for name in self.all_klee_var_names
                if name not in file_fixed_points
            ]
        else:
            if num_klee_inputs is not None:
                num_klee_inputs = min(num_klee_inputs, self.num_total_klee_inputs)
                selected_klee_inputs = random.sample(
                    self.all_klee_var_names, num_klee_inputs
                )
            else:
                selected_klee_inputs = self.all_klee_var_names
            print(
                f"Selected klee inputs: {selected_klee_inputs} from {self.num_total_klee_inputs}"
            )

        fixed_points = {}
        program = self.get_common_header({"num_nodes": num_nodes})
        program += """
        int main() {
            Topology topology = {0};\n"""

        for node in range(num_nodes):
            program += f"           add_node(&topology, {node});\n"

        for from_ in range(num_nodes):
            for to_ in range(num_nodes):
                if from_ != to_ and f"edge_{from_}_{to_}" in self.problem_config:
                    edge_capacity = int(self.problem_config[f"edge_{from_}_{to_}"])
                    program += f"          add_edge(&topology, {from_}, {to_}, {edge_capacity});\n"

        program += """
            Demand demands[MAX_NODES * MAX_NODES];
        """
        demand_count = 0
        demand_count_dict = {}
        for from_, to_ in self.possible_demands:
            demand_key = f"demand_{from_}_{to_}"
            paths = self.all_paths[(from_, to_)]
            # sort the paths by length, and then name
            paths.sort(key=lambda x: (len(x), "_".join(x)))
            shortest_path = paths[0]
            if len(shortest_path) > 2:
                prefered_value = int(self.problem_config["small_flow_cutoff"])
            else:
                prefered_value = capacity
            if demand_key in selected_klee_inputs:
                program += f"""
                int {demand_key};
                klee_make_symbolic(&{demand_key}, sizeof({demand_key}), "{demand_key}");
                klee_assume({demand_key} >= {prefered_value} & {demand_key} <= {max_flow});
                """
                if not DISABLE_CUSTOMIZATION:
                    program += f"""
                    klee_prefer_cex(&demand_{from_}_{to_}, demand_{from_}_{to_} == {prefered_value});
                    """
                program += f"""
                demands[{demand_count}] = (Demand){{.from = {from_}, .to = {to_}, .demand = demand_{from_}_{to_}}};
                """
            else:
                if file_fixed_points is not None:
                    value = file_fixed_points[demand_key]
                else:
                    value = prefered_value  # random.choice([int(self.problem_config["small_flow_cutoff"]), max_flow])
                fixed_points[demand_key] = value
                program += f"""
                int {demand_key} = {value};
                demands[{demand_count}] = (Demand){{.from = {from_}, .to = {to_}, .demand = {value}}};
                """
            demand_count_dict[(from_, to_)] = demand_count
            demand_count += 1

        if not DISABLE_CUSTOMIZATION:
            non_zero_x = []
            for from_, to_ in self.possible_demands:
                program += f"""
                int x_{from_}_{to_} = 0;
                """
                edge_demands = get_all_demands_for_edge(self.all_paths, (from_, to_))
                if len(edge_demands) > 1:
                    non_zero_x.append((from_, to_))
                    for demand in edge_demands:
                        program += f"x_{from_}_{to_} += demand_{demand[0]}_{demand[1]};"

            # assume that at least one edge has a demand greater than {max_flow}
            assumption_string = ""
            for from_, to_ in self.possible_demands:
                if (from_, to_) in non_zero_x:
                    assumption_string += f"x_{from_}_{to_} > {capacity} | "
            assumption_string = assumption_string[:-3]
            program += f"klee_assume({assumption_string});"
            # program += f"klee_assume(x_0_1 > {capacity} & x_1_3 > {capacity});"

        for input_path in list_of_input_paths_to_exclude:
            # read the json file
            with open(input_path, "r") as f:
                test_cases = json.load(f)
            for _, test in test_cases.items():
                excluding_string = ""
                for key, value in test.items():
                    translated_key = key.strip("'").strip()
                    if key in selected_klee_inputs:
                        excluding_string += f"{translated_key} != {value} | "
                excluding_string = excluding_string[:-3]

                program += f"""
                klee_assume({excluding_string});
                """
        if self.problem_config["heuristic_name"] == "DemandPinning":
            program += self.get_demand_pinning_program(demand_count)
        elif self.problem_config["heuristic_name"] == "PoP":
            program += self.get_PoP_program(demand_count_dict)
        elif self.problem_config["heuristic_name"] == "LLM":
            program += self.get_LLM_program(demand_count_dict)
        elif self.problem_config["heuristic_name"] == "DOTE":
            program += self.get_DOTE_program(demand_count_dict)
        return {"program": program, "fixed_points": fixed_points}


# Timing utilities for performance tracking
class TimingTracker:
    def __init__(self):
        self.timings = {}
        self.start_times = {}

    def start(self, operation_name):
        self.start_times[operation_name] = time.time()

    def end(self, operation_name):
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(duration)
            print(f"{operation_name} took {duration:.4f} seconds")
            del self.start_times[operation_name]

    def get_summary(self):
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
        return summary

    def print_summary(self):
        summary = self.get_summary()
        print("\n=== TIMING SUMMARY ===")
        for operation, stats in summary.items():
            print(f"{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total time: {stats['total_time']:.4f} seconds")
            print(f"  Average time: {stats['avg_time']:.4f} seconds")
            print(f"  Min time: {stats['min_time']:.4f} seconds")
            print(f"  Max time: {stats['max_time']:.4f} seconds")
        print("=====================\n")


# Global timing tracker instance
timing_tracker = TimingTracker()
def optimal_TE_standalone(
    num_nodes, edges, demands, possible_demands=None, given_all_paths=None
):
    """
    Standalone version of optimal_TE for use by demand_pinning_TE and pop_TE functions.
    This is the original implementation without the optimization.
    """
    start_time = time.time()
    # print(f"Starting optimal_TE_standalone with {num_nodes} nodes, {len(demands)} demands")

    # Create the solver
    solver_start = time.time()
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise Exception("Solver could not be created!")
    solver_end = time.time()
    if ENABLE_PRINT:
        print(f"Solver creation took {solver_end - solver_start:.4f} seconds")

    all_vars = {}
    flow_on_path_vars = {}
    all_paths = given_all_paths if given_all_paths is not None else {}
    nodes = range(num_nodes)

    # Variable creation phase
    var_creation_start = time.time()
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                if (from_, to_) in all_paths:
                    paths = all_paths[(from_, to_)]
                else:
                    paths = find_all_paths(
                        {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                    )
                    all_paths[(from_, to_)] = paths
                for path in paths:
                    path_string = "_".join(path)
                    var_name = f"aux_flowpath_{path_string}"
                    min_capacity_on_path = get_min_capacity_on_path(path, edges)
                    flow_on_path_vars[var_name] = solver.NumVar(
                        0, min_capacity_on_path, var_name
                    )
    var_creation_end = time.time()
    # print(f"Variable creation took {var_creation_end - var_creation_start:.4f} seconds")

    # Create flow variables for
    if possible_demands is None:
        possible_demands, _ = find_possible_demands({"num_nodes": num_nodes, **edges})
    for from_, to_ in demands:
        if (from_, to_) in possible_demands:
            all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]
    flow_pair_vars = {}
    flowpath_constraints = {}
    flow_demand_constraints = {}

    # Constraint creation phase
    constraint_creation_start = time.time()
    for from_, to_ in possible_demands:
        if (from_, to_) in demands:
            flow_pair_vars[f"aux_flow_{from_}_{to_}"] = solver.NumVar(
                0, 2 * demands[(from_, to_)], f"aux_flow_{from_}_{to_}"
            )
            flow_demand_constraints[(from_, to_)] = solver.Add(
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] <= demands[(from_, to_)]
            )
            # the aux_flow_{from_}_{to_} is equal to the sum of the flow on all paths from from_ to to_
            paths = all_paths[(from_, to_)]
            path_names = ["_".join(path) for path in paths]
            sum_flow_on_paths = solver.Sum(
                flow_on_path_vars[f"aux_flowpath_{path_name}"]
                for path_name in path_names
            )
            constraint = solver.Add(
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] == sum_flow_on_paths
            )
            flowpath_constraints[(from_, to_)] = constraint

    capacity_constraints = {}
    # Add capacity constraints for each edge
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                edge = f"edge_{from_}_{to_}"
                if edge in edges:
                    capacity = edges[edge]
                    # find all the paths that use this edge
                    every_path = []
                    for key, value in all_paths.items():
                        every_path.extend("_".join(path) for path in value)
                    paths_with_this_edge = [
                        path for path in every_path if f"{from_}_{to_}" in path
                    ]
                    constraint = solver.Add(
                        solver.Sum(
                            flow_on_path_vars[f"aux_flowpath_{path}"]
                            for path in paths_with_this_edge
                        )
                        <= capacity
                    )
                    capacity_constraints[f"{from_}_{to_}"] = constraint
    constraint_creation_end = time.time()
    if ENABLE_PRINT:
        print(
            f"Constraint creation took {constraint_creation_end - constraint_creation_start:.4f} seconds"
        )

    # Objective: Maximize the total flow
    objective_start = time.time()
    total_flow = solver.Sum(flow_pair_vars.values())
    solver.Maximize(total_flow)
    objective_end = time.time()
    if ENABLE_PRINT:
        print(f"Objective setup took {objective_end - objective_start:.4f} seconds")

    # Solve the problem
    solve_start = time.time()
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("Solver did not find an optimal solution!")
    solve_end = time.time()
    if ENABLE_PRINT:
        print(f"Solver execution took {solve_end - solve_start:.4f} seconds")

    # Extract results
    extraction_start = time.time()
    flow_values = {
        edge: flow_pair_vars[edge].solution_value() for edge in flow_pair_vars
    }
    flowpath_values = {
        path: flow_on_path_vars[path].solution_value() for path in flow_on_path_vars
    }
    for key in flow_pair_vars:
        all_vars[key] = flow_pair_vars[key].solution_value()

    for key in flow_on_path_vars:
        all_vars[key] = flow_on_path_vars[key].solution_value()

    for key in flow_demand_constraints:
        all_vars[f"lambda_flow_demand_{key[0]}_{key[1]}"] = flow_demand_constraints[
            key
        ].dual_value()

    for key in flowpath_constraints:
        all_vars[f"lambda_flowpath_{key[0]}_{key[1]}"] = flowpath_constraints[
            key
        ].dual_value()

    for key in capacity_constraints:
        all_vars[f"lambda_capacity_{key}"] = capacity_constraints[key].dual_value()

    total_flow_value = total_flow.solution_value()
    extraction_end = time.time()
    if ENABLE_PRINT:
        print(f"Result extraction took {extraction_end - extraction_start:.4f} seconds")

    end_time = time.time()
    if ENABLE_PRINT:
        print(
        f"Total optimal_TE_standalone execution took {end_time - start_time:.4f} seconds"
    )

    return {
        "flow_values": flow_values,
        "flowpath_values": flowpath_values,
        "optimal_total_flow": total_flow_value,
        "all_vars": all_vars,
    }

def get_TE_lagrangian_gradient_wrapper(
    num_nodes, edges, input_dict, given_all_paths=None, use_optimized=True
):
    """
    Wrapper function that can choose between optimized and original gradient implementations.
    """
    if use_optimized:
        try:
            return get_TE_lagrangian_gradient_optimized(
                num_nodes, edges, input_dict, given_all_paths
            )
        except Exception as e:
            print(
                f"Optimized gradient version failed: {e}. Falling back to original implementation."
            )
            return get_TE_lagrangian_gradient(
                num_nodes, edges, input_dict, given_all_paths
            )
    else:
        return get_TE_lagrangian_gradient(num_nodes, edges, input_dict, given_all_paths)
