import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import ijson
from pathlib import Path
import random

from gradient_ascent import (
    fit_gaussian_process,
    flatten_dict,
    estimate_gradient_with_gp,
    update_with_closest_angle_to_gradient,
    predict_gaussian_process,
    RED,
    RESET,
)


LAMBDA_MAX_VALUE = 1000000

def MetaEase_specific_parameters() -> list[str]:
    """
    Returns the list of MetaEase-specific configuration parameter names.
    These parameters control the core behavior of the MetaEase search algorithm.
    """
    return [
            # Length of blocks used for partitioning the search space
            "block_length",
            # Maximum number of KLEE-generated inputs to consider for scalability
            "max_num_scalable_klee_inputs",
            # Whether to use gap values when filtering candidate inputs
            "use_gaps_in_filtering",
            # Whether to remove inputs that have zero gap from consideration
            "remove_zero_gap_inputs",
            # Learning rate for gradient ascent optimization
            "gradient_ascent_rate",
            # Maximum time (in seconds) allowed per KLEE point evaluation
            "max_time_per_klee_point",
            # Number of rounds with non-zero improvements before stopping
            "num_non_zero_rounds",
            # Maximum number of KLEE points to process per iteration
            "max_num_klee_points_per_iteration",
            ]

def get_problem_description(args) -> dict:
    """
    Returns a dictionary containing the problem configuration based on command-line arguments.

    The problem name (args.problem) determines the problem type and heuristic settings.

    HOW TO ADD A NEW HEURISTIC:
    ---------------------------
    1. Add a new elif branch in the appropriate problem type section (e.g., under TE, arrow, vbp, etc.)
    2. Set the heuristic_name key to your new heuristic's name
    3. Configure the relevant parameters for your heuristic (disable_klee, num_samples, etc.)
    4. Implement the heuristic logic in the corresponding problem module

    HOW TO ADD A NEW PROBLEM CLASS:
    -------------------------------
    1. Add a new elif branch at the end of this function checking for your problem prefix
       (e.g., elif args.problem.startswith("my_new_problem"):)
    2. Define a problem_description dictionary with at minimum:
       - "problem_type": your problem type identifier
       - "heuristic_name": the heuristic to use
       - MetaEase parameters from MetaEase_specific_parameters()
    3. Create a new problem module in src/problems/ implementing:
       - The problem's objective function
       - Input generation logic
       - Any heuristic-specific methods

    Args:
        args: Command-line arguments containing at least 'problem' and 'method' attributes

    Returns:
        dict: Problem configuration dictionary with all necessary parameters
    """
    if args.problem.startswith("TE"):
        heuristic_name = args.problem.split("_")[-2]
        topology = args.problem.split("_")[-1]
        # Clean topology name by removing invisible Unicode characters
        topology = ''.join(char for char in topology if char.isprintable())
        use_MetaOpt_cluster = False
        cluster_path = None
        max_num_paths = float("inf")
        # TODO: Instead of having these based on the name of the topology, to allow for more general use-cases check the size of the topology and decide based on that.
        # It is surprising to me that we are reducing the samples for the bigger problem instances, in part because it seems that those would be the scenario where we would need more samples to do proper search.
        # I get you probably did that to get the experiment to run faster but that is not the right way.
        # TODO: another comment, to allow for more general runtimes, I would suggest having a method that calls the partitioning algorithm for a new topology that needs it based on the topology size. This is not clean enough.
        if topology in ["Cogentco", "Uninet2010"]:
            num_samples = 20
            max_num_paths = 4
            use_MetaOpt_cluster = True
            if topology == "Cogentco":
                cluster_path = f"../topologies/partition_log/Cogentco_10_fm_partitioning"
            elif topology == "Uninet2010":
                cluster_path = f"../topologies/partition_log/Uninet2010_8_fm_partitioning"
        else:
            num_samples = 50
        problem_description = {
            "problem_type": "TE",
            "block_length": 1.0,
            "max_num_scalable_klee_inputs": 16,
            "use_gaps_in_filtering": True,
            "remove_zero_gap_inputs": True,
            "gradient_ascent_rate": 1.0,
            "disable_gradient_ascent": False,
            # "max_time_per_klee_point": 1800,  # 30 minutes timeout for complex TE problems
            "heuristic_name": heuristic_name,
            "num_samples": num_samples,
            "topology": topology,
            "use_MetaOpt_cluster": use_MetaOpt_cluster,
            "cluster_path": cluster_path,
            "max_num_paths": max_num_paths,
        }
        # TODO: instead of hardcoded numeric inputs have these as parameters in a config file maybe?
        if heuristic_name == "DemandPinning":
            problem_description["disable_klee"] = False
            problem_description["num_random_seed_samples"] = 0
            problem_description["num_samples"] = 20
            problem_description["num_iterations"] = 1000
            problem_description["disable_gradient_ascent"] = True
            problem_description["disable_gaussian_process"] = True
            # problem_description["max_num_scalable_klee_inputs"] = 1000000000
            # problem_description["randomized_gradient_ascent"] = True
            # problem_description["num_vars_in_randomized_gradient_ascent"] = 1000000000
        # TODO: have documetnation on the different heuristics that you benchmarks you have and describe what they are.
        elif heuristic_name == "LLM":
            problem_description["disable_klee"] = False
            problem_description["num_random_seed_samples"] = 10
            problem_description["num_samples"] = 20
            problem_description["num_iterations"] = 1000
            problem_description["disable_gradient_ascent"] = False
            problem_description["disable_gaussian_process"] = False
            problem_description["max_num_scalable_klee_inputs"] = 1000000000
            problem_description["max_num_klee_points_per_iteration"] = 1
            problem_description["use_MetaOpt_cluster"] = False
            # problem_description["randomized_gradient_ascent"] = True
            # problem_description["num_vars_in_randomized_gradient_ascent"] = 10
        if heuristic_name == "PoP":
            problem_description["block_length"] = 1
            problem_description["keep_redundant_code_paths"] = True # One code path only
            problem_description["max_num_scalable_klee_inputs"] = 1000000000 # One code path only
            problem_description["num_random_seed_samples"] = 20
            problem_description["num_partitions"] = 2 # Same as MetaOpt
            problem_description["num_random"] = 5     # For wrapping arrow, same as MetaOpt
            problem_description["partitions_file"] = f"../topologies/pop_partitions/MetaOpt/{topology}/partitions.json"
            problem_description["max_num_klee_points_per_iteration"] = 1
            problem_description["randomized_gradient_ascent"] = True
            problem_description["num_vars_in_randomized_gradient_ascent"] = 10
            problem_description["disable_gaussian_process"] = True
            problem_description["num_samples"] = 1
        elif heuristic_name == "DOTE":
            problem_description["num_samples"] = 20
            problem_description["disable_gaussian_process"] = False
            problem_description["disable_klee"] = False
            problem_description["disable_gradient_ascent"] = False
            problem_description["num_random_seed_samples"] = 20
            problem_description["max_num_klee_points_per_iteration"] = 1
            problem_description["max_num_scalable_klee_inputs"] = 100000000
            problem_description["num_iterations"] = 2000
            problem_description["actual_gap_save_interval"] = 1000
            problem_description["relaxed_gap_save_interval"] = 10
            problem_description["save_and_plot_interval"] = 10
    elif args.problem.startswith("arrow"):
        topology_name = args.problem.split("_")[-1]
        problem_description = {
            "problem_type": "arrow",
            "heuristic_name": "arrow",
            "max_num_scalable_klee_inputs": 1000,
            "relaxed_gap_save_interval": 10,
            "actual_gap_save_interval": 1000,
            "save_and_plot_interval": 10,
            "use_gaps_in_filtering": True,
            "keep_redundant_code_paths": True, # One code path only
            "remove_zero_gap_inputs": True,
            "num_tickets": 3,        # enough variety, still small
            "num_random": 10,        # For wrapping arrow
            "num_scenarios": 3,
            "gradient_ascent_rate": 1.0,
            "num_non_zero_rounds": 1,
            "num_samples": 40, # this is used for gaussian process
            "disable_gaussian_process": True, # this is used for direct derivative
            "randomized_gradient_ascent": True,
            "num_vars_in_randomized_gradient_ascent": 10,
            "max_time_per_klee_point": None,
            "num_random_seed_samples": 10, # this is used for random seed samples because of the one code-path issue
            "max_num_klee_points_per_iteration": 1,
            "disable_gradient_ascent": False,
            "disable_sequential_filtering": True,
        }
        if topology_name == "IBM":
            problem_description["topology_path"] = "../topologies/arrow_IBM_optical_topology.txt"
        elif topology_name == "B4":
            problem_description["topology_path"] = "../topologies/arrow_B4_optical_topology.txt"
        elif topology_name == "simple":
            problem_description["topology_path"] = "../topologies/arrow_simple_optical_topology.txt"
        else:
            raise ValueError(f"Invalid topology: {topology_name}")
    elif args.problem.startswith("vbp"):
        heuristic_name = args.problem.split("_")[1]
        num_items = int(args.problem.split("_")[2])
        num_dimensions = int(args.problem.split("_")[3])
        print(f"vbp num_items: {num_items}, num_dimensions: {num_dimensions}")
        problem_description = {
            "problem_type": "vbp",
            "heuristic_name": heuristic_name,
            "num_items": num_items,
            "num_dimensions": num_dimensions,
            "max_num_scalable_klee_inputs": 1000,
            "num_samples": 50, # this is used for gaussian process
            "disable_gaussian_process": False, # this is used for direct derivative
            "max_time_per_klee_point": None,
            "block_length": 1.0,
            "gradient_ascent_rate": 1.0,
            "use_gaps_in_filtering": True,
            "remove_zero_gap_inputs": True,
            "minimize_is_better": True,
            "early_stop": False,
            "keep_redundant_code_paths": True, # We want code path changes
            # Enable rounding strategies to explore discrete solutions
            "use_rounding_strategies": True,
            # Number of highest-fractional variables to exhaustively round (floor/ceil)
            # 0 = disable exhaustive combinations
            "rounding_exhaustive_k": 3,
            # Coefficient for smooth integrality penalty (encourages x_j -> nearest integer)
            "integrality_penalty_gamma": 0.1,
            "num_iterations": 1000,
        }
    elif args.problem.startswith("knapsack"):
        num_items = int(args.problem.split("_")[1])
        problem_description = {
            "problem_type": "knapsack",
            "heuristic_name": "greedy",
            "capacity": 50,
            "max_value": 50,
            "num_items": num_items,
            "max_num_scalable_klee_inputs": 200,
            "early_stop": True,
            "block_length": 0.1,
            "gradient_ascent_rate": 1.0,
            "use_gaps_in_filtering": True,
            "remove_zero_gap_inputs": True,
            "disable_gaussian_process": False,
            "num_non_zero_rounds": 5,
            "disable_gradient_ascent": False,
            "num_samples": 50,
            "num_random_seed_samples": 30,
            "max_num_klee_points_per_iteration": 10,
        }
    elif args.problem.startswith("mwm"):
        topology = args.problem.split("_")[-1]
        problem_description = {
            "problem_type": "mwm",
            "heuristic_name": "greedy",
            "topology": topology,
            "max_num_scalable_klee_inputs": 1000000,
            "use_gaps_in_filtering": True,
            "remove_zero_gap_inputs": True,
            "max_weight": 10,
            "disable_gradient_ascent": False,
            "disable_gaussian_process": True,
            "randomized_gradient_ascent": True,
            "num_vars_in_randomized_gradient_ascent": 10,
            "keep_redundant_code_paths": True,
            "num_random_seed_samples": 10,
            "num_samples": 20,
        }
        problem_description["max_value"] = problem_description["max_weight"]
        problem_description["min_value"] = 0
    else:
        raise ValueError(f"Invalid problem: {args.problem}")
    problem_description["method"] = args.method
    return problem_description
