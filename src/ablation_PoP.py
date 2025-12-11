from run import main
import time
import json

base_dir = "../ablation_PoP/"
table_name = "ablation_PoP_timing_table.json"
topologies = ["swan", "b4-teavar", "abilene"]
klee_task = "inputs_scale_fixed_points"
num_samples = 20
num_iterations = 1000
base_problem_description = {
    "problem_type": "TE",
    "block_length": 1.0,
    "max_num_scalable_klee_inputs": 16,
    "use_gaps_in_filtering": True,
    "remove_zero_gap_inputs": True,
    "gradient_ascent_rate": 1.0,
    "disable_gradient_ascent": False,
    "heuristic_name": "PoP",
    "num_samples": num_samples,
    "use_MetaOpt_cluster": False,
    "max_num_paths": float("inf"),
    "num_iterations": num_iterations,
    "method": "MetaEase",
    "early_stop": False,
    "relaxed_gap_save_interval": 100,
    "actual_gap_save_interval": 1000,
    "save_and_plot_interval": 100,
    "keep_redundant_code_paths": True,
    "num_partitions": 2,  # Same as MetaOpt
    "num_random": 5,  # For wrapping arrow, same as MetaOpt
    "max_num_klee_points_per_iteration": 1,
    "disable_sequential_filtering": True,
    "minimize_is_better": False,
}


# This vs Klee
def get_random_seed_config():
    return {
        "disable_klee": True,
        "num_random_seed_samples": 10,
        "max_num_scalable_klee_inputs": 1000000000,
        "num_samples": num_samples,
    }


# This vs Random
def get_klee_config(max_num_scalable_klee_inputs):
    return {
        "disable_klee": False,
        "num_random_seed_samples": 0,
        "max_num_scalable_klee_inputs": max_num_scalable_klee_inputs,
    }


# This vs Direct Gradient
def get_gp_config():
    return {
        "num_samples": num_samples,  # For GP
        "disable_gradient_ascent": False,
        "disable_guassian_process": False,
    }


# This vs GP
def get_direct_gradient_config():
    return {
        "disable_gradient_ascent": False,
        "disable_guassian_process": True,
    }

# Randomized Gradient
def get_randomized_gradient_config():
    return {
        "disable_gradient_ascent": False,
        "disable_guassian_process": True,
        "randomized_gradient_ascent": True,
        "num_vars_in_randomized_gradient_ascent": 10
    }

def no_gradient_ascent_config():
    return {
        "disable_gradient_ascent": True,
        "disable_guassian_process": True,
        "num_samples": 0,
    }


timing_table = {}
ablation_name = "Random_with_GP"
# Run
if ablation_name == "Random_with_GP":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_random_seed_config()
    gradient_config = get_gp_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    timing_table["Random_with_GP"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        problem_description["partitions_file"] = (
            f"../topologies/pop_partitions/MetaOpt/{topology}/partitions.json"
        )
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)

ablation_name = "Random_with_Direct_Gradient"
if ablation_name == "Random_with_Direct_Gradient":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_random_seed_config()
    gradient_config = get_direct_gradient_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    timing_table["Random_with_Direct_Gradient"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        problem_description["partitions_file"] = (
            f"../topologies/pop_partitions/MetaOpt/{topology}/partitions.json"
        )
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)
