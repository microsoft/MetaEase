from metaease import main
import time
import json
import os
import argparse

base_dir = "../ablation_DemandPinning/"
table_name = "ablation_DemandPinning_timing_table.json"
topologies = ["swan", "b4-teavar", "abilene"]
klee_task = "inputs_scale_fixed_points"
base_problem_description = {
    "problem_type": "TE",
    "block_length": 1.0,
    "max_num_scalable_klee_inputs": 16,
    "use_gaps_in_filtering": True,
    "remove_zero_gap_inputs": True,
    "gradient_ascent_rate": 1.0,
    "disable_gradient_ascent": False,
    "heuristic_name": "DemandPinning",
    "num_samples": 20,
    "use_MetaOpt_cluster": False,
    "max_num_paths": float("inf"),
    "num_iterations": 500,
    "method": "MetaEase",
    "early_stop": True,
    "relaxed_gap_save_interval": 100,
    "actual_gap_save_interval": 1000,
    "save_and_plot_interval": 100,
    "disable_sequential_filtering": False,
    "minimize_is_better": False,
    "use_gaps_in_filtering": True,
    "remove_zero_gap_inputs": True,
    "keep_redundant_code_paths": False,
}


# This vs Klee
def get_random_seed_config():
    return {
        "disable_klee": True,
        "num_random_seed_samples": 10,
        "max_num_scalable_klee_inputs": 1000000000,
        "num_samples": 20,
    }


# This vs Random
def get_klee_config(max_num_scalable_klee_inputs):
    return {
        "disable_klee": False,
        "num_random_seed_samples": 0,
        "max_num_scalable_klee_inputs": max_num_scalable_klee_inputs,
    }

def get_LLM_seed_config(topology):
    return {
        "seed_file": f"/home/ubuntu/MetaEase/MetaOptimize/LLM_{topology}.json",
        "disable_klee": True,
        "num_random_seed_samples": 0,
        "max_num_scalable_klee_inputs": 1000000000,
        "num_samples": 50,
        "early_stop": False,
    }

# This vs Direct Gradient
def get_gp_config():
    return {
        "num_samples": 20,  # For GP
        "disable_gradient_ascent": False,
        "disable_guassian_process": False,
    }


# This vs GP
def get_direct_gradient_config():
    return {
        "disable_gradient_ascent": False,
        "disable_guassian_process": True,
    }


def no_gradient_ascent_config():
    return {
        "disable_gradient_ascent": True,
        "disable_guassian_process": True,
        "num_samples": 0,
    }


timing_table = {}
if os.path.exists(table_name):
    with open(table_name, "r") as f:
        timing_table = json.load(f)

ablation_name = "LLM_with_GP"
if ablation_name == "LLM_with_GP":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    gradient_config = get_gp_config()
    problem_description.update(gradient_config)
    timing_table["LLM_with_GP"] = {}
    for topology in topologies:
        seed_config = get_LLM_seed_config(topology)
        problem_description.update(seed_config)
        problem_description["topology"] = topology
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)

timing_table = {}

ablation_name = "Klee_with_GP"
if ablation_name == "Klee_with_GP":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_klee_config(1000000)
    gradient_config = get_gp_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    timing_table["Klee_with_GP"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)


ablation_name = "Random_with_GP"
# Run
if ablation_name == "Random_with_GP":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_random_seed_config()
    gradient_config = get_gp_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    problem_description["ignore_code_path"] = True
    timing_table["Random_with_GP"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)


ablation_name = "varying_klee_inputs_with_no_gradient_ascent"
klee_inputs = [4, 8, 16, 32, 64, 128]
if ablation_name == "varying_klee_inputs_with_no_gradient_ascent":
    for klee_input in klee_inputs:
        save_dir = base_dir + ablation_name + "_" + str(klee_input)
        problem_description = base_problem_description.copy()
        seed_config = get_klee_config(klee_input)
        gradient_config = no_gradient_ascent_config()
        problem_description.update(seed_config)
        problem_description.update(gradient_config)
        problem_description["ignore_code_path"] = True
        timing_table[ablation_name + "_" + str(klee_input)] = {}
        for topology in topologies:
            problem_description["topology"] = topology
            start_time = time.time()
            main(problem_description, save_dir, klee_task)
            end_time = time.time()
            timing_table[ablation_name + "_" + str(klee_input)][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)

# Ignoring code path for direct gradient
ablation_name = "Random_with_Direct_Gradient"
if ablation_name == "Random_with_Direct_Gradient":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_random_seed_config()
    gradient_config = get_direct_gradient_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    problem_description["ignore_code_path"] = True
    timing_table["Random_with_Direct_Gradient"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)


ablation_name = "Klee_with_Direct_Gradient"
if ablation_name == "Klee_with_Direct_Gradient":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_klee_config(16)
    gradient_config = get_direct_gradient_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    problem_description["ignore_code_path"] = True
    timing_table["Klee_with_Direct_Gradient"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)

ablation_name = "Klee_Gap_Sample_Based"
from gap_sample_based import sample_based_gradient_main
if ablation_name == "Klee_Gap_Sample_Based":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    args = argparse.Namespace()
    args.base_save_dir = save_dir
    timing_table["Klee_Gap_Sample_Based"] = {}
    for topology in topologies:
        problem_description["topology"] = topology
        start_time = time.time()
        if topology == "swan":
            seed_path = "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_with_GP/20250907_170845__MetaEase__TE__AGSI-1000_BL-1_DGA-False_DGP-False_DK-False_DSF-False_ES-True_GAR-1_HN-DemandPinning_KRCP-False_MNP-inf_MNSKI-1000000_MIB-False_NI-500_NRSS-0_NS-20_RGSI-100_RZGI-True_SAPI-100_T-swan_UMC-False_UGIF-True/klee_inputs_0_0_1757266187.json"
        elif topology == "b4-teavar":
            seed_path = "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_with_GP/20250907_175223__MetaEase__TE__AGSI-1000_BL-1_DGA-False_DGP-False_DK-False_DSF-False_ES-True_GAR-1_HN-DemandPinning_KRCP-False_MNP-inf_MNSKI-1000000_MIB-False_NI-500_NRSS-0_NS-20_RGSI-100_RZGI-True_SAPI-100_T-b4-teavar_UMC-False_UGIF-True/klee_inputs_0_0_1757267561.json"
        elif topology == "abilene":
            seed_path = "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_with_GP/20250907_185103__MetaEase__TE__AGSI-1000_BL-1_DGA-False_DGP-False_DK-False_DSF-False_ES-True_GAR-1_HN-DemandPinning_KRCP-False_MNP-inf_MNSKI-1000000_MIB-False_NI-500_NRSS-0_NS-20_RGSI-100_RZGI-True_SAPI-100_T-abilene_UMC-False_UGIF-True/klee_inputs_0_0_1757271915.json"
        args.problem = "TE_DemandPinning_" + topology
        args.method = "SampleBasedGradient"
        best_sample = sample_based_gradient_main(args, num_iterations=problem_description["num_iterations"], seed_path=seed_path, enforce_num_iterations=True)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time


