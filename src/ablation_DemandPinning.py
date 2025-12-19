"""
Ablation script for MetaEase's DemandPinning heuristic (Section 6 of the paper).

This file runs a sequence of *design-choice ablations* to answer:
  - Which pieces of MetaEase are necessary?
  - What happens (performance + runtime) when each is removed or replaced?

Each ablation is implemented as a separate block keyed by `ablation_name`.
To re-run a specific experiment in the artifact, set the corresponding
`ablation_name` and execute this script.

High-level mapping to the paper:
  1) Seed generation (KLEE vs Random vs LLM), Figure 14
  2) Path-aware vs path-agnostic / direct gradients, Figure 13
  3) GP surrogate vs sample-based / direct gradients, Figure 13
  4) Varying KLEE inputs / projected dimensions, Figure 15
"""

from metaease import main
import time
import json
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ablation_data_dir = os.path.join(SCRIPT_DIR, "ablation_data")

base_dir = "../ablation_DemandPinning/"
# Timing table (JSON) written incrementally as each ablation finishes.
table_name = "ablation_DemandPinning_timing_table.json"
# Topologies used in the paper for the TE DemandPinning heuristic.
topologies = ["swan", "b4-teavar", "abilene"]
# KLEE task: generate scalability-aware inputs that can be scaled to larger sizes.
klee_task = "inputs_scale_fixed_points"

# Baseline MetaEase configuration for DemandPinning.
# Individual ablations *modify* this dictionary with seed/gradient settings.
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


# ---------------------------------------------------------------------------
# Seed generation ablations (Section 6.1): KLEE vs Random vs LLM
# ---------------------------------------------------------------------------
# These configurations control how MetaEase is initialized before gradient ascent.
# Good seeds = better local optima; bad seeds = poor worst-case gaps.

def get_random_seed_config():
    """
    Random multi-start seeds (no symbolic execution).

    Used to compare KLEE-based seeds against pure random initialization.
    This corresponds to the **Random** baseline in the ablation section.
    """
    return {
        "disable_klee": True,
        "num_random_seed_samples": 10,
        "max_num_scalable_klee_inputs": 1000000000,
        "num_samples": 20,
    }


def get_klee_config(max_num_scalable_klee_inputs):
    """
    Symbolic-execution-based seeds from KLEE.

    `max_num_scalable_klee_inputs` controls how many distinct KLEE inputs
    (i.e., path-based equivalence classes) we allow before handing off to
    gradient-based search.
    """
    return {
        "disable_klee": False,
        "num_random_seed_samples": 0,
        "max_num_scalable_klee_inputs": max_num_scalable_klee_inputs,
    }

def get_LLM_seed_config(topology):
    """
    LLM-generated seeds for a given topology.

    Seeds are produced offline by prompting an LLM to reason over the
    DemandPinning code and output candidate bad cases. In the paper,
    this serves as the **LLM** seeding baseline and is compared against
    KLEE and Random seeding.
    """
    return {
        "seed_file": f"{ablation_data_dir}/LLM_{topology}.json",
        "disable_klee": True,
        "num_random_seed_samples": 0,
        "max_num_scalable_klee_inputs": 1000000000,
        "num_samples": 50,
        "early_stop": False,
    }

def get_gp_config():
    """
    Enable Gaussian-Process-based surrogate gradients.

    This configuration keeps MetaEase's GP surrogate on and tunes the
    number of samples used for fitting the GP model.
    """
    return {
        "num_samples": 20,  # For GP
        "disable_gradient_ascent": False,
        "disable_gaussian_process": False,
    }


def get_direct_gradient_config():
    """
    Direct gradient ascent without a GP surrogate.

    Conceptually, this is the "path-aware but non-surrogate" setting:
    MetaEase tries to optimize directly in heuristic space, which is
    useful for contrasting with the GP-based gradient (Section 6.3).
    """
    return {
        "disable_gradient_ascent": False,
        "disable_gaussian_process": True,
    }


def no_gradient_ascent_config():
    """
    Disable *all* gradient ascent and GP surrogate.

    This corresponds to pure KLEE-based or seed-only search
    (e.g., for projected-dimension / K variations without gradient steps).
    """
    return {
        "disable_gradient_ascent": True,
        "disable_gaussian_process": True,
        "num_samples": 0,
    }

timing_table = {}
if os.path.exists(table_name):
    with open(table_name, "r") as f:
        timing_table = json.load(f)

# ---------------------------------------------------------------------------
# Ablation 1: LLM seeds + GP (LLM_with_GP)
#
# Compares KLEE-based seeds vs LLM-generated seeds, *holding the GP
# surrogate fixed*. This tests whether "LLM reasoning about code" can
# replace symbolic execution. In the paper, LLM seeds do not help and
# sometimes hurt performance.
# ---------------------------------------------------------------------------
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
        # Fix topology and run MetaEase once per topology.
        problem_description["topology"] = topology
        start_time = time.time()
        main(problem_description, save_dir, klee_task)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time

with open(table_name, "w") as f:
    json.dump(timing_table, f)

timing_table = {}

# ---------------------------------------------------------------------------
# Ablation 2: KLEE seeds + GP (Klee_with_GP)
#
# This is the "full" MetaEase seeding story: path-based KLEE seeds +
# GP-based path-aware gradient ascent. It is the main reference line
# for seed-generation comparisons (vs Random / LLM).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Ablation 3: Random seeds + GP (Random_with_GP)
#
# Random multi-start initialization with the same GP-based gradient as
# MetaEase. This isolates the effect of *symbolic seeding* by replacing
# KLEE with uniform random seeds.
# ---------------------------------------------------------------------------
ablation_name = "Random_with_GP"
# Run
if ablation_name == "Random_with_GP":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_random_seed_config()
    gradient_config = get_gp_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    # NOTE: `ignore_code_path=True` explicitly disables path-aware
    # equivalence classes during the seeding stage. This emphasizes
    # that these runs are purely random, path-agnostic initializations.
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


# ---------------------------------------------------------------------------
# Ablation 4: Varying KLEE inputs with no gradient ascent
#            (varying_klee_inputs_with_no_gradient_ascent)
#
# This block studies the effect of *dimensionality / number of KLEE seeds*
# on performance and runtime. It disables gradient ascent entirely so
# we can cleanly observe how many KLEE inputs (paths) are required.
# This corresponds to the projected-gradient / K-variation discussion
# in the ablation section.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Ablation 5: Random seeds + Direct Gradient (Random_with_Direct_Gradient)
#
# This configuration removes both KLEE *and* the GP surrogate:
#   - Initialization: random seeds
#   - Optimization: direct gradient ascent (no GP)
#
# It highlights how unstable / inefficient path-agnostic direct gradients
# can be on real heuristics, compared to MetaEase's GP-based path-aware
# gradients.
# ---------------------------------------------------------------------------
ablation_name = "Random_with_Direct_Gradient"
if ablation_name == "Random_with_Direct_Gradient":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_random_seed_config()
    gradient_config = get_direct_gradient_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    # Again, treat inputs as purely numeric vectors and ignore which
    # code path they came from, i.e., path-agnostic optimization.
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


# ---------------------------------------------------------------------------
# Ablation 6: KLEE seeds + Direct Gradient (Klee_with_Direct_Gradient)
#
# Uses symbolic-execution-based seeds, but disables the GP surrogate.
# This isolates the value of the *GP model itself* while keeping
# KLEE-based path-aware seeding.
# ---------------------------------------------------------------------------
ablation_name = "Klee_with_Direct_Gradient"
if ablation_name == "Klee_with_Direct_Gradient":
    save_dir = base_dir + ablation_name
    problem_description = base_problem_description.copy()
    seed_config = get_klee_config(16)
    gradient_config = get_direct_gradient_config()
    problem_description.update(seed_config)
    problem_description.update(gradient_config)
    # Here, KLEE still discovers path-diverse seeds, but the follow-up
    # optimization uses direct gradients instead of a GP surrogate.
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

# ---------------------------------------------------------------------------
# Ablation 7: KLEE + Gap Sample-Based Gradient (Klee_Gap_Sample_Based)
#
# This uses sample-based finite-difference style gradients instead of
# the GP surrogate. It corresponds to the "sample-based gradient"
# baseline in Section 6.3 and demonstrates that GP-based gradients
# achieve similar (or better) gaps with significantly fewer heuristic
# evaluations (up to 4× fewer calls, 16.6× faster on large topologies).
# ---------------------------------------------------------------------------
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
        # Each topology has a pre-computed KLEE seed file that covers a
        # diverse set of code paths for the corresponding TE instance.
        if topology == "swan":
            seed_path = os.path.join(
                ablation_data_dir, "klee_inputs_0_0_1757266187.json"
            )
        elif topology == "b4-teavar":
            seed_path = os.path.join(
                ablation_data_dir, "klee_inputs_0_0_1757267561.json"
            )
        elif topology == "abilene":
            seed_path = os.path.join(
                ablation_data_dir, "klee_inputs_0_0_1757271915.json"
            )
        args.problem = "TE_DemandPinning_" + topology
        # Name used inside `gap_sample_based.sample_based_gradient_main`;
        # corresponds to the "Sample-Based Gradient" baseline in the paper.
        args.method = "SampleBasedGradient"
        best_sample = sample_based_gradient_main(args, num_iterations=problem_description["num_iterations"], seed_path=seed_path, enforce_num_iterations=True)
        end_time = time.time()
        timing_table[ablation_name][topology] = end_time - start_time


