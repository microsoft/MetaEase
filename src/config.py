from problems.programs_TE import *
from problems.programs_vbp import *
from problems.programs_knapsack import *
from problems.programs_max_weighted_matching import *
from problems.programs_arrow import *
import os
import json

# Get the directory of the current file
BASE_DIR = os.path.dirname(__file__)
# Build the relative path to the topologies directory
TOPOLOGY_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR, "..", "topologies"
    )
)


def load_topology_info(topology_name):
    """Load topology information from the topology file."""
    topology_file = os.path.join(TOPOLOGY_DIR, f"{topology_name}.json")
    if not os.path.exists(topology_file):
        raise ValueError(f"Topology file not found: {topology_file}")

    with open(topology_file, "r") as f:
        topology_description = json.load(f)

    topology_data = {}
    num_nodes = len(topology_description["nodes"])
    avg_capacity = sum(
        link["capacity"] for link in topology_description["links"]
    ) / len(topology_description["links"])
    topology_data["num_nodes"] = num_nodes
    topology_data["capacity"] = avg_capacity

    # get all the edges
    edges = []
    for link in topology_description["links"]:
        topology_data[f"edge_{link['source']}_{link['target']}"] = link["capacity"]

    return topology_data

# TODO: it seems some of our heuristics are missing here? Dote? also, I think once more I see the importance of in the main readme describing how a user can analyze (1) a new heuristic (2) a new problem class (3) a new problem instance. And describe what
# changes they need to make to the code to do so.
PROBLEM_CONFIGS = {
    "vbp": {
        "num_items": 20,
        "heuristic_name": "FF",
        "num_dimensions": 1,
        "bin_size": 10.0,
    },
    "knapsack": {
        "min_value": 0.0,
    },
    "arrow": {
        "heuristic_name": "arrow",
    },
    "TE": {
        "heuristic_name": "DemandPinning",
        "top_k": 2,
    },
    "mwm": {
        "heuristic_name": "Greedy",
    },
}

COMMON_PARAMETERS = {
    # Optimization parameters
    "block_length": 0.1, # The size of the block around the current best sample to sample from
    "num_samples": 50, # The number of samples to generate in each block, this will be multiplied by a multiplier (currently 1.1) to account for the samples that are within the block but with a different code path # TODO: this was not clear.
    "num_iterations": 2000, # The number of iterations to run the gradient ascent for on one klee point
    "gradient_ascent_rate": 0.2, # The learning rate for the gradient ascent, the same rate will be used for all variables
    "disable_guassian_process": False, # If True, the guassian process will not be used, direct gradient computation will be used (for samples with a lot of variables, up to num_keys_for_gradient=20)
    "randomized_gradient_ascent": False, # If True, the gradient ascent will be run on randomized inputs, if False, one step gradient ascent is done on all variables at once # TODO: this was not clear. needs a more detailed description.
    "num_vars_in_randomized_gradient_ascent": 10, # The number of variables to run the gradient ascent on at once
    "disable_klee": False, # If True, the klee will not be run, only the gradient ascent will run on random samples
    "num_random_seed_samples": 30, # When disable_klee is True, the number of random samples to generate for the gradient ascent
    "freeze_cluster_fixed_keys": True, # If True, the fixed keys will not be updated # TODO: this is not clear and needs more clarified explanation.
    # Logging parameters
    "relaxed_gap_save_interval": 10, # The interval to save the relaxed gap
    "actual_gap_save_interval": 100, # The interval to save the actual gap
    "save_and_plot_interval": 10, # The interval to save the and plot the results
    "enable_plotting": True, # If True, the results will be plotted
    # Filtering parameters
    "use_gaps_in_filtering": True, # If True, the gaps will be computed and used to filter the kleesamples # TODO: it would be good to include in this comment "how" they will be used to do so.
    "remove_zero_gap_inputs": True, # If True, the klee samples with zero gap will be removed
    "keep_redundant_code_paths": False, # If True, the redundant code paths will be kept, True used mostly for VBP since we want the code-path changes to happen. Due to klee time constraints, two paths can be similar up to a point but then diverge after that, but we see them as inputs generating the same code-paths because it never went all the way
    # Klee parameters
    "use_MetaOpt_cluster": False, # If True, the MetaOpt cluster will be used (mostly for TE). Make sure to also provide cluster_path in the problem description
    "ignore_gap_value_in_num_non_zero_rounds": False,
    "num_non_zero_rounds": 1, # Run one klee setting for at most num_non_zero_rounds round untill you find a non-zero gap, used alonside list_of_input_paths_to_exclude
    "max_num_scalable_klee_inputs": 16, # The maximum number of klee variables
    "preferred_values": None, # The preferred values for the klee variables, if None, the values will be randomly sampled from the min_value to max_value # TODO: would benefit from guidence on how to set it.
    # Time budget parameters
    "early_stop": True, # If True, the gradient ascent will stop when the relaxed gap converges based on has_converged function
    "max_num_klee_points_per_iteration": None, # The maximum number of klee points to run in each iteration
    "max_time_per_klee_point": None, # The maximum time to run gradient ascent on a klee point
    "max_total_time": None, # The maximum total time to run MetaEase
}
# TODO: could use a comment on what these are.
PARAMETERS = {
    "vbp": {
        "min_value": 0.0,
        "max_value": PROBLEM_CONFIGS["vbp"]["bin_size"],
    },
    "knapsack": {
        "min_value": 0.0,
    },
    "TE": {
        "min_value": 0.0,
    },
    "mwm": {
        "min_value": 0.0,
    },
    "arrow": {
        "min_value": 0.0,
        "max_value": 1200,
    },
}


def get_problem_instance(problem_type, config_path):
    """Create an instance of the appropriate problem class based on problem type."""
    if problem_type == "TE":
        return TEProblem(config_path)
    elif problem_type == "vbp":
        return VBPProblem(config_path)
    elif problem_type == "knapsack":
        return KnapsackProblem(config_path)
    elif problem_type == "mwm":
        return MWMProblem(config_path)
    elif problem_type == "arrow":
        return ArrowProblem(config_path)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def make_config(config_dict, config_path):
    with open(config_path, "w") as f:
        json.dump(config_dict, f)


def override_config(problem_type, config_path, override_keys_and_values):
    if not os.path.exists(config_path):
        config = PROBLEM_CONFIGS[problem_type].copy()
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    for key, value in override_keys_and_values.items():
        config[key] = value
    with open(config_path, "w") as f:
        json.dump(config, f)


def handle_TE_parameters(problem_description):
    # Load topology-specific parameters
    topology_name = problem_description.get("topology")
    if not topology_name:
        raise ValueError("Topology name must be provided for TE problems")

    topology_info = load_topology_info(topology_name)

    # Update TE-specific parameters
    te_params = PARAMETERS["TE"].copy()
    te_params["max_flow"] = 2 * topology_info["capacity"]
    te_params["max_value"] = te_params["max_flow"]
    te_params["small_flow_cutoff"] = int(0.05 * topology_info["capacity"])
    te_params["preferred_values"] = [
        te_params["max_flow"],
        te_params["small_flow_cutoff"],
    ]

    # Merge parameters
    te_params.update(topology_info)
    return te_params

def handle_MWM_parameters(problem_description):
    topology_name = problem_description.get("topology")
    if not topology_name:
        raise ValueError("Topology name must be provided for MWM problems")

    topology_info = load_topology_info(topology_name)
    return topology_info
# TODO: this seems like it is fragile especially when it comes to supporting new heuristic types.
# think through a proper design where the inputs are taken in a way that can generalize beyond the examples we have.
def get_parameters(problem_description):
    """Get parameters based on problem description including topology for TE problems."""
    problem_type = problem_description["problem_type"]
    params = COMMON_PARAMETERS.copy()
    params.update(PROBLEM_CONFIGS[problem_type])
    params.update(problem_description)

    if problem_type == "TE":
        te_params = handle_TE_parameters(problem_description)
        params.update(te_params)
    elif problem_type == "mwm":
        mwm_params = handle_MWM_parameters(problem_description)
        params.update(mwm_params)
    else:
        # For other problem types, use existing parameter logic
        params.update(PARAMETERS[problem_type])

    # Set minimize_is_better flag
    minimize_is_better = problem_type in ["vbp"]
    params["minimize_is_better"] = minimize_is_better

    return params
