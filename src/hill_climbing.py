from config import *
from common import *
from logger import AsyncLogger
from run_utils import get_save_dir, make_config
import random
import json
import os
import time
import copy
import numpy as np

random.seed(time.time())
stuck_limit = 1000000

def hill_climbing_main(args, num_iterations=1000, num_neighbors=10, step_size=1.0, max_time=float('+inf')):
    problem_description = get_problem_description(args)
    problem_description["method"] = "HillClimbing"
    metaease_params = MetaEase_specific_parameters()
    for param in metaease_params:
        if param in problem_description:
            del problem_description[param]

    problem_description["num_iterations"] = num_iterations
    problem_description["num_neighbors"] = num_neighbors
    problem_description["step_size"] = step_size
    problem_description["max_time"] = max_time
    # Configuration
    problem_type = problem_description["problem_type"]
    parameters = get_parameters(problem_description)
    save_dir = get_save_dir(args.base_save_dir, problem_description)

    # Initialize logger
    logger = AsyncLogger.init(save_dir)
    logger.debug("Experiment started")
    logger.info(f"Parameters: {parameters}")

    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, f"{problem_type}_config.json")
    make_config(parameters, config_path)

    # Verify config file was created
    if not os.path.exists(config_path):
        logger.error(f"Config file was not created at {config_path}")
        raise FileNotFoundError(f"Config file was not created at {config_path}")

    problem = get_problem_instance(problem_type, config_path)
    all_klee_vars = problem.all_klee_var_names
    logger.debug(f"All klee vars: {all_klee_vars}")

    # Get thresholds for hill climbing - we only need input variable thresholds
    # Create a dummy relaxed_all_vars with just the input variables to get their thresholds
    dummy_relaxed_vars = {var: 0.0 for var in all_klee_vars}
    thresholds = problem.get_thresholds(dummy_relaxed_vars)

    # Filter thresholds to only include input variables (klee variables)
    input_thresholds = {
        var: thresholds[var] for var in all_klee_vars if var in thresholds
    }
    logger.info(f"Using input thresholds: {input_thresholds}")

    # Initialize tracking variables
    max_global_gap = float("-inf")
    best_global_sample = None
    best_global_all_vars = None

    # Hill climbing parameters
    current_sample = {}

    # Initialize current sample randomly
    for var_name in all_klee_vars:
        if var_name in input_thresholds:
            min_val, max_val = input_thresholds[var_name]
            current_sample[var_name] = random.uniform(min_val, max_val)
        else:
            current_sample[var_name] = random.uniform(0.0, 100.0)

    # Initialize current state variables
    current_gap = float("-inf")
    current_opt_val = None
    current_heur_val = None
    current_all_vars = None
    current_code_path_num = None

    # Evaluate initial sample
    try:
        (
            initial_gap,
            _,
            (initial_opt_val, initial_heur_val),
            initial_all_vars,
            initial_code_path_num,
        ) = problem.get_gaps_process_input(current_sample)
        current_gap = initial_gap
        current_opt_val = initial_opt_val
        current_heur_val = initial_heur_val
        current_all_vars = initial_all_vars
        current_code_path_num = initial_code_path_num
        max_global_gap = initial_gap
        best_global_sample = copy.deepcopy(current_sample)
        best_global_all_vars = initial_all_vars
        logger.log_current_max_gap(initial_gap)
    except Exception as e:
        logger.error(f"Error evaluating initial sample: {e}")
        # Generate a new initial sample if the first one fails
        for var_name in all_klee_vars:
            if var_name in input_thresholds:
                min_val, max_val = input_thresholds[var_name]
                current_sample[var_name] = random.uniform(min_val, max_val)
            else:
                current_sample[var_name] = random.uniform(0.0, 100.0)
        # Don't set current_gap to -inf here, let the loop handle it

    # Hill climbing strategy
    logger.info(f"Starting hill climbing with {num_iterations} iterations")
    logger.info(
        f"Number of neighbors per iteration: {num_neighbors}, Step size: {step_size}"
    )
    start_time = time.time()

    # Tracking variables for results
    gaps = []
    gaps_times = []
    optimal_values = []
    heuristic_values = []
    all_optimal_vars = []
    code_path_nums = []
    successful_evaluations = 0
    improvements = 0
    stuck_count = 0
    actual_iters = 0
    actual_evals = 0
    iteration = 0

    while time.time() - start_time < max_time:
        iteration += 1
        actual_iters += 1

        # Generate neighbor samples
        neighbor_samples = []
        for _ in range(max(1, num_neighbors)):  # Guard against num_neighbors == 0
            neighbor = copy.deepcopy(current_sample)
            # Perturb one random variable
            var_to_perturb = random.choice(all_klee_vars)
            if var_to_perturb in input_thresholds:
                min_val, max_val = input_thresholds[var_to_perturb]
                # Add or subtract step_size with equal probability
                if random.random() < 0.5:
                    perturbation = step_size
                else:
                    perturbation = -step_size
                # Optional: scale step by range
                # perturbation *= (max_val - min_val)
                neighbor[var_to_perturb] += perturbation
                # Ensure bounds
                neighbor[var_to_perturb] = max(
                    min_val, min(max_val, neighbor[var_to_perturb])
                )
            else:
                # For variables without thresholds, use a smaller step and add upper cap
                if random.random() < 0.5:
                    perturbation = step_size
                else:
                    perturbation = -step_size
                neighbor[var_to_perturb] += perturbation
                neighbor[var_to_perturb] = max(0.0, min(1000.0, neighbor[var_to_perturb]))  # Add upper cap
            neighbor_samples.append(neighbor)

        # Evaluate all neighbors
        best_neighbor_gap = float("-inf")
        best_neighbor_sample = None
        best_neighbor_all_vars = None
        best_neighbor_opt_val = None
        best_neighbor_heur_val = None
        best_neighbor_code_path_num = None

        for neighbor in neighbor_samples:
            try:
                gap, _, (opt_val, heur_val), all_vars, code_path_num = (
                    problem.get_gaps_process_input(neighbor)
                )
                actual_evals += 1
                successful_evaluations += 1

                if gap > best_neighbor_gap:
                    best_neighbor_gap = gap
                    best_neighbor_sample = copy.deepcopy(neighbor)
                    best_neighbor_all_vars = all_vars
                    best_neighbor_opt_val = opt_val
                    best_neighbor_heur_val = heur_val
                    best_neighbor_code_path_num = code_path_num

            except Exception as e:
                logger.warning(
                    f"Error evaluating neighbor at iteration {iteration}: {e}"
                )
                continue

        # Hill climbing: only move if we found a better neighbor
        moved = False
        if best_neighbor_sample is not None and best_neighbor_gap > current_gap:
            current_sample = best_neighbor_sample
            current_gap = best_neighbor_gap
            current_opt_val = best_neighbor_opt_val
            current_heur_val = best_neighbor_heur_val
            current_all_vars = best_neighbor_all_vars
            current_code_path_num = best_neighbor_code_path_num
            improvements += 1
            stuck_count = 0  # Reset stuck counter
            moved = True
            logger.debug(
                f"Iteration {iteration}: Moved to better neighbor, gap: {current_gap}"
            )
        else:
            stuck_count += 1
            logger.debug(
                f"Iteration {iteration}: No better neighbor found, staying at gap: {current_gap}"
            )

        # Update global best if necessary
        if current_gap > max_global_gap:
            print(f"HC: Time: {time.time() - start_time}, iteration: {iteration}, gap: {current_gap}")
            max_global_gap = current_gap
            best_global_sample = copy.deepcopy(current_sample)
            best_global_all_vars = current_all_vars  # Single source of truth
            # logger.info(f"Iteration {iteration}: New best gap found: {max_global_gap}")
            logger.log_current_max_gap(max_global_gap)

        # Record results for this iteration - record the CURRENT state (post decision)
        gaps.append(current_gap)
        gaps_times.append(time.time() - start_time)
        optimal_values.append(current_opt_val)
        heuristic_values.append(current_heur_val)
        all_optimal_vars.append(current_all_vars)
        code_path_nums.append(current_code_path_num)

        # Log progress
        if (iteration + 1) % 1000 == 0:
            improvement_rate = improvements / (iteration + 1) if iteration > 0 else 0
            logger.info(
                f"Iteration {iteration + 1}/{num_iterations}. "
                f"Current gap: {current_gap:.4f}, Max gap: {max_global_gap:.4f}, "
                f"Improvements: {improvements}, Stuck count: {stuck_count}, "
                f"Improvement rate: {improvement_rate:.3f}"
            )

        # Early stopping if stuck for too long (optional)
        if stuck_count > stuck_limit:  # Stop if no improvement for 50 iterations
            logger.info(
                f"Stopping early at iteration {iteration} due to no improvements for {stuck_count} iterations"
            )
            break

    end_time = time.time()
    logger.info(f"Hill climbing completed in {end_time - start_time:.2f} seconds")
    logger.log_final_max_gap(max_global_gap)
    logger.info(
        f"Success rate: {successful_evaluations}/{actual_evals} ({100*successful_evaluations/max(1, actual_evals):.1f}%)"
    )
    logger.info(
        f"Total improvements: {improvements}, Improvement rate: {improvements/max(1, actual_iters):.3f}"
    )

    # Save results
    results = {
        "max_gap": max_global_gap,
        "best_sample": best_global_sample,
        "num_iterations": iteration,
        "num_neighbors": num_neighbors,
        "step_size": step_size,
        "successful_evaluations": successful_evaluations,
        "actual_evaluations": actual_evals,
        "actual_iterations": actual_iters,
        "success_rate": (
            successful_evaluations / max(1, actual_evals)
            if actual_evals > 0
            else 0
        ),
        "improvements": improvements,
        "improvement_rate": improvements / max(1, actual_iters) if actual_iters > 0 else 0,
        "stuck_count": stuck_count,
        "execution_time": end_time - start_time,
        "all_gaps": gaps,
        "gaps_times": gaps_times,
        "optimal_values": optimal_values,
        "heuristic_values": heuristic_values,
        "code_path_nums": code_path_nums,
    }

    # Save detailed results
    results_path = os.path.join(save_dir, "hill_climbing_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save best sample details
    if best_global_sample:
        best_sample_path = os.path.join(save_dir, "best_sample.json")
        with open(best_sample_path, "w") as f:
            json.dump(
                {
                    "sample": best_global_sample,
                    "gap": max_global_gap,
                    "optimal_value": (
                        optimal_values[gaps.index(max_global_gap)]
                        if max_global_gap in gaps
                        else None
                    ),
                    "heuristic_value": (
                        heuristic_values[gaps.index(max_global_gap)]
                        if max_global_gap in gaps
                        else None
                    ),
                },
                f,
                indent=2,
            )

    # Save gaps in the same format as other methods for consistency
    gaps_path = os.path.join(save_dir, "get_gap_outputs.json")
    with open(gaps_path, "w") as f:
        json.dump(gaps, f)

    # Save optimal values
    optimal_values_path = os.path.join(save_dir, "get_gap_outputs_optimal_values.json")
    with open(optimal_values_path, "w") as f:
        json.dump(optimal_values, f)

    # Save heuristic values
    heuristic_values_path = os.path.join(
        save_dir, "get_gap_outputs_heuristic_values.json"
    )
    with open(heuristic_values_path, "w") as f:
        json.dump(heuristic_values, f)

    # Save code path numbers
    code_path_nums_path = os.path.join(save_dir, "get_gap_outputs_code_path_nums.json")
    with open(code_path_nums_path, "w") as f:
        json.dump(code_path_nums, f)

    logger.info(f"Results saved to {save_dir}")
    logger.info(f"Hill climbing completed. Best gap: {max_global_gap}")

    # Cleanup
    logger.shutdown()
