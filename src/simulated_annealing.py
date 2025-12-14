from config import *
from common import *
from logger import AsyncLogger
from run_utils import get_save_dir, make_config
import random
import json
import os
import time
import math
import copy
random.seed(time.time())

def simulated_annealing_main(
    args,
    num_iterations=1000,
    initial_temperature=1.0,
    cooling_rate=0.99,
    num_neighbors=1,
    max_time=float('+inf')
):
    problem_description = get_problem_description(args)
    problem_description["method"] = "SimulatedAnnealing"
    metaease_params = MetaEase_specific_parameters()
    for param in metaease_params:
        if param in problem_description:
            del problem_description[param]

    problem_description["num_iterations"] = num_iterations
    problem_description["initial_temperature"] = initial_temperature
    problem_description["cooling_rate"] = cooling_rate
    problem_description["num_neighbors"] = num_neighbors
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

    # Get thresholds for simulated annealing - we only need input variable thresholds
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
    best_global_all_vars = None  # Single source of truth for best solution

    # Simulated annealing parameters (minimal version)
    temperature = initial_temperature
    min_temperature = 1e-8
    effective_cooling_rate = cooling_rate * 0.98
    current_sample = {}

    # Initialize current sample randomly within bounds when available
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
                current_sample[var_name] = min(max_val, max(min_val, random.uniform(min_val, max_val)))
            else:
                current_sample[var_name] = min(100.0, max(0.0, random.uniform(0.0, 100.0)))
        # Keep current_gap as -inf, other variables as None

    # Simulated annealing strategy
    logger.info(f"Starting simulated annealing with {num_iterations} iterations")
    logger.info(
        f"Initial temperature: {initial_temperature}, Cooling rate: {cooling_rate}"
    )
    start_time = time.time()

    # Tracking variables for results
    gaps = []
    gaps_times = []
    optimal_values = []
    heuristic_values = []
    all_optimal_vars = []
    code_path_nums = []
    temperatures = []
    successful_evaluations = 0
    accepted_moves = 0
    iteration = 0

    for iteration in range(num_iterations * 1000):
        if time.time() - start_time >= max_time: # or temperature <= min_temperature:
            break

        # Generate a single neighbor by perturbing one variable with Gaussian noise
        neighbor = copy.deepcopy(current_sample)
        var_to_perturb = random.choice(all_klee_vars)
        if var_to_perturb in input_thresholds:
            min_val, max_val = input_thresholds[var_to_perturb]
            std_dev = max(1e-12, (max_val - min_val) * 0.02)
            neighbor[var_to_perturb] = max(
                min_val,
                min(max_val, neighbor[var_to_perturb] + random.gauss(0, std_dev)),
            )
        else:
            std_dev = 0.2
            neighbor[var_to_perturb] = max(0.0, neighbor[var_to_perturb] + random.gauss(0, std_dev))

        # Evaluate neighbor
        try:
            (
                neighbor_gap,
                _,
                (neighbor_opt_val, neighbor_heur_val),
                neighbor_all_vars,
                neighbor_code_path_num,
            ) = problem.get_gaps_process_input(neighbor)
            successful_evaluations += 1
        except Exception as e:
            logger.warning(
                f"Error evaluating neighbor at iteration {iteration}: {e}"
            )
            # Record current state and continue (temperature still cools)
            gaps.append(current_gap)
            gaps_times.append(time.time() - start_time)
            optimal_values.append(current_opt_val)
            heuristic_values.append(current_heur_val)
            all_optimal_vars.append(current_all_vars)
            code_path_nums.append(current_code_path_num)
            temperatures.append(temperature)
            temperature = max(temperature * effective_cooling_rate, min_temperature)
            continue

        # Metropolis accept/reject decision (classic SA)
        if temperature < 0.01 and random.random() < 0.3:
            gaps.append(current_gap)
            gaps_times.append(time.time() - start_time)
            optimal_values.append(current_opt_val)
            heuristic_values.append(current_heur_val)
            all_optimal_vars.append(current_all_vars)
            code_path_nums.append(current_code_path_num)
            temperatures.append(temperature)
            temperature = max(temperature * effective_cooling_rate, min_temperature)
            continue

        gap_difference = neighbor_gap - current_gap
        if gap_difference >= 0:
            accept = True
        else:
            T = max(temperature, min_temperature)
            accept = random.random() < math.exp(0.8 * gap_difference / T)

        if accept:
            current_sample = neighbor
            current_gap = neighbor_gap
            current_opt_val = neighbor_opt_val
            current_heur_val = neighbor_heur_val
            current_all_vars = neighbor_all_vars
            current_code_path_num = neighbor_code_path_num
            accepted_moves += 1

            logger.debug(
                f"Iteration {iteration}: Accepted neighbor, gap: {current_gap}"
            )

            # Update global best if necessary
            if current_gap > max_global_gap:
                if abs(current_gap - max_global_gap) < 1e-6 and random.random() < 0.1:
                    pass
                else:
                    print(f"SA: Time: {time.time() - start_time}, iteration: {iteration}, gap: {current_gap}")
                    max_global_gap = current_gap
                    best_global_sample = copy.deepcopy(current_sample)
                    best_global_all_vars = current_all_vars
                    # logger.info(
                    #     f"Iteration {iteration}: New best gap found: {max_global_gap}"
                    # )
                    logger.log_current_max_gap(max_global_gap)

        # Record and cool
        gaps.append(current_gap)
        gaps_times.append(time.time() - start_time)
        optimal_values.append(current_opt_val)
        heuristic_values.append(current_heur_val)
        all_optimal_vars.append(current_all_vars)
        code_path_nums.append(current_code_path_num)
        temperatures.append(temperature)

        temperature = max(temperature * effective_cooling_rate, min_temperature)

        # Minimal periodic logging
        if (iteration + 1) % 1000 == 0:
            acceptance_rate = accepted_moves / (iteration + 1)
            logger.info(
                f"Iteration {iteration + 1}/{num_iterations}. "
                f"Current gap: {current_gap:.4f}, Max gap: {max_global_gap:.4f}, "
                f"Temperature: {temperature:.4f}, Acceptance rate: {acceptance_rate:.3f}"
            )

    end_time = time.time()
    logger.info(f"Simulated annealing completed in {end_time - start_time:.2f} seconds")
    logger.log_final_max_gap(max_global_gap)
    logger.info(
        f"Success rate: {successful_evaluations}/{iteration} ({(100*successful_evaluations/max(1, iteration)):.1f}%)"
    )
    logger.info(f"Final acceptance rate: {accepted_moves/max(1, iteration):.3f}")

    # Save results
    results = {
        "max_gap": max_global_gap,
        "best_sample": best_global_sample,
        "num_iterations": iteration,
        "num_neighbors": num_neighbors,
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "min_temperature": min_temperature,
        "successful_evaluations": successful_evaluations,
        "total_evaluations": iteration,
        "success_rate": (
            successful_evaluations / iteration
            if iteration > 0
            else 0
        ),
        "accepted_moves": accepted_moves,
        "acceptance_rate":  accepted_moves / iteration if iteration > 0 else 0,
        "execution_time": end_time - start_time,
        "all_gaps": gaps,
        "gaps_times": gaps_times,
        "optimal_values": optimal_values,
        "heuristic_values": heuristic_values,
        "code_path_nums": code_path_nums,
        "temperatures": temperatures,
    }

    # Save detailed results
    results_path = os.path.join(save_dir, "simulated_annealing_results.json")
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

    # Save temperature history
    temperatures_path = os.path.join(save_dir, "temperature_history.json")
    with open(temperatures_path, "w") as f:
        json.dump(temperatures, f)

    logger.info(f"Results saved to {save_dir}")
    logger.info(f"Simulated annealing completed. Best gap: {max_global_gap}")

    # Cleanup
    logger.shutdown()
