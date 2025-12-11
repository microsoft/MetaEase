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
import concurrent.futures
from functools import partial

random.seed(time.time())

def sample_based_gradient_main(args, num_iterations=1000, k_samples=5, step_size=1.0, gradient_epsilon=1e-6, max_time=float('+inf'), seed_path=None, enforce_num_iterations=False):
    """
    Sample-based gradient optimization method.
    
    Args:
        args: Command line arguments
        num_iterations: Maximum number of iterations
        k_samples: Number of random samples to start with and use for gradient estimation
        step_size: Step size for gradient updates
        gradient_epsilon: Small value for numerical gradient estimation
        max_time: Maximum time limit for optimization
    """
    problem_description = get_problem_description(args)
    problem_description["method"] = "SampleBasedGradient"
    metaease_params = MetaEase_specific_parameters()
    for param in metaease_params:
        if param in problem_description:
            del problem_description[param]

    problem_description["num_iterations"] = num_iterations
    problem_description["k_samples"] = k_samples
    problem_description["step_size"] = step_size
    problem_description["gradient_epsilon"] = gradient_epsilon
    problem_description["max_time"] = max_time
    
    # Configuration
    problem_type = problem_description["problem_type"]
    parameters = get_parameters(problem_description)
    save_dir = get_save_dir(args.base_save_dir, problem_description)

    # Initialize logger
    logger = AsyncLogger.init(save_dir)
    logger.debug("Sample-based gradient optimization started")
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

    # Get thresholds for optimization variables
    dummy_relaxed_vars = {var: 0.0 for var in all_klee_vars}
    thresholds = problem.get_thresholds(dummy_relaxed_vars)
    input_thresholds = {
        var: thresholds[var] for var in all_klee_vars if var in thresholds
    }
    logger.info(f"Using input thresholds: {input_thresholds}")

    # Initialize tracking variables
    max_global_gap = float("-inf")
    best_global_sample = None
    best_global_all_vars = None

    # Counter for get_gaps_process_input calls
    get_gaps_call_count = 0

    # Initialize k random samples
    current_samples = []
    if seed_path is not None:
        with open(seed_path, "r") as f:
            seed_samples = json.load(f)
        for sample_name, sample in seed_samples.items():
            current_samples.append(sample)
    else:
        for i in range(k_samples):
            sample = {}
            for var_name in all_klee_vars:
                if var_name in input_thresholds:
                    min_val, max_val = input_thresholds[var_name]
                    sample[var_name] = random.uniform(min_val, max_val)
                else:
                    sample[var_name] = random.uniform(0.0, 100.0)
            current_samples.append(sample)

    logger.info(f"Starting sample-based gradient optimization with {num_iterations} iterations with {enforce_num_iterations}")
    logger.info(f"Initial samples: {len(current_samples)}, Step size: {step_size}, Gradient epsilon: {gradient_epsilon}")
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
    actual_iters = 0
    actual_evals = 0
    iteration = 0

    # Evaluate initial samples in parallel
    initial_results, get_gaps_call_count = evaluate_samples_parallel(problem, current_samples, logger, get_gaps_call_count)
    
    # Find best initial sample
    best_initial_idx = -1
    best_initial_gap = float("-inf")
    for i, result in enumerate(initial_results):
        if result is not None:
            gap, opt_val, heur_val, all_vars, code_path_num = result
            successful_evaluations += 1
            actual_evals += 1
            
            if gap > best_initial_gap:
                best_initial_gap = gap
                best_initial_idx = i
                max_global_gap = gap
                gaps_times.append(time.time() - start_time)
                best_global_sample = copy.deepcopy(current_samples[i])
                best_global_all_vars = all_vars
                logger.log_current_max_gap(gap)

    if best_initial_idx == -1:
        logger.error("All initial samples failed to evaluate")
        raise RuntimeError("All initial samples failed to evaluate")

    logger.info(f"Best initial sample has gap: {best_initial_gap}")

    while (time.time() - start_time < max_time) or (enforce_num_iterations and iteration < num_iterations):
        iteration += 1
        actual_iters += 1

        # Estimate gradient for each sample using finite differences
        gradients = []
        
        for sample_idx, sample in enumerate(current_samples):
            gradient, get_gaps_call_count = estimate_gradient_finite_diff(
                problem, sample, all_klee_vars, input_thresholds, 
                gradient_epsilon, logger, get_gaps_call_count
            )
            gradients.append(gradient)
            actual_evals += len(all_klee_vars) * 2  # Each gradient estimation uses 2*num_vars evaluations

        # Update each sample using its gradient
        new_samples = []
        for sample_idx, (sample, gradient) in enumerate(zip(current_samples, gradients)):
            if gradient is not None:
                new_sample = update_sample_with_gradient(
                    sample, gradient, step_size, input_thresholds, all_klee_vars
                )
                new_samples.append(new_sample)
            else:
                # If gradient estimation failed, keep the original sample
                new_samples.append(copy.deepcopy(sample))

        current_samples = new_samples

        # Evaluate new samples in parallel
        results, get_gaps_call_count = evaluate_samples_parallel(problem, current_samples, logger, get_gaps_call_count)
        actual_evals += len(current_samples)

        # Track results and find best sample in this iteration
        iteration_best_gap = float("-inf")
        iteration_best_sample = None
        iteration_best_all_vars = None
        
        for i, result in enumerate(results):
            if result is not None:
                gap, opt_val, heur_val, all_vars, code_path_num = result
                successful_evaluations += 1
                
                # Record results
                gaps.append(gap)
                gaps_times.append(time.time() - start_time)
                optimal_values.append(opt_val)
                heuristic_values.append(heur_val)
                all_optimal_vars.append(all_vars)
                code_path_nums.append(code_path_num)
                
                # Check if this is the best in this iteration
                if gap > iteration_best_gap:
                    iteration_best_gap = gap
                    iteration_best_sample = copy.deepcopy(current_samples[i])
                    iteration_best_all_vars = all_vars

        # Update global best if we found a better sample
        if iteration_best_gap > max_global_gap:
            print(f"SBG: Time: {time.time() - start_time}, iteration: {iteration}, gap: {iteration_best_gap}, get_gaps_calls: {get_gaps_call_count}")
            max_global_gap = iteration_best_gap
            best_global_sample = iteration_best_sample
            best_global_all_vars = iteration_best_all_vars
            improvements += 1
            logger.log_current_max_gap(max_global_gap)

        # Log progress
        if iteration % 100 == 0:
            improvement_rate = improvements / iteration if iteration > 0 else 0
            logger.info(
                f"Iteration {iteration}/{num_iterations}. "
                f"Best gap this iteration: {iteration_best_gap:.4f}, "
                f"Global best gap: {max_global_gap:.4f}, "
                f"Improvements: {improvements}, "
                f"Improvement rate: {improvement_rate:.3f}, "
                f"get_gaps_calls: {get_gaps_call_count}"
            )

    end_time = time.time()
    logger.info(f"Sample-based gradient optimization completed in {end_time - start_time:.2f} seconds")
    logger.log_final_max_gap(max_global_gap)
    logger.info(f"Total get_gaps_process_input calls: {get_gaps_call_count}")
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
        "k_samples": k_samples,
        "step_size": step_size,
        "gradient_epsilon": gradient_epsilon,
        "successful_evaluations": successful_evaluations,
        "actual_evaluations": actual_evals,
        "actual_iterations": actual_iters,
        "gap_computations": get_gaps_call_count,
        "success_rate": (
            successful_evaluations / max(1, actual_evals)
            if actual_evals > 0
            else 0
        ),
        "improvements": improvements,
        "improvement_rate": improvements / max(1, actual_iters) if actual_iters > 0 else 0,
        "execution_time": end_time - start_time,
        "all_gaps": gaps,
        "gaps_times": gaps_times,
        "optimal_values": optimal_values,
        "heuristic_values": heuristic_values,
        "code_path_nums": code_path_nums,
    }

    # Save detailed results
    results_path = os.path.join(save_dir, "sample_based_gradient_results.json")
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
    logger.info(f"Sample-based gradient optimization completed. Best gap: {max_global_gap}")
    print(f"SBG: Total get_gaps_process_input calls: {get_gaps_call_count}")

    # Cleanup
    logger.shutdown()


def evaluate_samples_parallel(problem, samples, logger, get_gaps_call_count, max_workers=None):
    """
    Evaluate multiple samples in parallel.

    Returns:
        Tuple of (results, updated_call_count) where results is a list of results,
        each result is either (gap, opt_val, heur_val, all_vars, code_path_num) or None if failed
    """
    call_count = 0

    def evaluate_single_sample(sample):
        nonlocal call_count
        try:
            call_count += 1
            gap, _, (opt_val, heur_val), all_vars, code_path_num = problem.get_gaps_process_input(sample)
            return (gap, opt_val, heur_val, all_vars, code_path_num)
        except Exception as e:
            return None

    # Use ThreadPoolExecutor for parallel evaluation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_sample, sample) for sample in samples]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"Error in parallel evaluation: {e}")
                results.append(None)
    
    return results, get_gaps_call_count + call_count


def estimate_gradient_finite_diff(problem, sample, all_klee_vars, input_thresholds, epsilon, logger, get_gaps_call_count):
    """
    Estimate gradient using finite differences.

    Returns:
        Tuple of (gradient, updated_call_count) where gradient is a dictionary mapping
        variable names to gradient values, or (None, updated_call_count) if estimation failed
    """
    print("estimate_gradient_finite_diff")
    gradient = {}
    base_gap = None
    call_count = get_gaps_call_count

    try:
        # Get base function value
        call_count += 1
        base_gap, _, _, _, _ = problem.get_gaps_process_input(sample)
    except Exception as e:
        logger.warning(f"Failed to evaluate base sample for gradient estimation: {e}")
        return None, call_count
    
    for var_name in all_klee_vars:
        try:
            # Create perturbed samples
            sample_plus = copy.deepcopy(sample)
            sample_minus = copy.deepcopy(sample)
            
            # Determine step size based on variable bounds
            if var_name in input_thresholds:
                min_val, max_val = input_thresholds[var_name]
                step = epsilon * (max_val - min_val)
            else:
                step = epsilon * 100.0  # Default range
            
            sample_plus[var_name] += step
            sample_minus[var_name] -= step
            
            # Ensure bounds
            if var_name in input_thresholds:
                min_val, max_val = input_thresholds[var_name]
                sample_plus[var_name] = max(min_val, min(max_val, sample_plus[var_name]))
                sample_minus[var_name] = max(min_val, min(max_val, sample_minus[var_name]))
            else:
                sample_plus[var_name] = max(0.0, min(1000.0, sample_plus[var_name]))
                sample_minus[var_name] = max(0.0, min(1000.0, sample_minus[var_name]))
            
            # Evaluate perturbed samples
            call_count += 1
            gap_plus, _, _, _, _ = problem.get_gaps_process_input(sample_plus)
            call_count += 1
            gap_minus, _, _, _, _ = problem.get_gaps_process_input(sample_minus)
            
            # Compute finite difference gradient
            if abs(sample_plus[var_name] - sample_minus[var_name]) > 1e-12:
                gradient[var_name] = (gap_plus - gap_minus) / (sample_plus[var_name] - sample_minus[var_name])
            else:
                gradient[var_name] = 0.0
                
        except Exception as e:
            logger.warning(f"Failed to compute gradient for variable {var_name}: {e}")
            gradient[var_name] = 0.0
    
    return gradient, call_count


def update_sample_with_gradient(sample, gradient, step_size, input_thresholds, all_klee_vars):
    """
    Update sample using gradient ascent (since we want to maximize the gap).
    
    Returns:
        Updated sample dictionary
    """
    new_sample = copy.deepcopy(sample)
    
    for var_name in all_klee_vars:
        if var_name in gradient:
            # Gradient ascent update
            update = step_size * gradient[var_name]
            new_sample[var_name] += update
            
            # Ensure bounds
            if var_name in input_thresholds:
                min_val, max_val = input_thresholds[var_name]
                new_sample[var_name] = max(min_val, min(max_val, new_sample[var_name]))
            else:
                new_sample[var_name] = max(0.0, min(1000.0, new_sample[var_name]))
    
    return new_sample