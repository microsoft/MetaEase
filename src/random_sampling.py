from config import *
from common import *
from logger import AsyncLogger
from run_utils import get_save_dir, make_config
import random
import json
import os
import time


def random_sampling_main(args, num_samples=8000, max_time=float('+inf')):
    problem_description = get_problem_description(args)
    problem_description["method"] = "Random"
    metaease_params = MetaEase_specific_parameters()
    for param in metaease_params:
        if param in problem_description:
            del problem_description[param]

    problem_description["num_samples"] = num_samples
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

    # Get thresholds for random sampling - we only need input variable thresholds
    # Create a dummy relaxed_all_vars with just the input variables to get their thresholds
    dummy_relaxed_vars = {var: 0.0 for var in all_klee_vars}
    thresholds = problem.get_thresholds(dummy_relaxed_vars)

    # Filter thresholds to only include input variables (klee variables)
    input_thresholds = {
        var: thresholds[var] for var in all_klee_vars if var in thresholds
    }
    logger.info(f"Using input thresholds: {input_thresholds}")

    max_global_gap = float("-inf")
    best_global_sample = None
    best_global_optimal_all_vars = None
    best_global_heuristic_all_vars = None

    # Random sampling strategy
    logger.info(f"Starting random sampling with {num_samples} samples and max time {max_time}")
    start_time = time.time()

    # Generate random samples
    random_samples = []
    gaps = []
    gaps_times = []
    optimal_values = []
    heuristic_values = []
    all_optimal_vars = []
    code_path_nums = []
    successful_evaluations = 0
    while time.time() - start_time < max_time:
        sample = {}
        for var_name in all_klee_vars:
            if var_name in input_thresholds:
                min_val, max_val = input_thresholds[var_name]
                # Generate random value within threshold bounds
                sample[var_name] = random.uniform(min_val, max_val)
            else:
                sample[var_name] = random.uniform(0.0, 100.0)
        random_samples.append(sample)

        try:
            # Compute gap for this sample
            gap, _, (opt_val, heur_val), all_vars, code_path_num = (
                problem.get_gaps_process_input(sample)
            )

            gaps.append(gap)
            gaps_times.append(time.time() - start_time)
            optimal_values.append(opt_val)
            heuristic_values.append(heur_val)
            all_optimal_vars.append(all_vars)
            code_path_nums.append(code_path_num)
            successful_evaluations += 1

            # Update best sample if this gap is larger
            if gap > max_global_gap:
                print(f"RS: Time: {time.time() - start_time}, num_samples: {len(random_samples)}, gap: {gap}")
                max_global_gap = gap
                best_global_sample = sample
                best_global_optimal_all_vars = all_vars
                best_global_heuristic_all_vars = (
                    all_vars  # This would need to be computed separately if needed
                )
                # logger.info(f"New best gap found: {max_global_gap} at sample {i}")
                logger.log_current_max_gap(max_global_gap)

        except Exception as e:
            logger.warning(f"Error processing sample {len(random_samples)}: {e}")
            continue

    end_time = time.time()
    logger.info(f"Random sampling completed in {end_time - start_time:.2f} seconds")
    logger.log_final_max_gap(max_global_gap)
    logger.info(
        f"Success rate: {successful_evaluations}/{len(random_samples)} ({100*successful_evaluations/len(random_samples):.1f}%)"
    )

    # Save results
    results = {
        "max_gap": max_global_gap,
        "best_sample": best_global_sample,
        "num_samples_evaluated": len(gaps),
        "total_samples_generated": len(random_samples),
        "successful_evaluations": successful_evaluations,
        "success_rate": (
            successful_evaluations / len(random_samples) if random_samples else 0
        ),
        "execution_time": end_time - start_time,
        "all_gaps": gaps,
        "gaps_times": gaps_times,
        "optimal_values": optimal_values,
        "heuristic_values": heuristic_values,
        "code_path_nums": code_path_nums,
    }

    # Save detailed results
    results_path = os.path.join(save_dir, "random_sampling_results.json")
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
    logger.info(f"Random sampling baseline completed. Best gap: {max_global_gap}")

    # Cleanup
    logger.shutdown()
