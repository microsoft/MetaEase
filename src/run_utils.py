from analysis_utils import *
from opt_utils import *
from config import *
from common import *
from multiprocessing import Pool, cpu_count, Manager
import random
import time

random.seed(91)
from clustering_utils import *
import sys

DO_OPTIMAL_COMBINATION_SEARCH = False
MAX_NUM_CORES = int(cpu_count() * 0.8)


def process_klee_batch(args):
    """Process a batch of klee inputs in parallel.

    Args:
        args (tuple): Contains (batch_inputs, problem, save_dir, parameters, assigned_fixed_keys, start_idx, optimization_start_time, filtered_results)

    Returns:
        list: Results from processing each input in the batch
    """
    try:
        (
            batch_inputs,
            problem,
            save_dir,
            parameters,
            assigned_fixed_keys,
            start_idx,
            optimization_start_time,
            filtered_results,
        ) = args
    except Exception as e:
        print(f"Error unpacking batch arguments: {str(e)}")
        return []

    results = []
    # Pre-create all batch directories at once
    batch_dirs = [
        f"{save_dir}/klee_input_{start_idx + i}" for i in range(len(batch_inputs))
    ]
    for dir_path in batch_dirs:
        os.makedirs(dir_path, exist_ok=True)

    for i, klee_input_value in enumerate(batch_inputs):
        # Check time limit at the start of each input processing
        if (
            optimization_start_time is not None
            and parameters.get("max_total_time") is not None
        ):
            if time.time() - optimization_start_time > parameters["max_total_time"]:
                print(
                    f"Time limit of {parameters['max_total_time']} seconds reached in process_klee_batch"
                )
                return results

        current_idx = start_idx + i
        batch_dir = batch_dirs[i]

        try:
            # Get pre-computed values if available
            pre_computed_values = None
            if filtered_results and parameters.get("use_gaps_in_filtering", False):
                # Find matching result in filtered_results
                for result in filtered_results:
                    if result["input_value"] == klee_input_value:
                        pre_computed_values = result
                        break

            # Cache the keys to avoid repeated dictionary operations
            input_keys = list(klee_input_value.keys()) if klee_input_value else []
            if not input_keys:
                print(f"Warning: Empty input keys for klee input {current_idx}")
                results.append((current_idx, [], None, None, None))
                continue

            (
                gap_list,
                relaxed_gap_list,
                best_sample_list,
                optimal_all_vars,
                heuristic_all_vars,
            ) = gradient_ascent_loop(
                problem,
                klee_input_value,
                input_keys,
                batch_dir,
                parameters,
                assigned_fixed_keys=assigned_fixed_keys if parameters.get("freeze_cluster_fixed_keys", False) else None,
                pre_computed_values=pre_computed_values,
            )
            if gap_list and isinstance(gap_list, list):
                try:
                    # Find the maximum gap value and its corresponding index
                    valid_gaps = [(i, g) for i, g in gap_list if g is not None]
                    if valid_gaps:
                        # Find the maximum gap and its original iteration index
                        max_gap_entry = max(valid_gaps, key=lambda x: x[1])
                        best_idx = max_gap_entry[0]  # This is the original iteration index
                        max_gap = max_gap_entry[1]
                        # Find corresponding best sample
                        best_sample = None
                        for sample_idx, sample in best_sample_list:
                            if sample_idx == best_idx:
                                best_sample = sample
                                break
                        # If no exact match found, try to find the closest sample
                        if best_sample is None and best_sample_list:
                            # Find the sample with the closest index
                            closest_idx = min(best_sample_list, key=lambda x: abs(x[0] - best_idx))
                            best_sample = closest_idx[1]
                            print(f"Warning: No exact sample match for idx {best_idx}, using closest idx {closest_idx[0]}")

                        if best_sample is not None:
                            print(f"Found best sample for klee input {current_idx}: gap={max_gap}, iteration={best_idx}")
                            best_result = (
                                current_idx,
                                gap_list,  # Return the full gap_list instead of just max_gap
                                best_sample,
                                optimal_all_vars,
                                heuristic_all_vars,
                            )

                            # Batch write operations
                            try:
                                with open(
                                    f"{batch_dir}/gap_list.json", "w"
                                ) as f1, open(
                                    f"{batch_dir}/best_sample_list.json", "w"
                                ) as f2:
                                    json.dump(gap_list, f1)
                                    json.dump(best_sample_list, f2)
                            except Exception as e:
                                print(
                                    f"Error saving results for klee input {current_idx}: {str(e)}"
                                )

                            results.append(best_result)
                        else:
                            print(
                                f"Warning: No valid best sample found for klee input {current_idx}"
                            )
                            results.append((current_idx, gap_list, None, None, None))
                    else:
                        print(
                            f"Warning: No valid gaps found for klee input {current_idx}"
                        )
                        results.append((current_idx, gap_list, None, None, None))
                except Exception as e:
                    print(
                        f"Error processing gap list for klee input {current_idx}: {str(e)}"
                    )
                    results.append((current_idx, gap_list, None, None, None))
            else:
                print(
                    f"Warning: Invalid or empty gap list for klee input {current_idx}"
                )
                results.append((current_idx, gap_list, None, None, None))

        except Exception as e:
            print(f"Error processing klee input {current_idx}: {str(e)}")
            results.append((current_idx, [], None, None, None))

    # Return results along with counter values from this worker process
    return results, problem.num_compute_optimal_value_called, problem.num_compute_heuristic_value_called


def process_klee_inputs_parallel(
    filtered_klee_input_values,
    problem,
    save_dir,
    parameters,
    assigned_fixed_keys,
    batch_size=None,
    optimization_start_time=None,
    filtered_results=None,
):
    """Process all klee inputs in parallel using dynamic batching.

    Args:
        filtered_klee_input_values (list): List of klee input values to process
        problem: Problem instance
        save_dir (str): Directory to save results
        parameters (dict): Parameters for processing
        batch_size (int, optional): Override automatic batch size calculation
        optimization_start_time (float, optional): Start time of optimization for time limit checks
        filtered_results (list, optional): List of filtered results containing pre-computed values

    Returns:
        tuple: (final_results, max_gap_found, best_sample_found, best_optimal_all_vars_found, best_heuristic_all_vars_found)
    """
    # Check time limit if specified
    if (
        optimization_start_time is not None
        and parameters.get("max_total_time") is not None
    ):
        if time.time() - optimization_start_time > parameters["max_total_time"]:
            print(
                f"Time limit of {parameters['max_total_time']} seconds reached in process_klee_inputs_parallel"
            )
            return [], float("-inf"), None, None, None

    # Use 80% of available cores to avoid system overload
    num_cores = min(MAX_NUM_CORES, max(1, int(cpu_count() * 0.8)))
    num_inputs = len(filtered_klee_input_values)

    # Dynamic batch size calculation based on input size and memory constraints
    if batch_size is None:
        # Calculate optimal batch size based on system memory and input size
        target_batch_memory = 100 * 1024 * 1024  # 100MB in bytes
        estimated_input_size = sys.getsizeof(filtered_klee_input_values[0])
        batch_size = max(
            1,
            min(
                MAX_NUM_CORES,  # Max batch size cap
                target_batch_memory // (estimated_input_size * num_cores),
                num_inputs // (num_cores * 4),  # Ensure at least 4 batches per core
            ),
        )

    print(
        f"Processing {num_inputs} inputs using {num_cores} cores with batch size {batch_size}"
    )

    # Pre-calculate batches to reduce overhead
    batches = [
        (
            filtered_klee_input_values[i : min(i + batch_size, num_inputs)],
            problem,
            save_dir,
            parameters,
            assigned_fixed_keys,
            i,
            optimization_start_time,
            filtered_results,  # Pass filtered_results to each batch
        )
        for i in range(0, num_inputs, batch_size)
    ]

    # Initialize shared memory manager for progress tracking
    manager = Manager()
    progress_dict = manager.dict()
    progress_lock = manager.Lock()  # Create a separate lock for synchronization
    progress_dict["completed"] = 0
    progress_dict["best_gap"] = float("-inf")

    def update_progress(batch_results):
        try:
            with progress_lock:  # Use the separate lock for synchronization
                progress_dict["completed"] += 1
                current_gaps = []
                for result in batch_results:
                    if result[1] is not None:
                        gap_value = result[1]
                        if isinstance(gap_value, (list, tuple)):
                            gaps = [
                                g[1]
                                for g in gap_value
                                if isinstance(g, (list, tuple)) and len(g) > 1
                            ]
                            if gaps:
                                current_gaps.append(max(gaps))
                        else:
                            current_gaps.append(gap_value)

                if current_gaps:
                    current_max = max(current_gaps)
                    progress_dict["best_gap"] = max(
                        progress_dict["best_gap"], current_max
                    )

                progress = (progress_dict["completed"] / len(batches)) * 100
                if (
                    progress_dict["completed"] % max(1, len(batches) // 10) == 0
                ):  # Print every 10%
                    print(
                        f"Progress: {progress:.1f}% ({progress_dict['completed']}/{len(batches)} batches)"
                    )
                    print(f"Current best gap: {progress_dict['best_gap']:.3f}")
        except Exception as e:
            print(f"Error in update_progress: {str(e)}")

    # Process batches in parallel with improved memory management
    all_results = []
    max_gap_found = float("-inf")
    best_sample_found = None
    best_optimal_all_vars_found = None
    best_heuristic_all_vars_found = None

    # Accumulate counter values from all worker processes
    total_optimal_calls = 0
    total_heuristic_calls = 0

    with Pool(processes=num_cores) as pool:
        try:
            # Use imap_unordered for better load balancing
            for batch_results in pool.imap_unordered(
                process_klee_batch,
                batches,
                chunksize=max(1, len(batches) // (num_cores * 4)),
            ):
                # Check time limit after each batch
                if (
                    optimization_start_time is not None
                    and parameters.get("max_total_time") is not None
                ):
                    if (
                        time.time() - optimization_start_time
                        > parameters["max_total_time"]
                    ):
                        print(
                            f"Time limit reached after batch, terminating remaining batches"
                        )
                        pool.terminate()
                        break

                if batch_results:
                    # Unpack the new return format: (results, optimal_calls, heuristic_calls)
                    if len(batch_results) == 3 and not isinstance(batch_results[0], tuple):
                        # New format with counters
                        actual_results, optimal_calls, heuristic_calls = batch_results
                        total_optimal_calls += optimal_calls
                        total_heuristic_calls += heuristic_calls
                        print(f"Current total optimal calls: {total_optimal_calls}")
                        print(f"Current total heuristic calls: {total_heuristic_calls}")
                    else:
                        # Old format or empty results
                        actual_results = batch_results

                    for result in actual_results:
                        idx, gap, sample, optimal_all_vars, heuristic_all_vars = result
                        if gap is not None and gap:  # Check that gap is not None and not empty
                            if isinstance(gap, (list, tuple)):
                                # Handle case where gap is a list of (iteration, gap) tuples
                                max_gap_in_batch = max(
                                    g[1]
                                    for g in gap
                                    if isinstance(g, (tuple, list)) and len(g) > 1
                                )
                            else:
                                max_gap_in_batch = gap

                            if max_gap_in_batch >= max_gap_found:
                                print(f"Updating best sample: max_gap_in_batch={max_gap_in_batch}, sample={sample is not None}")
                                max_gap_found = max_gap_in_batch
                                best_sample_found = sample
                                best_optimal_all_vars_found = optimal_all_vars
                                best_heuristic_all_vars_found = heuristic_all_vars

                    all_results.extend(actual_results)
                    update_progress(actual_results)

        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            pool.terminate()
            pool.join()

    # Sort and filter results
    valid_results = [
        (idx, gap, sample, optimal_all_vars, heuristic_all_vars)
        for idx, gap, sample, optimal_all_vars, heuristic_all_vars in all_results
        if gap is not None and gap  # Check that gap is not None and not empty
    ]
    valid_results.sort(key=lambda x: x[0])

    # Convert to final format
    final_results = []
    for idx, gap, sample, optimal_all_vars, heuristic_all_vars in valid_results:
        if isinstance(gap, (list, tuple)):
            final_results.append(
                (idx, gap, None, [sample], optimal_all_vars, heuristic_all_vars)
            )
        else:
            final_results.append(
                (
                    idx,
                    [(0, gap)],
                    None,
                    [(0, sample)],
                    optimal_all_vars,
                    heuristic_all_vars,
                )
            )

    print(f"Successfully processed {len(valid_results)}/{num_inputs} inputs")
    print(f"Maximum gap found across all batches: {max_gap_found}")

    if best_sample_found is None:
        print(f"\033[91mERROR: best_sample_found is None!\033[0m")

    # Save the best result found
    with open(f"{save_dir}/best_result.json", "w") as f:
        json.dump(
            {
                "max_gap": max_gap_found,
                "best_sample": best_sample_found,
                "best_optimal_all_vars": best_optimal_all_vars_found,
                "best_heuristic_all_vars": best_heuristic_all_vars_found,
            },
            f,
        )

    return (
        final_results,
        max_gap_found,
        best_sample_found,
        best_optimal_all_vars_found,
        best_heuristic_all_vars_found,
        total_optimal_calls,
        total_heuristic_calls,
    )


def filter_single_klee_input(args):
    """Process a single KLEE input for filtering."""
    klee_input_value, problem, use_gaps_in_filtering, minimize_is_better = args
    args_dict = problem.convert_input_dict_to_args(klee_input_value)
    heuristic_result = problem.compute_heuristic_value(args_dict)
    heuristic_value = heuristic_result["heuristic_value"]
    code_path = heuristic_result["code_path_num"]

    if use_gaps_in_filtering:
        optimal_result = problem.compute_optimal_value(args_dict)
        optimal_value = optimal_result["optimal_value"]
        optimal_all_vars = optimal_result["all_vars"]
        gap = optimal_value - heuristic_value if optimal_value is not None else None
        if minimize_is_better:
            gap = -gap
    else:
        gap = None
        optimal_value = None
        optimal_all_vars = None

    return {
        "gap": gap,
        "code_path": code_path,
        "input_value": klee_input_value,
        "heuristic_value": heuristic_value,
        "optimal_value": optimal_value,
        "optimal_all_vars": optimal_all_vars,
    }


def filter_klee_inputs(klee_input_values, problem, use_gaps_in_filtering=False, minimize_is_better=False, remove_zero_gap_inputs=False, keep_redundant_code_paths=False):
    """Filter KLEE inputs in parallel with improved memory management and progress tracking."""
    print(f"Filtering {len(klee_input_values)} klee inputs in parallel")

    # Use 90% of cores but cap to prevent system overload
    num_cores = min(MAX_NUM_CORES, max(1, int(cpu_count() * 0.9)))

    # Calculate optimal chunk size based on available memory
    total_inputs = len(klee_input_values)
    chunk_size = max(100, min(1000, total_inputs // (num_cores * 4)))
    num_chunks = (total_inputs + chunk_size - 1) // chunk_size

    filtered_results = []  # Store all results including gaps and values
    added_code_paths = set()

    # Create a manager for shared progress tracking
    manager = Manager()
    progress = manager.Value("i", 0)
    total_processed = manager.Value("i", 0)
    progress_lock = manager.Lock()

    def update_progress(chunk_results):
        with progress_lock:
            total_processed.value += len(chunk_results)
            current_progress = (total_processed.value / total_inputs) * 100
            if current_progress - progress.value >= 10:
                progress.value = int(current_progress)
                print(
                    f"Progress: {progress.value}% ({total_processed.value}/{total_inputs} inputs)"
                )
    all_results = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_inputs)
        chunk = klee_input_values[start_idx:end_idx]

        # Prepare batch arguments
        args_list = [
            (input_value, problem, use_gaps_in_filtering, minimize_is_better) for input_value in chunk
        ]
        optimal_batch_size = max(1, len(chunk) // num_cores)

        with Pool(processes=num_cores) as pool:
            try:
                chunk_results = list(
                    pool.imap_unordered(
                        filter_single_klee_input,
                        args_list,
                        chunksize=optimal_batch_size,
                    )
                )

                # Process results for this chunk
                with progress_lock:
                    for result in chunk_results:
                        if result["code_path"] not in added_code_paths or (
                            result["gap"] is not None
                            and result["gap"] > 0.01
                            and use_gaps_in_filtering
                        ):
                            if remove_zero_gap_inputs and result["gap"] == 0:
                                continue
                            filtered_results.append(result)
                            if not keep_redundant_code_paths:
                                added_code_paths.add(result["code_path"])
                all_results.extend(chunk_results)
                update_progress(chunk_results)

                # Save intermediate results every 5 chunks or if it's the last chunk
                if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
                    try:
                        with open("filtered_intermediate.json", "w") as f:
                            json.dump(
                                {
                                    "filtered_results": filtered_results,
                                    "processed_count": total_processed.value,
                                    "unique_code_paths": len(added_code_paths),
                                },
                                f,
                            )
                    except Exception as e:
                        print(f"Warning: Failed to save intermediate results: {e}")

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                continue

    print(f"Filtering complete. Found {len(filtered_results)} unique code paths")
    # Handle cases with no filtered results
    if len(filtered_results) == 0:
        if len(all_results) > 0:
            # select a random result from all_results
            random_index = int(np.random.randint(0, len(all_results)))
            filtered_results = [all_results[random_index]]
            print(f"\033[91mWarning: No valid gaps found in filtering, selecting a random result from all_results\033[0m")
        else:
            # No results at all; return empty outputs gracefully
            return [], [], []
    # Extract just the input values and gaps for backward compatibility
    filtered_klee_input_values = [result["input_value"] for result in filtered_results]
    gaps = [result["gap"] for result in filtered_results if result["gap"] is not None]

    return filtered_klee_input_values, gaps, filtered_results


def filter_klee_inputs_sequential(klee_input_values, problem, use_gaps_in_filtering=False, minimize_is_better=False, remove_zero_gap_inputs=False, keep_redundant_code_paths=False):
    """Filter KLEE inputs sequentially with progress tracking."""
    print(f"Filtering {len(klee_input_values)} klee inputs sequentially")

    total_inputs = len(klee_input_values)
    filtered_results = []  # Store all results including gaps and values
    added_code_paths = set()
    all_results = []

    # Process inputs sequentially
    for i, klee_input_value in enumerate(klee_input_values):
        try:
            # Process single input using the same logic as filter_single_klee_input
            args_dict = problem.convert_input_dict_to_args(klee_input_value)
            heuristic_result = problem.compute_heuristic_value(args_dict)
            heuristic_value = heuristic_result["heuristic_value"]
            code_path = heuristic_result["code_path_num"]

            if use_gaps_in_filtering:
                optimal_result = problem.compute_optimal_value(args_dict)
                optimal_value = optimal_result["optimal_value"]
                optimal_all_vars = optimal_result["all_vars"]
                gap = optimal_value - heuristic_value if optimal_value is not None else None
                if minimize_is_better:
                    gap = -gap
            else:
                gap = None
                optimal_value = None
                optimal_all_vars = None

            result = {
                "gap": gap,
                "code_path": code_path,
                "input_value": klee_input_value,
                "heuristic_value": heuristic_value,
                "optimal_value": optimal_value,
                "optimal_all_vars": optimal_all_vars,
            }

            all_results.append(result)

            # Apply filtering logic
            if result["code_path"] not in added_code_paths or (
                result["gap"] is not None
                and result["gap"] > 0.01
                and use_gaps_in_filtering
            ):
                if remove_zero_gap_inputs and result["gap"] == 0:
                    continue
                filtered_results.append(result)
                if not keep_redundant_code_paths:
                    added_code_paths.add(result["code_path"])

            # Progress tracking
            if (i + 1) % max(1, total_inputs // 10) == 0:  # Print every 10%
                progress = ((i + 1) / total_inputs) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{total_inputs} inputs)")

            # Save intermediate results every 1000 inputs or if it's the last input
            if (i + 1) % 1000 == 0 or i == total_inputs - 1:
                try:
                    with open("filtered_intermediate.json", "w") as f:
                        json.dump(
                            {
                                "filtered_results": filtered_results,
                                "processed_count": i + 1,
                                "unique_code_paths": len(added_code_paths),
                            },
                            f,
                        )
                except Exception as e:
                    print(f"Warning: Failed to save intermediate results: {e}")

        except Exception as e:
            print(f"Error processing input {i}: {e}")
            continue

    print(f"Filtering complete. Found {len(filtered_results)} unique code paths")

    # Handle cases with no filtered results
    if len(filtered_results) == 0:
        if len(all_results) > 0:
            # select a random result from all_results
            random_index = int(np.random.randint(0, len(all_results)))
            filtered_results = [all_results[random_index]]
            print(f"\033[91mWarning: No valid gaps found in filtering, selecting a random result from all_results\033[0m")
        else:
            # No results at all; return empty outputs gracefully
            return [], [], []

    # Extract just the input values and gaps for backward compatibility
    filtered_klee_input_values = [result["input_value"] for result in filtered_results]
    gaps = [result["gap"] for result in filtered_results if result["gap"] is not None]

    return filtered_klee_input_values, gaps, filtered_results


def get_best_sample_and_gap(gap_list, best_sample_list):
    """Get the best sample and gap from the results.
    Handles both formats:
    - Old format: gap_list is list of (iteration, gap) tuples
    - New format: gap_list is a single float (max gap)
    """
    if isinstance(gap_list, (int, float)):
        # New format: gap_list is already the max gap
        return best_sample_list, gap_list
    elif isinstance(gap_list, list):
        # Old format: list of (iteration, gap) tuples
        try:
            # Find the maximum gap and its iteration
            max_gap_in_list = max(gap_list, key=lambda x: x[1])
            best_iteration = max_gap_in_list[0]  # Get iteration number
            max_gap = max_gap_in_list[1]
            best_sample = None
            for iter_num, sample in best_sample_list:
                if iter_num == best_iteration:
                    best_sample = sample
                    break
            return best_sample, max_gap
        except (IndexError, TypeError) as e:
            # If there's any error parsing the old format, try direct max
            try:
                return best_sample_list[0][1], max(gap_list)
            except:
                print(f"Warning: Could not parse gap_list format: {gap_list}")
                return None, 0.0
    else:
        print(f"Warning: Unexpected gap_list format: {type(gap_list)}")
        return None, 0.0


def maximize_values_for_klee_path(
    klee_path,
    problem,
    logger,
    save_dir,
    parameters,
    best_global_sample_path,
    assigned_fixed_keys,
    optimization_start_time=None,
):
    """
    This function maximizes the values for a given klee path.
    Returns the best sample and the gap values across all klee inputs.

    Returns:
        tuple: (best_sample, max_gap, best_optimal_all_vars, best_heuristic_all_vars) where:
            - best_sample is the input values that achieved the best gap
            - max_gap is the best gap achieved across all KLEE paths
            - best_optimal_all_vars is the optimal variables from the best iteration
            - best_heuristic_all_vars is the heuristic variables from the best iteration
    """
    logger.info(f"=== Starting maximize_values_for_klee_path for {klee_path} ===")

    # Check time limit if specified
    if (
        optimization_start_time is not None
        and parameters.get("max_total_time") is not None
    ):
        if time.time() - optimization_start_time > parameters["max_total_time"]:
            logger.info(
                f"Time limit of {parameters['max_total_time']} seconds reached in maximize_values_for_klee_path"
            )
            return None, float("-inf"), None, None

    with open(klee_path, "r") as f:
        klee_input_values = list(json.load(f).values())

    logger.info(f"Loaded {len(klee_input_values)} KLEE input values")
    logger.log(logger.log_entry(f"Filtering klee inputs"))
    start_time = time.time()
    filtered_results_path = os.path.join(save_dir, "filtered_results.json")
    if not os.path.exists(filtered_results_path):
        if parameters.get("disable_sequential_filtering", False):
            filtered_klee_input_values, gaps, filtered_results = filter_klee_inputs_sequential(
                klee_input_values,
                problem,
                use_gaps_in_filtering=parameters.get("use_gaps_in_filtering", True),
                minimize_is_better=parameters["minimize_is_better"],
                remove_zero_gap_inputs=parameters.get("remove_zero_gap_inputs", True),
                keep_redundant_code_paths=parameters.get("keep_redundant_code_paths", False),
            )
        else:
            filtered_klee_input_values, gaps, filtered_results = filter_klee_inputs(
                klee_input_values,
                problem,
                use_gaps_in_filtering=parameters.get("use_gaps_in_filtering", True),
                minimize_is_better=parameters["minimize_is_better"],
                remove_zero_gap_inputs=parameters.get("remove_zero_gap_inputs", True),
                keep_redundant_code_paths=parameters.get("keep_redundant_code_paths", False),
            )
        # save filtered results
        with open(filtered_results_path, "w") as f:
            json.dump(filtered_results, f)
        # save filtered klee inputs and gaps for backward compatibility
        with open(os.path.join(save_dir, "filtered_klee_input_values.json"), "w") as f:
            json.dump(filtered_klee_input_values, f)
        with open(os.path.join(save_dir, "filtered_gaps.json"), "w") as f:
            json.dump(gaps, f)
    else:
        with open(filtered_results_path, "r") as f:
            filtered_results = json.load(f)
        filtered_klee_input_values = [
            result["input_value"] for result in filtered_results
        ]
        gaps = [
            result["gap"] for result in filtered_results if result["gap"] is not None
        ]

    logger.info(f"Filtered results: {len(filtered_results)} total, {len(gaps)} with valid gaps")
    logger.log(
        logger.log_entry(f"Filtering time: {time.time() - start_time:.3f} seconds")
    )

    # Initialize best sample and max gap (will be updated if use_gaps_in_filtering is True)
    best_sample = None
    max_gap = float("-inf")
    best_optimal_all_vars = None
    best_heuristic_all_vars = None

    if parameters["use_gaps_in_filtering"]:
        logger.info("Using gaps in filtering mode")
        # Sort filtered_results by gap
        filtered_results.sort(
            key=lambda x: x["gap"] if x["gap"] is not None else float("-inf"),
            reverse=True,
        )
        initial_max_gap = filtered_results[0]["gap"] if filtered_results else None
        best_klee_input = (
            filtered_results[0]["input_value"] if filtered_results else None
        )
        logger.info(f"Initial max gap from filtering: {initial_max_gap}")
        # logger.info(f"Best KLEE input from filtering: {best_klee_input}")

        if gaps:
            print(
                f"Max gap of {initial_max_gap} and min gap of {min(gaps)} out of {len(klee_input_values)} klee inputs"
            )
            logger.log(logger.log_entry(f"Min gap: {min(gaps)}"))
        else:
            print(
                f"Max gap of {initial_max_gap} and no valid gaps out of {len(klee_input_values)} klee inputs"
            )
            logger.log(logger.log_entry("Min gap: None (no valid gaps)"))
        # Log max gap and number of inputs
        logger.log(logger.log_max_gap(initial_max_gap, len(klee_input_values)))
        # logger.log(
        #     logger.log_entry(
        #         f"Best klee input saved in {best_global_sample_path}: {best_klee_input}"
        #     )
        # )
        # save as the bestglobal sample for that klee iteration
        # Create complete sample by combining best_klee_input with assigned_fixed_keys
        complete_sample = dict(assigned_fixed_keys) if assigned_fixed_keys else {}
        complete_sample.update(best_klee_input)
        with open(best_global_sample_path, "w") as f:
            json.dump(complete_sample, f)

        # Initialize best sample and max gap from filtered results
        if filtered_results and filtered_results[0]["gap"] is not None:
            max_gap = filtered_results[0]["gap"]
            best_sample = filtered_results[0]["input_value"]
            best_optimal_all_vars = filtered_results[0].get("optimal_all_vars")
            best_heuristic_all_vars = None  # heuristic_all_vars not stored in filtered_results
            # logger.info(f"Set best_sample from filtering: {best_sample}")
            logger.info(f"Set max_gap from filtering: {max_gap}")
        else:
            logger.warning("No valid gaps found in filtered results!")
            # logger.warning(f"filtered_results[0]: {filtered_results[0] if filtered_results else 'No results'}")

    logger.log(
        logger.log_entry(
            f"Number of filtered klee inputs: {len(filtered_klee_input_values)}"
        )
    )

    if parameters["max_num_klee_points_per_iteration"] is not None:
        # sort filtered_results by gap, highest to lowest
        filtered_results.sort(key=lambda x: x["gap"], reverse=True)
        filtered_results = filtered_results[: int(parameters["max_num_klee_points_per_iteration"])]
        filtered_klee_input_values = [
            result["input_value"] for result in filtered_results
        ]
        logger.log(
            logger.log_entry(
                f"Selected first {len(filtered_klee_input_values)} klee inputs from {len(klee_input_values)} with highest gaps"
            )
        )

    # logger.info(f"Before processing: best_sample={best_sample}, max_gap={max_gap}")

    if not DO_OPTIMAL_COMBINATION_SEARCH:
        # logger.info("Using parallel processing mode")
        # Always run gradient ascent, but use filtering results as starting point if available
        if parameters["use_gaps_in_filtering"] and max_gap > float("-inf"):
            logger.info(f"Found gap {max_gap} from filtering, but will still run gradient ascent to potentially improve it")

        logger.info("Starting parallel processing of KLEE inputs")
        (
            results,
            batch_max_gap,
            batch_best_sample,
            batch_best_optimal_all_vars,
            batch_best_heuristic_all_vars,
            worker_optimal_calls,
            worker_heuristic_calls,
        ) = process_klee_inputs_parallel(
            filtered_klee_input_values,
            problem,
            save_dir,
            parameters,
            assigned_fixed_keys,
            batch_size=None,
            optimization_start_time=optimization_start_time,
            filtered_results=filtered_results,  # Pass the filtered results
        )

        # Add the worker process counter values to the main problem instance
        problem.num_compute_optimal_value_called += worker_optimal_calls
        problem.num_compute_heuristic_value_called += worker_heuristic_calls
        logger.info(f"Added {worker_optimal_calls} optimal calls and {worker_heuristic_calls} heuristic calls from worker processes")
        logger.info(f"Parallel processing completed. Results: {len(results)}")
        logger.info(f"Batch max gap: {batch_max_gap}")
        # logger.info(f"Batch best sample: {batch_best_sample}")
        # print(f"Completed processing all {len(results)} klee inputs")
        # print(f"Maximum gap found in this batch: {batch_max_gap}")

        # Update best sample and max gap
        if batch_max_gap >= max_gap:
            max_gap = batch_max_gap
            best_sample = batch_best_sample
            best_optimal_all_vars = batch_best_optimal_all_vars
            best_heuristic_all_vars = batch_best_heuristic_all_vars
            # logger.info(f"Updated best_sample from parallel processing: {best_sample}")
            logger.info(f"Updated max_gap from parallel processing: {max_gap}")
        else:
            logger.info(f"Parallel processing did not improve gap. Keeping existing best_sample: {best_sample}")

    else:
        logger.info("Using optimal combination search mode")
        process_klee_inputs_with_optimal_combination_search(
            filtered_klee_input_values,
            problem,
            save_dir,
            parameters,
            batch_size=batch_size,
        )
        # For optimal combination search, we need to read the results from saved files
        # since the function doesn't return results directly
        for klee_idx in range(len(filtered_klee_input_values)):
            results_dir = os.path.join(save_dir, f"klee_input_{klee_idx}")
            if os.path.exists(os.path.join(results_dir, "gap_list.json")):
                with open(os.path.join(results_dir, "gap_list.json"), "r") as f:
                    gap_list = json.load(f)
                with open(os.path.join(results_dir, "best_sample_list.json"), "r") as f:
                    best_sample_list = json.load(f)
                if gap_list:
                    current_best_sample, current_max_gap = get_best_sample_and_gap(
                        gap_list, best_sample_list
                    )
                    if current_max_gap >= max_gap:
                        max_gap = current_max_gap
                        best_sample = current_best_sample
                        # logger.info(f"Updated best_sample from optimal search: {best_sample}")

    # logger.info(f"Final values before saving: best_sample={best_sample}, max_gap={max_gap}")

    # Save the best result for this KLEE path
    with open(os.path.join(save_dir, "best_result.json"), "w") as f:
        json.dump(
            {
                "max_gap": max_gap,
                "best_sample": best_sample,
                "best_optimal_all_vars": best_optimal_all_vars,
                "best_heuristic_all_vars": best_heuristic_all_vars,
            },
            f,
        )

    logger.log(
        logger.log_entry(f"Results finished for {klee_path} with best gap: {max_gap}")
    )
    logger.info(f"=== Finished maximize_values_for_klee_path. Returning: max_gap={max_gap} ===")
    return best_sample, max_gap, best_optimal_all_vars, best_heuristic_all_vars


def get_save_dir(base_save_dir, problem_description):
    # sort problem_description by key
    problem_description = dict(sorted(problem_description.items()))
    # remove the keys that have None value
    problem_description = {k: v for k, v in problem_description.items() if v is not None}
    method = problem_description.pop("method")
    problem_type = problem_description.pop("problem_type")

    for key in list(problem_description.keys()):
        if "switching_cost" in key:
            problem_description.pop(key)
    for key in list(problem_description.keys()):
        if "const_switching_cost" in key:
            problem_description.pop(key)
    for key in list(problem_description.keys()):
        if key not in ["method", "problem_type", "num_items", "topology", "heuristic_name", "num_cities", "num_dimensions", "num_tasks", "topology_path"]:
            problem_description.pop(key)

    # drop the keys that have Dictionary value
    problem_description = {k: v for k, v in problem_description.items() if not isinstance(v, dict)}
    folder_name = "_".join(
        [
            f"{''.join([word.capitalize()[0] for word in key.split('_')])}-{str(problem_description[key]).replace('/', '_').replace(' ', '_').split('.')[0]}"
            for key in problem_description.keys()
        ]
    )
    # add timestamp to the folder name
    folder_name = f"{time.strftime('%Y%m%d_%H%M%S')}__{method}__{problem_type}__{folder_name}"
    return os.path.join(base_save_dir, folder_name)


def identify_gap_contributing_variables(
    problem, input_dict, logger, optimal_all_vars=None, heuristic_all_vars=None
):
    """Identify which variables contribute to the gap between optimal and heuristic solutions.

    A variable contributes to the gap if:
    1. It leads to different decisions in optimal vs heuristic solutions
    2. Changing it affects the gap significantly

    Args:
        problem: Problem instance
        input_dict: Dictionary of input values
        logger: Logger instance
        optimal_all_vars: Optional pre-computed optimal variables
        heuristic_all_vars: Optional pre-computed heuristic variables

    Returns:
        set: Set of variable names that contribute to the gap
    """
    # If optimal_all_vars and heuristic_all_vars are not provided, compute them
    if optimal_all_vars is None or heuristic_all_vars is None:
        args_dict = problem.convert_input_dict_to_args(input_dict)
        if optimal_all_vars is None:
            # Get optimal solution
            optimal_result = problem.compute_optimal_value(args_dict)
            optimal_all_vars = optimal_result["all_vars"]
        if heuristic_all_vars is None:
            # Get heuristic solution
            heuristic_result = problem.compute_heuristic_value(args_dict)
            heuristic_all_vars = heuristic_result["all_vars"]

    decision_to_input_map = problem.get_decision_to_input_map(optimal_all_vars)
    contributing_vars = set()

    # Compare decisions in optimal and heuristic solutions
    for var_name, input_var in decision_to_input_map.items():
        # Check if this variable leads to different decisions
        opt_decision = optimal_all_vars.get(var_name)
        heur_decision = heuristic_all_vars.get(var_name)

        if opt_decision is not None and heur_decision is not None:
            # If the decisions are different, this variable contributes to the gap
            if (
                abs(opt_decision - heur_decision) > 0.1
            ):  # Use small epsilon for float comparison
                if isinstance(input_var, list):
                    contributing_vars.update(input_var)
                else:
                    contributing_vars.add(input_var)
                # logger.debug(
                #     f"Variable {input_var} contributes to gap because {var_name} has different values: optimal={opt_decision}, heuristic={heur_decision}"
                # )

    return contributing_vars


def handle_fixed_variables(
    cluster_variables,
    fixed_variables,
    prev_best_sample_path,
    already_optimized_vars,
    parameters,
    logger,
    problem=None,
    prev_best_input=None,
):
    """Initialize fixed variables with random values from thresholds, considering which variables contribute to the gap."""
    assigned_fixed_keys = {}
    original_cluster_variables = cluster_variables.copy()
    if os.path.exists(prev_best_sample_path):
        with open(prev_best_sample_path, "r") as f:
            prev_best_data = json.load(f)
            prev_best_sample = prev_best_data["sample"]
            prev_best_optimal_all_vars = prev_best_data.get("optimal_all_vars")
            prev_best_heuristic_all_vars = prev_best_data.get("heuristic_all_vars")

        # If we have the problem instance and previous input, identify contributing variables
        contributing_vars = set()
        if problem is not None and prev_best_input is not None:
            contributing_vars = identify_gap_contributing_variables(
                problem,
                prev_best_input,
                logger,
                optimal_all_vars=prev_best_optimal_all_vars,
                heuristic_all_vars=prev_best_heuristic_all_vars,
            )
            # logger.debug(f"Variables contributing to gap: {contributing_vars}")
        for var in original_cluster_variables:
            if var in contributing_vars:
                cluster_variables.remove(var)
                fixed_variables.add(var)
                # logger.debug(
                #     f"Variable {var} is contributing to gap, adding to fixed variables"
                # )
        # logger.debug(f"Cluster variables: {cluster_variables}")
        # logger.debug(f"Fixed variables: {fixed_variables}")
        # Use values from previous best sample for fixed variables that were already optimized
        for var in fixed_variables:
            if var in prev_best_sample:
                # Only keep previous value if the variable contributed to the gap
                if var in contributing_vars:
                    assigned_fixed_keys[var] = prev_best_sample[var]
                    # logger.debug(
                    #     f"Using previous contributing value {prev_best_sample[var]} for variable {var}"
                    # )
                else:
                    assigned_fixed_keys[var] = prev_best_sample[var]
                    # logger.debug(
                    #     f"Previous variable {var} did not contribute to gap much, using previous optimized value"
                    # )
            else:
                raise ValueError(
                    f"Variable {var} not found in previous best sample"
                )
    else:
        # For all variables in a new iteration, use random values
        for var in fixed_variables:
            if parameters["preferred_values"]:
                assigned_fixed_keys[var] = random.choice(parameters["preferred_values"])
            else:
                assigned_fixed_keys[var] = random.randint(max(parameters["min_value"], 1), parameters["max_value"])
        logger.debug(f"Assigned random or preferred fixed keys: {assigned_fixed_keys}")

    return assigned_fixed_keys, fixed_variables, cluster_variables

