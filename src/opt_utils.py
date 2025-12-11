from common import *
import json
import math
import random
import itertools

ENABLE_PRINT = False
GRANULARITY = 10000  # multiplied by numbers, the values will be integers
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def compute_heuristic_code_path(problem, input_demand_values):
    converted_input_values = problem.convert_input_dict_to_args(input_demand_values)
    heuristic_result = problem.compute_heuristic_value(converted_input_values)
    return (heuristic_result["code_path_num"], heuristic_result["heuristic_value"])

def get_heuristic_and_optimal_values(problem, input_demand_values, only_relaxed=False):
    """Get heuristic, optimal, and relaxed optimal values for any problem type."""
    converted_input_values = problem.convert_input_dict_to_args(input_demand_values)
    heuristic_value = problem.compute_heuristic_value(converted_input_values)
    if not only_relaxed:
        try:
            optimal = problem.compute_optimal_value(converted_input_values)
            optimal_value = optimal["optimal_value"]
            optimal_all_vars = optimal["all_vars"]
        except Exception as e:
            print(f"Warning: Failed to compute optimal value: {str(e)}")
            optimal_value = None
            optimal_all_vars = None
    else:
        optimal_value = None
        optimal_all_vars = None
    try:
        relaxed_optimal = problem.compute_relaxed_optimal_value(converted_input_values)
        relaxed_optimal_value = relaxed_optimal["relaxed_optimal_value"]
        relaxed_all_vars = relaxed_optimal["relaxed_all_vars"]
    except Exception as e:
        print(f"Warning: Failed to compute relaxed optimal value: {str(e)}")
        relaxed_optimal_value = None
        relaxed_all_vars = None

    return (
        heuristic_value["heuristic_value"],
        heuristic_value["code_path_num"],
        optimal_value,
        optimal_all_vars,
        relaxed_optimal_value,
        relaxed_all_vars,
        heuristic_value.get("all_vars")  # Add heuristic variables to return tuple
    )


def get_heuristic_values(problem, input_demand_values):
    """Get just the heuristic values for any problem type."""
    converted_input_values = problem.convert_input_dict_to_args(input_demand_values)
    heuristic_value = problem.compute_heuristic_value(converted_input_values)
    return (
        heuristic_value["heuristic_value"],
        heuristic_value["code_path_num"],
    )


def get_relaxed_optimal_gradient(problem, relaxed_all_vars):
    """Get the optimal gradient for any problem type."""
    converted_input_values = problem.convert_input_dict_to_args(relaxed_all_vars)
    try:
        gradient = problem.compute_lagrangian_gradient(converted_input_values)
        if gradient is None:
            print("Warning: compute_lagrangian_gradient returned None")
            return {}

        # Check for None values in gradient and replace with 0
        for key in gradient:
            if gradient[key] is None:
                print(f"Warning: Found None value in gradient for key {key}, replacing with 0")
                gradient[key] = 0.0

        return gradient
    except Exception as e:
        print(f"Error in get_relaxed_optimal_gradient: {str(e)}")
        return {}


def get_relaxed_optimal_values(problem, input_demand_values):
    """Get the relaxed optimal values for any problem type."""
    converted_input_values = problem.convert_input_dict_to_args(input_demand_values)
    relaxed_optimal_values = problem.compute_relaxed_optimal_value(
        converted_input_values
    )
    return relaxed_optimal_values["relaxed_all_vars"]


def generate_block_samples(
    anchor_point, block_length, num_samples, thresholds, assigned_fixed_keys=None, anchor_heu_code_path_num=None
):
    """Generate random samples within a block around the anchor point."""
    samples = []
    for _ in range(num_samples):
        sample = {}
        for key, value in anchor_point.items():
            if assigned_fixed_keys is not None and key in assigned_fixed_keys:
                sample[key] = value
            else:
                min_val = max(thresholds[key][0], value - block_length / 2)
                max_val = min(thresholds[key][1], value + block_length / 2)
                if anchor_heu_code_path_num is not None and key.startswith("demand_"):
                    try:
                        if key.startswith("demand_"):
                            _, from_node, to_node = key.split("_")
                            if f"{from_node}-{to_node}_" in anchor_heu_code_path_num:
                                # For DemandPinning, we don't want to increase the pinned demand values more than pinning thresholds
                                # because that would change the code path
                                min_val = value
                                max_val = value
                    except:
                        pass
                sample[key] = (
                    int(np.random.uniform(min_val, max_val) * GRANULARITY) / GRANULARITY
                )
        samples.append(sample)
    # if there's an all-zero sample, remove it
    samples = [
        sample for sample in samples if not all(value == 0 for value in sample.values())
    ]
    if ENABLE_PRINT:
        print(f"Generated {len(samples)} samples in generate_block_samples")
    return samples

def get_multiplier(num_vars):
    return 1.1
    # if num_vars <= 1000:
    #     return 20
    # else:
    #     return 1.5

def get_gaussian_process(
    problem,
    anchor_input_values,
    original_keys,
    block_length,
    num_samples,
    anchor_heu_code_path_num,
    thresholds,
    assigned_fixed_keys=None,
    disable_guassian_process=False,
):
    """Get Gaussian process model for any problem type."""
    start_time = time.time()

    if ENABLE_PRINT:
        print(f"{RED}Starting Gaussian Process analysis for {num_samples} samples{RESET}")

    multiplier = get_multiplier(len(anchor_input_values))
    # Generate random samples within the block
    samples = generate_block_samples(
        anchor_input_values,
        block_length,
        int(num_samples * multiplier), # we generate more samples to know we have enough samples after filtering
        thresholds,
        assigned_fixed_keys,
        anchor_heu_code_path_num
    )
    if ENABLE_PRINT:
        print(
            f"{RED}Time taken to generate samples: {time.time() - start_time:.3f} seconds{RESET}"
        )

    # Evaluate each sample
    y = []
    start_time = time.time()
    filtered_samples = []
    for sample in samples:
        if ENABLE_PRINT:
            print(f"len(filtered_samples): {len(filtered_samples)}")
        if len(filtered_samples) >= num_samples:
            break
        try:
            heuristic_value, heu_code_path_num = get_heuristic_values(problem, sample)
        except Exception as e:
            print(f"{RED} {sample} {RESET}")
            continue
        if heu_code_path_num == anchor_heu_code_path_num: # or disable_guassian_process:
            y.append(heuristic_value)
            filtered_samples.append(sample)

    if ENABLE_PRINT:
        print(f"filtered_samples after get_heuristic_values: {len(filtered_samples)}")

    if ENABLE_PRINT:
        print(
            f"{RED}Time taken to evaluate samples: {time.time() - start_time:.3f} seconds{RESET}"
        )

    start_time = time.time()
    y = np.array(y)
    keys_for_heuristic = original_keys
    if assigned_fixed_keys is not None:
        keys_for_heuristic = [
            key for key in keys_for_heuristic if key not in assigned_fixed_keys
        ]
    # Fit Gaussian Process
    if not disable_guassian_process:
        gp, scaler_x, scaler_y = fit_gaussian_process(
            filtered_samples, y.tolist(), keys_for_heuristic
        )
    else:
        gp, scaler_x, scaler_y = None, None, None
    if ENABLE_PRINT:
        print(
            f"{RED}Time taken to fit Gaussian process: {time.time() - start_time:.3f} seconds{RESET}"
        )
    end_time = time.time()
    if ENABLE_PRINT:
        print(f"{RED}Gaussian Process analysis took {end_time - start_time:.4f} seconds{RESET}")
    return gp, scaler_x, scaler_y, keys_for_heuristic, filtered_samples


def update_anchor_input_values(
    problem,
    anchor_input_values,
    original_keys,
    save_dir,
    parameters,
    assigned_fixed_keys=None,
):
    """Update anchor input values with proper error handling.

    Args:
        problem: Problem instance
        anchor_input_values: Initial input values
        original_keys: Original keys for the input values
        save_dir: Directory to save results
        parameters: Parameters for the optimization
        assigned_fixed_keys: Keys that should not be modified

    Returns:
        dict: Updated anchor input values, or None if optimization fails
    """
    start_time = time.time()
    if ENABLE_PRINT:
        print(f"{RED}Updating anchor input values for {len(original_keys)} variables{RESET}")
    try:
        block_length = parameters["block_length"]
        num_samples = parameters["num_samples"]
        disable_guassian_process = parameters.get("disable_guassian_process", False)
        # print in red
        if ENABLE_PRINT:
            print(f"{RED}disable_guassian_process: {disable_guassian_process}{RESET}")
        # Get initial values and handle potential failures
        try:
            relaxed_all_vars = get_relaxed_optimal_values(problem, anchor_input_values)
            thresholds = problem.get_thresholds(relaxed_all_vars)
            if relaxed_all_vars is None:
                print(
                    f"Warning: Failed to get relaxed optimal values for input {anchor_input_values}"
                )
                return None

            current_best_sample = relaxed_all_vars
            heu_result = get_heuristic_values(problem, anchor_input_values)
            if heu_result is None:
                print(f"Warning: Failed to get heuristic values")
                return None

            heu_code_path_num = heu_result[1]
        except Exception as e:
            print(f"Error getting initial values: {str(e)}")
            return None

        # Get optimal gradient with error handling
        try:
            optimal_gradient = get_relaxed_optimal_gradient(problem, relaxed_all_vars)
            if optimal_gradient is None:
                print(f"Warning: Failed to get optimal gradient")
                return None

            # Check for None values in gradient and replace with 0
            for key in optimal_gradient:
                if optimal_gradient[key] is None:
                    print(f"Warning: Found None value in gradient for key {key}, replacing with 0")
                    optimal_gradient[key] = 0.0

        except Exception as e:
            print(f"Error getting optimal gradient: {str(e)}")
            # Instead of returning None, try to continue with a zero gradient
            print("Warning: Continuing with zero gradient due to error")
            optimal_gradient = {key: 0.0 for key in relaxed_all_vars.keys()}

        if parameters.get("randomized_gradient_ascent", False):
            variable_keys = set(original_keys) - set(assigned_fixed_keys)
            if len(variable_keys) > parameters.get("num_vars_in_randomized_gradient_ascent", 10):
                variable_keys = random.sample(variable_keys, min(len(variable_keys), parameters.get("num_vars_in_randomized_gradient_ascent", 10)))
            for key in original_keys:
                if key not in variable_keys and key not in assigned_fixed_keys:
                    assigned_fixed_keys[key] = current_best_sample[key]

        for key in current_best_sample.keys():
            if key.startswith("const_") and key not in assigned_fixed_keys:
                assigned_fixed_keys[key] = current_best_sample[key]

        # Get Gaussian process with error handling
        try:
            gp_result = get_gaussian_process(
                problem,
                current_best_sample,
                original_keys,
                block_length,
                num_samples,
                heu_code_path_num,
                thresholds,
                assigned_fixed_keys,
                disable_guassian_process,
            )

            if gp_result is None or len(gp_result) != 5:
                print(f"Warning: Invalid Gaussian process result")
                return None

            gp, scaler_x, scaler_y, keys_for_heuristic, samples = gp_result

            if len(samples) == 0:
                print(f"Warning: No valid samples generated")
                return None

        except Exception as e:
            print(f"Error in Gaussian process: {str(e)}")
            return None

        if not disable_guassian_process:
            # Get heuristic gradient
            try:
                point = np.array([anchor_input_values[k] for k in keys_for_heuristic])
                if gp is not None:
                    heuristic_gradient = estimate_gradient_with_gp(gp, scaler_x, scaler_y, point)
                else:
                    heuristic_gradient = [0] * len(keys_for_heuristic)
            except Exception as e:
                print(f"Error getting heuristic gradient: {str(e)}")
                return None

        # Calculate combined gradient
        try:
            gradient_dict = {
                key: optimal_gradient[key] for key in optimal_gradient.keys()
            }

            if not disable_guassian_process:
                for index, key in enumerate(keys_for_heuristic):
                    # Ensure heuristic_gradient[index] is not None
                    heuristic_val = heuristic_gradient[index] if heuristic_gradient[index] is not None else 0.0
                    gradient_dict[key] -= heuristic_val

                if parameters.get("minimize_is_better", False):
                    gradient_dict = {
                        key: -gradient_dict[key] for key in gradient_dict.keys()
                    }

            if assigned_fixed_keys is not None:
                for key in assigned_fixed_keys:
                    gradient_dict[key] = 0

            # Final check: ensure no None values in gradient_dict
            for key in gradient_dict:
                if gradient_dict[key] is None:
                    print(f"Warning: Found None value in final gradient_dict for key {key}, replacing with 0")
                    gradient_dict[key] = 0.0

        except Exception as e:
            print(f"Error calculating combined gradient: {str(e)}")
            return None

        # Integrality penalty: encourage demands toward nearest integer
        gamma = parameters.get("integrality_penalty_gamma", 0.0)
        if gamma and gamma > 0:
            for key in original_keys:
                if key.startswith("demand_") and key in anchor_input_values:
                    delta = anchor_input_values[key] - round(anchor_input_values[key])
                    # add derivative of gamma*(delta^2): 2*gamma*delta
                    gradient_dict[key] += 2 * gamma * delta

        # Find best sample
        try:
            # Create a lambda that takes a sample and returns just the code path
            exec = lambda sample: compute_heuristic_code_path(problem, sample)
            best_sample = update_with_closest_angle_to_gradient(
                population=samples,
                best_sample=current_best_sample,
                gradient_dict=gradient_dict,
                keys_for_heuristic=keys_for_heuristic,
                thresholds=thresholds,
                assigned_fixed_keys=assigned_fixed_keys,
                compute_heuristic_code_path=(exec, heu_code_path_num),
                gradient_ascent_rate=parameters.get("gradient_ascent_rate", 0.2),
                disable_guassian_process=disable_guassian_process,
                minimize_is_better=parameters["minimize_is_better"],
                block_length=parameters["block_length"],
                ignore_code_path=parameters.get("ignore_code_path", False),
                early_stop=parameters["early_stop"]
            )
            # print(f"new_all_vars: {best_sample}")
            if best_sample is None:
                print(f"Warning: Failed to find best sample")
                return None

            end_time = time.time()
            if ENABLE_PRINT:
                print(f"{RED}Anchor input values update took {end_time - start_time:.4f} seconds{RESET}")
            return best_sample

        except Exception as e:
            print(f"Error finding best sample: {str(e)}")
            end_time = time.time()
            if ENABLE_PRINT:
                print(f"{RED}Anchor input values update failed after {end_time - start_time:.4f} seconds{RESET}")
            return None

    except Exception as e:
        print(f"Fatal error in update_anchor_input_values: {str(e)}")
        end_time = time.time()
        if ENABLE_PRINT:
            print(f"{RED}Anchor input values update failed after {end_time - start_time:.4f} seconds{RESET}")
        return None


def calculate_slope(data):
    """Calculate the slope of a dataset using linear regression.

    Args:
        data (list): List of numerical values

    Returns:
        float: Slope of the linear regression line
    """
    if len(data) < 2:
        return 0
    x = np.arange(len(data))
    y = np.array(data)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope


def apply_ewma(data, alpha=0.3):
    """Apply Exponentially Weighted Moving Average smoothing.

    Args:
        data (list): List of numerical values
        alpha (float): Smoothing factor between 0 and 1

    Returns:
        list: Smoothed values
    """
    # Filter out None values
    filtered_data = [x for x in data if x is not None]
    if not filtered_data:
        return []

    result = [filtered_data[0]]  # Initialize with first value
    for n in range(1, len(filtered_data)):
        result.append(alpha * filtered_data[n] + (1 - alpha) * result[n - 1])
    return result


def has_negative_growth_in_windows(
    max_fitness_scores, window_size=5, slope_threshold=-1e-3
):
    """Check if the past three consecutive windows exhibit negative slopes.

    Args:
        max_fitness_scores (list): List of fitness scores
        window_size (int): Size of each window to analyze
        slope_threshold (float): Threshold for considering a slope negative

    Returns:
        bool: True if three consecutive windows show negative growth
    """
    if len(max_fitness_scores) < 3 * window_size:
        return False

    # Apply EWMA smoothing
    smoothed_scores = apply_ewma(max_fitness_scores)

    # Get the last three windows
    windows = [
        smoothed_scores[-3 * window_size : -2 * window_size],
        smoothed_scores[-2 * window_size : -window_size],
        smoothed_scores[-window_size:],
    ]

    # Calculate slopes for each window
    slopes = [calculate_slope(window) for window in windows]

    # Check if all slopes are negative (below threshold)
    return all(slope < slope_threshold for slope in slopes)


def has_converged(relaxed_gap_list, window_size=10, block_length_threshold=1e-6, min_iterations=10, stagnation_threshold=1e-4):
    """Check if the optimization has converged based on the relaxed gap values.

    Args:
        relaxed_gap_list (list): List of relaxed gap values
        window_size (int): Size of the window to analyze
        block_length_threshold (float): Threshold for relative change in values
        min_iterations (int): Minimum number of iterations before considering convergence
        stagnation_threshold (float): Minimum absolute change threshold to detect stagnation

    Returns:
        bool: True if convergence criteria are met
    """
    # Need minimum number of iterations and enough values for a window
    if len(relaxed_gap_list) < max(window_size, min_iterations):
        return False

    # Get the last window of values
    window = relaxed_gap_list[-window_size:]

    # Apply EWMA smoothing
    smoothed_window = apply_ewma(window)

    # If no valid data after filtering, return False (no convergence)
    if not smoothed_window:
        return False

    # Calculate basic statistics
    mean_val = np.mean(smoothed_window)
    std_val = np.std(smoothed_window)

    # Avoid division by zero for relative calculations
    if abs(mean_val) < 1e-10:
        relative_change = 0
    else:
        relative_change = std_val / abs(mean_val)

    # Calculate absolute changes
    abs_changes = np.abs(np.diff(smoothed_window))
    max_abs_change = np.max(abs_changes) if len(abs_changes) > 0 else 0

    # Calculate slope using robust linear regression
    x = np.arange(len(smoothed_window))
    y = np.array(smoothed_window)
    slope, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]

    # Check for stagnation (very small absolute changes)
    is_stagnating = max_abs_change < stagnation_threshold

    # # IMPROVED oscillation detection - only consider significant oscillations
    if len(smoothed_window) >= 4:
        differences = np.diff(smoothed_window)
        sign_changes = np.sum(np.diff(np.signbit(differences)))

        # Calculate the range of oscillation
        value_range = np.max(smoothed_window) - np.min(smoothed_window)
        relative_range = value_range / abs(mean_val) if abs(mean_val) > 1e-10 else 0

        # Only consider it oscillating if:
        # 1. There are many sign changes AND
        # 2. The relative range is significant (> 10%)
        is_oscillating = (sign_changes > len(differences) * 0.5) and (relative_range > 0.1)
    else:
        is_oscillating = False

    # # Print debug information
    # print in red
    if ENABLE_PRINT:
        print(f"{RED}Convergence metrics - Relative change: {relative_change:.2e}, "
              f"Slope: {slope:.2e}, Max abs change: {max_abs_change:.2e}, "
              f"Stagnating: {is_stagnating}, Oscillating: {is_oscillating}{RESET}")

    # Convergence criteria:
    # 1. Relative change is small enough OR we're seeing stagnation
    # 2. Slope is non-positive (not diverging)
    # 3. Not oscillating significantly (improved detection)
    return ((relative_change < block_length_threshold or is_stagnating)
            and slope <= 0
            and not is_oscillating)


def gradient_ascent_loop(
    problem,
    anchor_input_values,
    original_keys,
    save_dir,
    parameters,
    assigned_fixed_keys=None,
    pre_computed_values=None,
):
    """Run gradient ascent loop with proper error handling.

    Args:
        problem: Problem instance
        anchor_input_values: Initial input values
        original_keys: Original keys for the input values
        save_dir: Directory to save results
        parameters: Parameters for the optimization
        assigned_fixed_keys: Keys that should not be modified
        pre_computed_values: Optional dict containing pre-computed values (heuristic_value, optimal_value, etc.)

    Returns:
        tuple: (gap_list, relaxed_gap_list, best_sample_list, best_optimal_all_vars, best_heuristic_all_vars)
    """
    start_time = time.time()
    if ENABLE_PRINT:
        print(f"assigned_fixed_keys in gradient_ascent_loop: {assigned_fixed_keys} and parameters.get('freeze_cluster_fixed_keys', False): {parameters.get('freeze_cluster_fixed_keys', False)}")
        print(f"{RED}Starting gradient ascent loop with {len(original_keys)} variables{RESET}")
    try:
        loop_start_time = time.time()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Use pre-computed values if available, otherwise compute them
        start_time = time.time()
        initial_gap = None
        relaxed_initial_gap = None
        best_optimal_all_vars = None
        best_heuristic_all_vars = None

        if pre_computed_values is not None:
            heuristic_value = pre_computed_values["heuristic_value"]
            optimal_value = pre_computed_values["optimal_value"]
            optimal_all_vars = pre_computed_values["optimal_all_vars"]
            anchor_input_values = optimal_all_vars
            initial_gap = optimal_value - heuristic_value if optimal_value is not None else None
            if parameters.get("minimize_is_better", False) and initial_gap is not None:
                initial_gap = -initial_gap
            best_optimal_all_vars = optimal_all_vars
            best_heuristic_all_vars = pre_computed_values.get("heuristic_all_vars")
        else:
            # Initialize with the initial gap calculation
            initial_values = get_heuristic_and_optimal_values(problem, anchor_input_values, only_relaxed=False)
            if initial_values and len(initial_values) == 7:  # Updated to check for 7 values
                heuristic_value, heu_code_path_num, optimal_value, optimal_all_vars, relaxed_optimal_value, relaxed_all_vars, heuristic_all_vars = initial_values
                anchor_input_values = optimal_all_vars
                initial_gap = optimal_value - heuristic_value if optimal_value is not None else None
                relaxed_initial_gap = relaxed_optimal_value - heuristic_value
                print(optimal_value, heuristic_value)
                print("X"*100)
                if parameters.get("minimize_is_better", False) and initial_gap is not None:
                    initial_gap = -initial_gap
                    relaxed_initial_gap = -relaxed_initial_gap
                best_optimal_all_vars = optimal_all_vars
                best_heuristic_all_vars = heuristic_all_vars
            else:
                initial_gap = None

        gap_list = [(-1, initial_gap)]
        relaxed_gap_list = [(-1, relaxed_initial_gap)]
        best_sample_list = [(-1, anchor_input_values)]
        iteration_time_list = [(0, "Getting initial gaps", time.time() - start_time)]
        exit_reason = ""
        save_name = f"{save_dir}/gap_list.png"
        i = 1
        while os.path.exists(save_name):
            save_name = f"{save_dir}/gap_list_{i}.png"
            i += 1

        # Track best gap and sample across all iterations
        best_gap = initial_gap if initial_gap is not None else float("-inf")
        best_sample = anchor_input_values
        current_anchor_input_values = anchor_input_values

        num_iterations = parameters.get("num_iterations", 100)
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            print(f"Warning: Invalid num_iterations {num_iterations}, using default 100")
            num_iterations = 100

        if parameters.get("disable_gradient_ascent", False):
            print(f"Gradient ascent disabled, skipping update_anchor_input_values for all iterations")
            num_iterations = 0

        for i in range(num_iterations):
            try:
                start_time = time.time()
                exit = False
                if parameters.get("max_time_per_klee_point", None) is not None:
                    time_elapsed = time.time() - loop_start_time
                    if time_elapsed > parameters["max_time_per_klee_point"]:
                        exit_reason = f"Time limit of {parameters['max_time_per_klee_point']} seconds reached at iteration {i}"
                        exit = True

                # Check for early stopping based on convergence
                if len(relaxed_gap_list) > 0 and parameters.get("early_stop", False):
                    relaxed_gaps = [g[1] for g in relaxed_gap_list]  # Extract just the gap values
                    if has_converged(relaxed_gaps, block_length_threshold=parameters.get("block_length", 0.1)):
                        exit_reason = f"Early stopping at iteration {i} - convergence criteria met"
                        exit = True

                if not exit:
                    # Get new anchor input values
                    try:
                        new_anchor_input_values = update_anchor_input_values(
                            problem,
                            current_anchor_input_values,
                            original_keys,
                            save_dir,
                            parameters,
                            assigned_fixed_keys,
                        )
                        if new_anchor_input_values is None:
                            print(f"Warning: update_anchor_input_values returned None at iteration {i}")
                            # Instead of continuing, try using the current best sample
                            new_anchor_input_values = best_sample
                    except Exception as e:
                        print(f"Error in update_anchor_input_values at iteration {i}: {str(e)}")
                        # Use the current best sample instead of exiting
                        new_anchor_input_values = best_sample

                    end_time = time.time()
                    iteration_time_list.append((i, "gradient ascent", end_time - start_time))
                    current_anchor_input_values = new_anchor_input_values

                    # Store sample if valid
                    if new_anchor_input_values:
                        best_sample_list.append((i, new_anchor_input_values))

                    # Check for zero values but don't exit immediately
                    if all(value == 0 for value in new_anchor_input_values.values()):
                        print(f"Warning: All zero values found at iteration {i}, no point with less than 90 degree were found")
                        new_anchor_input_values = best_sample
                        current_anchor_input_values = best_sample
                        exit_reason = f"All zero values found at iteration {i}, no point with less than 90 degree were found"
                        exit = True

                # Get gap of current sample
                if i % parameters.get("relaxed_gap_save_interval", 1) == 0 or exit:
                    only_relaxed = True
                    if i % parameters.get("actual_gap_save_interval", 10) == 0 or exit:
                        only_relaxed = False
                    try:
                        start_time = time.time()
                        values = get_heuristic_and_optimal_values(
                            problem, new_anchor_input_values, only_relaxed=only_relaxed
                        )
                        if not values or len(values) != 7:
                            print(f"Warning: Invalid values returned from get_heuristic_and_optimal_values at iteration {i}")
                            continue

                        (
                            heuristic_value,
                            heu_code_path_num,
                            optimal_value,
                            optimal_all_vars,
                            relaxed_optimal_value,
                            relaxed_all_vars,
                            heuristic_all_vars,
                        ) = values

                        # Get heuristic_all_vars from compute_heuristic_value
                        args_dict = problem.convert_input_dict_to_args(new_anchor_input_values)
                        heuristic_result = problem.compute_heuristic_value(args_dict)
                        heuristic_all_vars = heuristic_result.get("all_vars")

                        # Update input values with optimal or relaxed values if available
                        new_anchor_input_values = relaxed_all_vars if relaxed_all_vars is not None else new_anchor_input_values
                        if optimal_all_vars is not None:
                            new_anchor_input_values = optimal_all_vars
                        current_anchor_input_values = new_anchor_input_values

                        end_time = time.time()
                        iteration_time_list.append(
                            (i, f"get heuristic and optimal values only_relaxed={only_relaxed}", end_time - start_time)
                        )

                        # Calculate gaps
                        gap = None
                        relaxed_gap = None

                        if optimal_value is not None:
                            gap = optimal_value - heuristic_value
                        if relaxed_optimal_value is not None:
                            relaxed_gap = relaxed_optimal_value - heuristic_value

                        # Adjust for minimize_is_better
                        if parameters.get("minimize_is_better", False):
                            if gap is not None:
                                gap = -gap
                            if relaxed_gap is not None:
                                relaxed_gap = -relaxed_gap

                        # Update best gap and sample if we found a better solution
                        if gap is not None and gap > best_gap:
                            best_gap = gap
                            best_sample = new_anchor_input_values
                            best_optimal_all_vars = optimal_all_vars
                            best_heuristic_all_vars = heuristic_all_vars
                            print(f"New best gap found at iteration {i}: {best_gap}")

                        # Store valid gaps
                        if gap is not None:
                            # print(f"[PPPPPP] New Gap is: {gap}")
                            gap_list.append((i, gap))
                        if relaxed_gap is not None:
                            # print(f"[PPPPPP] New Relaxed Gap is: {relaxed_gap}")
                            relaxed_gap_list.append((i, relaxed_gap))

                        # Rounding strategies to improve discrete gap
                        variants = []  # Initialize variants list
                        # Identify demand variables for rounding (always define demand_keys)
                        demand_keys = original_keys
                        if parameters.get("use_rounding_strategies", False):
                            # Floor rounding
                            floor_in = {**new_anchor_input_values}
                            for k in demand_keys:
                                floor_in[k] = math.floor(floor_in[k])
                            variants.append(floor_in)
                            # Ceil rounding
                            ceil_in = {**new_anchor_input_values}
                            for k in demand_keys:
                                ceil_in[k] = math.ceil(ceil_in[k])
                            variants.append(ceil_in)
                            # Nearest integer rounding
                            nearest_in = {**new_anchor_input_values}
                            for k in demand_keys:
                                nearest_in[k] = int(round(nearest_in[k]))
                            variants.append(nearest_in)
                            # Stochastic rounding
                            stoch_in = {**new_anchor_input_values}
                            for k in demand_keys:
                                v = stoch_in[k]
                                lo = math.floor(v)
                                stoch_in[k] = lo + 1 if random.random() < (v - lo) else lo
                            variants.append(stoch_in)

                        # Evaluate each rounded variant
                        for rnd in variants:
                            try:
                                hv, _, ov, oav, _, _, hav = get_heuristic_and_optimal_values(
                                    problem, rnd, only_relaxed=False
                                )

                            except Exception:
                                continue
                            if ov is None or hv is None:
                                continue
                            r_gap = ov - hv
                            if parameters.get("minimize_is_better", False):
                                r_gap = -r_gap
                            if r_gap > best_gap:
                                best_gap = r_gap
                                best_sample = rnd
                                best_optimal_all_vars = oav
                                best_heuristic_all_vars = hav
                                print(f"New best discrete gap via rounding at iteration {i}: {best_gap}")
                                gap_list.append((i, best_gap))
                                best_sample_list.append((i, rnd))
                        # Exhaustive floor/ceil combos on top-fractional variables
                        k = parameters.get("rounding_exhaustive_k", 0)
                        if k and k > 0 and parameters.get("use_rounding_strategies", False):
                            # pick keys with fractional parts closest to 0.5
                            frac_parts = {
                                key: new_anchor_input_values[key] - math.floor(new_anchor_input_values[key])
                                for key in demand_keys
                            }
                            # keys sorted by distance from .5 (descending), then take k
                            sorted_keys = sorted(
                                frac_parts,
                                key=lambda x: abs(frac_parts[x] - 0.5),
                                reverse=False,
                            )[:k]
                            for pattern in itertools.product([0, 1], repeat=len(sorted_keys)):
                                combo = {**new_anchor_input_values}
                                for idx_key, key in enumerate(sorted_keys):
                                    combo[key] = (
                                        math.floor(combo[key]) if pattern[idx_key] == 0 else math.ceil(combo[key])
                                    )
                                try:
                                    hv, _, ov, oav, _, _, hav = get_heuristic_and_optimal_values(
                                        problem, combo, only_relaxed=False
                                    )
                                except Exception:
                                    continue
                                if ov is None or hv is None:
                                    continue
                                r_gap = ov - hv
                                if parameters.get("minimize_is_better", False):
                                    r_gap = -r_gap
                                if r_gap > best_gap:
                                    best_gap = r_gap
                                    best_sample = combo
                                    best_optimal_all_vars = oav
                                    best_heuristic_all_vars = hav
                                    print(f"New best discrete gap via exhaustive rounding at iteration {i}: {best_gap}")
                                    gap_list.append((i, best_gap))
                                    best_sample_list.append((i, combo))

                    except Exception as e:
                        print(f"Error calculating gaps at iteration {i}: {str(e)}")
                        continue
                # Save intermediate results
                if i % parameters.get("save_and_plot_interval", 10) == 0 or exit:
                    # print in red
                    if ENABLE_PRINT:
                        print(f"{RED}Saving and plotting at iteration {i}{RESET}")
                    try:
                        if parameters.get("enable_plotting", False) and len(gap_list) > 1:
                            # Plot results
                            if gap_list or relaxed_gap_list:
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

                                # Plot gap list if it exists
                                if gap_list:
                                    x1, y1 = zip(*gap_list)
                                    valid_points = [(x, y) for x, y in zip(x1, y1) if y is not None]
                                    if valid_points:
                                        x1, y1 = zip(*valid_points)
                                        ax1.plot(x1, y1)
                                    ax1.set_ylabel("Gap")

                                # Plot relaxed gap list if it exists
                                if relaxed_gap_list:
                                    x2, y2 = zip(*relaxed_gap_list)
                                    valid_points = [(x, y) for x, y in zip(x2, y2) if y is not None]
                                    if valid_points:
                                        x2, y2 = zip(*valid_points)
                                        ax2.plot(x2, y2)
                                    ax2.set_ylabel("Relaxed Gap")

                                plt.savefig(save_name)
                                plt.close()
                        # Save results to files
                        with open(f"{save_dir}/gap_list.json", "w") as f:
                            json.dump(gap_list, f)
                        with open(f"{save_dir}/relaxed_gap_list.json", "w") as f:
                            json.dump(relaxed_gap_list, f)
                        with open(f"{save_dir}/best_sample_list.json", "w") as f:
                            json.dump(best_sample_list, f)
                        with open(f"{save_dir}/iteration_time_list.json", "w") as f:
                            json.dump(iteration_time_list, f)
                        if exit:
                            with open(f"{save_dir}/exit_reason.txt", "w") as f:
                                f.write(exit_reason)
                        # Save current best result
                        with open(f"{save_dir}/best_result.json", "w") as f:
                            json.dump({
                                "max_gap": best_gap,
                                "best_sample": best_sample,
                                "best_optimal_all_vars": best_optimal_all_vars,
                                "best_heuristic_all_vars": best_heuristic_all_vars
                            }, f)
                    except Exception as e:
                        print(f"Error saving results at iteration {i}: {str(e)}")

                if exit:
                    print(f"DEBUG: Exiting at iteration {i} with reason: {exit_reason}")
                    break

            except Exception as e:
                print(f"Error in iteration {i}: {str(e)}")
                continue

        end_time = time.time()
        if ENABLE_PRINT:
            print(f"{RED}Gradient ascent loop took {end_time - start_time:.4f} seconds{RESET}")
        return gap_list, relaxed_gap_list, best_sample_list, best_optimal_all_vars, best_heuristic_all_vars

    except Exception as e:
        print(f"Fatal error in gradient_ascent_loop: {str(e)}")
        end_time = time.time()
        if ENABLE_PRINT:
            print(f"{RED}Gradient ascent loop failed after {end_time - start_time:.4f} seconds{RESET}")
        # If we have an initial gap, return it instead of empty lists
        try:
            return return_from_files(save_dir)
        except Exception as e:
            print(f"Error returning from files: {str(e)}")
            if initial_gap is not None:
                return [(-1, initial_gap)], [(-1, relaxed_initial_gap)], [(-1, anchor_input_values)], best_optimal_all_vars, best_heuristic_all_vars
            return [], [], [], None, None

def return_from_files(save_dir):
    with open(f"{save_dir}/gap_list.json", "r") as f:
        gap_list = json.load(f)
    with open(f"{save_dir}/relaxed_gap_list.json", "r") as f:
        relaxed_gap_list = json.load(f)
    with open(f"{save_dir}/best_sample_list.json", "r") as f:
        best_sample_list = json.load(f)
    return gap_list, relaxed_gap_list, best_sample_list

def chunks(iterable, size):
    """Yield successive chunks from iterable."""
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, size)), [])


def process_klee_inputs_with_optimal_combination_search(
    filtered_klee_input_values,
    problem,
    save_dir,
    parameters,
    batch_size=100,
):
    all_combinations_path = f"{save_dir}/all_combinations.json"
    if not os.path.exists(all_combinations_path):
        all_combinations = problem.get_all_binary_combinations()
        with open(all_combinations_path, "w") as f:
            json.dump(all_combinations, f)

    # Process combinations in parallel
    num_cores = cpu_count()
    if ENABLE_PRINT:
        print(f"{RED}Processing using {num_cores} cores{RESET}")

    # Process combinations in chunks to manage memory
    with open(all_combinations_path, "rb") as f:
        parser = ijson.items(f, "item")
        chunk_id = 0
        combinations_processed = 0

        # Create a pool of workers that will be reused for all chunks
        with Pool(processes=num_cores) as pool:
            for combinations_chunk in chunks(parser, batch_size * num_cores):
                # Split the chunk into batches and create work items for all KLEE inputs
                args_list = []
                for klee_index, anchor_input_values in enumerate(
                    filtered_klee_input_values
                ):
                    for i in range(0, len(combinations_chunk), batch_size):
                        batch = combinations_chunk[i : i + batch_size]
                        if batch:  # Only process non-empty batches
                            args_list.append(
                                (
                                    batch,
                                    problem,
                                    anchor_input_values,
                                    list(anchor_input_values.keys()),
                                    parameters,
                                    f"{save_dir}/klee_input_{klee_index}",
                                    combinations_processed + i,
                                    klee_index,
                                )
                            )

                # Process all batches in parallel
                results = pool.map(process_single_combination_batch, args_list)

                # Update progress
                combinations_processed += len(combinations_chunk)
                print(
                    f"Completed chunk {chunk_id} (total processed: {combinations_processed})"
                )
                chunk_id += 1

    print("Completed processing all combinations")


def process_single_combination_batch(args):
    """Process a batch of combinations for a single KLEE input."""
    (
        combinations,
        problem,
        anchor_input_values,
        original_keys,
        save_dir,
        parameters,
        start_index,
        klee_index,
    ) = args

    results = []
    for i, combination in enumerate(combinations):
        current_index = start_index + i
        new_anchor_input_values = anchor_input_values.copy()
        for key in combination.keys():
            new_anchor_input_values[key] = combination[key]
        assigned_fixed_keys = list(combination.keys())

        predicted_optimal_value = problem.get_optimal_value_based_on_combination(
            combination
        )
        heuristic_value = get_heuristic_values(problem, new_anchor_input_values)[0]
        print(
            f"Heuristic value: {heuristic_value}, Predicted optimal value: {predicted_optimal_value}"
        )
        if (
            # problem.is_input_feasible(new_anchor_input_values)
            predicted_optimal_value
            == 4
            # and heuristic_value - predicted_optimal_value > 1
        ):
            print(f"KLEE {klee_index} - Combination {current_index+1} is feasible")
            os.makedirs(save_dir, exist_ok=True)
            batch_dir = f"{save_dir}/combination_{current_index}"
            os.makedirs(batch_dir, exist_ok=True)
            gap_list, relaxed_gap_list, best_sample_list, best_optimal_all_vars, best_heuristic_all_vars = gradient_ascent_loop(
                problem,
                new_anchor_input_values,
                original_keys,
                batch_dir,
                parameters,
                assigned_fixed_keys,
            )

            results.append(True)
        else:
            results.append(False)

    num_feasible = sum(1 for r in results if r)
    if num_feasible > 0:
        print(
            f"KLEE {klee_index} - Found {num_feasible} feasible combinations in batch"
        )
    return klee_index, start_index, num_feasible
