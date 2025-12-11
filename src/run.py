from analysis_utils import *
from opt_utils import *
from config import *
from common import *
from run_utils import *
from logger import AsyncLogger
from clustering_utils import *
import argparse
import shutil
import sys

def main(
    problem_description,
    base_save_dir="../logs",
    klee_task="inputs_scale_fixed_points",
):
    # Configuration
    problem_type = problem_description["problem_type"]
    parameters = get_parameters(problem_description)
    save_dir = get_save_dir(base_save_dir, problem_description)

    # Initialize logger
    logger = AsyncLogger.init(save_dir)
    logger.debug("Experiment started")
    logger.info(f"Parameters: {parameters}")

    # Enable print capture to route all print statements through the logger
    logger.capture_prints(enable=True)

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

    max_global_gap = float("-inf")
    best_global_sample = None
    best_global_optimal_all_vars = None
    best_global_heuristic_all_vars = None
    already_optimized_vars = (
        set()
    )  # Initialize empty set for tracking optimized variables
    start_time = time.time()

    # Get clusters of variable names
    if klee_task == "inputs_scale_fixed_points":
        if parameters["use_MetaOpt_cluster"]:
            variable_names_clusters = get_clusters_from_directory(problem_type, all_klee_vars, parameters["cluster_path"])
        else:
            variable_names_clusters = get_clusters(
                problem_type,
                all_klee_vars,
                parameters["max_num_scalable_klee_inputs"],
            )
            # variable_names_clusters = [["demand_0_1", "demand_0_2", "demand_1_3", "demand_2_3", "demand_0_3"], ["demand_0_4", "demand_2_4", "demand_2_5", "demand_0_5","demand_4_5", "demand_3_5", "demand_1_5"]]
    else:
        variable_names_clusters = [all_klee_vars]
    end_time = time.time()

    logger.info(
        f"Time taken to get {len(variable_names_clusters)} clusters: {end_time - start_time:.3f} seconds"
    )
    logger.debug(f"Variable names clusters: {variable_names_clusters}")
    klee_iterations = len(variable_names_clusters)
    optimization_start_time = time.time()

    # Track the best input that led to the gap
    best_gap_input = None

    for klee_idx, cluster_variables in enumerate(variable_names_clusters):
        # Check time limit at the start of each major iteration
        if (
            parameters["max_total_time"] is not None
            and time.time() - optimization_start_time > parameters["max_total_time"]
        ):
            logger.info(f"Time limit of {parameters['max_total_time']} seconds reached")
            break

        logger.info(f"Processing cluster {klee_idx + 1}/{klee_iterations}")
        cluster_max_gap = float("-inf")
        cluster_best_sample = None
        cluster_best_optimal_all_vars = None
        cluster_best_heuristic_all_vars = None

        # Get all variable names not in current cluster
        all_variables = set(all_klee_vars)
        fixed_variables = all_variables - set(cluster_variables)
        # logger.info(f"Fixed variables for cluster {klee_idx + 1}: {fixed_variables}")
        # logger.info(f"Already optimized variables: {already_optimized_vars}")

        # Load best sample from previous iteration if available
        prev_best_sample_path = os.path.join(
            save_dir, f"best_global_sample_kleeIteration{klee_idx - 1}.json"
        )

        assigned_fixed_keys, fixed_variables, cluster_variables = (
            handle_fixed_variables(
                cluster_variables,
                fixed_variables,
                prev_best_sample_path,
                already_optimized_vars,
                parameters,
                logger,
                problem=problem,
                prev_best_input=best_gap_input,
            )
        )
        print(f"DEBUG: cluster_variables {cluster_variables}")
        if len(cluster_variables) == 0:
            logger.info("No variables to optimize for this cluster")
            # copy the json file from the previous cluster
            shutil.copy(
                os.path.join(save_dir, f"fixed_variables_kleeIteration{klee_idx - 1}.json"),
                os.path.join(save_dir, f"fixed_variables_kleeIteration{klee_idx}.json"),
            )
            continue

        # Save fixed variables for this iteration
        fixed_vars_path = os.path.join(
            save_dir, f"fixed_variables_kleeIteration{klee_idx}.json"
        )
        with open(fixed_vars_path, "w") as f:
            json.dump(assigned_fixed_keys, f)

        for non_zero_round in range(parameters["num_non_zero_rounds"]):
            # Check time limit at the start of each non-zero round
            if (
                parameters["max_total_time"] is not None
                and time.time() - optimization_start_time > parameters["max_total_time"]
            ):
                logger.info(
                    f"Time limit of {parameters['max_total_time']} seconds reached in non-zero round {non_zero_round}"
                )
                break

            # Only break if we've found a gap better than the previous best global gap
            if (
                cluster_max_gap > max_global_gap
                and not parameters["ignore_gap_value_in_num_non_zero_rounds"]
            ):
                break
            klee_path = os.path.join(
                save_dir, f"klee_inputs_{klee_idx}_{non_zero_round}.json"
            )
            # Only run klee if the klee_path does not exist and we are not using a seed file and we are not disabling klee
            if not os.path.exists(klee_path) and not parameters["disable_klee"] and not parameters.get("seed_file", False):
                # run klee program
                start_time = time.time()
                # Configure KLEE timeout based on parameters or use a reasonable default
                klee_timeout = parameters.get("max_time_per_klee_point", 1800)  # Default 30 minutes
                if klee_timeout is None:
                    klee_timeout = 1800 * 6 # 1.5 hours

                # Use the conda environment's Python
                python_executable = sys.executable

                args = [
                    python_executable,
                    "run_klee.py",
                    "--problem",
                    problem_type,
                    "--problem-config-path",
                    config_path,
                    "--save-name",
                    klee_path,
                    "--disable-print",
                    "--task",
                    klee_task,
                    "--use_num_bins",
                    "--time-out",
                    str(int(klee_timeout)),
                ]

                if non_zero_round != 0:
                    relative_exclusion_paths = [
                        os.path.join(save_dir, f"klee_inputs_{i}_{j}.json")
                        for i in range(klee_idx + 1)
                        for j in range(non_zero_round)
                    ]
                    all_exclusion_paths = [
                        os.path.abspath(path) for path in relative_exclusion_paths
                    ]
                    # only add the exclusion paths if they are not empty
                    exclusion_paths = []
                    for exclusion_path in all_exclusion_paths:
                        if os.path.exists(exclusion_path):
                            exclusion_paths.append(exclusion_path)
                        else:
                            logger.info(f"Exclusion path {exclusion_path} does not exist")

                    args.extend(
                        [
                            "--input-list-to-exclude",
                        ]
                        + exclusion_paths
                    )

                if klee_task == "inputs_scale":
                    args.extend(
                        [
                            "--max-num-scalable-klee-inputs",
                            str(parameters["max_num_scalable_klee_inputs"]),
                        ]
                    )
                elif klee_task == "inputs_scale_fixed_points":
                    args.extend(
                        [
                            "--path-to-assigned-fixed-points",
                            fixed_vars_path,
                        ]
                    )

                logger.debug(f"Running KLEE with config path: {config_path}")
                logger.debug(f"Saving results to: {klee_path}")
                logger.debug(f"Full command: {' '.join(args)}")

                # Get the directory of the current script
                current_dir = os.path.dirname(os.path.abspath(__file__))
                logger.debug(f"Running in directory: {current_dir}")

                # Check time limit before running KLEE
                if (
                    parameters["max_total_time"] is not None
                    and time.time() - optimization_start_time
                    > parameters["max_total_time"]
                ):
                    logger.info(
                        f"Time limit of {parameters['max_total_time']} seconds reached before running KLEE"
                    )
                    break

                # Capture output from the subprocess
                try:
                    result = subprocess.run(
                        args,
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=current_dir,  # Set the working directory
                    )
                    # Log stdout and stderr
                    if result.stdout:
                        logger.debug("STDOUT: " + result.stdout)
                    if result.stderr:
                        logger.warning("STDERR: " + result.stderr)
                except subprocess.CalledProcessError as e:
                    logger.error("Error running KLEE:")
                    logger.error("STDOUT: " + e.stdout)
                    logger.error("STDERR: " + e.stderr)
                    raise

                end_time = time.time()
                klee_time = end_time - start_time
                logger.info(
                    f"Time taken to run klee program for non-zero round {non_zero_round}: {klee_time:.3f} seconds"
                )

            # Read the klee_path and check if it is empty
            KLEE_ONLY_ONE_CODE_PATH_OR_FAILED = False
            klee_samples = []
            # Load the samples from the seed file if it is provided
            if parameters.get("seed_file", False):
                with open(parameters["seed_file"], "r") as f:
                    klee_samples = json.load(f)
                logger.info(f"Loaded {len(klee_samples)} samples from seed file {parameters['seed_file']}")
                with open(klee_path, "w") as f:
                    json.dump(klee_samples, f)

            # prasing and cleaning the klee_samples
            if os.path.exists(klee_path):
                with open(klee_path, "r") as f:
                    klee_samples = json.load(f)
                if len(klee_samples) > 0:
                    for key, value in klee_samples.items():
                        # disacrad any klee_var that is not in the problem.all_klee_var_names
                        old_value = value.copy()
                        for var in old_value.keys():
                            if var not in problem.all_klee_var_names:
                                logger.info(f"KLEE path: Removing {var} from {key}")
                                del klee_samples[key][var]
                        for var in problem.all_klee_var_names:
                            if var not in klee_samples[key]:
                                logger.info(f"KLEE path: Adding {var} to {key}")
                                klee_samples[key][var] = 0

                if len(klee_samples) == 0:
                    KLEE_ONLY_ONE_CODE_PATH_OR_FAILED = True
                    logger.info(f"KLEE path {klee_path} is empty, skipping")
                elif len(klee_samples) == 1:
                    KLEE_ONLY_ONE_CODE_PATH_OR_FAILED = True
                    logger.info(f"KLEE path {klee_path} has 1 sample, one code path is only available")

            # Only add the problem.all_klee_var_names = max_value to all free variables if we are not using a seed file and we are not disabling klee
            if not parameters["disable_klee"] and not parameters.get("seed_file", False):
                # Add the problem.all_klee_var_names = max_value to all free variables
                for value in [parameters["max_value"], parameters["min_value"]]:
                    new_klee_sample = {}
                    for var in problem.all_klee_var_names:
                        if var in assigned_fixed_keys:
                            new_klee_sample[var] = assigned_fixed_keys[var]
                        else:
                            new_klee_sample[var] = value
                    klee_samples[f"sample_value_{value}"] = new_klee_sample
                with open(klee_path, "w") as f:
                    json.dump(klee_samples, f)

            # Add random samples if we are not using a seed file or we are disabling klee or if klee has failed or has only one code-path or the problem type is arrow, PoP or knapsack
            if not parameters.get("seed_file", False) and ((not os.path.exists(klee_path) and parameters["disable_klee"]) or KLEE_ONLY_ONE_CODE_PATH_OR_FAILED or problem.problem_config["problem_type"] in ["arrow", "PoP", "knapsack"]):
                thresholds = problem.get_thresholds({var: 0.0 for var in problem.all_klee_var_names})
                logger.info(f"Adding Random samples because disable_klee is True or klee_path has one (one code-path) or 0 samples (failed klee)")
                num_random_seed_samples = parameters["num_random_seed_samples"]
                vars = problem.all_klee_var_names
                random_samples = {
                    f"random_sample_{i}": {var: random.randint(thresholds[var][0], thresholds[var][1]) for var in vars}
                    for i in range(num_random_seed_samples)
                }
                if len(klee_samples) > 0:
                    # add klee_samples to random_samples
                    random_samples.update(klee_samples)
                with open(klee_path, "w") as f:
                    json.dump(random_samples, f)
                klee_samples = random_samples

            print("Number of klee samples: ", len(klee_samples))

            cluster_round_dir = os.path.join(
                save_dir, f"cluster_{klee_idx}_{non_zero_round}"
            )
            os.makedirs(cluster_round_dir, exist_ok=True)
            try:
                (
                    round_best_sample,
                    round_max_gap,
                    round_best_optimal_all_vars,
                    round_best_heuristic_all_vars,
                ) = maximize_values_for_klee_path(
                    klee_path,
                    problem,
                    logger,
                    cluster_round_dir,
                    parameters,
                    best_global_sample_path=os.path.join(
                        save_dir, f"best_global_sample_kleeIteration{klee_idx}.json"
                    ),
                    assigned_fixed_keys=assigned_fixed_keys,
                    optimization_start_time=optimization_start_time,
                )
            except Exception as e:
                logger.error(f"Error maximizing values for klee path {klee_path}: {e}")
                logger.error(f"NOTICE: Setting the round_best_sample to the existing best_global_sample")
                # setting the round_best_sample to the existing best_global_sample
                round_best_sample = best_global_sample
                round_max_gap = max_global_gap
                round_best_optimal_all_vars = best_global_optimal_all_vars.copy() if best_global_optimal_all_vars is not None else None
                round_best_heuristic_all_vars = best_global_heuristic_all_vars.copy() if best_global_heuristic_all_vars is not None else None

            # Exit if failed to find any valid sample
            if round_best_sample is None and round_max_gap == float("-inf"):
                already_optimized_vars.update(cluster_variables)
                logger.info(f"Already optimized variables: {already_optimized_vars}")
                break

            # Update cluster best
            if round_max_gap >= cluster_max_gap:
                logger.info(f"Updating cluster best - round_max_gap: {round_max_gap}, cluster_max_gap: {cluster_max_gap}")
                cluster_max_gap = round_max_gap
                cluster_best_sample = round_best_sample
                cluster_best_optimal_all_vars = round_best_optimal_all_vars
                cluster_best_heuristic_all_vars = round_best_heuristic_all_vars

                if round_best_sample is None:
                    logger.error(f"round_best_sample is None! This should not happen.")
                    logger.error(f"round_max_gap: {round_max_gap}")
                    # Don't update if best_sample is None
                    continue

                complete_input = dict(assigned_fixed_keys)
                complete_input.update(round_best_sample)

                # Update best gap input if this is the best gap so far
                if round_max_gap >= max_global_gap:
                    best_gap_input = complete_input
                    # Make sure to update the global best variables when we find a better gap
                    best_global_optimal_all_vars = round_best_optimal_all_vars
                    best_global_heuristic_all_vars = round_best_heuristic_all_vars

            # logger.debug(
            #     f"Best sample for non-zero round {non_zero_round}: {round_best_sample}, Max gap: {round_max_gap}"
            # )

            # Break if we've reached the last non-zero round
            if non_zero_round == parameters["num_non_zero_rounds"] - 1:
                already_optimized_vars.update(cluster_variables)
                # logger.info(f"Already optimized variables: {already_optimized_vars}")
                break

        # Update global best
        if cluster_max_gap >= max_global_gap:
            max_global_gap = cluster_max_gap
            best_global_sample = cluster_best_sample
            best_global_optimal_all_vars = cluster_best_optimal_all_vars
            best_global_heuristic_all_vars = cluster_best_heuristic_all_vars

        logger.info(f"Cluster {klee_idx + 1} max gap: {cluster_max_gap}")
        logger.log_current_max_gap(cluster_max_gap)
        already_optimized_vars.update(cluster_variables)
        # logger.info(f"Already optimized variables: {already_optimized_vars}")

        # Save the cluster-level best result
        cluster_dir = os.path.join(save_dir, f"cluster_{klee_idx}_{non_zero_round}")
        if os.path.exists(cluster_dir):
            logger.info(f"About to save cluster-level best_result.json:")
            logger.info(f"cluster_max_gap: {cluster_max_gap}")
            # logger.info(f"cluster_best_sample: {cluster_best_sample}")

            with open(os.path.join(cluster_dir, "best_result.json"), "w") as f:
                json.dump(
                    {
                        "max_gap": cluster_max_gap,
                        "best_sample": cluster_best_sample,
                        "best_optimal_all_vars": cluster_best_optimal_all_vars,
                        "best_heuristic_all_vars": cluster_best_heuristic_all_vars,
                    },
                    f,
                )
            # logger.info(f"Saved cluster-level best_result.json with max_gap: {cluster_max_gap}")

        # Save the best sample for this iteration
        with open(
            os.path.join(save_dir, f"best_global_sample_kleeIteration{klee_idx}.json"),
            "w",
        ) as f:
            json.dump(
                {
                    "sample": best_global_sample,
                    "gap": max_global_gap,
                    "optimal_all_vars": best_global_optimal_all_vars,
                    "heuristic_all_vars": best_global_heuristic_all_vars,
                },
                f,
            )

        # Check if we hit the time limit in the non-zero round loop
        if (
            parameters["max_total_time"] is not None
            and time.time() - optimization_start_time > parameters["max_total_time"]
        ):
            break

    # logger.info(f"Final best global sample: {best_global_sample}")
    # find the optimal and heuristic values for the final best global sample
    gap, gradient, (optimal_value, heuristic_value), all_vars, code_path_num = problem.get_gaps_process_input(best_global_sample)
    logger.info(f"Optimal values: {optimal_value}")
    logger.info(f"Heuristic values: {heuristic_value}")
    logger.info(f"Number of optimal value calls: {problem.num_compute_optimal_value_called}")
    logger.info(f"Number of heuristic value calls: {problem.num_compute_heuristic_value_called}")
    ratio = heuristic_value / (optimal_value + 1e-10)
    logger.log_final_max_gap(max_global_gap)

    # Save final results
    with open(os.path.join(save_dir, "final_results.json"), "w") as f:
        json.dump(
            {
                "max_global_gap": max_global_gap,
                "optimal_values": optimal_value,
                "heuristic_values": heuristic_value,
                "ratio (heuristic / optimal)": ratio,
                "number of optimal value calls": problem.num_compute_optimal_value_called,
                "number of heuristic value calls": problem.num_compute_heuristic_value_called,
                "best_global_sample": best_global_sample,
                "best_global_optimal_all_vars": best_global_optimal_all_vars,
                "best_global_heuristic_all_vars": best_global_heuristic_all_vars,
                "start_time": start_time,
                "end_time": time.time(),
                "execution_time": time.time() - start_time,
            },
            f,
        )

    # Cleanup
    logger.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-description-path", type=str, required=True)
    parser.add_argument("--base-save-dir", type=str, default="../logs")
    parser.add_argument("--klee-task", type=str, default="inputs_scale_fixed_points")
    args = parser.parse_args()
    problem_description = json.loads(args.problem_description_path)
    main(problem_description, args.base_save_dir, args.klee_task)
