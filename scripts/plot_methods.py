from plot_common import setup_plot_style, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, METHOD_HATCHES
from log_parser import parse_log_file, parse_metaopt_log_file, parse_log_file_from_results_json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

setup_plot_style()
ENABLE_TIME_CROPPING = True

def get_scale_factor(problem, experiment_name):
    problem = problem.lower()
    experiment_name = experiment_name.lower()
    if "dote" in experiment_name or "dote" in problem:
        return 200 * 14 * 2
    if experiment_name.startswith("te_") or "llm" in experiment_name:
        # Total capacity in the network
        if problem == "cogentco":
            return 200 * 486 
        elif "b4" in problem:
            return 200 * 38
        elif problem == "uninet2010":
            return 200 * 202
        elif problem == "swan":
            return 200 * 24
        elif problem == "abilene":
            return 200 * 26
        elif problem == "abilenedote":
            return 200 * 14 * 2
        else:
            return 1
    elif experiment_name.startswith("knapsack"):
        # Total possible value in the knapsack
        return int(problem) * 50
    elif experiment_name.startswith("mwm"):
        if problem == "cogentco":
            return 10 * 486 
        elif "b4" in problem:
            return 10 * 38
        elif problem == "uninet2010":
            return 10 * 202
        elif problem == "swan":
            return 10 * 24
        elif problem == "abilene":
            return 10 * 26
        elif problem == "abilenedote":
            return 10 * 14 * 2
        else:
            return 1
    elif experiment_name.startswith("arrow"):
        if "b4" in problem:
            return 19 * 400
        elif problem == "ibm":
            return 154 * 400
        else:
            return 1
    return 1

def get_color_for_method(method):
    """Get the predefined color for a method."""
    return METHOD_COLORS.get(method, "#666666")  # Default gray if method not found

def get_label_for_method(method):
    """Get the predefined label for a method."""
    return METHOD_LABELS.get(method, method)  # Return original name if not found

def format_problem_name(problem):
    """Format problem name for better display."""
    # Replace underscores with spaces and capitalize
    if problem.lower() == "ibm":
        return "IBM"
    return problem.replace('_', ' ').title()

def count_baseline_gap_computations(log_dir, method):
    """Count gap computations for baseline methods (Random, SimulatedAnnealing, HillClimbing)."""
    # Look for results.json files in the log directory
    results_files = []

    # Check for different possible results file names
    possible_files = [
        "random_sampling_results.json",
        "simulated_annealing_results.json", 
        "hill_climbing_results.json",
        "results.json",
        "sample_based_gradient_results.json",
    ]
    
    for filename in possible_files:
        file_path = os.path.join(log_dir, filename)
        if os.path.exists(file_path):
            results_files.append(file_path)
    
    total_gaps = 0
    for results_file in results_files:
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                if "gap_computations" in data:
                    total_gaps += data["gap_computations"]
                elif 'all_gaps' in data and isinstance(data['all_gaps'], list):
                    total_gaps += len(data['all_gaps'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {results_file}: {e}")

    # if no results files found, explore experiments.log
    if len(results_files) == 0 and os.path.exists(os.path.join(log_dir, "experiment.log")):
        # print(f"No results files found for {method} in {log_dir}, exploring experiment.log")
        # parse Starting XXX with {num_samples} samples, parse Starting random sampling with {num_samples} samples
        with open(os.path.join(log_dir, "experiment.log"), 'r') as f:
            for line in f:
                if "Starting random sampling" in line:
                    total_gaps += int(line.split("with ")[1].split(" samples")[0])
                elif "Starting HillClimbing" in line:
                    total_gaps += int(line.split("with ")[1].split(" samples")[0])
                elif "Starting SimulatedAnnealing" in line:
                    total_gaps += int(line.split("with ")[1].split(" samples")[0])
    return total_gaps

def walk_metaease_gap_directory(log_dir):
    """Count gap computations for MetaEase by scanning all gap_list.json files."""
    total_gaps = 0
    log_data = {}
    # Get the birth time (creation time) of the log_dir
    try:
        import subprocess
        result = subprocess.run(['stat', '-c', '%W', log_dir], capture_output=True, text=True)
        if result.returncode == 0:
            birth_time = float(result.stdout.strip())
            if birth_time > 0:  # Birth time is available (0 means not available)
                creation_time = birth_time
            else:
                # Fallback to directory change time if birth time not available
                creation_time = os.path.getctime(log_dir)
        else:
            creation_time = os.path.getctime(log_dir)
    except:
        # Fallback to directory change time if stat command fails
        creation_time = os.path.getctime(log_dir)

    log_data['start_time'] = creation_time
    log_data['gap_values'] = []
    log_data['all_gap_times'] = []
    log_data['time_from_start'] = [0]
    log_data['gaps'] = [0]
    log_data["number of heuristic value calls"] = 0
    log_data["number of optimal value calls"] = 0
    # read the file that ends with _config.json
    config_file = os.path.join(log_dir, [f for f in os.listdir(log_dir) if f.endswith('_config.json')][0])
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
            log_data["num_iterations"] = data["num_iterations"]
            log_data["num_samples"] = data["num_samples"]

    log_data["explored_klee_inputs"] = 0
    # Walk through all subdirectories to find gap_list.json files
    for root, dirs, _ in os.walk(log_dir):
        # dirs includes the cluster_x_y folders
        for dir in dirs:
            if not dir.startswith('cluster_'):
                continue
            cluster_dir = os.path.join(root, dir)
            for cluster_root, klee_input_dirs, _ in os.walk(cluster_dir):
                # cluster_dirs are the klee_input_i_j folders
                for klee_input_dir in klee_input_dirs:
                    if not klee_input_dir.startswith('klee_input_'):
                        continue
                    file_path = os.path.join(cluster_root, klee_input_dir, "gap_list.json")
                    try:
                        time_file_last_modified = os.path.getctime(file_path)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                total_gaps += len(data)
                                log_data["explored_klee_inputs"] += 1
                                log_data["number of optimal value calls"] += len(data)
                                gaps = [gap for iteration, gap in data]
                                max_iteration = max(iteration for iteration, gap in data)
                                log_data["number of heuristic value calls"] += max_iteration * log_data["num_samples"] * 1.1 # the get_multiplier is 1.1
                                log_data['gap_values'].extend(gaps)
                                log_data['all_gap_times'].extend([time_file_last_modified] * len(gaps))
                                relaxed_file_path = file_path.replace("gap_list.json", "relaxed_gap_list.json")
                                if os.path.exists(relaxed_file_path):
                                    with open(relaxed_file_path, 'r') as f:
                                        data = json.load(f)
                                        if isinstance(data, list):
                                            log_data["number of heuristic value calls"] += len(data)
                    except (json.JSONDecodeError, Exception) as e:
                        # print(f"Warning: Could not parse {file_path}: {e}")                 
                        pass

    # find the max envelope of the gap_values
    final_gap = -float('inf')
    for index, gap_value in enumerate(log_data['gap_values']):
        if gap_value > final_gap:
            final_gap = gap_value
            log_data['final_gap'] = final_gap
            log_data['gaps'].append(final_gap)
            log_data['time_from_start'].append(log_data['all_gap_times'][index] - log_data['start_time'])

    # if final_results.json exists, parse it
    final_results_file = os.path.join(log_dir, "final_results.json")
    if os.path.exists(final_results_file):
        with open(final_results_file, 'r') as f:
            data = json.load(f)
            log_data['final_gap'] = data['max_global_gap']
            if "number of optimal value calls" in data:
                log_data["number of optimal value calls"] = data["number of optimal value calls"]
            if "number of heuristic value calls" in data:
                log_data["number of heuristic value calls"] = data["number of heuristic value calls"]

    log_data['end_time'] = max(log_data['all_gap_times'])
    log_data['total_time'] = log_data['end_time'] - log_data['start_time']
    log_data['is_experiment_finished'] = False
    # print(f"Total total time: {log_data['total_time']}")
    return {
        "gap_computations": total_gaps,
        "log_data": log_data
    }

def get_log_data_for_one_experiment(log_dir, method):
    log_file = os.path.join(log_dir, "experiment.log")
    if method != "MetaOpt":
        log_data = None
        # First try to parse the results.json files
        try:
            possible_files = [
                "final_results.json",
                "results.json",
                "random_sampling_results.json",
                "simulated_annealing_results.json",
                "hill_climbing_results.json",
                "sample_based_gradient_results.json",
            ]
            for file in possible_files:
                file_path = os.path.join(log_dir, file)
                if os.path.exists(file_path):
                    log_data = parse_log_file_from_results_json(file_path)
                    break
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        if log_data is not None:
            print(f"{method} Max Gap: {log_data['final_gap']}")

        # Then try to parse the experiment.log file if the above is not successful
        try:
            if log_data is None and os.path.exists(log_file) or log_data['gap_times_unavailable']:
                print(f"Parsing {log_file} because json parsing failed")
                log_data = parse_log_file(log_file)
        except Exception as e:
            print(f"Error parsing {log_dir}: {e}")

    else:
        log_file = os.path.join(log_dir, "MetaOpt_log.txt")
        if os.path.exists(log_file):
            log_data = parse_metaopt_log_file(log_file)
        else:
            return None

    # Count gap computations based on method type
    if method == "MetaEase":
        experiment_log_walk = walk_metaease_gap_directory(log_dir)
        gap_computations = experiment_log_walk['gap_computations']
        if log_data is None or (log_data is not None and not log_data['is_experiment_finished']) or os.path.exists(log_file + ".1"):
            # If experiment is not finished or bad logging
            log_data = experiment_log_walk["log_data"]
        log_data['gap_computations'] = gap_computations
    elif method == "MetaOpt":
        gap_computations = 1
        log_data['gap_computations'] = gap_computations
    elif 'gap_computations' not in log_data or (log_data['gap_computations'] is None or log_data['gap_computations'] < 100):
        gap_computations = count_baseline_gap_computations(log_dir, method)
        log_data['gap_computations'] = gap_computations
    return log_data

def get_problem_dict(experiment_file):
    problem_dict = {}
    with open(experiment_file, "r") as f:
        # Parse the header row
        _, setting, _ = f.readline().strip().split(",")
        print(f"Setting: {setting}")
        # Parse the data rows
        for line in f:
            if line.startswith("#"):
                continue
            method, problem, log_dir = line.strip().split(",")
            # print(f"Processing {method} for {problem} in {log_dir}")
            try:
                log_data = get_log_data_for_one_experiment(log_dir, method)
            except Exception as e:
                print(f"Error processing {method} for {problem} in {log_dir}: {e}")
                continue
            if log_data is None:
                continue
            if problem not in problem_dict:
                problem_dict[problem] = {}
            if method not in problem_dict[problem]:
                problem_dict[problem][method] = []
            problem_dict[problem][method].append(log_data)

    # for any problem and method, if gap_computations is 0 for a case, set that tho the avg gap_computations of non-zero gap_computations
    for problem in problem_dict:
        for method in problem_dict[problem]:
            non_zero_gap_computations = [log_data['gap_computations'] for log_data in problem_dict[problem][method] if log_data['gap_computations'] != 0]
            if len(non_zero_gap_computations) > 0:
                avg_gap_computations = np.mean(non_zero_gap_computations)
            else:
                continue
            for idx, log_data in enumerate(problem_dict[problem][method]):
                if log_data['gap_computations'] == 0:
                    problem_dict[problem][method][idx]['gap_computations'] = avg_gap_computations

    # for any problem, make sure to cut values of gaps for all methods to the same length of time_from_start of MetaEase
    for problem in problem_dict:
        for method in problem_dict[problem]:
            max_time_from_start = max(problem_dict[problem]["MetaEase"][0]["time_from_start"])
            if method not in ["MetaEase", "MetaOpt"] and ENABLE_TIME_CROPPING:
                for count, log_data in enumerate(problem_dict[problem][method]):
                    indexes = [i for i, time_from_start in enumerate(problem_dict[problem][method][count]["time_from_start"]) if time_from_start <= max_time_from_start]
                    problem_dict[problem][method][count]["time_from_start"] = [problem_dict[problem][method][count]["time_from_start"][i] for i in indexes]
                    problem_dict[problem][method][count]["gaps"] = [problem_dict[problem][method][count]["gaps"][i] for i in indexes]
                    problem_dict[problem][method][count]["final_gap"] = max(problem_dict[problem][method][count]["gaps"])
                    problem_dict[problem][method][count]["total_time"] = max(problem_dict[problem][method][count]["time_from_start"])

    experiment_name = experiment_file.split("/")[-1].split(".")[0]
    problem_dict = scale_problem_dict(problem_dict, experiment_name)
    return summarize_problem_dict(problem_dict), setting

def scale_log_data(log_data, experiment_name, problem):
    for key in log_data:
        if key in ["max_gap", "max_global_gap", "final_gap", "gaps", "all_gaps"] and "time" not in key:
            if isinstance(log_data[key], list):
                log_data[key] = [value / get_scale_factor(problem, experiment_name) * 100 for value in log_data[key]]
            else:
                log_data[key] = log_data[key] / get_scale_factor(problem, experiment_name) * 100
    return log_data

def scale_problem_dict(problem_dict, experiment_name):
    for problem in problem_dict:
        for method in problem_dict[problem]:
            for index, log_data in enumerate(problem_dict[problem][method]):
                problem_dict[problem][method][index] = scale_log_data(log_data, experiment_name, problem)
    return problem_dict

def summarize_problem_dict(problem_dict):
    """Each problem and method could have multiple log data, we need to summarize them."""
    # 'gap_computations', gaps, time_from_start, final_gap
    # average final_gap and gap_computations
    # average gaps and time_from_start curves for the same method and problem
    avg_problem_dict = {}
    for problem in problem_dict:
        avg_problem_dict[problem] = {}
        for method in problem_dict[problem]:
            curves = problem_dict[problem][method]
            # Initialize the averaged data structure
            avg_problem_dict[problem][method] = {}
            # Average final_gap and gap_computations (scalar values)
            gap_computations = [log_data['gap_computations'] for log_data in curves if log_data['gap_computations'] is not None]
            if gap_computations:
                avg_problem_dict[problem][method]['gap_computations'] = np.mean(gap_computations)
            else:
                avg_problem_dict[problem][method]['gap_computations'] = None

            final_gaps = [log_data['final_gap'] for log_data in curves if 'final_gap' in log_data and log_data['final_gap'] is not None]
            if final_gaps:
                avg_problem_dict[problem][method]['final_gap'] = np.mean(final_gaps)
            else:
                # Calculate final gap from the last gap value
                final_gaps = [log_data['gaps'][-1] for log_data in curves if len(log_data['gaps']) > 0 and log_data['gaps'][-1] is not None]
                avg_problem_dict[problem][method]['final_gap'] = np.mean(final_gaps) if final_gaps else 0.0
            # Average the time series curves (gaps vs time_from_start)
            if len(curves) == 1:
                # If only one curve, just copy it
                avg_problem_dict[problem][method]['time_from_start'] = curves[0]['time_from_start']
                avg_problem_dict[problem][method]['gaps'] = curves[0]['gaps']
            else:
                # Multiple curves - need to interpolate and average
                # Find the common time range
                max_time = max(max(log_data['time_from_start']) for log_data in curves)
                min_time = min(min(log_data['time_from_start']) for log_data in curves)
                # Create a common time grid for interpolation
                # Use the union of all time points, sorted
                all_times = set()
                for log_data in curves:
                    all_times.update(log_data['time_from_start'])
                common_times = sorted(list(all_times))
                # Interpolate each curve onto the common time grid
                interpolated_gaps = []
                for log_data in curves:
                    # For each curve, interpolate gaps at common_times
                    curve_gaps = np.interp(common_times, log_data['time_from_start'], log_data['gaps'])
                    interpolated_gaps.append(curve_gaps)
                # Average the interpolated curves
                avg_gaps = np.max(interpolated_gaps, axis=0)
                avg_problem_dict[problem][method]['time_from_start'] = common_times
                avg_problem_dict[problem][method]['gaps'] = avg_gaps.tolist()

    return avg_problem_dict


def plot_baselines(problem_dict, output_dir, problem_type, remove_metaopt=False):
    with open(f"{output_dir}/{problem_type}_problem_dict.json", "w") as f:
        json.dump(problem_dict, f)
    for problem in problem_dict:
        # Create a more visually appealing plot
        fig, ax = plt.subplots(figsize=(7, 5))
        methods_line_style ={
            "MetaEase": "solid",
            "HillClimbing": "-.",
            "SampleBasedGradient": "dashed",
            "Random": "dashdot",
            "SimulatedAnnealing": "dotted",
        }
        methods = list(problem_dict[problem].keys())
        if remove_metaopt:
            if "MetaOpt" in methods:
                methods.remove("MetaOpt")

        max_time_from_start = max([max(problem_dict[problem][method]["time_from_start"]) for method in methods])
        print(f"Max time from start for {problem}: {max_time_from_start}")
        
        for method in methods:
            method_max_time = max(problem_dict[problem][method]["time_from_start"])
            # If the method's max time is greater than the max time from start, cut them
            while method_max_time > max_time_from_start and method != "MetaOpt":
                problem_dict[problem][method]["time_from_start"].pop(-1)
                problem_dict[problem][method]["gaps"].pop(-1)
                method_max_time = max(problem_dict[problem][method]["time_from_start"])
            # If the method's max time is less than the max time from start, append the same value of gap from the last point in time_from_start for the missing time points
            if method_max_time < max_time_from_start and method != "MetaOpt":
                problem_dict[problem][method]["time_from_start"].append(max_time_from_start)
                problem_dict[problem][method]["gaps"].append(problem_dict[problem][method]["gaps"][-1])
            if problem_dict[problem][method]["gaps"][0] == 0:
                # remove the first gap and time_from_start
                problem_dict[problem][method]["time_from_start"].pop(0)
                problem_dict[problem][method]["gaps"].pop(0)
            # Plot with enhanced styling using predefined colors and labels
            ax.plot(problem_dict[problem][method]["time_from_start"], 
                   problem_dict[problem][method]["gaps"], 
                   label=get_label_for_method(method),
                   linestyle=methods_line_style[method],
                   color=get_color_for_method(method), 
                   linewidth=3, 
                   alpha=0.9,
                #    marker='o', 
                #    markersize=4, 
                   markevery=max(1, len(problem_dict[problem][method]["time_from_start"]) // 20))

        # if "MetaOpt" in problem_dict[problem].keys() and not remove_metaopt:
        #     max_gap = max(problem_dict[problem]["MetaOpt"]["gaps"])
        #     print(f"Max gap for MetaOpt {problem}: {max_gap}")
        #     # add a horizontal line at the gap of MetaOpt
        #     ax.axhline(y=max_gap, color='black', linestyle='--', linewidth=2)
        #     # add a text annotation for the gap of MetaOpt, left aligned
        #     ax.text(0.05, max_gap, f'MetaOpt Gap: {max_gap:.2f}', 
        #             ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Enhanced plot styling
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Norm Max Gap Log (%)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        # make y_ticks to be 0.1, 1, 10, 100
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
        # ax.set_title(f'Method Comparison: {format_problem_name(problem)}', 
        #             fontsize=16, fontweight='bold', pad=20)
        
        # Improve grid and background
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')
        # put axis outside the plot, top and center
        # Enhanced legend
        legend = ax.legend(loc='upper center', 
                          fontsize=11, 
                          frameon=False,
                          bbox_to_anchor=(0.5, 1.3),
                          ncol=3)
        
        # Add an upward arrow that says "Better" (rotated)
        # Position the arrow in the upper right area of the plot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Arrow position (90% to the right, starting from 80% height going to 90% height)
        arrow_x = xlim[0] + 0.9 * (xlim[1] - xlim[0])
        arrow_y_start = ylim[0] + 0.01 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.1 * (ylim[1] - ylim[0])
        
        # Draw the upward arrow
        ax.annotate('', xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # Add rotated "Better" text next to the arrow
        ax.text(arrow_x - 0.02 * (xlim[1] - xlim[0]), 
               (arrow_y_start + arrow_y_end) / 2,
               'Better', rotation=90, va='center', ha='right', 
               fontsize=12, fontweight='bold')

        # Improve axis formatting
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add subtle background color
        fig.patch.set_facecolor('white')
        
        # Save the plotting data to JSON
        plot_data = {
            "problem_type": problem_type,
            "problem": problem,
            "methods": methods,
            "method_labels": {method: get_label_for_method(method) for method in methods},
            "method_colors": {method: get_color_for_method(method) for method in methods},
            "method_line_styles": {method: methods_line_style.get(method, "solid") for method in methods},
            "time_series_data": {}
        }
        
        # Add time series data for each method
        for method in methods:
            if method in problem_dict[problem]:
                plot_data["time_series_data"][method] = {
                    "time_from_start": problem_dict[problem][method]["time_from_start"],
                    "gaps": problem_dict[problem][method]["gaps"],
                    "final_gap": problem_dict[problem][method].get("final_gap", max(problem_dict[problem][method]["gaps"]) if problem_dict[problem][method]["gaps"] else 0)
                }
        
        with open(f"{output_dir}/{problem_type}_{problem}_baseline_data.json", "w") as f:
            json.dump(plot_data, f, indent=2)

        # Adjust layout and save with high DPI
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{problem_type}_{problem}.pdf", 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none')
        plt.close()


def plot_gap_computations(problem_dict, output_dir, problem_type, setting):
    """Create a bar plot showing gap computation counts for each method and problem."""
    # Collect data for plotting
    problems = list(problem_dict.keys())
    problems = sorted(problems, key=str.lower)
    # drop MetaOpt from methods
    methods = set()
    for problem in problems:
        methods.update(problem_dict[problem].keys())
    methods = sorted(list(methods), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else len(METHOD_ORDER))
    if "MetaOpt" in methods:
        methods.remove("MetaOpt")
    
    # Create data matrix: problems x methods
    gap_computations_data = {}
    for method in methods:
        gap_computations_data[method] = []
        for problem in problems:
            if method in problem_dict[problem]:
                gap_computations_data[method].append(problem_dict[problem][method]['gap_computations'])
            else:
                gap_computations_data[method].append(0)
    
    # Create the plot
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Set up bar positions
    x = np.arange(len(problems))
    total_width = 0.8
    bar_width = total_width / max(1, len(methods))
    offset_start = - (total_width - bar_width) / 2.0
    
    # Plot bars for each method
    for idx, method in enumerate(methods):
        offsets = x + offset_start + idx * bar_width
        bars = ax.bar(offsets, gap_computations_data[method], bar_width, 
                     label=get_label_for_method(method), 
                     color=get_color_for_method(method),
                     hatch=METHOD_HATCHES[method],
                     alpha=0.9,
                     edgecolor='black',
                     linewidth=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, gap_computations_data[method]):
            if value > 0:  # Only show labels for non-zero values
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=7)
    
    # Enhanced plot styling
    ax.set_xlabel(setting, fontsize=14, fontweight='bold')
    ax.set_ylabel('# Benchmark Calls', fontsize=14, fontweight='bold')
    # ax.set_title('Gap Computation Counts by Method and Problem', fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels - center x-ticks with the group of bars
    center_x = x + offset_start + (len(methods) - 1) * bar_width * 0.7
    for index, problem in enumerate(problems):
        if len(problem) >= 7:
            center_x[index] = x[index] + offset_start + (len(methods) - 1) * bar_width
    ax.set_xticks(center_x)
    ax.set_xticklabels([format_problem_name(problem) for problem in problems], rotation=0, ha='right', fontsize=12)
    
    # Improve grid and background
    ax.grid(False)  # Turn off all grids first
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')

    # Enhanced legend - 2 columns at top outside
    legend = ax.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 1.3),
                      ncol=3,
                      fontsize=12, 
                      frameon=False)

    # Improve axis formatting
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to log scale if there's a large range
    max_value = max([max(values) for values in gap_computations_data.values() if values])
    min_value = min([min([v for v in values if v > 0]) for values in gap_computations_data.values() if any(v > 0 for v in values)])
    is_log_scale = max_value / min_value > 100
    if is_log_scale:  # Use log scale if range is large
        ax.set_yscale('log')
        # make the y-label new line
        ax.set_ylabel('# Benchmark Calls (Log)', fontsize=12)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    arrow_x = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    if is_log_scale:
        arrow_y_start = ylim[0] + 0.3 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    else:
        arrow_y_start = ylim[0] + 0.65 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.45 * (ylim[1] - ylim[0])
    
    # Draw the upward arrow
    ax.annotate('', xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Add rotated "Better" text next to the arrow
    ax.text(arrow_x - 0.02 * (xlim[1] - xlim[0]), 
            (arrow_y_start + arrow_y_end) / 2,
            'Better', rotation=90, va='center', ha='right', 
            fontsize=12, fontweight='bold')

    # Add subtle background color
    fig.patch.set_facecolor('white')
    
    # Adjust layout and save with high DPI
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{problem_type}_gap_computations.pdf", 
               dpi=300, 
               bbox_inches='tight', 
               facecolor='white',
               edgecolor='none')
    plt.close()
    
    # Print summary statistics
    print(f"\nGap Computation Summary:")
    print("-" * 50)
    for method in methods:
        total_computations = sum(gap_computations_data[method])
        avg_computations = total_computations / len(problems)
        print(f"{get_label_for_method(method):<20}: Total={total_computations:>8}, Average={avg_computations:>8.1f}")


def plot_against_metaopt(problem_dict, output_dir, problem_type):
    """Create a scatter plot comparing MetaEase vs MetaOpt performance."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Collect data for scatter plot
    problems = []
    metaopt_times = []
    metaopt_gaps = []
    metaease_times = []
    metaease_gaps = []
    
    for problem in problem_dict:
        if "MetaOpt" in problem_dict[problem] and "MetaEase" in problem_dict[problem]:
            problems.append(problem)
            
            # Get MetaOpt data (max gap and corresponding time)
            metaopt_gaps_data = problem_dict[problem]["MetaOpt"]["gaps"]
            metaopt_times_data = problem_dict[problem]["MetaOpt"]["time_from_start"]
            max_gap_idx = metaopt_gaps_data.index(max(metaopt_gaps_data))
            metaopt_gaps.append(metaopt_gaps_data[max_gap_idx])
            metaopt_times.append(metaopt_times_data[max_gap_idx])
            
            # Get MetaEase data (max gap and corresponding time)
            metaease_gaps_data = problem_dict[problem]["MetaEase"]["gaps"]
            metaease_times_data = problem_dict[problem]["MetaEase"]["time_from_start"]
            max_gap_idx = metaease_gaps_data.index(max(metaease_gaps_data))
            metaease_gaps.append(metaease_gaps_data[max_gap_idx])
            metaease_times.append(metaease_times_data[max_gap_idx])
    
    if not problems:
        print("No problems with both MetaOpt and MetaEase data found.")
        return
    
    # Calculate percentage and time ratio
    gap_percentages = []
    time_ratios = []
    
    for i in range(len(problems)):
        # MetaEase gap as percentage of MetaOpt gap
        gap_percentage = (metaease_gaps[i] / metaopt_gaps[i]) * 100
        gap_percentages.append(gap_percentage)
        
        # Time ratio (MetaEase time / MetaOpt time)
        time_ratio = (metaease_times[i] - metaopt_times[i]) / 3600
        time_ratios.append(time_ratio)
    
    # Create scatter plot
    ax.scatter(time_ratios, gap_percentages, 
              c=get_color_for_method("MetaEase"), 
              s=200, alpha=0.8,
            #   marker='o',
              edgecolors='black', linewidth=1)
    
    # Add problem labels
    for i, problem in enumerate(problems):
        ax.annotate(format_problem_name(problem), 
                   (time_ratios[i], gap_percentages[i]),
                   xytext=(4, 6), textcoords='offset points',
                   fontsize=7, alpha=0.8, fontweight='bold')
    
    # Enhanced plot styling
    ax.set_xlabel('Time (MetaEase - MetaOpt) (h)', fontsize=12)
    ax.set_ylabel('MetaEase Gap (% of MetaOpt Gap)', fontsize=12)
    ax.set_ylim(0, 110)
    # ax.set_title('MetaEase Performance: Gap Percentage vs Time Ratio\n(Higher % + Lower Ratio = Better)', 
    #             fontsize=16, fontweight='bold', pad=20)
    
    # Add reference lines
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label='MetaOpt Baseline (100%)')
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Equal Time (0.0s)')
    
    # Improve grid and background
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # Enhanced legend
    legend = ax.legend(loc='lower right',
                      fontsize=10, 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True,
                      framealpha=0.9,
                      edgecolor='gray')
    
    # Improve axis formatting
    ax.tick_params(axis='both', which='major', labelsize=11)
    # yticks every 10
    ax.set_yticks(np.arange(0, 110, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle background color
    fig.patch.set_facecolor('white')
    
    # Add performance summary text
    better_metaease = sum(1 for i in range(len(problems)) 
                         if gap_percentages[i] > 100 and time_ratios[i] < 1.0)
    total_problems = len(problems)
    
    # summary_text = f'MetaEase better: {better_metaease}/{total_problems} problems'
    # ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
    #         fontsize=12, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save with high DPI
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{problem_type}_metaease_vs_metaopt_comparison.pdf", 
               dpi=300, 
               bbox_inches='tight', 
               facecolor='white',
               edgecolor='none')
    plt.close()
    with open(f"{output_dir}/{problem_type}_metaease_vs_metaopt_comparison.txt", "w") as f:
        f.write(f"MetaEase better: {better_metaease}/{total_problems} problems\n")
        f.write(f"Topologies: {', '.join(problems)}\n")
        f.write(f"Time ratio: {', '.join([f'{time_ratios[i]:.2f}' for i in range(len(time_ratios))])} h\n")
        f.write(f"Gap percentage: {', '.join([f'{gap_percentages[i]:.2f}' for i in range(len(gap_percentages))])} %\n")


def plot_max_gaps_bar(problem_dict, output_dir, problem_type, setting, remove_metaopt=False):
    """Plot grouped bar chart of max gaps per method across problems.

    - X-axis: problems (pretty formatted via format_problem_name)
    - Bars: one per method for each problem, height = max gap for that method/problem
    """
    print(f"**********plot_max_gaps_bar**********")
    # Collect the union of methods across all problems to keep ordering consistent
    setup_plot_style()
    method_set = set()
    for problem in problem_dict:
        method_set.update(problem_dict[problem].keys())
    methods = sorted(list(method_set), key=lambda x: METHOD_ORDER.index(x))
    if remove_metaopt and "MetaOpt" in methods:
        methods.remove("MetaOpt")

    if not methods:
        print("No methods found to plot.")
        return

    problems = sorted(list(problem_dict.keys()), key=str.lower)
    num_problems = len(problems)
    num_methods = len(methods)

    # Prepare data matrix [num_methods x num_problems]
    gaps_matrix = []
    for method in methods:
        method_gaps = []
        for problem in problems:
            # sort the methods by METHOD_ORDER
            if method in problem_dict[problem]:
                # use max gap for consistency with other plots
                gaps = problem_dict[problem][method]["gaps"]
                method_gaps.append(max(gaps) if len(gaps) > 0 else np.nan)
            else:
                # import pdb; pdb.set_trace()
                method_gaps.append(0)
        gaps_matrix.append(method_gaps)

    # sort the problem alphabetically
    problems = sorted(problems, key=str.lower)
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_problems)
    total_width = 0.8
    bar_width = total_width / max(1, num_methods)
    offset_start = - (total_width - bar_width) / 2.0
    max_total_gap = max(max(gaps_matrix[i]) for i in range(num_methods))
    for idx, method in enumerate(methods):
        offsets = x + offset_start + idx * bar_width
        # Ensure zero-height bars are still visible: plot a tiny height but label true value
        true_vals = gaps_matrix[idx]
        display_vals = [
            (v if (v is not None and not np.isnan(v)) else np.nan)
            for v in true_vals
        ]
        print(f"problems: {problems}")
        print(f"{method}: {display_vals}")
        bars = ax.bar(offsets,
                      display_vals,
                      width=bar_width,
                      color=get_color_for_method(method),
                      label=get_label_for_method(method),
                      hatch=METHOD_HATCHES[method],
                      alpha=0.9,
                      edgecolor='black',
                      linewidth=0.6)
        # Add value labels
        for j, bar in enumerate(bars):
            v = true_vals[j]
            if np.isnan(v):
                continue
            if v == 0 or True:
                # instaed of 0.2 show .2
                display_val = f"{v:.1f}" if v < 1 else f"{v:.0f}"
                if display_val.startswith("0."):
                    display_val = display_val[1:]
                if display_val == ".0":
                    display_val = "0"
                fontsize = 18 if v < 1 else 19
                if (j >= 10 or  j < 15) and problem_type == "LLM":
                    fontsize = 16
                ax.text(bar.get_x() + bar.get_width()/2.0,
                        bar.get_height(),
                        display_val,
                        ha='center', va='bottom', fontsize=fontsize)

    # Axis and styling
    # ax.set_xlabel(setting, fontweight="bold", fontsize=20)
    ax.set_ylabel('Norm Max Gap (%)', fontweight="bold", fontsize=22)
    center_x = x + offset_start + (len(methods) - 1) * bar_width * 0.7
    for index, problem in enumerate(problems):
        if len(problem) >= 7:
            center_x[index] = x[index] + offset_start + (len(methods)) * bar_width
        if problem == "Cogentco" and problem_type == "LLM":
            center_x[index] = x[index] + offset_start + (len(methods) + 1) * bar_width * 0.7 + 0.05
    ax.set_xticks(center_x)
    # set ytick font size to 19, make the color dark black
    ax.tick_params(axis='y', labelsize=19, color='black', labelcolor='black')
    ax.set_xticklabels([format_problem_name(p) for p in problems], size=19, rotation=0, ha='right', fontweight="bold")
    ax.grid(False)  # Turn off all grids first
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5)  # Only horizontal grids
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Arrow position (90% to the right, starting from 80% height going to 90% height)
    arrow_x = xlim[0] + 0.95 * (xlim[1] - xlim[0])
    arrow_y_start = ylim[0] + 0.45 * (ylim[1] - ylim[0])
    arrow_y_end = ylim[0] + 0.65 * (ylim[1] - ylim[0])
    better_font_size = 20

    if problem_type == "TE_PoP":
        arrow_x = xlim[0] + 0.67 * (xlim[1] - xlim[0])
        arrow_y_start = ylim[0] + 0.7 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.9 * (ylim[1] - ylim[0])
        better_font_size = 18
    # Draw the upward arrow
    ax.annotate('', xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    if problem_type == "LLM":
        ax.set_ylim(0, 36)
    # Add rotated "Better" text next to the arrow
    ax.text(arrow_x - 0.02 * (xlim[1] - xlim[0]), 
            (arrow_y_start + arrow_y_end) / 2,
            'Better', rotation=90, va='center', ha='right', 
            fontsize=better_font_size, fontweight='bold')

    # legend = ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True,
    #                    framealpha=0.9, edgecolor='gray')
    # two columns, up and outside
    legend = ax.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 1.2),
                      ncol=3,
                      fontsize=18, 
                      frameon=False,
                      columnspacing=0.8,   # reduce space between columns (default ~2.0)
                      handletextpad=0.5,   # reduce space between legend symbol and text
                      borderaxespad=0.2    # reduce space between legend and plot
                      )

    # Save the plotting data to JSON
    plot_data = {
        "problem_type": problem_type,
        "setting": setting,
        "problems": problems,
        "methods": methods,
        "gaps_matrix": gaps_matrix,
        "method_labels": {method: get_label_for_method(method) for method in methods},
        "method_colors": {method: get_color_for_method(method) for method in methods},
        "method_hatches": {method: METHOD_HATCHES.get(method, "") for method in methods}
    }
    
    with open(f"{output_dir}/{problem_type}_max_gap_bars_data.json", "w") as f:
        json.dump(plot_data, f, indent=2)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{problem_type}_max_gap_bars.pdf",
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="method_comparison")
    args = parser.parse_args()
    experiment_name = args.experiment_file.split("/")[-1].split(".")[0]
    args.output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
 
    problem_dict, setting = get_problem_dict(args.experiment_file)
    if experiment_name == "TE_DemandPinning":
        # read and add to problem_dict
        with open("/data1/pantea/MetaEase/MetaOptimize/scripts/paper_logs/TE_DemandPinning_large_problem_dict.json", "r") as f:
            problem_dict_large = json.load(f)
        problem_dict.update(problem_dict_large)

    # print method and max gap for each problem
    for problem in problem_dict:
        for method in problem_dict[problem]:
            print(f"{method}: {problem_dict[problem][method]['gaps'][-1]} at time {problem_dict[problem][method]['time_from_start'][-1]}")
    # sort the problem_dict keys
    problem_dict = dict(sorted(problem_dict.items(), key=lambda x: x[0]))
    if experiment_name == "LLM":
        problem_dict["Cogentco"]["SampleBasedGradient"]["gap_computations"] = 9
        problem_dict["Uninet2010"]["SampleBasedGradient"]["gap_computations"] = 10
    plot_baselines(problem_dict, args.output_dir, problem_type=experiment_name, remove_metaopt=True)
    plot_gap_computations(problem_dict, args.output_dir, problem_type=experiment_name, setting=setting)
    # plot_against_metaopt(problem_dict, args.output_dir, problem_type=experiment_name)
    plot_max_gaps_bar(problem_dict, args.output_dir, problem_type=experiment_name, setting=setting, remove_metaopt=True)


if __name__ == "__main__":
    main()








