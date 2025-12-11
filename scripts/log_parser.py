#!/usr/bin/env python3

import re
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
from plot_common import setup_plot_style
import json
setup_plot_style()

def parse_metaopt_log_file(log_file_path, print_gap_values=False):
    """
    Parse MetaOpt log file format which contains:
    - Optimal and heuristic objectives
    - Gap value
    - Demand values
    - Time taken
    """
    # Initialize variables
    optimal_objective = None
    heuristic_objective = None
    gap_value = None
    demand_values = {}
    time_taken_ms = None

    # Regular expressions
    objective_pattern = r"Optimal objective: ([\d.]+), Heuristic objective: ([\d.]+)"
    gap_pattern = r"Gap: ([\d.]+)"
    demand_pattern = r"Demand values: ({.*})"
    time_pattern = r"Time taken: ([\d]+) ms"

    with open(log_file_path, "r") as f:
        content = f.read()
        
        # Extract objectives
        objective_match = re.search(objective_pattern, content)
        if objective_match:
            optimal_objective = float(objective_match.group(1))
            heuristic_objective = float(objective_match.group(2))

        # Extract gap value
        gap_match = re.search(gap_pattern, content)
        if gap_match:
            gap_value = float(gap_match.group(1))

        # Extract demand values
        demand_match = re.search(demand_pattern, content)
        if demand_match:
            try:
                import json
                demand_str = demand_match.group(1)
                demand_values = json.loads(demand_str)
            except json.JSONDecodeError:
                print("Warning: Could not parse demand values JSON")

        # Extract time taken
        time_match = re.search(time_pattern, content)
        if time_match:
            time_taken_ms = int(time_match.group(1))

    if print_gap_values:
        print(f"Optimal Objective: {optimal_objective}")
        print(f"Heuristic Objective: {heuristic_objective}")
        print(f"Gap Value: {gap_value}")
        print(f"Time Taken: {time_taken_ms} ms")
        print(f"Number of Demand Values: {len(demand_values)}")

    time_from_start = time_taken_ms / 1000.0 if time_taken_ms else None
    x = [0.0, time_from_start]
    y = [0.0, gap_value]
    return {
        "optimal_objective": optimal_objective,
        "heuristic_objective": heuristic_objective,
        "gap_value": gap_value,
        "demand_values": demand_values,
        "time_taken_ms": time_taken_ms,
        "time_from_start": x,
        "gaps": y,
    }

def get_curve(all_gaps, gap_times):
    max_gap = 0
    x = [0]
    y = [0]
    for gap, time in zip(all_gaps, gap_times):
        if gap < 10**(-12):
            gap = gap * 10**(12)
        if float(gap) > max_gap:
            max_gap = float(gap)
            x.append(gap)
            y.append(time)
    return x, y

def parse_log_file_from_results_json(results_json_path):
    with open(results_json_path, "r") as f:
        results = json.load(f)
    try:
        gap_computations = results["gap_computations"] if "gap_computations" in results else len(results["all_gaps"])
        gaps, time_from_start = get_curve(results["all_gaps"], results["gaps_times"]) if "gaps_times" in results else get_curve([0, results["max_gap"] if "max_gap" in results else results["max_global_gap"]], [0, results["execution_time"]])
        return {
            "gap_computations": gap_computations,
            "final_gap": max(gaps),
            "time_from_start": time_from_start,
            "gaps": gaps,
            "gap_times_unavailable": "gaps_times" not in results,
        }
    except:
        return None

def parse_log_file(log_file_path, print_gap_values=False):
    # Initialize variables
    start_time = None
    end_time = None
    final_gap = None
    gap_values = []

    # Regular expressions
    timestamp_pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
    gap_pattern = r"(?:Current (?:max (?:global )?|best )|Final max (?:global )?)gap: (\d+\.?\d*)" # matches Current max gap, Current max global gap, Current best gap, or Final max gap, Final max global gap
    final_gap_pattern = r"Final max (?:global )?gap: (\d+\.?\d*)"
    is_experiment_finished = True

    with open(log_file_path, "r") as f:
        for line in f:
            # Extract timestamp
            timestamp_match = re.search(timestamp_pattern, line)
            if timestamp_match:
                current_time = datetime.strptime(
                    timestamp_match.group(1), "%Y-%m-%d %H:%M:%S"
                )

                # Set start time if not set
                if not start_time:
                    start_time = current_time
                end_time = current_time

            # Extract gap values
            gap_match = re.search(gap_pattern, line)
            if gap_match:
                gap = float(gap_match.group(1))
                if current_time:  # Only add if we have a valid timestamp
                    gap_values.append((current_time, gap))

            # Extract final gap
            final_gap_match = re.search(final_gap_pattern, line)
            if final_gap_match:
                final_gap = float(final_gap_match.group(1))
                is_experiment_finished = True

    # Calculate total experiment time
    total_time = end_time - start_time if start_time and end_time else None
    if print_gap_values:
        # Print results
        print(f"Total Experiment Time: {total_time}")
        print(f"Final Gap Value: {final_gap}")
        print("Timestamp         Time from Start(s)      Gap")
        print("-" * 55)

    x = [0.0]
    y = [0.0]
    for timestamp, gap in gap_values:
        time_from_start = (timestamp - start_time).total_seconds()
        if gap > y[-1]:
            if print_gap_values:
                print(f"{timestamp}    {time_from_start}    {gap:.4f}")
            x.append(time_from_start)
            y.append(gap)

    return {
        "start_time": start_time,
        "end_time": end_time,
        "total_time": total_time,
        "final_gap": final_gap,
        "gap_values": gap_values,
        "time_from_start": x,
        "gaps": y,
        "is_experiment_finished": is_experiment_finished,
    }


def plot_gap_over_time(x, y, output_file):
    # Set style
    plt.style.use("bmh")  # Using a built-in style

    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the data with a line and markers
    ax.plot(
        x,
        y,
        "b-o",
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="Gap Value",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    # Fill area under the curve with light blue
    ax.fill_between(x, y, alpha=0.2, color="blue")

    # Set labels and title with larger font sizes
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Gap Value", fontsize=12)
    ax.set_title("Evolution of Gap Value Over Time", fontsize=14, pad=15)

    # Format y-axis to use scientific notation for large numbers
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set x-axis ticks
    num_ticks = 6
    tick_positions = np.linspace(min(x), max(x), num_ticks)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{t:.0f}" for t in tick_positions], rotation=45)

    # Add legend
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description="Parse experiment log file")
    parser.add_argument("--log_file", type=str, help="Path to the experiment log file")
    args = parser.parse_args()

    results = parse_log_file(args.log_file, print_gap_values=True)

    # Create the plot
    output_file = args.log_file.replace("experiment.log", "gap_over_time.png")
    plot_gap_over_time(results["time_from_start"], results["gaps"], output_file)


if __name__ == "__main__":
    main()
