from common import *


def get_file_name_from_input_values(input_values):
    name = "_".join(f"{k}_{v:.2f}" for k, v in input_values.items())
    # if it's too long, truncate it
    if len(name) > 100:
        name = name[:100]
    return name


def plot_heuristic_and_optimal_values(steps_and_values, demand_key, save_path):
    """Plot analysis of heuristic, optimal, and relaxed optimal values."""
    # Create results directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Extract data from steps_and_values
    steps = [item[0] for item in steps_and_values]
    heuristic_values = [item[1] for item in steps_and_values]
    optimal_values = [item[3] for item in steps_and_values]
    relaxed_optimal_values = [item[4] for item in steps_and_values]
    gaps = [item[5] for item in steps_and_values]
    relaxed_gaps = [item[6] for item in steps_and_values]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Plot heuristic values
    ax1.plot(steps, heuristic_values, label="Heuristic", color="blue")
    ax1.set_xlabel(f"Value ({demand_key})")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f"Heuristic Values for {demand_key}")

    # Plot optimal and relaxed optimal values
    ax2.plot(steps, optimal_values, label="Optimal", color="red")
    ax2.plot(
        steps,
        relaxed_optimal_values,
        label="Relaxed Optimal",
        color="green",
        linestyle="--",
    )
    ax2.set_xlabel(f"Value ({demand_key})")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)
    ax2.set_title(f"Optimal and Relaxed Optimal Values for {demand_key}")

    # Plot gap with optimal
    ax3.plot(steps, gaps, label="Gap with Optimal", color="purple")
    ax3.set_xlabel(f"Value ({demand_key})")
    ax3.set_ylabel("Gap")
    ax3.legend()
    ax3.grid(True)
    ax3.set_title(f"Gap with Optimal for {demand_key}")

    # Plot gap with relaxed optimal
    ax4.plot(steps, relaxed_gaps, label="Gap with Relaxed Optimal", color="orange")
    ax4.set_xlabel(f"Value ({demand_key})")
    ax4.set_ylabel("Gap")
    ax4.legend()
    ax4.grid(True)
    ax4.set_title(f"Gap with Relaxed Optimal for {demand_key}")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{demand_key}_analysis.png")
    plt.close()

    # Save the data to a CSV file
    df = pd.DataFrame(
        {
            "value": steps,
            "heuristic": heuristic_values,
            "optimal": optimal_values,
            "relaxed_optimal": relaxed_optimal_values,
            "gap_optimal": gaps,
            "gap_relaxed": relaxed_gaps,
        }
    )
    df.to_csv(f"{save_path}/{demand_key}_data.csv", index=False)


def sweep_analysis(
    problem,
    anchor_input_values,
    max_value,
    step_size,
    anchor_heu_code_path_num,
    minimize_is_better,
    save_dir,
):
    """Perform sweep analysis for any problem type."""
    print("Starting sweep analysis...")
    for key, value in anchor_input_values.items():
        print(f"\nAnalyzing variable: {key}")
        steps_and_values = []

        # Calculate range with more points around the anchor value
        start_value = max(0, value - step_size)
        end_value = min(max_value, value + step_size)
        num_points = 200  # Increase number of points for smoother plots

        for i in np.linspace(start_value, end_value, num_points):
            input_values = anchor_input_values.copy()
            input_values[key] = i
            (
                heuristic_value,
                heu_code_path_num,
                optimal_value,
                optimal_all_vars,
                relaxed_optimal_value,
                relaxed_all_vars,
            ) = get_heuristic_and_optimal_values(problem, input_values)
            # Calculate gaps (positive gap means better performance)
            if minimize_is_better:
                optimal_gap = (
                    heuristic_value - optimal_value
                )  # Smaller is better, so flip the subtraction
                relaxed_optimal_gap = heuristic_value - relaxed_optimal_value
            else:
                optimal_gap = optimal_value - heuristic_value  # Larger is better
                relaxed_optimal_gap = relaxed_optimal_value - heuristic_value

            if heu_code_path_num == anchor_heu_code_path_num:
                steps_and_values.append(
                    [
                        i,
                        heuristic_value,
                        heu_code_path_num,
                        optimal_value,
                        relaxed_optimal_value,
                        optimal_gap,
                        relaxed_optimal_gap,
                    ]
                )

        plot_heuristic_and_optimal_values(
            steps_and_values,
            key,
            save_path=f"{save_dir}/sweep_results_{step_size}_{get_file_name_from_input_values(anchor_input_values)}",
        )
        print(f"Completed analysis for {key}")

    print(
        f"\nAnalysis complete! Results saved in {save_dir}/sweep_results_{step_size} directory."
    )


def plot_gp_vs_actual_values(steps_and_values, var_key, save_path):
    """Plot GP predictions vs actual values."""
    os.makedirs(save_path, exist_ok=True)

    # Extract data
    steps = [item[0] for item in steps_and_values]
    gp_predictions = [item[1] for item in steps_and_values]
    actual_values = [item[2] for item in steps_and_values]
    uncertainties = [item[3] for item in steps_and_values]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot GP predictions with uncertainty
    ax1.plot(steps, gp_predictions, label="GP Prediction", color="blue")
    ax1.fill_between(
        steps,
        np.array(gp_predictions) - 2 * np.array(uncertainties),
        np.array(gp_predictions) + 2 * np.array(uncertainties),
        color="blue",
        alpha=0.2,
        label="95% Confidence",
    )
    ax1.plot(steps, actual_values, label="Actual Value", color="red", linestyle="--")
    ax1.set_xlabel(f"Value ({var_key})")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f"GP Predictions vs Actual Values for {var_key}")

    # Plot prediction error
    prediction_errors = np.array(gp_predictions) - np.array(actual_values)
    ax2.plot(steps, prediction_errors, label="Prediction Error", color="green")
    ax2.fill_between(
        steps,
        -2 * np.array(uncertainties),
        2 * np.array(uncertainties),
        color="green",
        alpha=0.2,
        label="95% Confidence",
    )
    ax2.set_xlabel(f"Value ({var_key})")
    ax2.set_ylabel("Prediction Error")
    ax2.legend()
    ax2.grid(True)
    ax2.set_title(f"GP Prediction Error for {var_key}")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{var_key}_gp_analysis.png")
    plt.close()

    # Save the data
    df = pd.DataFrame(
        {
            "value": steps,
            "gp_prediction": gp_predictions,
            "actual_value": actual_values,
            "uncertainty": uncertainties,
        }
    )
    df.to_csv(f"{save_path}/{var_key}_gp_data.csv", index=False)


def plot_gradient_analysis(steps_and_values, var_key, save_path):
    """Plot gradient analysis."""
    os.makedirs(save_path, exist_ok=True)

    # Extract data
    steps = [item[0] for item in steps_and_values]
    gradients = [item[1] for item in steps_and_values]

    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    # Plot gradients
    ax1.plot(steps, gradients, label="Gradient", color="blue")
    ax1.set_xlabel(f"Value ({var_key})")
    ax1.set_ylabel("Gradient Value")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f"Gradient Estimates for {var_key}")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{var_key}_gradient_analysis.png")
    plt.close()

    # Save the data
    df = pd.DataFrame(
        {
            "value": steps,
            "gradient": gradients,
        }
    )
    df.to_csv(f"{save_path}/{var_key}_gradient_data.csv", index=False)


def gradient_analysis(
    problem,
    anchor_input_values,
    block_length,
    gp,
    scaler_x,
    scaler_y,
    keys_for_heuristic,
    num_samples,
    save_dir,
):
    """Perform gradient analysis for any problem type."""
    print("\nStarting Gradient analysis...")
    save_dir = f"{save_dir}/gradient_results_{block_length}_{num_samples}_{get_file_name_from_input_values(anchor_input_values)}"
    os.makedirs(save_dir, exist_ok=True)

    # For each dimension, calculate gradients around anchor point
    for idx, key in enumerate(anchor_input_values.keys()):
        print(f"\nAnalyzing gradients for {key}")
        steps_and_values = []

        # Calculate range around anchor value
        start_value = max(0, anchor_input_values[key] - block_length)
        end_value = min(max_value, anchor_input_values[key] + block_length)
        steps = np.linspace(start_value, end_value, 200)

        for step in steps:
            # Create point for gradient estimation
            point = np.array([anchor_input_values[k] for k in keys_for_heuristic])
            point[idx] = step

            # Estimate gradient
            gradient = estimate_gradient(gp, scaler_x, scaler_y, point.copy())[idx]
            steps_and_values.append([step, gradient])

        # Plot and save results
        plot_gradient_analysis(steps_and_values, key, save_dir)
        print(f"Completed gradient analysis for {key}")

    print(f"Gradient analysis complete! Results saved in {save_dir} directory.")
