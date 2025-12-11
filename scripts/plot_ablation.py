import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from log_parser import parse_log_file
from plot_common import *
from plot_methods import METHOD_COLORS as METHOD_COLORS_PLOT_METHODS
from plot_methods import get_log_data_for_one_experiment, scale_log_data

Problem = "KleeNotEnough"

TIMINGS_TABLE = {
    "Klee + GP": {"swan": 2618.0000693798065, "b4-teavar": 3519.6368210315704, "abilene": 1633.5674574375153},
    "Random + GP": {"swan": 607.923549413681, "b4-teavar": 4558.510195732117, "abilene": 692.6063709259033},
    "varying_klee_inputs_with_no_gradient_ascent_4": {"swan": 65.54589009284973, "b4-teavar": 340.6763803958893, "abilene": 98.1863112449646},
    "varying_klee_inputs_with_no_gradient_ascent_8": {"swan": 45.94328761100769, "b4-teavar": 206.2981779575348, "abilene": 62.954373598098755},
    "varying_klee_inputs_with_no_gradient_ascent_16": {"swan": 28.431153297424316, "b4-teavar": 103.27003169059753, "abilene": 38.86101222038269},
    "varying_klee_inputs_with_no_gradient_ascent_32": {"swan": 75.24105191230774, "b4-teavar": 79.14918613433838, "abilene": 34.95122289657593},
    "varying_klee_inputs_with_no_gradient_ascent_64": {"swan": 1237.1975500583649, "b4-teavar": 39.05155849456787, "abilene": 210.87700748443604},
    "varying_klee_inputs_with_no_gradient_ascent_128": {"swan": 1244.89590716362, "b4-teavar": 29.840561151504517, "abilene": 892.5809605121613},
    "Random + Direct Gradient": {"swan": 5.010131597518921, "b4-teavar": 96.82276344299316, "abilene": 8.752670049667358},
    "Klee + Direct Gradient": {"swan": 2254.5375435352325, "b4-teavar": 59617.95025730133, "abilene": 5014.899379253387},
    "LLM + GP": {"abilene": 1382.5396492481232, "swan": 546.1390287876129, "b4-teavar": 7719.654041051865},
    "Klee + SampleBasedGradient": {"swan": 94140.0, "b4-teavar": 83258.77625846863, "abilene": 89946.3962827},
    }

# "BlockLength": {"0.1": 621.7342035770416, "0.5": 620.8125612735748, "1.0": 622.2939355373383, "5.0": 620.2243139743805, "10.0": 622.6246445178986, "20.0": 622.1144859790802}
setup_plot_style()
tickz_rotation = 0

# Define ablation study directories
if Problem == "Seed":
    ablation_dirs = {
        "LLM + GP": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning_LLM/LLM_with_GP",
        "No Customization + Klee + GP": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning_no_customization/Klee_with_GP",
        "Klee + GP": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_with_GP",
        "Random + GP": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Random_with_GP",
    }
    x_label = ""
elif Problem == "Gradient":
    ablation_dirs = {
        "Klee + GP": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_with_GP",
        "Klee + Direct Gradient": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_with_Direct_Gradient",
        "Klee + SampleBasedGradient": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/Klee_Gap_Sample_Based",
    }
elif Problem == "Parameter_K":
    ablation_dirs = {
        "varying_klee_inputs_with_no_gradient_ascent_4": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/varying_klee_inputs_with_no_gradient_ascent_4",
        "varying_klee_inputs_with_no_gradient_ascent_8": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/varying_klee_inputs_with_no_gradient_ascent_8",
        "varying_klee_inputs_with_no_gradient_ascent_16": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/varying_klee_inputs_with_no_gradient_ascent_16",
        "varying_klee_inputs_with_no_gradient_ascent_32": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/varying_klee_inputs_with_no_gradient_ascent_32",
        "varying_klee_inputs_with_no_gradient_ascent_64": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/varying_klee_inputs_with_no_gradient_ascent_64",
        "varying_klee_inputs_with_no_gradient_ascent_128": "/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/varying_klee_inputs_with_no_gradient_ascent_128",
    }
elif Problem == "BlockLenght":
    ablation_dirs = {
        "0.1": "/data1/pantea/MetaEase/MetaOptimize/ablation_PoP/BlockLength/0.1",
        "0.5": "/data1/pantea/MetaEase/MetaOptimize/ablation_PoP/BlockLength/0.5",
        "1.0": "/data1/pantea/MetaEase/MetaOptimize/ablation_PoP/BlockLength/1.0",
        "5.0": "/data1/pantea/MetaEase/MetaOptimize/ablation_PoP/BlockLength/5.0",
        "10.0": "/data1/pantea/MetaEase/MetaOptimize/ablation_PoP/BlockLength/10.0",
        "20.0": "/data1/pantea/MetaEase/MetaOptimize/ablation_PoP/BlockLength/20.0",
    }

TOPOLOGIES = ["abilene", "b4-teavar", "swan"]
if Problem == "BlockLenght" or Problem == "NumSamples":
    TOPOLOGIES = ["abilene"]

TOPOLOGY_NAMES = {
    "swan": "Swan",
    "b4-teavar": "B4",
    "abilene": "Abilene",
}

# pick three fancy professional colors
TOPOLOGY_COLORS = {
    "swan": "#00508C",  # deep orange
    "b4-teavar": "#873600",  # dark burnt orange
    "abilene": "#4E9F50",  # dark forest green
}

HATCHES = {"swan": "/", "b4-teavar": "\\", "abilene": ""}

METHOD_COLORS = {
    "Klee + GP": METHOD_COLORS_PLOT_METHODS["MetaEase"],
    "Random + GP": METHOD_COLORS_PLOT_METHODS["Random"],
    "Klee + Direct Gradient": METHOD_COLORS_PLOT_METHODS["HillClimbing"],
    "MetaEase": METHOD_COLORS_PLOT_METHODS["MetaEase"],
    "No Customization + Klee + GP": METHOD_COLORS_PLOT_METHODS["SimulatedAnnealing"],
    "LLM + GP": "#767676",
    "Klee + SampleBasedGradient": METHOD_COLORS_PLOT_METHODS["SampleBasedGradient"],
    "varying_klee_inputs_with_no_gradient_ascent_4": "#009E73",
    "varying_klee_inputs_with_no_gradient_ascent_8": "#4E79A7",
    "varying_klee_inputs_with_no_gradient_ascent_16": "#0072B2",
    "varying_klee_inputs_with_no_gradient_ascent_32": "#E15759",
    "varying_klee_inputs_with_no_gradient_ascent_64": "#F28E2B",
    "varying_klee_inputs_with_no_gradient_ascent_128": "#767676",
    "0.1": "#009E73",
    "0.5": "#4E79A7",
    "1.0": "#0072B2",
    "5.0": "#E15759",
    "10.0": "#F28E2B",
    "20.0": "#767676",
}

METHOD_HATCHES = {
    "Klee + GP": "",
    "Random + GP": "\\",
    "Klee + Direct Gradient": "\\",
    "MetaEase": "",
    "No Customization + Klee + GP": "/",
    "LLM + GP": ".",
    "Klee + SampleBasedGradient": "/",
    "varying_klee_inputs_with_no_gradient_ascent_4": "",
    "varying_klee_inputs_with_no_gradient_ascent_8": "/",
    "varying_klee_inputs_with_no_gradient_ascent_16": "\\",
    "varying_klee_inputs_with_no_gradient_ascent_32": ".",
    "varying_klee_inputs_with_no_gradient_ascent_64": "x",
    "varying_klee_inputs_with_no_gradient_ascent_128": "o",
    "0.1": "",
    "0.5": "/",
    "1.0": "\\",
    "5.0": ".",
    "10.0": "x",
    "20.0": "o",
}

if Problem == "Seed":
    METHOD_NAMES = {
        "No Customization + Klee + GP": "Klee w/o Customization",
        "Klee + GP": "Klee w/ Customization",
        "Random + GP": "Random",
        "LLM + GP": "LLM",
    }
    METHOD_ORDER = [
        "No Customization + Klee + GP",
        "Klee + GP",
        "Random + GP",
        "LLM + GP",
    ]
elif Problem == "Gradient":
    METHOD_NAMES = {
        "Klee + Direct Gradient": "Direct Heuristic Gradient",
        "Klee + GP": "GP-based Gradient",
        "Klee + SampleBasedGradient": "Direct Gap Gradient",
    }
    METHOD_ORDER = [
        "Klee + GP",
        "Klee + Direct Gradient",
        "Klee + SampleBasedGradient",
    ]
elif Problem == "Parameter_K":
    METHOD_NAMES = {
        "varying_klee_inputs_with_no_gradient_ascent_4": "K = 4",
        "varying_klee_inputs_with_no_gradient_ascent_8": "K = 8",
        "varying_klee_inputs_with_no_gradient_ascent_16": "K = 16",
        "varying_klee_inputs_with_no_gradient_ascent_32": "K = 32",
        "varying_klee_inputs_with_no_gradient_ascent_64": "K = 64",
        "varying_klee_inputs_with_no_gradient_ascent_128": "K = 128",
    }
    METHOD_ORDER = [
        "varying_klee_inputs_with_no_gradient_ascent_4",
        "varying_klee_inputs_with_no_gradient_ascent_8",
        "varying_klee_inputs_with_no_gradient_ascent_16",
        "varying_klee_inputs_with_no_gradient_ascent_32",
        "varying_klee_inputs_with_no_gradient_ascent_64",
        "varying_klee_inputs_with_no_gradient_ascent_128",
    ]
elif Problem == "BlockLenght":
    METHOD_NAMES = {
        "0.1": "Block Length = 0.1",
        "0.5": "Block Length = 0.5",
        "1.0": "Block Length = 1.0",
        "5.0": "Block Length = 5.0",
        "10.0": "Block Length = 10.0",
        "20.0": "Block Length = 20.0",
    }
    METHOD_ORDER = [
        "0.1",
        "0.5",
        "1.0",
        "5.0",
        "10.0",
        "20.0",
    ]

def parse_ablation_data():
    """Parse all ablation study data"""
    ablation_data = {}

    for name, base_dir in ablation_dirs.items():
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} does not exist")
            continue
        ablation_data[name] = {}

        # Find all experiment directories
        exp_dirs = glob.glob(os.path.join(base_dir, "*"))
        exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

        for exp_dir in exp_dirs:
            log_file = os.path.join(exp_dir, "experiment.log")
            # Extract topology from directory name
            topology = "unknown"
            if "swan" in exp_dir:
                topology = "swan"
            elif "b4-teavar" in exp_dir:
                topology = "b4-teavar"
            elif "abilene" in exp_dir:
                topology = "abilene"
            log_data = get_log_data_for_one_experiment(exp_dir, "MetaEase")
            log_data = scale_log_data(log_data, "TE_DemandPinning", topology)
            if log_data is None:
                print(f"No log data found for {name} {exp_dir}")
                continue
            if topology not in ablation_data[name]:
                ablation_data[name][topology] = []
            ablation_data[name][topology].append(log_data)

    # reorder the ablation data
    ablation_data = {
        k: v
        for k, v in sorted(
            ablation_data.items(), key=lambda item: METHOD_ORDER.index(item[0])
        )
    }
    return ablation_data


def plot_max_time_per_ablation(ablation_data, output_dir="plots"):
    """Plot maximum time achieved per ablation version"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect data by ablation and topology
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                max_times = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if exp_data["time_from_start"]:
                        max_times.append(max(exp_data["time_from_start"]))

                if max_times:
                    means.append(np.mean(max_times))
                    stds.append(np.std(max_times))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)
        ax.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=TOPOLOGY_NAMES[topology],
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    # ax.set_xlabel("Ablation Version")
    ax.set_ylabel("Maximum Time", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "max_time_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_max_gap_per_ablation(ablation_data, output_dir="plots"):
    """Plot maximum gap achieved per ablation version"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect data by ablation and topology
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                max_gaps = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if "final_gap" in exp_data and exp_data["final_gap"] is not None:
                        max_gaps.append(exp_data["final_gap"])
                    elif exp_data.get("all_gaps"):
                        max_gaps.append(max(exp_data["all_gaps"]))
                    elif exp_data.get("gaps"):
                        max_gaps.append(max(exp_data["gaps"]))
                    else:
                        raise ValueError(
                            f"No gap data found for {ablation_name} {topology}"
                        )

                if max_gaps:
                    means.append(np.mean(max_gaps))
                    stds.append(np.std(max_gaps))
                else:
                    print(f"No gap data found for {ablation_name} {topology}")
                    means.append(0)
                    stds.append(0)
            else:
                print(f"No topology data found for {ablation_name} {topology}")
                means.append(0)
                stds.append(0)

        ax.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=TOPOLOGY_NAMES[topology],
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )
    # Make log scale
    # ax.set_yscale("log")
    # ax.set_xlabel("Ablation Version")
    ax.set_ylabel("Norm Max Gap (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width * 1.7)
    ax.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "max_gap_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    # save max
    max_gaps = {}
    for ablation_name in ablation_names:
        max_gaps[ablation_name] = []
        for topology in topologies:
            max_gaps[ablation_name].append(
                ablation_data[ablation_name][topology][0]["final_gap"]
            )
    with open(os.path.join(output_dir, "max_gaps.json"), "w") as f:
        json.dump(max_gaps, f)
    plt.close()


def plot_max_gap_vs_time_per_ablation(ablation_data, output_dir="plots"):
    """Plot max gap vs time for each topology with different lines for different methods"""
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()
    topologies = TOPOLOGIES

    # Create one plot per topology
    for topology in topologies:
        fig, ax = plt.subplots(figsize=(6, 4))

        # First pass: find max time across all methods for this topology
        max_time = 0
        for ablation_name, topology_data in ablation_data.items():
            if topology in topology_data:
                for exp_data in topology_data[topology]:
                    if exp_data["time_from_start"] and exp_data["gaps"]:
                        max_time = max(max_time, max(exp_data["time_from_start"]))

        # Second pass: extend shorter experiments to max_time
        for ablation_name, topology_data in ablation_data.items():
            if topology in topology_data:
                for exp_data in topology_data[topology]:
                    if exp_data["time_from_start"] and exp_data["gaps"]:
                        if max(exp_data["time_from_start"]) < max_time:
                            # Add points to extend to max_time
                            exp_data["time_from_start"].append(max_time)
                            exp_data["gaps"].append(exp_data["gaps"][-1])

        # Third pass: plot the data
        for idx, (ablation_name, topology_data) in enumerate(ablation_data.items()):
            if topology in topology_data:
                for exp_data in topology_data[topology]:
                    if exp_data["time_from_start"] and exp_data["gaps"]:
                        ax.plot(
                            [t / 3600 for t in exp_data["time_from_start"]],
                            exp_data["gaps"],
                            # alpha=0.7,
                            label=METHOD_NAMES[ablation_name],
                            color=METHOD_COLORS[ablation_name],
                            hatch=METHOD_HATCHES[ablation_name],
                            linewidth=4,
                        )

        ax.set_xlabel("Time Log (hours)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Norm Max Gap (%)", fontsize=14, fontweight="bold")
        # set x log scale
        ax.set_xscale("log")
        # put legend outside of plot at the top, 2 columns, bigger font
        ax.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", ncol=2, frameon=False)
        # ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"max_gap_vs_time_{topology}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_benchmark_calls_per_ablation(ablation_data, output_dir="plots"):
    """Plot number of optimal value calls per ablation version"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect data by ablation and topology
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                optimal_calls = []
                for exp_data in ablation_data[ablation_name][topology]:
                    # Count optimal value calls
                    if (
                        "number of optimal value calls" in exp_data
                        and exp_data["number of optimal value calls"] is not None
                    ):
                        optimal_calls.append(exp_data["number of optimal value calls"])
                    elif (
                        "gap_computations" in exp_data
                        and exp_data["gap_computations"] is not None
                    ):
                        # Fallback: use gap computations
                        optimal_calls.append(exp_data["gap_computations"])
                    elif exp_data.get("gaps"):
                        # Fallback: use number of gap evaluations
                        optimal_calls.append(len(exp_data["gaps"]))

                if optimal_calls:
                    means.append(np.mean(optimal_calls))
                    stds.append(np.std(optimal_calls))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=TOPOLOGY_NAMES[topology],
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    # ax.set_xlabel("Ablation Version")
    ax.set_ylabel("Number of Optimal Value Calls", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "optimal_value_calls_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_heuristic_calls_per_ablation(ablation_data, output_dir="plots"):
    """Plot number of heuristic value calls per ablation version"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect data by ablation and topology
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                heuristic_calls = []
                for exp_data in ablation_data[ablation_name][topology]:
                    # Count heuristic value calls
                    if (
                        "number of heuristic value calls" in exp_data
                        and exp_data["number of heuristic value calls"] is not None
                    ):
                        heuristic_calls.append(
                            exp_data["number of heuristic value calls"]
                        )

                if heuristic_calls:
                    means.append(np.mean(heuristic_calls))
                    stds.append(np.std(heuristic_calls))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=TOPOLOGY_NAMES[topology],
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    # ax.set_xlabel("Ablation Version")
    ax.set_ylabel("Number of Heuristic Value Calls", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heuristic_value_calls_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_ablation_by_topology(ablation_data, output_dir="plots"):
    """Plot ablation methods grouped by topology (bars = methods, x-axis = topologies)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Collect data by topology and ablation
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES
    topologies.sort(key=lambda x: x.lower())

    x_pos = np.arange(len(topologies))
    width = 0.12 if len(ablation_names) > 4 else 0.2

    for i, ablation_name in enumerate(ablation_names):
        means = []
        stds = []

        for topology in topologies:
            if topology in ablation_data[ablation_name]:
                max_gaps = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if "final_gap" in exp_data and exp_data["final_gap"] is not None:
                        max_gaps.append(exp_data["final_gap"])
                    elif exp_data.get("gaps"):
                        max_gaps.append(max(exp_data["gaps"]))

                if max_gaps:
                    print(f"Max gaps for {ablation_name} {topology}: {max_gaps}")
                    means.append(np.mean(max_gaps))
                    stds.append(np.std(max_gaps))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        # means = [m / 200 * 100 for m in means]
        # if an element in means is 0, set it to 0.01
        display_means = means#[1 + m * 1.1 if m < 5 else m for m in means]
        ax.bar(
            x_pos + i * width,
            display_means,
            width,
            # yerr=stds,
            label=METHOD_NAMES[ablation_name],
            color=METHOD_COLORS[ablation_name],
            hatch=METHOD_HATCHES[ablation_name],
            alpha=0.7,
            capsize=5,
        )
        # annonate the gap value on the bar
        if Problem == "Seed":
            for j, (x, y) in enumerate(zip(x_pos + i * width, means)):
                ax.text(x, y + 0.1 , f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    # set y as log scale
    # ax.set_yscale("log")
    # ax.set_xlabel("Topology")
    ax.set_ylabel("Norm Max Gap (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width * len(ablation_names) / 2)
    ax.set_xticklabels(
        [TOPOLOGY_NAMES[topology] for topology in topologies],
        rotation=tickz_rotation,
        ha="right",
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    # y ticks evey 200
    ax.grid(False)
    ax.grid("y", alpha=0.3)
    # ax.set_yticks(np.arange(0, 1800, 400))

    # add an upward arrow on the top right corner
    # Position the arrow in the upper right area of the plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Arrow position (90% to the right, starting from 80% height going to 90% height)
    arrow_x = xlim[0] + 0.9 * (xlim[1] - xlim[0])
    arrow_y_start = ylim[0] + 0.45 * (ylim[1] - ylim[0])
    arrow_y_end = ylim[0] + 0.65 * (ylim[1] - ylim[0])
    if Problem == "Parameter_K":
        arrow_y_start = ylim[0] + 0.35 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.55 * (ylim[1] - ylim[0])

    # Draw the upward arrow
    ax.annotate(
        "",
        xy=(arrow_x, arrow_y_end),
        xytext=(arrow_x, arrow_y_start),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )

    # Add rotated "Better" text next to the arrow
    ax.text(
        arrow_x - 0.02 * (xlim[1] - xlim[0]),
        (arrow_y_start + arrow_y_end) / 2,
        "Better",
        rotation=90,
        va="center",
        ha="right",
        fontsize=8,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ablation_by_topology.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    # print avg of explored_klee_inputs for each ablation method, avg over topologies
    for ablation_name in ablation_names:
        if "klee" in ablation_name.lower():
            avg_explored_klee_inputs = []
            for topology in topologies:
                if topology in ablation_data[ablation_name]:
                    for exp_data in ablation_data[ablation_name][topology]:
                        if "explored_klee_inputs" in exp_data:
                            avg_explored_klee_inputs.append(
                                np.mean(exp_data["explored_klee_inputs"])
                            )
            print(
                f"Avg explored klee inputs for {ablation_name}: {np.mean(avg_explored_klee_inputs)}"
            )


def plot_timings_by_topology(timings_table, ablation_data, output_dir="plots"):
    """Plot timing data grouped by topology (bars = methods, x-axis = topologies)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Get methods and topologies from timings_table
    methods = ablation_dirs.keys()  # list(ablation_data.keys())
    topologies = TOPOLOGIES # list(ablation_data[methods[0]].keys()) if methods else []
    topologies.sort(key=lambda x: x.lower())
    x_pos = np.arange(len(topologies))
    width = 0.12 if len(methods) > 4 else 0.2

    for i, method in enumerate(methods):
        timings = []
        for topology in topologies:
            if method not in timings_table:
                # get it from ablation_data
                timings.append(
                    np.mean(
                        [
                            exp_data["time_from_start"][-1]
                            for exp_data in ablation_data[method][topology]
                        ]
                    )
                )
            else:
                if topology in timings_table[method]:
                    timings.append(timings_table[method][topology])
                else:
                    timings.append(0)

        ax.bar(
            x_pos + i * width,
            [t / 3600 for t in timings],
            width,
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method],
            hatch=METHOD_HATCHES[method],
            alpha=0.7,
            capsize=5,
        )
        # # annotate the timing value on the bar
        # for j, (x, y) in enumerate(zip(x_pos + i * width, timings)):
        #     if y > 0:  # Only annotate non-zero values
        #         ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    # xtick positions are off
    ax.set_ylabel("Runtime (hours)", fontsize=14, fontweight="bold")
    # ax.set_ylim(0, 30)
    # show y ticks as 10^0, 10^1, 10^2, 10^3
    max_y = max([t / 3600 for t in timings])
    if max_y > 1:
        ax.set_ylabel("Runtime Log (hours)", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        # ax.set_ylim(0.001, 30)
        ax.set_yticks([0.5, 1, 5, 10, 24, 80])
        ax.set_yticklabels([0.5, 1, 5, 10, 24, ""])
    # set y log scale
    ax.set_xticks(x_pos + width * len(methods) / 2)
    ax.set_xticklabels(
        [TOPOLOGY_NAMES[topology] for topology in topologies],
        rotation=tickz_rotation,
        ha="right",
        fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper left", ncol=2)
    ax.grid(False)
    ax.grid("y", alpha=0.3)
    # ax.set_yticks(np.arange(0, 18, 2))

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "timings_by_topology.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_convergence_speed(ablation_data, output_dir="plots"):
    """Plot convergence speed (time to reach 90% of max gap)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect data by ablation and topology
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                convergence_times = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if (
                        exp_data["time_from_start"]
                        and exp_data["gaps"]
                        and len(exp_data["gaps"]) > 1
                    ):
                        max_gap = max(exp_data["gaps"])
                        target_gap = 0.9 * max_gap

                        # Find time to reach 90% of max gap
                        convergence_time = None
                        for j, gap in enumerate(exp_data["gaps"]):
                            if gap >= target_gap:
                                convergence_time = exp_data["time_from_start"][j]
                                break

                        if convergence_time is not None:
                            convergence_times.append(convergence_time)

                if convergence_times:
                    means.append(np.mean(convergence_times))
                    stds.append(np.std(convergence_times))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=topology,
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    # ax.set_xlabel("Ablation Version")
    ax.set_ylabel("Convergence Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "convergence_speed_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_efficiency_analysis(ablation_data, output_dir="plots"):
    """Plot efficiency analysis: gap per benchmark call"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect data by ablation and topology
    ablation_names = list(ablation_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                efficiencies = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if exp_data["gaps"]:
                        max_gap = max(exp_data["gaps"])

                        # Calculate efficiency
                        if (
                            "gap_computations" in exp_data
                            and exp_data["gap_computations"] is not None
                        ):
                            efficiency = max_gap / exp_data["gap_computations"]
                        else:
                            efficiency = max_gap / len(exp_data["gaps"])

                        efficiencies.append(efficiency)

                if efficiencies:
                    means.append(np.mean(efficiencies))
                    stds.append(np.std(efficiencies))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=topology,
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    # ax.set_xlabel("Ablation Version")
    ax.set_ylabel("Efficiency (Gap per Call)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=45,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "efficiency_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_topology_comparison(ablation_data, output_dir="plots"):
    """Plot comparison across topologies for each ablation"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    topologies = TOPOLOGIES

    for idx, (ablation_name, topology_data) in enumerate(ablation_data.items()):
        if idx >= 4:
            break

        ax = axes[idx]

        for topo_idx, topology in enumerate(topologies):
            if topology in topology_data:
                max_gaps = []
                for exp_data in topology_data[topology]:
                    if exp_data["gaps"]:
                        max_gaps.append(max(exp_data["gaps"]))

                if max_gaps:
                    ax.hist(
                        max_gaps,
                        alpha=0.7,
                        label=topology,
                        color=TOPOLOGY_COLORS[topology],
                        hatch=HATCHES[topology],
                        bins=10,
                    )

        ax.set_xlabel("Maximum Gap", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topology_comparison_per_ablation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_summary_comparison(ablation_data, output_dir="plots"):
    """Create a comprehensive summary comparison plot"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Collect all data for summary
    summary_data = {}

    for ablation_name, topology_data in ablation_data.items():
        summary_data[ablation_name] = {
            "max_gaps": [],
            "benchmark_calls": [],
            "convergence_times": [],
            "efficiencies": [],
        }

        for topology, experiments in topology_data.items():
            for exp_data in experiments:
                if exp_data["gaps"]:
                    max_gap = max(exp_data["gaps"])
                    summary_data[ablation_name]["max_gaps"].append(max_gap)

                    # Benchmark calls
                    if (
                        "gap_computations" in exp_data
                        and exp_data["gap_computations"] is not None
                    ):
                        benchmark_calls = exp_data["gap_computations"]
                    else:
                        benchmark_calls = len(exp_data["gaps"])
                    summary_data[ablation_name]["benchmark_calls"].append(
                        benchmark_calls
                    )

                    # Efficiency
                    efficiency = max_gap / benchmark_calls
                    summary_data[ablation_name]["efficiencies"].append(efficiency)

                    # Convergence time
                    if exp_data["time_from_start"] and len(exp_data["gaps"]) > 1:
                        target_gap = 0.9 * max_gap
                        convergence_time = None
                        for i, gap in enumerate(exp_data["gaps"]):
                            if gap >= target_gap:
                                convergence_time = exp_data["time_from_start"][i]
                                break
                        if convergence_time is not None:
                            summary_data[ablation_name]["convergence_times"].append(
                                convergence_time
                            )

    # Plot 1: Max Gap
    ax1 = axes[0, 0]
    ablation_names = list(summary_data.keys())
    # sort the ablation names by the METHOD_ORDER
    ablation_names = [name for name in METHOD_ORDER if name in ablation_names]
    topologies = TOPOLOGIES

    x_pos = np.arange(len(ablation_names))
    width = 0.25

    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            # Get data for this topology from the original ablation_data
            if topology in ablation_data[ablation_name]:
                max_gaps = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if exp_data["gaps"]:
                        max_gaps.append(max(exp_data["gaps"]))

                if max_gaps:
                    means.append(np.mean(max_gaps))
                    stds.append(np.std(max_gaps))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax1.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=topology,
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    ax1.set_ylabel("Maximum Gap", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=45,
        ha="right",
    )
    ax1.legend()

    # Plot 2: Benchmark Calls
    ax2 = axes[0, 1]
    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                benchmark_calls = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if (
                        "gap_computations" in exp_data
                        and exp_data["gap_computations"] is not None
                    ):
                        benchmark_calls.append(exp_data["gap_computations"])
                    elif exp_data["gaps"]:
                        benchmark_calls.append(len(exp_data["gaps"]))

                if benchmark_calls:
                    means.append(np.mean(benchmark_calls))
                    stds.append(np.std(benchmark_calls))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax2.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=topology,
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    ax2.set_ylabel("Benchmark Calls", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax2.legend()

    # Plot 3: Efficiency
    ax3 = axes[0, 2]
    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                efficiencies = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if exp_data["gaps"]:
                        max_gap = max(exp_data["gaps"])
                        if (
                            "gap_computations" in exp_data
                            and exp_data["gap_computations"] is not None
                        ):
                            efficiency = max_gap / exp_data["gap_computations"]
                        else:
                            efficiency = max_gap / len(exp_data["gaps"])
                        efficiencies.append(efficiency)

                if efficiencies:
                    means.append(np.mean(efficiencies))
                    stds.append(np.std(efficiencies))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax3.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=topology,
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    ax3.set_ylabel("Efficiency", fontsize=14, fontweight="bold")
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax3.legend()

    # Plot 4: Convergence Time
    ax4 = axes[1, 0]
    for i, topology in enumerate(topologies):
        means = []
        stds = []

        for ablation_name in ablation_names:
            if topology in ablation_data[ablation_name]:
                convergence_times = []
                for exp_data in ablation_data[ablation_name][topology]:
                    if (
                        exp_data["time_from_start"]
                        and exp_data["gaps"]
                        and len(exp_data["gaps"]) > 1
                    ):
                        max_gap = max(exp_data["gaps"])
                        target_gap = 0.9 * max_gap

                        convergence_time = None
                        for j, gap in enumerate(exp_data["gaps"]):
                            if gap >= target_gap:
                                convergence_time = exp_data["time_from_start"][j]
                                break

                        if convergence_time is not None:
                            convergence_times.append(convergence_time)

                if convergence_times:
                    means.append(np.mean(convergence_times))
                    stds.append(np.std(convergence_times))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax4.bar(
            x_pos + i * width,
            means,
            width,
            # yerr=stds,
            label=topology,
            color=TOPOLOGY_COLORS[topology],
            hatch=HATCHES[topology],
            alpha=0.7,
            capsize=5,
        )

    ax4.set_ylabel("Time (seconds)", fontsize=14, fontweight="bold")
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels(
        [METHOD_NAMES[name] for name in ablation_names],
        rotation=tickz_rotation,
        ha="right",
    )
    ax4.legend()

    # Plot 5: Max Gap vs Benchmark Calls scatter
    ax5 = axes[1, 1]
    for i, (name, data) in enumerate(summary_data.items()):
        ax5.scatter(
            data["benchmark_calls"],
            data["max_gaps"],
            alpha=0.6,
            label=METHOD_NAMES[name],
            color=METHOD_COLORS[name],
            s=50,
        )
    ax5.set_xlabel("Benchmark Calls", fontsize=14, fontweight="bold")
    ax5.set_ylabel("Maximum Gap", fontsize=14, fontweight="bold")
    ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Efficiency vs Convergence Time
    ax6 = axes[1, 2]
    for i, (name, data) in enumerate(summary_data.items()):
        if data["efficiencies"] and data["convergence_times"]:
            # Ensure both arrays have the same length
            min_len = min(len(data["efficiencies"]), len(data["convergence_times"]))
            if min_len > 0:
                ax6.scatter(
                    data["convergence_times"][:min_len],
                    data["efficiencies"][:min_len],
                    alpha=0.6,
                    label=METHOD_NAMES[name],
                    color=METHOD_COLORS[name],
                    s=50,
                )
    ax6.set_xlabel("Convergence Time (seconds)", fontsize=14, fontweight="bold")
    ax6.set_ylabel("Efficiency", fontsize=14, fontweight="bold")
    ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ablation_summary_comparison.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_klee_not_enough(output_dir):
    setup_plot_style()
    data1 = [[0, 10], [1, 11.075799999999958], [10, 13.163499999999999], [20, 16.20270000000005], [30, 18.816999999999894], [40, 20.0], [50, 20.0], [60, 20.0], [70, 19.879300000000057], [80, 19.223600000000033], [90, 19.021299999999997], [100, 19.24090000000001], [110, 20.0], [120, 20.0], [130, 20.0], [131, 20.0]]
    data2 = [[0, 10.0], [10, 0], [20, 0.0], [30, 0.0], [40, 0.0], [50, 0.0], [60, 0.0], [70, 0.0], [80, 0.0], [81, 0.0]]
    data3 = [[0, 10.0], [5, 0.0], [10, 0.0], [15, 0.0], [20, 0.0], [25, 0.0], [30, 0.0], [50, 0.0]]
    fig, ax = plt.subplots(figsize=(6,4))
    x1 = [row[0] for row in data1]
    y1 = [row[1] / (200 * 4) * 100 for row in data1]
    x2 = [row[0] for row in data2]
    y2 = [row[1] / (200 * 4) * 100 for row in data2]
    x3 = [row[0] for row in data3]
    y3 = [row[1] / (200 * 4) * 100 for row in data3]
    ax.plot(x1, y1, color=METHOD_COLORS["Klee + GP"],  linewidth=2.5, label="Path-Aware GP-Based")
    ax.plot(x3, y3, color="darkblue", linestyle="--", linewidth=2.5, label="Path-Agnostic GP-Based")
    ax.plot(x2, y2, color="darkorange", linestyle=":", linewidth=2.5, label="Sample-Based")

    ax.set_ylim(-0.1, 3)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Norm Max Gap (%)", fontweight="bold")
    # add a point at 0, 10 and have an arrow pointing to it saying "Initial Gap"
    ax.scatter(x1[0], y1[0], color="red", s=100, marker="o", zorder=10) #cover what's below it
    # make the arrow bold
    ax.annotate("Klee Seed", (x1[0] + 5, y1[0]), xytext=(x1[0] + 30, y1[0]), arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=0, shrinkB=0), fontweight="bold", fontsize=12)
    # ax.set_title("KleeNotEnough")
    ax.grid(True, alpha=0.3)
    # move the legend outside the plot, top center,no frame
    # ax.legend(bbox_to_anchor=(0.5, 1.3), loc="upper center", ncol=2, frameon=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),   # relative to figure, not axes
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False
    )
    
    # fig.subplots_adjust(top=1)   # manage margins yourself
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "klee_not_enough.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    """Main function to run all ablation analysis plots"""
    print("Parsing ablation study data...")
    output_dir = f"ablation_plots_{Problem}"
    os.makedirs(output_dir, exist_ok=True)
    if Problem != "KleeNotEnough":
        if Problem != "Gradient":
            ablation_data = parse_ablation_data()
        else:
            ablation_data = None
        print("Creating plots...")
        # save the ablation data
        if ablation_data is not None:
            with open(os.path.join(output_dir, "ablation_data.txt"), "w") as f:
                f.write(str(ablation_data))
            plot_max_gap_per_ablation(ablation_data, output_dir)
            # plot_max_gap_vs_time_per_ablation(ablation_data, output_dir)
            plot_max_time_per_ablation(ablation_data, output_dir)
            plot_benchmark_calls_per_ablation(ablation_data, output_dir)
            plot_heuristic_calls_per_ablation(ablation_data, output_dir)
            plot_ablation_by_topology(ablation_data, output_dir)
        plot_timings_by_topology(TIMINGS_TABLE, ablation_data, output_dir)
        # plot_convergence_speed(ablation_data, output_dir)
        # plot_efficiency_analysis(ablation_data, output_dir)
        # plot_topology_comparison(ablation_data, output_dir)
        # plot_summary_comparison(ablation_data, output_dir)
    else:
        plot_klee_not_enough(output_dir)
    print("All ablation analysis plots created successfully!")


if __name__ == "__main__":
    main()
