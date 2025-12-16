# MetaEase: Heuristic Analysis from Source Code via Symbolic-Guided Optimization
MetaEase is a framework for automatically finding worst-case inputs of heuristics by maximizing the gap between heuristic and optimal solutions. It combines symbolic execution (KLEE) with gradient-based optimization to efficiently explore the input space.

This codebase supports multiple problems:
- Traffic Engineering (TE): DemandPinning, PoP, Sort-based Algorithm in the paper, DOTE
- Vector Bin Packing (VBP): First Fit and First Fit Decreasing
- Knapsack: Sort-based Greedy
- Maximum Weighted Matching (MWM): Greedy
- IP--Optical Network Optimization: Arrow Heuristic

# Code Structure

```
MetaEase_MSR/
├── src/                          # Main source code directory
│   ├── paper.py                  # Main entry point for running experiments
│   ├── metaease.py               # Core MetaEase algorithm implementation
│   ├── config.py                 # Problem configuration and parameter management
│   ├── common.py                 # Shared utilities and helper functions
│   ├── logger.py                 # Async logging functionality
│   │
│   ├── problems/                 # Problem-specific implementations
│   │   ├── problem.py            # Base Problem class (abstract interface)
│   │   ├── programs_TE.py        # Traffic Engineering problem implementations
│   │   ├── programs_vbp.py       # Vector Bin Packing problem implementations
│   │   ├── programs_knapsack.py  # Knapsack problem implementations
│   │   ├── programs_max_weighted_matching.py  # MWM problem implementations
│   │   ├── programs_arrow.py     # Arrow heuristic (IP-Optical) implementations
│   │   └── DOTE/                 # DOTE (Deep Optimization for Traffic Engineering) problem implementation
│   │
│   ├── gradient_ascent.py        # Gradient ascent optimization (MetaEase)
│   ├── hill_climbing.py          # Hill climbing baseline
│   ├── simulated_annealing.py    # Simulated annealing baseline
│   ├── random_sampling.py        # Random sampling baseline
│   ├── gap_sample_based.py       # Sample-based gradient baseline
│   │
│   ├── run_klee.py               # KLEE symbolic execution integration
│   ├── run_utils.py              # Execution and compilation utilities
│   ├── opt_utils.py              # Optimization helper functions
│   ├── clustering_utils.py       # Variable clustering/partitioning utilities
│   ├── analysis_utils.py         # Analysis and result processing utilities
│   ├── utils.py                  # General utility functions
│   └── ablation_DemandPinning.py # Ablation study script for DemandPinning
│
├── scripts/                      # Analysis and visualization scripts
│   ├── generate_experiment_file.py  # Aggregates experiment results
│   ├── plot_methods.py           # Generates comparison plots across methods
│   ├── plot_ablation.py          # Generates ablation study plots
│   ├── plot_comparison.py        # Comparison visualization utilities
│   ├── plot_common.py            # Shared plotting utilities
│   └── log_parser.py             # Log parsing utilities
│
├── topologies/                   # Network topology data files
│
├── paper.sh                      # Script to run all paper experiments from section 5 of the paper
├── paper_subset.sh               # Script to run subset of experiments from section 5 of the paper
├── environment.yml               # Conda environment configuration
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```


# Setting up the environment
The following instructions are for Linux and macOS. If you are using Windows, you can use the Windows Subsystem for Linux (WSL) to run the code.

## Install Conda (if not already installed)
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

## Install brew and clang (if not already installed)
```bash
NONINTERACTIVE=1 /bin/bash -c “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)”
test -d ~/.linuxbrew && eval “$(~/.linuxbrew/bin/brew shellenv)”
test -d /home/linuxbrew/.linuxbrew && eval “$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)”
echo “eval \“\$($(brew --prefix)/bin/brew shellenv)\“” >> ~/.bashrc

# Install clang
# Update package lists
sudo apt update

# Install clang and build essentials
sudo apt install -y clang llvm build-essential cmake
```

## Install Klee
```bash
brew install klee
```

This may take a while.

## Set up the environment
```bash
conda env create -f environment.yml
conda activate metaease
pip install -r requirements.txt
echo 'conda activate metaease' >> ~/.bashrc
```

# Instructions to run the code

To run the code, you can use the following command:
```bash
cd src # This is important to run the code from the src directory
python paper.py --problem <problem> --method <method> --base-save-dir <base-save-dir>
```

For example, to run the code for the DemandPinning problem on the Abilene topology with the MetaEase method, you can use the following command:
```bash
cd src
python paper.py --problem TE_DemandPinning_abilene --method MetaEase --base-save-dir ../logs_final_TE
```

List of all problems:
- TE_<DemandPinning|PoP|LLM>_<abilene|b4-teavar|swan|Cogentco|Uninet2010>
- TE_DOTE_AbileneDOTE
- vbp_<FF|FFD>_<num_items>_<num_dimensions>
- knapsack_<num_items>
- mwm_<abilene|b4-teavar|swan|Cogentco|Uninet2010>
- arrow_<IBM|B4|simple>

List of all methods:
- MetaEase
- Random
- SimulatedAnnealing
- HillClimbing
- GradientSampleBased

Note: The random seed used to produce the plots were time, so you may not get the exact results for the baselines.

## Plotting Results
To visualize and compare results across methods, use the plotting scripts:
```bash
cd scripts
python generate_experiment_file.py --logs-dir <LOGS_DIR> --output-name <EXPERIMENT_NAME>
python plot_methods.py --experiment_file <EXPERIMENT_NAME>.txt --output_dir <OUTPUT_DIR>
```

This generates plots showing gap progression over time for each method, allowing you to compare MetaEase against the baseline methods.

# Quick Example: Running MetaEase and the baselines
To test the code, you can run MetaEase and the baselines on Traffic Engineering (TE) problem on Abilene topology for Demand Pinning heuristic:

```bash
cd src
python paper.py --problem TE_DemandPinning_abilene --method MetaEase --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method HillClimbing --baseline-max-time 400 --baseline-repeat 1 --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method SimulatedAnnealing --baseline-max-time 400 --baseline-repeat 1 --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method Random --baseline-max-time 400 --baseline-repeat 1 --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method GradientSampleBased --baseline-max-time 400 --baseline-repeat 1 --base-save-dir ../logs_final_TE

# To aggregate the results for plotting, you can run the following command:
cd ../scripts
python generate_experiment_file.py --logs-dir ../logs_final_TE --output-name TE_DemandPinning
python plot_methods.py --experiment_file TE_DemandPinning.txt --output_dir ../plots/TE_DemandPinning
```

# Output Files Structure
How does the output look like?

When you run an experiment, the output is saved in a directory under `--base-save-dir` (default: `../logs_final`). The directory name follows the pattern:
```
<TIMESTAMP>__<METHOD>__<PROBLEM_TYPE>__<PROBLEM_CONFIG>
```

For example: `20251212_171601__MetaEase__TE__HN-DemandPinning_T-abilene`

## Top-Level Files

- **`experiment.log`**: Complete log of the experiment execution, including all print statements, debug messages, and progress updates.

- **`<PROBLEM_TYPE>_config.json`**: Configuration file containing all problem-specific parameters (e.g., topology, heuristic type, optimization hyperparameters).

- **`final_metaease_results.json`**: Summary of the final results, containing:
  - `max_global_gap`: The maximum performance gap found across all iterations
  - `best_global_sample`: The input values that produced the maximum gap
  - `best_global_optimal_all_vars`: Variables from the optimal solution for the best input
  - `best_global_heuristic_all_vars`: Variables from the heuristic solution for the best input
  - `total_execution_time`: Total time taken for the experiment

- **`best_global_sample_kleeIteration<N>.json`**: Best input sample found after each KLEE iteration (cluster). Contains the input variable assignments that led to the best gap for that iteration.

- **`fixed_variables_kleeIteration<N>.json`**: Variables that were fixed (not optimized) during each KLEE iteration, along with their assigned values.

- **`klee_inputs_<KLEE_IDX>_<NON_ZERO_ROUND>.json`**: Input seeds generated by KLEE symbolic execution for each cluster and non-zero round. Each file contains a dictionary where keys are test case names and values are dictionaries of variable assignments.

## Cluster Directories

For each cluster (partition of variables), a directory is created:
- **`cluster_<KLEE_IDX>_<NON_ZERO_ROUND>/`**: Contains results for a specific cluster and non-zero round.

Within each cluster directory:
- **`best_result.json`**: Best result found for this cluster, containing:
  - `max_gap`: Maximum gap found in this cluster
  - `best_sample`: Input values that produced this gap
  - `optimal_all_vars`: Optimal solution variables
  - `heuristic_all_vars`: Heuristic solution variables

- **`klee_input_<SAMPLE_IDX>/`**: Directory for each KLEE-generated seed input, containing:
  - **`gap_list.json`**: List of gaps found during gradient ascent iterations. Each entry is `[iteration_number, gap_value]`.
  - **`best_sample_list.json`**: List of best samples found during gradient ascent. Each entry is `[iteration_number, sample_dictionary]`, where the sample dictionary contains all input variable assignments.

# Output Files for Baseline Methods

## Random Sampling
- **`random_sampling_results.json`**: Contains a list of results, where each entry includes:
  - `sample`: Input variable assignments
  - `gap`: Performance gap for this sample
  - `heuristic_value`: Heuristic solution value
  - `optimal_value`: Optimal solution value
  - `time`: Timestamp when this sample was evaluated

## Simulated Annealing, Hill Climbing, GradientSampleBased
Similar structure to Random, with method-specific result files containing the optimization history and best results found.

# Understanding the Results

- **Gap**: The performance gap is defined as `optimal_value - heuristic_value`. A larger gap indicates a worse-case input for the heuristic.

- **Code Path**: Each heuristic execution follows a specific code path (determined by conditional branches). MetaEase tracks code paths to ensure path-aware gradient updates.

- **KLEE Iterations**: MetaEase partitions variables into clusters and optimizes them sequentially. Each cluster optimization is called a "KLEE iteration" because KLEE is run to generate seeds for that cluster.

## Running Subset of Paper Experiments

If you only want to run a subset of the experiments (useful for quick testing or when you have limited time/resources), you can use the `paper_subset.sh` script. This script runs a smaller selection of experiments that still demonstrate MetaEase's capabilities across different problem types.

```bash
bash paper_subset.sh
```

### What Gets Executed in the Subset

The subset script runs the following experiments:

- **TE_DemandPinning** (on swan, abilene only)
  - Generates data for **Figure 7** and **Figure 8** (partial)
- **TE_DOTE** (on AbileneDOTE)
  - Generates data for **Table 4**
- **TE_PoP** (on swan only)
  - Generates data for **Figure 9** and **Table 5** (partial)
- **TE_LLM** (on swan, abilene, b4-teavar)
  - Generates data for **Table 6**
- **mwm** (on swan, abilene, b4-teavar)
  - Generates data for **Figure 12** (partial)

### Differences from Full Script

- **Fewer topologies**: Only runs on smaller/medium-sized topologies (excludes Cogentco and Uninet2010)
- **Fewer problem types**: Excludes Vector Bin Packing, Knapsack, and Arrow heuristic experiments
- **Sequential baseline execution**: Baseline methods run one at a time (not in parallel), which is slower but uses less system resources
- **Faster execution**: Typically completes in a fraction of the time compared to the full script

### When to Use

Use `paper_subset.sh` when you want to:
- Quickly verify the codebase works correctly
- Test on a limited hardware setup
- Generate a subset of the paper's figures/tables
- Understand the workflow without running the full experiment suite

For complete paper reproduction, use `paper.sh` instead.

# Running All Paper Experiments

The `paper.sh` script automates running all experiments needed to reproduce the paper's results. It systematically runs MetaEase and baseline methods across all problem types and configurations, then generates the plots and tables used in the paper.

```bash
bash paper.sh
```

**Note**: This script will run all experiments, which may take a significant amount of time (potentially days depending on your hardware). The script runs baseline methods in parallel to speed up execution.

### What Gets Executed

The script runs experiments for the following problems and generates data for specific paper figures/tables:

- **TE_DemandPinning** (on swan, abilene, b4-teavar, Cogentco, Uninet2010)
  - Generates data for **Figure 7**, **Figure 8**, and **Table 3**
- **TE_DOTE** (on AbileneDOTE)
  - Generates data for **Table 4**
- **TE_PoP** (on swan, abilene, b4-teavar, Cogentco, Uninet2010)
  - Generates data for **Figure 9** and **Table 5**
- **TE_LLM** (on swan, abilene, b4-teavar)
  - Generates data for **Table 6**
- **vbp_FFD** (on settings: 10_1, 10_2, 15_1, 15_2, 20_1, 20_2)
  - Generates data for **Table 7**
- **knapsack** (on 20, 30, 40, 50 items)
  - Generates data for **Figure 10**
- **arrow** (on IBM, B4)
  - Generates data for **Figure 11**
- **mwm** (on swan, abilene, b4-teavar, Cogentco, Uninet2010)
  - Generates data for **Figure 12**

### Differences from Subset Script

- **All topologies**: Includes large topologies (Cogentco, Uninet2010) in addition to smaller ones
- **All problem types**: Includes Vector Bin Packing, Knapsack, and Arrow heuristic experiments
- **Parallel baseline execution**: Baseline methods run in parallel (4 processes) to reduce total time
- **Complete results**: Generates all figures and tables from the paper

Results are saved in `../logs_final_<PROBLEM_TYPE>/` directories, and plots are automatically generated in `../plots/<PROBLEM_TYPE>/`.

# Ablation Studies

This artifact includes a dedicated script to reproduce ablations studies from Section 6 of the paper.

- **Script location**: `src/ablation_DemandPinning.py`
- **Purpose**: runs a sequence of ablation experiments by varying:
  - seed generation (KLEE vs Random vs LLM)
  - gradient configuration (GP surrogate vs direct / sample-based gradients)
  - number of KLEE inputs (projected dimension \(K\))

### Running the ablation experiments

From the repository root:

```bash
cd src # This is important to run the code from the src directory
python ablation_DemandPinning.py
```

The script will:

- sequentially run all configured ablations in `ablation_DemandPinning.py`
- write results into directories under `../ablation_DemandPinning/`
- incrementally update a timing summary JSON: `ablation_DemandPinning_timing_table.json`

If you only want to run a **single** ablation, you can comment out the other ablation blocks and keep only the one whose `ablation_name` you are interested in (e.g., `"Klee_with_GP"`, `"Random_with_GP"`, `"LLM_with_GP"`, `"varying_klee_inputs_with_no_gradient_ascent"`, etc.).

### Plotting ablation results

The plotting helper for ablations is `scripts/plot_ablation.py`. It can generate timing and gap-comparison plots similar to those in the paper.

1. **Edit the paths**  
   - `plot_ablation.py` currently contains **hard-coded absolute paths** in the `ablation_dirs` dictionaries (e.g., under the `Problem == "Seed"`, `"Gradient"`, or `"Parameter_K"` branches).  
   - Change these directory strings so they point to your local `ablation_DemandPinning` output directory, for example:
     - replace paths like `/home/ubuntu/MetaEase/MetaOptimize/ablation_DemandPinning/...`
     - with your local path (or whatever path you used).

2. **Choose which ablation family to plot**  
   At the top of `scripts/plot_ablation.py`, set:

   ```python
   Problem = "Seed"         # for seed-generation ablations (LLM vs Random vs KLEE)
   # or
   Problem = "Gradient"     # for gradient-style ablations (GP vs direct vs sample-based)
   # or
   Problem = "Parameter_K"  # for varying number of KLEE inputs K
   # or
   Problem = "BlockLenght"  # for PoP block-length ablations (if desired)
   ```

3. **Run the plotting script**

   From the repository root:

   ```bash
   cd scripts
   python plot_ablation.py
   ```

   This will parse the ablation logs in the directories you configured and produce PDF plots (e.g., `ablation_by_topology.pdf`, `timings_by_topology.pdf`, `max_gap_per_ablation.pdf`) in a folder named:

   ```text
   ablation_plots_<Problem>/
   ```

   For example, if `Problem = "Seed"`, the plots will be under `ablation_plots_Seed/`.

# Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{karimi2026metaease,
  title={MetaEase: Heuristic Analysis from Source Code via Symbolic-Guided Optimization},
  author={Karimi, Pantea and Kakarla, Siva Kesava Reddy and Namyar, Pooria and Segarra, Santiago and Beckett, Ryan and Alizadeh, Mohammad and Arzani, Behnaz},
  journal = {USENIX Symposium on Networked Systems Design and Implementation 2026},
  year={2026}
}
```

Also, you can checkout my other heuristic analysis tool:
XPlain: Towards Safer Heuristics With XPlain

```bibtex
@inproceedings{10.1145/3696348.3696884,
author = {Karimi, Pantea and Pirelli, Solal and Kakarla, Siva Kesava Reddy and Beckett, Ryan and Segarra, Santiago and Li, Beibin and Namyar, Pooria and Arzani, Behnaz},
title = {Towards Safer Heuristics With XPlain},
year = {2024},
isbn = {9798400712722},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696348.3696884},
doi = {10.1145/3696348.3696884},
abstract = {Many problems that cloud operators solve are computationally expensive, and operators often use heuristic algorithms (that are faster and scale better than optimal) to solve them more efficiently. Heuristic analyzers enable operators to find when and by how much their heuristics underperform. However, these tools do not provide enough detail for operators to mitigate the heuristic's impact in practice: they only discover a single input instance that causes the heuristic to underperform (and not the full set) and they do not explain why. We propose XPlain, a tool that extends these analyzers and helps operators understand when and why their heuristics underperform. We present promising initial results that show such an extension is viable.},
booktitle = {Proceedings of the 23rd ACM Workshop on Hot Topics in Networks},
pages = {68–76},
numpages = {9},
keywords = {Domain-Specific Language, Explainable Analysis, Heuristic Analysis},
location = {Irvine, CA, USA},
series = {HotNets '24}
}