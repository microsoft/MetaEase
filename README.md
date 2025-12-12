# Introduction 
MetaEase is a framework for automatically finding worst-case inputs of heuristics by maximizing the gap between heuristic and optimal solutions. It combines symbolic execution (KLEE) with gradient-based optimization to efficiently explore the input space.

This codebase supports multiple problems:
- Traffic Engineering (TE): DemandPinning, PoP, Sort-based Algorithm in the paper, DOTE
- Vector Bin Packing (VBP): First Fit and First Fit Decreasing
- Knapsack: Sort-based Greedy
- Maximum Weighted Matching (MWM): Greedy
- IP--Optical Network Optimization: Arrow Heuristic


# Setting up the environment
The following instructions are for Linux.

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
sudo apt install clang
```

## Install Klee
```bash
brew install klee
```

This may take a while.

## Set up the environment
```bash
cd MetaOptimize
conda env create -f environment.yml
conda activate metaease
pip install -r requirements.txt
echo 'conda activate metaease' >> ~/.bashrc
```

# Instructions to run the code

To run the code, you can use the following command:
```bash
python src/paper.py --problem <problem> --method <method>
```

For example, to run the code for the DemandPinning problem on the Abilene topology with the MetaEase method, you can use the following command:
```bash
python src/paper.py --problem TE_DemandPinning_abilene --method MetaEase
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

# Running Paper Experiments
To geterate the plots for the paper, you can run use the commands in `paper.sh`.