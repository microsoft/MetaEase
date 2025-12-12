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
- TE_<DemandPinning|PoP|LLM|DOTE>_<abilene|b4-teavar|swan|Cogentco|Uninet2010>
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

## Traffice Engineering (TE) plots
```bash
python paper.py --problem TE_DemandPinning_abilene --method MetaEase --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method HillClimbing --baseline-max-time 600 --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method SimulatedAnnealing --baseline-max-time 600 --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method Random --baseline-max-time 600 --base-save-dir ../logs_final_TE
python paper.py --problem TE_DemandPinning_abilene --method GradientSampleBased --baseline-max-time 600 --base-save-dir ../logs_final_TE
```