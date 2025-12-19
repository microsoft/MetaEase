# How to Add a New Problem

This guide explains how to add a completely new problem type to MetaEase (e.g., Vector Bin Packing, Knapsack). Adding a new problem requires implementing several components and registering the problem in the configuration system.

## Overview

When adding a new problem, you need to:
1. Create a problem class that inherits from `Problem`
2. Implement all required abstract methods
3. Register the problem in `config.py`
4. Add problem parsing logic in `common.py`
5. Implement KLEE program generation for symbolic execution

## Step-by-Step Guide

### Step 1: Create the Problem Module

Create a new file `src/problems/programs_<problem_name>.py` (e.g., `programs_vbp.py`, `programs_knapsack.py`).

**Example structure:**
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import numpy as np
from .problem import Problem
from common import LAMBDA_MAX_VALUE

class YourProblem(Problem):
    def __init__(self, problem_config_path):
        super().__init__(problem_config_path)
        # Initialize problem-specific attributes
        # e.g., self.num_items = self.problem_config["num_items"]
        
    # Implement all required methods (see below)
```

### Step 2: Implement Required Abstract Methods

Your problem class must implement all methods from the `Problem` base class:

#### 2.1 `convert_input_dict_to_args(input_dict)`

Converts the input dictionary (used by MetaEase) into the format expected by your problem's computation functions.

**Example (Knapsack):**
```python
def convert_input_dict_to_args(self, input_dict):
    values = [input_dict[f"value_{i}"] for i in range(self.num_items)]
    weights = [input_dict[f"weight_{i}"] for i in range(self.num_items)]
    capacity = self.problem_config["capacity"]
    return {
        "values": values,
        "weights": weights,
        "capacity": capacity,
        "input_dict": input_dict
    }
```

#### 2.2 `compute_optimal_value(args_dict)`

Computes the optimal solution value (e.g., using an ILP solver) and returns:
- `optimal_value`: The optimal objective value
- `all_vars`: Dictionary of all variables in the optimal solution
- `gradient`: Gradient information for optimization (optional but recommended)

**Example (Knapsack):**
```python
def compute_optimal_value(self, args_dict):
    values = args_dict["values"]
    weights = args_dict["weights"]
    capacity = args_dict["capacity"]
    
    # Use OR-Tools or another solver
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample"
    )
    solver.init(values, [weights], [capacity])
    computed_value = solver.solve()
    
    # Extract solution
    packed_items = []
    for i in range(len(values)):
        if solver.best_solution_contains(i):
            packed_items.append(i)
    
    # Construct all_vars
    all_vars = {}
    for i in range(len(values)):
        all_vars[f"value_{i}"] = values[i]
        all_vars[f"weight_{i}"] = weights[i]
        all_vars[f"aux_x_{i}"] = 1 if i in packed_items else 0
    
    return {
        "optimal_value": computed_value,
        "all_vars": all_vars,
        "gradient": {}  # Can be computed from Lagrangian
    }
```

#### 2.3 `compute_heuristic_value(args_dict)`

Computes the heuristic solution and returns:
- `heuristic_value`: The heuristic objective value
- `all_vars`: Dictionary of all variables in the heuristic solution
- `code_path_num`: A unique identifier for the code path taken (critical for path-aware optimization)

**Example (Knapsack):**
```python
def compute_heuristic_value(self, args_dict):
    self.num_compute_heuristic_value_called += 1
    values = args_dict["values"]
    weights = args_dict["weights"]
    capacity = args_dict["capacity"]
    
    # Implement your heuristic (e.g., greedy by value/weight ratio)
    items = list(zip(values, weights, range(len(values))))
    items.sort(key=lambda x: x[0] / x[1], reverse=True)  # Sort by density
    
    packed_items = []
    total_value = 0
    total_weight = 0
    
    for value, weight, idx in items:
        if total_weight + weight <= capacity:
            packed_items.append(idx)
            total_value += value
            total_weight += weight
    
    # Compute code_path_num based on decisions made
    # This should uniquely identify the execution path
    code_path_num = hash(tuple(sorted(packed_items))) % (2**31)
    
    # Construct all_vars
    all_vars = {}
    for i in range(len(values)):
        all_vars[f"value_{i}"] = values[i]
        all_vars[f"weight_{i}"] = weights[i]
        all_vars[f"aux_x_{i}"] = 1 if i in packed_items else 0
    
    return {
        "heuristic_value": total_value,
        "all_vars": all_vars,
        "code_path_num": code_path_num
    }
```

**Important:** The `code_path_num` should uniquely identify the execution path. It's used by MetaEase to ensure path-aware gradient updates.

#### 2.4 `compute_lagrangian_value(args_dict, give_relaxed_gap=False)`

Computes the Lagrangian relaxation value for gradient-based optimization. Returns:
- `lagrange`: The Lagrangian value
- `constraints`: Dictionary of constraint violations

**Example (Knapsack):**
```python
def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
    values = args_dict["values"]
    weights = args_dict["weights"]
    capacity = args_dict["capacity"]
    
    # Solve relaxed knapsack (fractional items allowed)
    # This is typically a linear programming relaxation
    relaxed_sol = optimal_relaxed_knapsack(values, weights, capacity)
    
    return {
        "lagrange": relaxed_sol["relaxed_optimal_value"],
        "constraints": {
            "capacity": sum(weights[i] * relaxed_sol["relaxed_all_vars"][f"aux_x_{i}"] 
                          for i in range(len(values))) - capacity
        }
    }
```

#### 2.5 `compute_lagrangian_gradient(args_dict)`

Computes gradients for the Lagrangian with respect to input variables. This is used for gradient ascent.

**Example:**
```python
def compute_lagrangian_gradient(self, args_dict):
    # Compute gradients using the dual variables from the relaxed solution
    # The gradient for input variable x_i is typically:
    # dL/dx_i = d(optimal_value)/dx_i - lambda * d(constraint)/dx_i
    
    values = args_dict["values"]
    weights = args_dict["weights"]
    capacity = args_dict["capacity"]
    
    relaxed_sol = optimal_relaxed_knapsack(values, weights, capacity)
    lambda_val = relaxed_sol["lambda"]
    
    gradient = {}
    for i in range(len(values)):
        # Gradient with respect to value_i
        gradient[f"value_{i}"] = relaxed_sol["relaxed_all_vars"][f"aux_x_{i}"]
        # Gradient with respect to weight_i
        gradient[f"weight_{i}"] = -lambda_val * relaxed_sol["relaxed_all_vars"][f"aux_x_{i}"]
    
    return gradient
```

#### 2.6 `compute_relaxed_optimal_value(args_dict)`

Computes the relaxed optimal value (typically LP relaxation). Returns:
- `relaxed_optimal_value`: The relaxed optimal value
- `relaxed_all_vars`: Variables from the relaxed solution

#### 2.7 `get_thresholds(relaxed_all_vars)`

Returns bounds for each variable used during optimization. Format: `{variable_name: (min_value, max_value)}`

**Example:**
```python
def get_thresholds(self, relaxed_all_vars):
    thresholds = {}
    for i in range(self.num_items):
        thresholds[f"value_{i}"] = (0, self.problem_config["max_value"])
        thresholds[f"weight_{i}"] = (0, self.problem_config["capacity"])
        thresholds[f"aux_x_{i}"] = (0, 1)
    thresholds["lambda"] = (0, LAMBDA_MAX_VALUE)
    return thresholds
```

#### 2.8 `is_input_feasible(input_dict)`

Checks if an input satisfies problem constraints.

**Example:**
```python
def is_input_feasible(self, input_dict):
    # Check if the input satisfies all constraints
    # For knapsack, we might check if weights are positive, etc.
    for i in range(self.num_items):
        if input_dict[f"weight_{i}"] < 0:
            return False
        if input_dict[f"value_{i}"] < 0:
            return False
    return True
```

#### 2.9 `get_decision_to_input_map(all_vars)`

Maps decision variables (from optimal/heuristic solutions) back to input variables. This is used for understanding which inputs affect which decisions.

**Example:**
```python
def get_decision_to_input_map(self, all_vars):
    # Map each decision variable to the input variables it depends on
    # For knapsack: aux_x_i depends on value_i and weight_i
    mapping = {}
    for i in range(self.num_items):
        mapping[f"aux_x_{i}"] = [f"value_{i}", f"weight_{i}"]
    return mapping
```

#### 2.10 `get_common_header(args_dict)`

Generates the common header for the C program that will be analyzed by KLEE. This header defines data structures, constants, and helper functions.

**Example (Knapsack):**
```python
def get_common_header(self, args_dict=None):
    num_items = self.problem_config["num_items"]
    capacity = self.problem_config["capacity"]
    max_value = self.problem_config["max_value"]
    
    header = f"""
    #include <klee/klee.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdbool.h>
    
    #define NUM_ITEMS {num_items}
    #define CAPACITY {capacity}
    #define MAX_VALUE {max_value}
    
    typedef struct {{
        double value;
        double weight;
    }} Item;
    
    // Helper functions for the heuristic
    int compare_items(const void* a, const void* b) {{
        Item* item_a = (Item*)a;
        Item* item_b = (Item*)b;
        double ratio_a = item_a->value / item_a->weight;
        double ratio_b = item_b->value / item_b->weight;
        if (ratio_a > ratio_b) return -1;
        if (ratio_a < ratio_b) return 1;
        return 0;
    }}
    """
    return header
```

#### 2.11 `generate_heuristic_program(program_type, list_of_input_paths_to_exclude=[], num_klee_inputs=None, path_to_assigned_fixed_points=None)`

Generates the complete C program for KLEE symbolic execution. This is the most complex method and is critical for MetaEase's path-aware optimization.

**Key requirements:**
- Use `klee_make_symbolic()` to mark input variables as symbolic
- Implement the heuristic algorithm in C
- Use `klee_assume()` to exclude already-explored paths
- Return a complete, compilable C program

**Example (Knapsack):**
```python
def generate_heuristic_program(
    self, 
    program_type, 
    list_of_input_paths_to_exclude=[],
    num_klee_inputs=None,
    path_to_assigned_fixed_points=None
):
    num_items = self.problem_config["num_items"]
    capacity = self.problem_config["capacity"]
    max_value = self.problem_config["max_value"]
    
    # Determine which inputs to make symbolic
    if path_to_assigned_fixed_points:
        with open(path_to_assigned_fixed_points, "r") as f:
            file_fixed_points = json.load(f)
        selected_klee_inputs = [
            name for name in self.all_klee_var_names 
            if name not in file_fixed_points
        ]
    else:
        if num_klee_inputs is not None:
            num_klee_inputs = min(num_klee_inputs, self.num_total_klee_inputs)
            selected_klee_inputs = random.sample(
                self.all_klee_var_names, num_klee_inputs
            )
        else:
            selected_klee_inputs = self.all_klee_var_names
    
    # Get fixed point values
    fixed_points = {}
    if path_to_assigned_fixed_points:
        fixed_points = file_fixed_points
    
    # Build the program
    program = self.get_common_header()
    program += """
    int main() {
        Item items[NUM_ITEMS];
        unsigned int totalValue = 0;
        unsigned int totalWeight = 0;
    """
    
    # Add symbolic inputs
    for i in range(num_items):
        var_name_value = f"value_{i}"
        var_name_weight = f"weight_{i}"
        
        if var_name_value in selected_klee_inputs:
            program += f"""
            klee_make_symbolic(&items[{i}].value, sizeof(double), "{var_name_value}");
            klee_assume(items[{i}].value >= 0);
            klee_assume(items[{i}].value <= MAX_VALUE);
            """
        else:
            fixed_val = fixed_points.get(var_name_value, random.uniform(0, max_value))
            program += f"        items[{i}].value = {fixed_val};\n"
        
        if var_name_weight in selected_klee_inputs:
            program += f"""
            klee_make_symbolic(&items[{i}].weight, sizeof(double), "{var_name_weight}");
            klee_assume(items[{i}].weight >= 0);
            klee_assume(items[{i}].weight <= CAPACITY);
            """
        else:
            fixed_val = fixed_points.get(var_name_weight, random.uniform(0, capacity))
            program += f"        items[{i}].weight = {fixed_val};\n"
    
    # Add path exclusion constraints
    for path_constraint in list_of_input_paths_to_exclude:
        program += f"        klee_assume(!({path_constraint}));\n"
    
    # Implement the heuristic
    program += """
        // Sort items by value/weight ratio (greedy heuristic)
        qsort(items, NUM_ITEMS, sizeof(Item), compare_items);
        
        // Pack items greedily
        for (int i = 0; i < NUM_ITEMS; i++) {
            if (totalWeight + items[i].weight <= CAPACITY) {
                totalValue += items[i].value;
                totalWeight += items[i].weight;
            }
        }
        
        return 0;
    }
    """
    
    return program
```

**Important notes:**
- Initialize `self.all_klee_var_names` in `__init__` with all variable names that can be made symbolic
- Initialize `self.num_total_klee_inputs` with the total number of symbolic inputs
- The program must compile and run with KLEE

### Step 3: Register Problem in `config.py`

Add your problem to the configuration system:

#### 3.1 Add to `PROBLEM_CONFIGS`

```python
PROBLEM_CONFIGS = {
    # ... existing problems ...
    "your_problem": {
        "heuristic_name": "your_heuristic",
        "num_items": 20,  # Your problem-specific parameters
        # ... other parameters ...
    },
}
```

#### 3.2 Add to `PARAMETERS`

```python
PARAMETERS = {
    # ... existing problems ...
    "your_problem": {
        "min_value": 0.0,
        "max_value": 100.0,
        # ... other parameters ...
    },
}
```

#### 3.3 Add to `get_problem_instance()`

```python
def get_problem_instance(problem_type, config_path):
    if problem_type == "TE":
        return TEProblem(config_path)
    # ... existing problems ...
    elif problem_type == "your_problem":
        return YourProblem(config_path)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
```

#### 3.4 Add parameter handling (if needed)

If your problem needs special parameter handling (like TE with topologies), add a handler function:

```python
def handle_YourProblem_parameters(problem_description):
    # Extract and process problem-specific parameters
    params = PARAMETERS["your_problem"].copy()
    # ... custom logic ...
    return params
```

Then call it in `get_parameters()`:

```python
def get_parameters(problem_description):
    problem_type = problem_description["problem_type"]
    params = COMMON_PARAMETERS.copy()
    params.update(PROBLEM_CONFIGS[problem_type])
    params.update(problem_description)
    
    if problem_type == "TE":
        te_params = handle_TE_parameters(problem_description)
        params.update(te_params)
    elif problem_type == "your_problem":
        your_params = handle_YourProblem_parameters(problem_description)
        params.update(your_params)
    else:
        params.update(PARAMETERS[problem_type])
    
    # Set minimize_is_better flag if needed
    minimize_is_better = problem_type in ["vbp", "your_problem"]
    params["minimize_is_better"] = minimize_is_better
    
    return params
```

### Step 4: Add Problem Parsing in `common.py`

Add parsing logic in `get_problem_description()` to handle command-line problem specifications:

```python
def get_problem_description(args) -> dict:
    # ... existing problem parsing ...
    
    elif args.problem.startswith("your_problem"):
        # Parse problem specification from args.problem
        # e.g., "your_problem_20_100" -> num_items=20, capacity=100
        parts = args.problem.split("_")
        num_items = int(parts[1])
        capacity = int(parts[2])
        
        problem_description = {
            "problem_type": "your_problem",
            "heuristic_name": "your_heuristic",
            "num_items": num_items,
            "capacity": capacity,
            "max_value": 100,
            # MetaEase optimization parameters
            "max_num_scalable_klee_inputs": 200,
            "num_samples": 50,
            "disable_gaussian_process": False,
            "block_length": 0.1,
            "gradient_ascent_rate": 1.0,
            "use_gaps_in_filtering": True,
            "remove_zero_gap_inputs": True,
            "num_iterations": 1000,
            # ... other parameters ...
        }
    
    return problem_description
```

### Step 5: Import in Required Files

#### 5.1 Import in `config.py`

```python
from problems.programs_your_problem import *
```

#### 5.2 Import in `run_klee.py`

```python
from problems.programs_your_problem import *
```

### Step 6: Test Your Implementation

1. **Test basic functionality:**
   ```bash
   cd src
   python paper.py --problem your_problem_20_100 --method Random --base-save-dir ../test_logs
   ```

2. **Test with MetaEase:**
   ```bash
   python paper.py --problem your_problem_20_100 --method MetaEase --base-save-dir ../test_logs
   ```

3. **Verify outputs:**
   - Check that results are saved correctly
   - Verify gaps are computed
   - Ensure KLEE programs compile and run

## Common Pitfalls

1. **Code path numbering:** Ensure `code_path_num` uniquely identifies execution paths. Use hash functions or bit masks based on heuristic decisions.

2. **KLEE program compilation:** Test that your generated C program compiles with `clang` and runs with KLEE before integrating.

3. **Gradient computation:** Ensure gradients are computed correctly for your problem's objective and constraints.

4. **Variable naming:** Use consistent naming conventions for input variables (e.g., `value_{i}`, `weight_{i}`) that match between Python and C code.

5. **Feasibility checking:** Implement proper constraint checking in `is_input_feasible()`.

## Example: Complete Knapsack Problem

See `src/problems/programs_knapsack.py` for a complete reference implementation of a new problem type.

## Next Steps

After adding a new problem, you may want to:
- Add multiple heuristics for the same problem (see `HOW_TO_ADD_A_NEW_HEURISTIC.md`)
- Tune MetaEase parameters for your problem
- Add problem-specific optimizations

