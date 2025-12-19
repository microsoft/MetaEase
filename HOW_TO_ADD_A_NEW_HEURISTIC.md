# How to Add a New Heuristic to an Existing Problem

This guide explains how to add a new heuristic algorithm to an existing problem type (e.g., adding a new heuristic to Traffic Engineering problems like DemandPinning, PoP, DOTE, or LLM).

## Overview

When adding a new heuristic to an existing problem, you need to:
1. Implement the heuristic function in the problem module
2. Add heuristic selection logic in `compute_heuristic_value()`
3. Add problem parsing logic in `common.py` (if needed)
4. Optionally implement KLEE program generation for the new heuristic

## Step-by-Step Guide

### Step 1: Implement the Heuristic Function

Add your heuristic implementation to the problem module (e.g., `src/problems/programs_TE.py`).

**Example: Adding a new "MyNewHeuristic" to TE problems**

```python
def my_new_heuristic_TE(num_nodes, edges, demands, possible_demands=None, given_all_paths=None):
    """
    Your new heuristic implementation.
    
    Args:
        num_nodes: Number of nodes in the network
        edges: List of edges with capacities
        demands: Dictionary of demands to route
        possible_demands: Optional precomputed possible demands
        given_all_paths: Optional precomputed paths
    
    Returns:
        dict: {
            "heuristic_value": objective value,
            "code_path_num": unique code path identifier,
            "all_vars": dictionary of solution variables,
            # ... other heuristic-specific outputs
        }
    """
    # Implement your heuristic algorithm here
    # ...
    
    # Compute code_path_num based on decisions made
    # This should uniquely identify the execution path
    code_path_num = compute_code_path_number(decisions)
    
    return {
        "heuristic_value": objective_value,
        "code_path_num": code_path_num,
        "all_vars": all_vars,
        # ... other outputs
    }
```

**Important considerations:**

1. **Code Path Numbering:** The `code_path_num` must uniquely identify the execution path. It's used by MetaEase for path-aware optimization. Common approaches:
   - Hash of decision sequence
   - Bit mask of branch decisions
   - Enumeration of path choices

   ```python
   def compute_code_path_number(decisions):
       # Example: hash the sequence of decisions
       return hash(tuple(sorted(decisions))) % (2**31)
   ```

2. **Return Format:** Always return a dictionary with at minimum:
   - `heuristic_value`: The objective value
   - `code_path_num`: Unique path identifier
   - `all_vars`: Dictionary of all variables in the solution

3. **Consistency:** Ensure the heuristic uses the same input/output format as other heuristics for the same problem.

### Step 2: Add Heuristic Selection Logic

Modify the `compute_heuristic_value()` method in your problem class to include your new heuristic.

**Example (TE Problem):**

```python
def compute_heuristic_value(self, args_dict):
    self.num_compute_heuristic_value_called += 1
    
    num_nodes = args_dict["num_nodes"]
    edges = args_dict["edges"]
    demands = args_dict["demands"]
    heuristic_name = self.problem_config["heuristic_name"]
    
    heuristic_start = time.time()
    
    if heuristic_name == "DemandPinning":
        heuristic_sol = demand_pinning_TE(
            num_nodes, edges, demands,
            self.problem_config["small_flow_cutoff"],
            possible_demands=self.possible_demands,
            given_all_paths=self.all_paths,
        )
    elif heuristic_name == "PoP":
        heuristic_sol = pop_TE_wrapper(
            num_nodes, edges, demands,
            self.partition_lists,
            possible_demands=self.possible_demands,
            given_all_paths=self.all_paths
        )
    elif heuristic_name == "LLM":
        heuristic_sol = LLM_TE(
            num_nodes, edges, demands,
            possible_demands=self.possible_demands,
            given_all_paths=self.all_paths,
        )
    elif heuristic_name == "DOTE":
        heuristic_sol = DOTE_wrapper(
            topology_name=self.problem_config["topology"],
            demands=demands
        )
    elif heuristic_name == "MyNewHeuristic":  # Add your new heuristic here
        heuristic_sol = my_new_heuristic_TE(
            num_nodes, edges, demands,
            possible_demands=self.possible_demands,
            given_all_paths=self.all_paths,
            # Add any heuristic-specific parameters from problem_config
            param1=self.problem_config.get("my_param1", default_value),
        )
    else:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")
    
    heuristic_end = time.time()
    if ENABLE_PRINT:
        print(f"{heuristic_name} execution took {heuristic_end - heuristic_start:.4f} seconds")
    
    return {
        "code_path_num": heuristic_sol["code_path_num"],
        "heuristic_value": heuristic_sol["heuristic_value"],
        "all_vars": heuristic_sol["all_vars"],
        # Include any other required fields
    }
```

### Step 3: Add Problem Parsing Logic

Add parsing logic in `src/common.py` to handle command-line specifications for your new heuristic.

**Example: Adding "MyNewHeuristic" to TE problems**

```python
def get_problem_description(args) -> dict:
    if args.problem.startswith("TE"):
        heuristic_name = args.problem.split("_")[-2]
        topology = args.problem.split("_")[-1]
        
        # ... existing TE setup code ...
        
        problem_description = {
            "problem_type": "TE",
            "heuristic_name": heuristic_name,
            "topology": topology,
            # ... other common parameters ...
        }
        
        # Add heuristic-specific configurations
        if heuristic_name == "DemandPinning":
            problem_description["disable_klee"] = False
            problem_description["num_random_seed_samples"] = 0
            problem_description["num_samples"] = 20
            problem_description["num_iterations"] = 1000
            # ... other DemandPinning-specific parameters ...
        
        elif heuristic_name == "MyNewHeuristic":  # Add your heuristic configuration
            problem_description["disable_klee"] = False  # Set based on your needs
            problem_description["num_random_seed_samples"] = 10
            problem_description["num_samples"] = 30
            problem_description["num_iterations"] = 2000
            problem_description["disable_gradient_ascent"] = False
            problem_description["disable_gaussian_process"] = False
            problem_description["max_num_scalable_klee_inputs"] = 16
            # Add any heuristic-specific parameters
            problem_description["my_param1"] = some_value
        
        # ... other heuristics ...
    
    return problem_description
```

**Common MetaEase parameters to configure:**

- `disable_klee`: Set to `True` if KLEE symbolic execution is not needed
- `num_random_seed_samples`: Number of random seed samples when KLEE is disabled
- `num_samples`: Number of samples for Gaussian Process surrogate
- `num_iterations`: Number of gradient ascent iterations per KLEE point
- `disable_gradient_ascent`: Set to `True` to disable gradient-based optimization
- `disable_gaussian_process`: Set to `True` to use direct gradients instead of GP surrogate
- `max_num_scalable_klee_inputs`: Maximum number of symbolic variables for KLEE
- `gradient_ascent_rate`: Learning rate for gradient ascent
- `block_length`: Size of the block around current best sample

### Step 4: Implement KLEE Program Generation (Optional but Recommended)

If you want MetaEase to use KLEE symbolic execution for your heuristic, implement KLEE program generation in the `generate_heuristic_program()` method.

**Example: Adding KLEE support for "MyNewHeuristic" in TE**

```python
def generate_heuristic_program(
    self,
    program_type,
    list_of_input_paths_to_exclude=[],
    num_klee_inputs=None,
    path_to_assigned_fixed_points=None,
):
    num_nodes = self.problem_config["num_nodes"]
    max_flow = int(self.problem_config["max_flow"])
    heuristic_name = self.problem_config["heuristic_name"]
    
    # ... existing code to select symbolic inputs ...
    
    # Generate program based on heuristic
    if heuristic_name == "DemandPinning":
        program = self._get_demand_pinning_program(...)
    elif heuristic_name == "PoP":
        program = self._get_pop_program(...)
    elif heuristic_name == "MyNewHeuristic":  # Add your heuristic's KLEE program
        program = self._get_my_new_heuristic_program(
            selected_klee_inputs,
            fixed_points,
            list_of_input_paths_to_exclude
        )
    else:
        raise ValueError(f"KLEE program generation not implemented for {heuristic_name}")
    
    return program

def _get_my_new_heuristic_program(self, selected_klee_inputs, fixed_points, excluded_paths):
    """
    Generate C program for MyNewHeuristic that KLEE can analyze.
    
    The program should:
    1. Use klee_make_symbolic() for input variables
    2. Implement the heuristic algorithm in C
    3. Use klee_assume() to exclude already-explored paths
    """
    num_nodes = self.problem_config["num_nodes"]
    max_flow = int(self.problem_config["max_flow"])
    
    program = self.get_common_header({"num_nodes": num_nodes})
    program += f"""
    int main() {{
        // Declare variables
        double demands[NUM_DEMANDS];
        // ... other variables ...
        
        // Make inputs symbolic
    """
    
    # Add symbolic inputs
    for demand_key in self.all_klee_var_names:
        if demand_key in selected_klee_inputs:
            program += f"""
            klee_make_symbolic(&demands[{demand_idx}], sizeof(double), "{demand_key}");
            klee_assume(demands[{demand_idx}] >= 0);
            klee_assume(demands[{demand_idx}] <= {max_flow});
            """
        else:
            fixed_val = fixed_points.get(demand_key, 0.0)
            program += f"        demands[{demand_idx}] = {fixed_val};\n"
    
    # Add path exclusion constraints
    for path_constraint in excluded_paths:
        program += f"        klee_assume(!({path_constraint}));\n"
    
    # Implement your heuristic in C
    program += """
        // Implement MyNewHeuristic algorithm here
        // ... heuristic logic ...
        
        return 0;
    }
    """
    
    return program
```

**Important notes for KLEE programs:**

1. **Symbolic variables:** Use `klee_make_symbolic(&var, sizeof(type), "name")` to mark inputs as symbolic
2. **Constraints:** Use `klee_assume(condition)` to add constraints and exclude paths
3. **Path exclusion:** Use `list_of_input_paths_to_exclude` to avoid re-exploring known paths
4. **Compilation:** Ensure the program compiles with `clang` and runs with KLEE

### Step 5: Test Your Implementation

1. **Test basic heuristic execution:**
   ```bash
   cd src
   python paper.py --problem TE_MyNewHeuristic_abilene --method Random --base-save-dir ../test_logs
   ```

2. **Test with MetaEase:**
   ```bash
   python paper.py --problem TE_MyNewHeuristic_abilene --method MetaEase --base-save-dir ../test_logs
   ```

3. **Verify outputs:**
   - Check that heuristic values are computed correctly
   - Verify code paths are tracked properly
   - Ensure gaps are computed (optimal - heuristic)

## Examples from the Codebase

### Example 1: Adding DOTE to TE

See how DOTE was added to TE problems:
- Heuristic function: `DOTE_wrapper()` in `src/problems/programs_TE.py`
- Selection logic: Added in `TEProblem.compute_heuristic_value()`
- Configuration: Added in `common.py` with DOTE-specific parameters

### Example 2: Adding First Fit Decreasing to VBP

See how FFD was added to Vector Bin Packing:
- Heuristic function: `first_fit_decreasing()` in `src/problems/programs_vbp.py`
- Selection logic: Added in `VBPProblem.compute_heuristic_value()`
- KLEE program: Implemented in `VBPProblem.generate_heuristic_program()`

## Common Pitfalls

1. **Code path numbering:** Ensure `code_path_num` is computed consistently and uniquely identifies execution paths. Test with multiple inputs to verify uniqueness.

2. **Parameter consistency:** Ensure heuristic-specific parameters are properly passed from `problem_config` to your heuristic function.

3. **KLEE program correctness:** If implementing KLEE support, test the generated C program separately before integrating:
   ```bash
   clang -emit-llvm -c your_program.c -o test.bc
   klee test.bc
   ```

4. **Return format:** Always match the return format expected by the problem class. Check existing heuristics for reference.

5. **Performance:** Consider the computational cost of your heuristic. MetaEase will call it many times during optimization.

## Heuristic-Specific Optimizations

Some heuristics may benefit from special MetaEase configurations:

- **Heuristics with single code path:** Set `keep_redundant_code_paths = True` and `max_num_scalable_klee_inputs` to a large value
- **Heuristics that don't benefit from gradients:** Set `disable_gradient_ascent = True`
- **Heuristics with many variables:** Use `randomized_gradient_ascent = True` with `num_vars_in_randomized_gradient_ascent` set appropriately

## Next Steps

After adding a new heuristic:
- Compare its performance with existing heuristics using the plotting scripts
- Tune MetaEase parameters for optimal worst-case input discovery
- Consider adding ablation studies if the heuristic has interesting properties

