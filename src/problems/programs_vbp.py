import sys
import os
# Add parent directory to path for common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from .problem import Problem
from ortools.linear_solver import pywraplp
from typing import List, Dict
import itertools
from itertools import product
import random
from common import LAMBDA_MAX_VALUE

SCALE_FACTOR = 10000  # Reduced scale factor for better numerical stability
SOLVER_TOLERANCE = 1e-4  # Increased tolerance for better convergence
ENABLE_PRINT = False


class Item:
    def __init__(self, item_sizes: List[float]):
        self.item_sizes = item_sizes
        self.num_dimensions = len(item_sizes)

    def __str__(self):
        return f"Item([{','.join(str(size) for size in self.item_sizes)}])"

    def get_dimension(self, dimension_index: int) -> float:
        return self.item_sizes[dimension_index]


class Bin:
    def __init__(self, num_dimensions: int, bin_size: float):
        self.num_dimensions = num_dimensions
        self.bin_size = bin_size
        self.remaining_capacities = [bin_size for _ in range(num_dimensions)]
        self.items = []

    def __str__(self):
        return f"Bin(remaining_capacities={self.remaining_capacities}, items={','.join(str(item) for item in self.items)})"

    def can_place_item(self, item: Item) -> bool:
        for i in range(self.num_dimensions):
            if self.remaining_capacities[i] < item.get_dimension(i):
                return False
        return True

    def get_remaining_capacity(self, dimension_index: int) -> float:
        return self.remaining_capacities[dimension_index]

    def place_item(self, item: Item) -> bool:
        if not self.can_place_item(item):
            return False
        for i in range(self.num_dimensions):
            self.remaining_capacities[i] -= item.get_dimension(i)
        self.items.append(item)
        return True


def optimal_vbp(items: List[Item], bin_size: float, relaxed: bool = False):
    # items is a list of lists, each inner contains the dimensions of an item
    # bin_size is size of each dimension of the bin
    if ENABLE_PRINT:
        print(f"\nSolving VBP with bin_size: {bin_size}")
        print(f"Items: {[str(item) for item in items]}")

    scaled_bin_size = int(bin_size * SCALE_FACTOR) * (1 + SOLVER_TOLERANCE)
    scaled_items = [
        Item(
            [
                int(item.get_dimension(i) * SCALE_FACTOR)
                for i in range(item.num_dimensions)
            ]
        )
        for item in items
    ]
    # TODO: how much is the runtime of SCIP a problem for the runtime for VBP? Gurobi is better optimized for mixed integer problems and runs faster.
    # It may be good to have a default of gurobi for mixed integer problems instead of ORTools.

    # TODO: for a cleaner implementation, it may be good to have a separate class for the optimal implementation and modularize it more. Right now this is a giant function blurb.
    if ENABLE_PRINT:
        print(f"Scaled bin size: {scaled_bin_size}")
        print(f"Scaled items: {[str(item) for item in scaled_items]}")

    if relaxed:
        solver = pywraplp.Solver.CreateSolver("GLOP")
    else:
        solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Failed to create solver")
        return None

    # create variables
    x = {}
    y = {}
    constraints = {}
    all_vars = {}
    num_items = len(scaled_items)
    num_dimensions = scaled_items[0].num_dimensions
    max_num_bins = num_items

    # Add a small tolerance to bin capacity for numerical stability
    capacity_tolerance = int(scaled_bin_size * SOLVER_TOLERANCE)

    for j in range(max_num_bins):
        # y[j] = 1 if bin j is used
        if relaxed:
            y[j] = solver.NumVar(0, 1, f"y_{j}")
        else:
            y[j] = solver.IntVar(0, 1, f"y_{j}")
        for i in range(num_items):
            # x[i, j] = 1 if item i is placed in bin j
            if relaxed:
                x[i, j] = solver.NumVar(0, 1, f"x_{i}_{j}")
            else:
                x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # x[i, j] <= y[j]
    for i in range(num_items):
        for j in range(max_num_bins):
            constraint = solver.Add(x[i, j] <= y[j])
            constraints[f"lambda_bin_used_item_{i}_bin_{j}"] = constraint

    # I want to start filling from the first bin
    for bin1 in range(max_num_bins):
        for bin2 in range(bin1 + 1, max_num_bins):
            constraint = solver.Add(y[bin1] >= y[bin2])

    # constraint for each item with tolerance
    for i in range(num_items):
        constraint = solver.Add(solver.Sum([x[i, j] for j in range(max_num_bins)]) == 1)
        constraints[f"lambda_placement_item_{i}"] = constraint

    # capacity constraint with tolerance
    for j in range(max_num_bins):
        for d in range(num_dimensions):
            constraint = solver.Add(
                solver.Sum(
                    [
                        x[i, j] * scaled_items[i].get_dimension(d)
                        for i in range(num_items)
                    ]
                )
                <= scaled_bin_size * y[j] + capacity_tolerance
            )
            constraints[f"lambda_capacity_bin_{j}_dim_{d}"] = constraint

    # minimize the number of bins used (sum of y[j])
    total_bins_used = solver.Sum(y.values())
    solver.Minimize(total_bins_used)

    status = solver.Solve()
    if ENABLE_PRINT:
        print(f"Solver status: {status}")
    # TODO: Separate solution parsing into a different function? Again seems like this one function is doing way too much, break it up into smaller pieces for better understanding and maybe wrap in a class to encapsulate everything that is related to the optimal encoding.
    if status == pywraplp.Solver.OPTIMAL:
        if relaxed:
            # Store variable values
            for j in range(max_num_bins):
                all_vars[f"aux_bin_used_{j}"] = y[j].solution_value()
                for i in range(num_items):
                    all_vars[f"aux_placement_item_{i}_bin_{j}"] = x[
                        i, j
                    ].solution_value()
            num_bins_used = sum(
                all_vars[f"aux_bin_used_{j}"] for j in range(max_num_bins)
            )

            if ENABLE_PRINT:
                # Print detailed solution for debugging
                print("\nDetailed solution:")
                for j in range(max_num_bins):
                    bin_usage = sum(
                        x[i, j].solution_value() * scaled_items[i].get_dimension(0)
                        for i in range(num_items)
                    )
                    if bin_usage > 0:
                        print(
                            f"Bin {j}: Usage = {bin_usage/SCALE_FACTOR:.4f}/{bin_size}"
                        )
                        for i in range(num_items):
                            if x[i, j].solution_value() > 0:
                                print(
                                    f"  Item {i}: Size = {items[i].get_dimension(0):.4f}, Fraction = {x[i, j].solution_value():.4f}"
                                )

            # Store actual dual values for relaxed case
            for key, constraint in constraints.items():
                if "capacity" in key:
                    all_vars[key] = constraint.dual_value() * SCALE_FACTOR
                else:
                    all_vars[key] = constraint.dual_value()
        else:
            for j in range(max_num_bins):
                bin_used = 0
                for i in range(num_items):
                    all_vars[f"aux_placement_item_{i}_bin_{j}"] = (
                        1 if x[i, j].solution_value() > 0.5 else 0
                    )
                    bin_used += all_vars[f"aux_placement_item_{i}_bin_{j}"]
                all_vars[f"aux_bin_used_{j}"] = 1 if bin_used > 0 else 0
            num_bins_used = sum(
                all_vars[f"aux_bin_used_{j}"] for j in range(max_num_bins)
            )

            if ENABLE_PRINT:
                # Print detailed solution for debugging
                print("\nDetailed solution:")
                for j in range(max_num_bins):
                    bin_usage = sum(
                        all_vars[f"aux_placement_item_{i}_bin_{j}"]
                        * items[i].get_dimension(0)
                        for i in range(num_items)
                    )
                    if bin_usage > 0:
                        print(f"Bin {j}: Usage = {bin_usage:.4f}/{bin_size}")
                        for i in range(num_items):
                            if all_vars[f"aux_placement_item_{i}_bin_{j}"] > 0:
                                print(
                                    f"  Item {i}: Size = {items[i].get_dimension(0):.4f}"
                                )

            # Estimate dual values for integer case using constraint violations/slack
            # For bin usage constraints: x[i,j] <= y[j]
            for i in range(num_items):
                for j in range(max_num_bins):
                    placement = all_vars[f"aux_placement_item_{i}_bin_{j}"]
                    bin_used = all_vars[f"aux_bin_used_{j}"]
                    slack = bin_used - placement  # Should be >= 0
                    all_vars[f"lambda_bin_used_item_{i}_bin_{j}"] = (
                        -slack
                    )  # Negative of slack

            # For placement constraints: sum(x[i,j]) == 1
            for i in range(num_items):
                sum_placements = sum(
                    all_vars[f"aux_placement_item_{i}_bin_{j}"]
                    for j in range(max_num_bins)
                )
                slack = abs(1 - sum_placements)  # Should be 0
                all_vars[f"lambda_placement_item_{i}"] = -slack

            # For capacity constraints
            for j in range(max_num_bins):
                for d in range(num_dimensions):
                    used_capacity = sum(
                        all_vars[f"aux_placement_item_{i}_bin_{j}"]
                        * scaled_items[i].get_dimension(d)
                        for i in range(num_items)
                    )
                    slack = (
                        scaled_bin_size * all_vars[f"aux_bin_used_{j}"] - used_capacity
                    ) / SCALE_FACTOR
                    all_vars[f"lambda_capacity_bin_{j}_dim_{d}"] = -slack

        # Store item dimensions
        for i in range(num_items):
            for d in range(num_dimensions):
                all_vars[f"demand_item_{i}_dim_{d}"] = items[i].get_dimension(d)

        return_key_name = "relaxed_all_vars" if relaxed else "all_vars"
        return {
            "num_bins_used": num_bins_used,
            return_key_name: all_vars,
        }
    else:
        print(f"Failed to find optimal solution. Status: {status}")
    return None


def first_fit(items: List[Item], bin_size: float):
    # put the items in the first bin that has enough space, if no bin has enough space, create a new bin
    num_items = len(items)
    num_dimensions = items[0].num_dimensions
    bins = [Bin(num_dimensions, bin_size)]
    code_path_num = ""
    all_vars = {}

    # Initialize all_vars with demand values
    for i in range(num_items):
        for d in range(num_dimensions):
            all_vars[f"demand_item_{i}_dim_{d}"] = items[i].get_dimension(d)

    # Initialize placement and bin usage variables to 0
    for i in range(num_items):
        for j in range(num_items):  # Maximum possible bins is num_items
            all_vars[f"aux_placement_item_{i}_bin_{j}"] = 0
    for j in range(num_items):
        all_vars[f"aux_bin_used_{j}"] = 0

    for i in range(num_items):
        for bin_index, bin in enumerate(bins):
            if bin.can_place_item(items[i]):
                bin.place_item(items[i])
                # code_path_num = f"{code_path_num}{bin_index}"
                all_vars[f"aux_placement_item_{i}_bin_{bin_index}"] = 1
                all_vars[f"aux_bin_used_{bin_index}"] = 1
                break
        else:
            new_bin = Bin(num_dimensions, bin_size)
            new_bin.place_item(items[i])
            bins.append(new_bin)
            bin_index = len(bins) - 1
            # code_path_num = f"{code_path_num}{bin_index}"
            all_vars[f"aux_placement_item_{i}_bin_{bin_index}"] = 1
            all_vars[f"aux_bin_used_{bin_index}"] = 1

    return {
        "num_bins_used": len(bins),
        "code_path_num": "",
        "bins": [str(bin) for bin in bins],
        "all_vars": all_vars
    }


def first_fit_decreasing(items: List[Item], bin_size: float):
    # first sort the items in decreasing order
    items.sort(key=lambda x: x.get_dimension(0), reverse=True)
    return first_fit(items, bin_size)

# TODO: this needs to have in the comment a discussion of how we are doing this given that the problem is discrete and the lagrangian is not differentiable.
# This reminds me that when we are writing how a user adds new problems we also need to give guidance on how to handle discrete objectives/lagrangians.
# TODO: another comment that also applies to programs_TE, is that your only computing the lagrangian for the optimal here right? or are you also computing the gradient for the gaussian process?
# If the former, change your function name to reflect that.
# TODO: since you are hardcoding gradients it would be good to put the equation for each problem of the gradient as a comment.
def get_vbp_lagrangian_gradient(
    items: List[Item], bin_size: float, input_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute the gradient of the Lagrangian at a specific point.

    Args:
        items: List of items with their dimensions
        bin_size: Size of each bin
        input_values: Dictionary of variable names and their values at the point of evaluation

    Returns:
        Dictionary of variable names and their corresponding gradient values
    """
    gradient_values = {}
    num_items = len(items)
    num_dimensions = len(items[0].item_sizes)
    num_bins = num_items  # Maximum number of bins needed is number of items

    # Process each variable in input_values
    for var in input_values.keys():
        if "aux_placement_item" in var and "lambda" not in var:
            # Extract item and bin IDs from variable name
            item_id = int(var.split("_")[3])
            bin_id = int(var.split("_")[5])

            # Calculate gradient for placement variables
            gradient_values[var] = (
                input_values[f"lambda_placement_item_{item_id}"]
                - input_values[f"lambda_bin_used_item_{item_id}_bin_{bin_id}"]
            )

            # Add contribution from capacity constraints
            for dimension in range(num_dimensions):
                gradient_values[var] -= (
                    input_values[f"lambda_capacity_bin_{bin_id}_dim_{dimension}"]
                    * input_values[f"demand_item_{item_id}_dim_{dimension}"]
                )

        elif "demand_item" in var and "lambda" not in var:
            # Extract item and dimension IDs
            item_id = int(var.split("_")[2])
            dimension = int(var.split("_")[4])

            # Calculate gradient for demand variables
            gradient_values[var] = 0
            for bin_id in range(num_bins):
                gradient_values[var] -= (
                    input_values[f"aux_placement_item_{item_id}_bin_{bin_id}"]
                    * input_values[f"lambda_capacity_bin_{bin_id}_dim_{dimension}"]
                )

        elif "bin_used" in var and "lambda" not in var:
            # Extract bin ID
            bin_id = int(var.split("_")[3])

            # Calculate gradient for bin usage variables
            gradient_values[var] = 1

            # Subtract contribution from capacity constraints
            for dimension in range(num_dimensions):
                gradient_values[var] += (
                    input_values[f"lambda_capacity_bin_{bin_id}_dim_{dimension}"]
                    * bin_size
                )

            # Subtract contribution from bin usage constraints
            for item_id in range(num_items):
                gradient_values[var] += input_values[
                    f"lambda_bin_used_item_{item_id}_bin_{bin_id}"
                ]

        elif "lambda_capacity" in var:
            # Extract bin and dimension IDs
            bin_id = int(var.split("_")[3])
            dimension = int(var.split("_")[5])

            # Calculate gradient for capacity constraint multipliers
            gradient_values[var] = 0
            for item_id in range(num_items):
                gradient_values[var] -= (
                    input_values[f"aux_placement_item_{item_id}_bin_{bin_id}"]
                    * input_values[f"demand_item_{item_id}_dim_{dimension}"]
                )
            gradient_values[var] += input_values[f"aux_bin_used_{bin_id}"] * bin_size

        elif "lambda_placement" in var:
            # Extract item ID
            item_id = int(var.split("_")[3])

            # Calculate gradient for placement constraint multipliers
            gradient_values[var] = -1
            for bin_id in range(num_bins):
                gradient_values[var] += input_values[
                    f"aux_placement_item_{item_id}_bin_{bin_id}"
                ]

        elif "lambda_bin_used_item" in var:
            # Extract item and bin IDs
            item_id = int(var.split("_")[4])
            bin_id = int(var.split("_")[6])

            # Calculate gradient for bin usage constraint multipliers
            gradient_values[var] = (
                -input_values[f"aux_placement_item_{item_id}_bin_{bin_id}"]
                + input_values[f"aux_bin_used_{bin_id}"]
            )
        else:
            print(f"Warning: Unrecognized variable name: {var}")
            gradient_values[var] = 0

    # Negate the gradient values for lambda variables
    for key in gradient_values:
        if "lambda" in key:
            gradient_values[key] = -gradient_values[key]
    return gradient_values

# TODO: can use a comment to describe when it is used. It doesn't look to me like you use this to compute the gradient of the lagrangian since you are hardcoding that, so would be good for a user to know why you need it.
# TODO: I wonder if as part of cleanup you want to build either a clean interface to the optimal's lagrangian and its gradient. Specifically so that in the MeteEase code you can always call the same function. There are two ways to do this, one is through call backs (which i am assuming is what your currently doing). The problem
# with callbacks is that it doesn't force the user to specify the input, the other option is something similar to C# interfaces where you make sure the user has to define certain functions for registering a problem. The latter may be a cleaner way.
def get_vbp_lagrangian(items, bin_size, input_values, give_relaxed_gap=False):
    lagrange = 0
    constraints = {}
    num_items = len(items)
    num_dimensions = len(items[0].item_sizes)
    num_bins = num_items  # Maximum number of bins needed is number of items

    # Sum up the bin usage variables
    for bin_id in range(num_bins):
        lagrange += input_values[f"aux_bin_used_{bin_id}"]

    if give_relaxed_gap:
        return {"lagrange": lagrange, "constraints": constraints}

    # Add contribution from placement constraints
    for item_id in range(num_items):
        constraint_value = -1
        for bin_id in range(num_bins):
            constraint_value += input_values[
                f"aux_placement_item_{item_id}_bin_{bin_id}"
            ]
        lagrange += input_values[f"lambda_placement_item_{item_id}"] * constraint_value
        constraints[f"lambda_placement_item_{item_id}"] = constraint_value

    # Add contribution from capacity constraints
    for bin_id in range(num_bins):
        for dimension in range(num_dimensions):
            constraint_value = bin_size * input_values[f"aux_bin_used_{bin_id}"]
            for item_id in range(num_items):
                constraint_value -= (
                    input_values[f"demand_item_{item_id}_dim_{dimension}"]
                    * input_values[f"aux_placement_item_{item_id}_bin_{bin_id}"]
                )
            lagrange += (
                constraint_value
                * input_values[f"lambda_capacity_bin_{bin_id}_dim_{dimension}"]
            )
            constraints[f"lambda_capacity_bin_{bin_id}_dim_{dimension}"] = (
                constraint_value
            )

    # Add contribution from bin usage constraints
    for item_id in range(num_items):
        for bin_id in range(num_bins):
            constraint_value = (
                -input_values[f"aux_placement_item_{item_id}_bin_{bin_id}"]
                + input_values[f"aux_bin_used_{bin_id}"]
            )
            lagrange += (
                input_values[f"lambda_bin_used_item_{item_id}_bin_{bin_id}"]
                * constraint_value
            )
            constraints[f"lambda_bin_used_item_{item_id}_bin_{bin_id}"] = (
                constraint_value
            )

    return {"lagrange": lagrange, "constraints": constraints}


def get_bin_packing_common_header(num_items, bin_capacity=100, num_dimensions=1):
    common_header = f"""
    #include <stdio.h>
    #include <stdlib.h>
    #include <klee/klee.h>
    #define NUM_ITEMS {str(num_items)}
    #define NUM_DIMENSIONS {str(num_dimensions)}
    #define BIN_CAPACITY {str(bin_capacity)}
    """

    common_header += """

    typedef struct {
        unsigned long dimensions[NUM_DIMENSIONS];
    } Item;

    typedef struct {
        unsigned long remaining_capacity[NUM_DIMENSIONS];
    } Bin;

    // Function to check if an item fits in a bin
    int fits_in_bin(Item item, Bin bin) {
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            if (item.dimensions[i] > bin.remaining_capacity[i]) {
                return 0;
            }
        }
        return 1;
    }

    // Function to place an item in a bin
    void place_in_bin(Item item, Bin *bin) {
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            bin->remaining_capacity[i] -= item.dimensions[i];
        }
    }
    """

    return common_header

# TODO: the "all_klee_var_names" requirements for registering a heuristic are a bit concerning to me, as in MetaEase is not as "touch-free as we claim it to be." In our meeting lets go through all the requirements when someone wants to register a heuristic/problem and see if there are ways we can address these a bit better.
class VBPProblem(Problem):
    def __init__(self, problem_config_path):
        super().__init__(problem_config_path)
        self.num_items = int(self.problem_config["num_items"])
        self.bin_size = self.problem_config["bin_size"]
        self.bin_size = self.bin_size
        self.num_dimensions = int(self.problem_config["num_dimensions"])
        self.num_total_klee_inputs = self.num_items * self.num_dimensions
        self.all_klee_var_names = []
        for i in range(self.num_items):
            for j in range(self.num_dimensions):
                self.all_klee_var_names.append(f"demand_item_{i}_dim_{j}")
    # TODO: what are thresholds used for and what are they in the bin packing context?
    def get_thresholds(self, relaxed_all_vars):
        thresholds = {key: (0, self.bin_size) for key in self.all_klee_var_names}
        for key in relaxed_all_vars.keys():
            if "aux" in key:
                thresholds[key] = (0, 1)
            elif "lambda" in key:
                thresholds[key] = (0, LAMBDA_MAX_VALUE)
        return thresholds

    def is_input_feasible(self, input_dict):
        # check if the sum of the items is less than the bin size
        for bin_id in range(self.num_items):
            for dimension in range(self.num_dimensions):
                sum_demand = 0
                for item_id in range(self.num_items):
                    sum_demand += (
                        input_dict[f"demand_item_{item_id}_dim_{dimension}"]
                        * input_dict[f"aux_placement_item_{item_id}_bin_{bin_id}"]
                    )
                if self.bin_size < sum_demand:
                    return False
        return True
    # TODO: expand this function's documentation to describe when and where it should be used.
    def get_item_sizes_for_bins(self, num_items, num_bins, random_seed=None):
        """Generate item sizes that will perfectly fill num_bins bins.
        Each bin will be filled exactly to capacity by splitting items."""
        assert num_items >= num_bins
        if random_seed is not None:
            random.seed(random_seed)

        bin_size = float(self.bin_size)  # Make sure we're working with float
        print(
            f"Generating {num_items} items to fill {num_bins} bins of size {bin_size}"
        )

        # First, generate num_bins items that exactly fill bins
        perfect_items = []
        unsplit_indices = set()  # Track which indices haven't been split yet

        # Start with full bins
        for i in range(num_bins):
            perfect_items.append(bin_size)
            unsplit_indices.add(i)  # Add index to unsplit set

        # Then split some bins randomly to create the remaining items
        items_needed = num_items - num_bins
        while items_needed > 0:
            # Prioritize splitting unsplit items if available
            if unsplit_indices:
                split_idx = random.choice(list(unsplit_indices))
                unsplit_indices.remove(split_idx)
            else:
                # If all items have been split at least once, choose randomly
                split_idx = random.randint(0, len(perfect_items) - 1)

            item_to_split = perfect_items[split_idx]

            # Generate a random split point between 30-70% of the item size
            # This ensures we don't get too small or too large pieces
            min_split = item_to_split * 0.1
            max_split = item_to_split * 0.9
            split_point = round(random.uniform(min_split, max_split), 3)

            perfect_items[split_idx] = split_point
            perfect_items.append(item_to_split - split_point)
            items_needed -= 1

        print(f"Generated items: {perfect_items}")
        return perfect_items

    def get_permutations_of_item_sizes(
        self, num_items, num_bins, random_seed=None, max_permutations=40000
    ):
        """Get a random subset of permutations of the item sizes that perfectly fill num_bins bins."""
        if random_seed is not None:
            random.seed(random_seed)

        item_sizes = self.get_item_sizes_for_bins(num_items, num_bins, random_seed)

        # Calculate total number of possible permutations
        import math

        total_perms = math.factorial(len(item_sizes))
        print(f"Total possible permutations: {total_perms}")

        if total_perms <= max_permutations:
            # If total permutations is manageable, return all of them
            return list(itertools.permutations(item_sizes))

        # Otherwise, generate random permutations
        seen = set()
        result = []
        attempts = 0
        max_attempts = max_permutations * 10  # Allow some extra attempts for duplicates

        while len(result) < max_permutations and attempts < max_attempts:
            # Generate a random permutation by shuffling
            perm = tuple(random.sample(item_sizes, len(item_sizes)))
            if perm not in seen:
                seen.add(perm)
                result.append(perm)
            attempts += 1

            if attempts % 1000 == 0:
                print(
                    f"Generated {len(result)} unique permutations out of {attempts} attempts"
                )

        print(f"Generated {len(result)} unique permutations")
        return result
    # TODO: this function needs documentation as to when and where it should be used.
    def get_optimal_value_based_on_combination(self, combination):
        bin_ids = set()
        for key, value in combination.items():
            if "aux_placement_item" in key and int(value) == 1:
                bin_ids.add(int(key.split("_")[5]))
        return len(bin_ids)
    # TODO: this function needs documentation as to when and where it should be used.
    def get_all_binary_combinations(self):
        all_combinations = []
        num_items = self.num_items
        num_bins = self.num_items
        # Each element in `product` is a tuple of bin indices (one for each item)
        for bin_assignment in product(range(num_bins), repeat=num_items):
            combination = {}
            for item_idx in range(num_items):
                for bin_idx in range(num_bins):
                    key = f"aux_placement_item_{item_idx}_bin_{bin_idx}"
                    combination[key] = 1 if bin_assignment[item_idx] == bin_idx else 0
            all_combinations.append(combination)

        return all_combinations

    def convert_input_dict_to_args(self, input_dict):
        items = []
        for i in range(self.num_items):
            item_sizes = []
            for j in range(self.num_dimensions):
                item_sizes.append(input_dict[f"demand_item_{i}_dim_{j}"])
            items.append(Item(item_sizes))
        return {
            "items": items,
            "input_dict": input_dict,
            "num_items": self.num_items,
            "bin_size": self.bin_size,
            "heuristic_name": self.problem_config["heuristic_name"],
            "num_dimensions": self.num_dimensions,
        }

    def compute_optimal_value(self, args_dict):
        self.num_compute_optimal_value_called += 1
        items = args_dict["items"]
        bin_size = self.bin_size
        optimal_sol = optimal_vbp(items, bin_size)
        gradient = get_vbp_lagrangian_gradient(items, bin_size, optimal_sol["all_vars"])
        return {
            "optimal_value": optimal_sol["num_bins_used"],
            "gradient": gradient,
            "all_vars": optimal_sol["all_vars"],
        }

    def compute_heuristic_value(self, args_dict):
        self.num_compute_heuristic_value_called += 1
        items = args_dict["items"]
        bin_size = self.bin_size
        heuristic_name = self.problem_config["heuristic_name"]
        if heuristic_name == "FF":
            heuristic_sol = first_fit(items, bin_size)
        elif heuristic_name == "FFD":
            heuristic_sol = first_fit_decreasing(items, bin_size)
        else:
            raise ValueError("Invalid heuristic name. Use 'FF' or 'FFD'.")
        return {
            "code_path_num": heuristic_sol["code_path_num"],
            "heuristic_value": heuristic_sol["num_bins_used"],
            "all_vars": heuristic_sol["all_vars"]
        }

    def compute_lagrangian_gradient(self, args_dict):
        items = args_dict["items"]
        bin_size = self.bin_size
        input_dict = args_dict["input_dict"]
        gradient = get_vbp_lagrangian_gradient(items, bin_size, input_dict)
        return gradient

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        items = args_dict["items"]
        bin_size = self.bin_size
        input_dict = args_dict["input_dict"]
        value = get_vbp_lagrangian(items, bin_size, input_dict, give_relaxed_gap)
        return value

    def compute_relaxed_optimal_value(self, args_dict):
        items = args_dict["items"]
        bin_size = self.bin_size
        input_dict = args_dict["input_dict"]
        value = optimal_vbp(items, bin_size, relaxed=True)
        return {
            "relaxed_optimal_value": value["num_bins_used"] if value is not None else None,
            "relaxed_all_vars": value["relaxed_all_vars"] if value is not None else None,
        }

    def get_common_header(self, args_dict):
        num_items = args_dict["num_items"]
        num_dimensions = args_dict["num_dimensions"]
        bin_size = args_dict["bin_size"]
        return get_bin_packing_common_header(num_items, bin_size, num_dimensions)

    def generate_heuristic_program(
        self,
        program_type,
        list_of_input_paths_to_exclude=[],
        num_klee_inputs=None,
        path_to_assigned_fixed_points=None,
        num_optimal_bins=None,
    ):
        num_items = self.num_items
        file_fixed_points = None
        if path_to_assigned_fixed_points:
            with open(path_to_assigned_fixed_points, "r") as f:
                file_fixed_points = json.load(f)
            selected_klee_inputs = [
                name
                for name in self.all_klee_var_names
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
            print(
                f"Selected klee inputs: {selected_klee_inputs} from {self.num_total_klee_inputs}"
            )

        fixed_points = {}
        program = self.get_common_header(
            {
                "num_items": num_items,
                "num_dimensions": self.num_dimensions,
                "bin_size": self.bin_size,
            }
        )

        program += """
        int main() {
            // Initialize items with symbolic values
            Item items[NUM_ITEMS];
            int num_bins = 0;

            // Initialize all bins with full capacity
            Bin bins[NUM_ITEMS];
            for (int i = 0; i < NUM_ITEMS; i++) {
                for (int j = 0; j < NUM_DIMENSIONS; j++) {
                    bins[i].remaining_capacity[j] = BIN_CAPACITY;
                }
            }
        """
        for i in range(num_items):
            program += f"""
            Item item{i};
            """
            for d in range(self.num_dimensions):
                demand_key = f"demand_item_{i}_dim_{d}"
                if demand_key in selected_klee_inputs:
                    program += f"""
                    int {demand_key};
                    klee_make_symbolic(&{demand_key}, sizeof({demand_key}), "{demand_key}");
                    klee_assume({demand_key} >= {int(self.bin_size * 0.1)} & {demand_key} <= {int(self.bin_size * 0.9)});  // Limit max item size to 80% of bin
                    item{i}.dimensions[{d}] = {demand_key};
                    """
                else:
                    if file_fixed_points is not None:
                        value = file_fixed_points[demand_key]
                    else:
                        value = random.randint(int(self.bin_size * 0.1), int(self.bin_size * 0.9))  # Limit fixed items too
                    fixed_points[demand_key] = value
                    program += f"""
                    int {demand_key} = {value};
                    item{i}.dimensions[{d}] = {demand_key};
                    """

        if num_optimal_bins is not None:
            # Add constraints to ensure feasible packing
            assumption_string = ""
            for d in range(self.num_dimensions):
                assumption_string += "("
                for i in range(num_items):
                    assumption_string += f"demand_item_{i}_dim_{d} + "
                assumption_string = assumption_string[:-3]
                assumption_string += f" == {int(num_optimal_bins * self.bin_size)}) & "
            assumption_string = assumption_string[:-3]
            program += f"""
            klee_assume({assumption_string});
            """

        for input_path in list_of_input_paths_to_exclude:
            with open(input_path, "r") as f:
                test_cases = json.load(f)
            for _, test in test_cases.items():
                excluding_string = ""
                for key, value in test.items():
                    translated_key = key.strip("'").strip()
                    if key in selected_klee_inputs:
                        excluding_string += f"{translated_key} != {value} | "
                excluding_string = excluding_string[:-3]

                program += f"""
                klee_assume({excluding_string});
                """

        for i in range(num_items):
            program += f"""
            items[{i}] = item{i};
            """

        if self.problem_config["heuristic_name"] == "FF":
            program += """
            // First-fit algorithm
            for (int i = 0; i < NUM_ITEMS; i++) {
                int placed = 0;
                for (int j = 0; j < num_bins; j++) {
                    if (fits_in_bin(items[i], bins[j])) {
                        place_in_bin(items[i], &bins[j]);
                        placed = 1;
                        break;
                    }
                }
                if (!placed) {
                    place_in_bin(items[i], &bins[num_bins]);
                    num_bins++;
                }
            }
            """
        elif self.problem_config["heuristic_name"] == "BF":
            program += """
            // Best-fit algorithm
            for (int i = 0; i < NUM_ITEMS; i++) {
                int best_bin = -1;
                unsigned long min_remaining = BIN_CAPACITY + 1;
                for (int j = 0; j < num_bins; j++) {
                    if (fits_in_bin(items[i], bins[j])) {
                        unsigned long remaining = bins[j].remaining_capacity[0] - items[i].dimensions[0];
                        if (remaining < min_remaining) {
                            min_remaining = remaining;
                            best_bin = j;
                        }
                    }
                }
                if (best_bin == -1) {
                    place_in_bin(items[i], &bins[num_bins]);
                    num_bins++;
                } else {
                    place_in_bin(items[i], &bins[best_bin]);
                }
            }
            """
        elif self.problem_config["heuristic_name"] == "FFD":
            program += """
            // First-Fit Decreasing algorithm
            // First sort items by their maximum dimension in decreasing order
            for (int i = 0; i < NUM_ITEMS - 1; i++) {
                for (int j = 0; j < NUM_ITEMS - i - 1; j++) {
                    // Find max dimension for both items
                    unsigned long max_dim_j = items[j].dimensions[0];
                    unsigned long max_dim_j1 = items[j + 1].dimensions[0];
                    for (int d = 1; d < NUM_DIMENSIONS; d++) {
                        if (items[j].dimensions[d] > max_dim_j) {
                            max_dim_j = items[j].dimensions[d];
                        }
                        if (items[j + 1].dimensions[d] > max_dim_j1) {
                            max_dim_j1 = items[j + 1].dimensions[d];
                        }
                    }
                    // Swap if next item is larger
                    if (max_dim_j1 > max_dim_j) {
                        Item temp = items[j];
                        items[j] = items[j + 1];
                        items[j + 1] = temp;
                    }
                }
            }

            // Now apply First-Fit on sorted items
            for (int i = 0; i < NUM_ITEMS; i++) {
                int placed = 0;
                for (int j = 0; j < num_bins; j++) {
                    if (fits_in_bin(items[i], bins[j])) {
                        place_in_bin(items[i], &bins[j]);
                        placed = 1;
                        break;
                    }
                }
                if (!placed) {
                    place_in_bin(items[i], &bins[num_bins]);
                    num_bins++;
                }
            }
            """
        else:
            raise ValueError("Invalid algorithm type. Use FF or FFD.")

        if num_optimal_bins is not None:
            program +=f"klee_assume(num_bins - {int(num_optimal_bins)} >= 1);"

        program += f"""
            return 0;
        }}
        """
        # print(program)
        return {"program": program, "fixed_points": fixed_points}
  # TODO: delete commented codes.
    def get_decision_to_input_map(self, all_vars):
        # Create a mapping of decision variables to their corresponding input variables
        decision_to_input_map = {}
        # Map aux_placement_item variables to their corresponding demand variables
        for key in all_vars:
            if key.startswith('aux_placement_item_'):
                # For aux_placement_item_X_bin_Y, the corresponding inputs are all demand_item_X_dim_Z
                parts = key.split('_')
                item_id = parts[3]
                for dim in range(self.num_dimensions):
                    input_var = f"demand_item_{item_id}_dim_{dim}"
                    if key not in decision_to_input_map:
                        decision_to_input_map[key] = []
                    decision_to_input_map[key].append(input_var)
            # elif key.startswith('aux_bin_used_'):
            #     # For aux_bin_used_X, the corresponding inputs are all demands that could be placed in that bin
            #     bin_id = key.split('_')[3]
            #     for item_id in range(self.num_items):
            #         for dim in range(self.num_dimensions):
            #             input_var = f"demand_item_{item_id}_dim_{dim}"
            #             if key not in decision_to_input_map:
            #                 decision_to_input_map[key] = []
            #             decision_to_input_map[key].append(input_var)
        return decision_to_input_map


def get_demand_gap(problem, demand):
    args_dict = problem.convert_input_dict_to_args(demand)
    optimal_sol = problem.compute_optimal_value(args_dict)
    heuristic_sol = problem.compute_heuristic_value(args_dict)
    return heuristic_sol["heuristic_value"] - optimal_sol["optimal_value"]


def convert_permutation_to_demand(permutation):
    demand = {}
    for i, size in enumerate(permutation):
        demand[f"demand_item_{i}_dim_0"] = size
    return demand