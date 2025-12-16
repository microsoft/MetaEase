import sys
import os
# Add parent directory to path for common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import numpy as np
from ortools.algorithms.python import knapsack_solver
from ortools.linear_solver import pywraplp
from .problem import Problem
from common import LAMBDA_MAX_VALUE

# Scaling factor to convert floating points to integers
SCALE_FACTOR = 100000


def optimal_relaxed_knapsack(values, weights, capacity):
    """
    Solves the relaxed knapsack problem using OR-Tools.

    Args:
        values (list of float): List of item values.
        weights (list of float): List of item weights.
        capacity (float): Capacity of the knapsack.

    Returns:
        dict: Solution containing the optimal value, selected fractions, and lambda.
    """
    # Create the solver
    solver = pywraplp.Solver.CreateSolver("GLOP")  # GLOP is a linear programming solver
    if not solver:
        raise Exception("Solver not found!")

    # Number of items
    n = len(values)

    # Define the variables: x[i] represents the fraction of item i selected (0 <= x[i] <= 1)
    x = [solver.NumVar(0, 1, f"aux_x_{i}_") for i in range(n)]

    # Objective function: Maximize sum(values[i] * x[i])
    solver.Maximize(solver.Sum(values[i] * x[i] for i in range(n)))

    # Constraint: sum(weights[i] * x[i]) <= capacity
    constraint = solver.Add(solver.Sum(weights[i] * x[i] for i in range(n)) <= capacity)
    # Solve the problem
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("The solver did not find an optimal solution.")

    # Extract the results
    selected_fractions = [x[i].solution_value() for i in range(n)]
    optimal_value = solver.Objective().Value()

    # Calculate the lagrange multiplier
    lambda_value = constraint.dual_value()

    all_vars = {"lambda": lambda_value}
    for i in range(n):
        all_vars[f"aux_x_{i}"] = selected_fractions[i]
        all_vars[f"value_{i}"] = values[i]
        all_vars[f"weight_{i}"] = weights[i]

    return {
        "relaxed_optimal_value": optimal_value,
        "relaxed_all_vars": all_vars,
        "lambda": lambda_value,
    }


def closed_form_relaxed_optimal_knapsack(values, weights, capacity):
    # Calculate value densities (value-to-weight ratio) and sort items by density in descending order
    items = list(enumerate(zip(values, weights)))
    items.sort(key=lambda x: x[1][0] / x[1][1], reverse=True)

    selected_fractions = [0] * len(values)
    total_value = 0
    remaining_capacity = capacity
    lambda_value = None

    # Iterate through the sorted items and fill the knapsack
    for i, (value, weight) in items:
        if weight <= remaining_capacity:
            # Take the full item
            selected_fractions[i] = 1
            total_value += value
            remaining_capacity -= weight
        else:
            # Take a fraction of the item to fill the remaining capacity
            selected_fractions[i] = remaining_capacity / weight
            total_value += selected_fractions[i] * value
            lambda_value = value / weight  # Value density of the fractional item
            break

    return {
        "optimal_value": total_value,
        "selected_fractions": selected_fractions,
        "lambda": lambda_value,
    }

# TODO: needs better documentation for why this function exits, what it is used for and when.
def knapsack_relaxed_optimal_solution_derivatives(values, weights, capacity):
    # Calculate value densities (value-to-weight ratio) and sort items by density in descending order
    items = list(enumerate(zip(values, weights)))
    items.sort(key=lambda x: x[1][0] / x[1][1], reverse=True)

    selected_fractions = [0] * len(values)
    total_value = 0
    remaining_capacity = capacity
    lambda_value = None
    partial_index = -1

    # Iterate through the sorted items and fill the knapsack
    for i, (value, weight) in items:
        if weight <= remaining_capacity:
            # Take the full item
            selected_fractions[i] = 1
            total_value += value
            remaining_capacity -= weight
        else:
            # Take a fraction of the item to fill the remaining capacity
            selected_fractions[i] = remaining_capacity / weight
            total_value += selected_fractions[i] * value
            lambda_value = value / weight  # Value density of the fractional item
            partial_index = i
            break

    # Compute derivatives
    derivatives = {}
    for i, (value, weight) in enumerate(zip(values, weights)):
        if selected_fractions[i] == 1:  # Fully included
            d_fraction_d_value = 0
            d_fraction_d_weight = 0
        elif selected_fractions[i] > 0:  # Partially included
            d_fraction_d_value = 0
            d_fraction_d_weight = -remaining_capacity / (weight**2)
        else:  # Not included
            d_fraction_d_value = 0
            d_fraction_d_weight = 0

        if partial_index == i:  # Lambda corresponds to the partially included item
            d_lambda_d_value = 1 / weight
            d_lambda_d_weight = -value / (weight**2)
        else:
            d_lambda_d_value = 0
            d_lambda_d_weight = 0

        derivatives[i] = {
            "d_fraction_d_value": d_fraction_d_value,
            "d_fraction_d_weight": d_fraction_d_weight,
            "d_lambda_d_value": d_lambda_d_value,
            "d_lambda_d_weight": d_lambda_d_weight,
        }

    return {
        "selected_fractions": selected_fractions,
        "lambda_value": lambda_value,
        "derivatives": derivatives,
    }

# TODO: documentation needs improvement to discuss the difference between this function and the one before it.
def get_knapsack_optimal_solution_gradient(input_dict, num_items, capacity):
    """
    Computes the gradient of the optimal knapsack solution with respect to value[i] and weight[i].

    Parameters:
        input_dict (dict): Dictionary with keys like 'value_i' and 'weight_i' for each item, and 'aux_x_i' for fractions.
        num_items (int): Number of items in the knapsack.
        capacity (float): Total capacity of the knapsack.

    Returns:
        gradient (dict): Gradients of `aux_x_i` and `lambda` with respect to `value_i` and `weight_i`.
    """
    # Parse values and weights from input_dict
    values = [input_dict[f"value_{i}"] for i in range(num_items)]
    weights = [input_dict[f"weight_{i}"] for i in range(num_items)]

    # Solve the knapsack problem
    items = list(enumerate(zip(values, weights)))
    items.sort(
        key=lambda x: x[1][0] / x[1][1], reverse=True
    )  # Sort by value-to-weight ratio (density)

    selected_fractions = [0] * num_items
    remaining_capacity = capacity
    lambda_value = None
    partial_index = -1

    # Determine optimal fractions and lambda
    for i, (value, weight) in items:
        if weight <= remaining_capacity:
            selected_fractions[i] = 1
            remaining_capacity -= weight
        else:
            selected_fractions[i] = remaining_capacity / weight
            lambda_value = value / weight
            partial_index = i
            break

    # Compute gradients
    gradient = {key: 0.0 for key in input_dict.keys()}

    for i in range(num_items):
        if selected_fractions[i] == 1:  # Fully included
            d_fraction_d_value = 0
            d_fraction_d_weight = 0
        elif selected_fractions[i] > 0:  # Partially included
            d_fraction_d_value = 0
            d_fraction_d_weight = -remaining_capacity / (weights[i] ** 2)
        else:  # Not included
            d_fraction_d_value = 0
            d_fraction_d_weight = 0

        # Populate gradient for aux_x
        gradient[f"aux_x_{i}"] = {
            "value": d_fraction_d_value,
            "weight": d_fraction_d_weight,
        }

    # Populate gradient for lambda
    gradient["lambda"] = {}
    for i in range(num_items):
        if i == partial_index:
            d_lambda_d_value = 1 / weights[i]
            d_lambda_d_weight = -values[i] / (weights[i] ** 2)
        else:
            d_lambda_d_value = 0
            d_lambda_d_weight = 0

        gradient["lambda"][f"value_{i}"] = d_lambda_d_value
        gradient["lambda"][f"weight_{i}"] = d_lambda_d_weight

    return gradient

# TODO: you say the lagrange function is defined as and then describe an optimization problem as opposed to a true lagrangian. You should probably clean up the function documentation.
def optimal_knapsack(values, weights, capacity):
    """
    The lagrange function is defined as:
    Maximize sum_{i=1}^{n} value_i * x_i
    subject to sum_{i=1}^{n} weight_i * x_i - capacity <= 0 and x_i = 0 or 1
    lambda >= 0
    x_i = 0 or 1 --> relaxed to 0 <= x_i <= 1
    automatically convert the sign of the inequality to <= 0
    automatically negate the sign of gradient for lambda
    """

    # Scale the floating point values and weights to integers
    scaled_values = [int(v * SCALE_FACTOR) for v in values]
    scaled_weights = [[int(w * SCALE_FACTOR) for w in weights]]
    scaled_capacity = int(capacity * SCALE_FACTOR)

    # Create the solver
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, "KnapsackExample"
    )

    # Initialize the solver with scaled values, weights, and capacity
    solver.init(scaled_values, scaled_weights, [scaled_capacity])

    # Solve the problem
    computed_value = solver.solve()

    # Get the items to include in the knapsack
    packed_items = []
    packed_weights = []
    total_weight = 0
    total_value = 0
    all_vars = {}

    # write a code that prints the lagrange multipliers values for the solution
    for i in range(len(values)):
        all_vars[f"value_{i}"] = values[i]
        all_vars[f"weight_{i}"] = weights[i]
        if solver.best_solution_contains(i) or weights[i] == 0:
            packed_items.append(i)
            packed_weights.append(weights[i])
            total_value += scaled_values[i]
            lambda_i = values[i] / (weights[i] + 10e-8)
            all_vars[f"aux_x_{i}"] = 1
            total_weight += weights[i]
        else:
            all_vars[f"aux_x_{i}"] = 0

    all_vars["lambda"] = 0

    # Scale the result values back to original floating-point values
    return (
        all_vars,
        total_value / SCALE_FACTOR,
        packed_items,
        packed_weights,
        total_weight,
    )

# TODO: make it clear that this is a heuristic in the name maybe? Should probably devise a naming convention for how functions that are heuristic are separated from optimal, what I see is that you have Optimal, optimal_relaxed, and then your using the conventional names for heuristics. you may want to have a heuristic pre-fix for those too.
def greedy_knapsack(values, weights, capacity):
    # Scale the floating point values and weights to integers to make compatible with the knapsack solver
    values = [int(v * SCALE_FACTOR) for v in values]
    weights = [int(w * SCALE_FACTOR) for w in weights]
    capacity = int(capacity * SCALE_FACTOR)

    # Calculate value-to-weight ratio
    items = list(zip(values, weights))
    items = sorted(items, key=lambda x: x[0] / (x[1] + 10e-8), reverse=True)

    total_value = 0
    total_weight = 0
    packed_values = []
    packed_weights = []
    packed_items = []

    for (value, weight), index in zip(items, range(len(items))):
        if total_weight + weight <= capacity:
            packed_items.append(index)
            packed_values.append(value)
            packed_weights.append(weight)
            total_value += value
            total_weight += weight

    total_value /= SCALE_FACTOR
    packed_values = [v / SCALE_FACTOR for v in packed_values]
    packed_weights = [w / SCALE_FACTOR for w in packed_weights]
    total_weight /= SCALE_FACTOR

    sorted_packed_items = sorted(packed_items)
    # a unique code path number for the packed items
    # hash each matching to a unique code path number
    code_path_num = hash(tuple(sorted_packed_items))
    return total_value, code_path_num, packed_values, packed_weights, total_weight, packed_items


def get_knapsack_lagrangian_gradient(input_dict, num_items, capacity):
    gradient = {key: 0.0 for key in input_dict.keys()}
    for key, value in input_dict.items():
        if "aux_x" in key:
            i = int(key.split("_")[-1])
            gradient[key] = (
                input_dict[f"value_{i}"]
                - input_dict["lambda"] * input_dict[f"weight_{i}"]
            )
        elif "value" in key:
            i = int(key.split("_")[-1])
            gradient[key] = input_dict[f"aux_x_{i}"]
        elif "weight" in key:
            i = int(key.split("_")[-1])
            gradient[key] = -input_dict["lambda"] * input_dict[f"aux_x_{i}"]

    gradient["lambda"] = capacity - sum(
        [input_dict[f"weight_{i}"] * input_dict[f"aux_x_{i}"] for i in range(num_items)]
    )
    for key, value in gradient.items():
        if "lambda" in key:
            gradient[key] = -gradient[key]

    return gradient


def get_common_header(num_items, capacity, max_value):
    program = f"""
    #include <stdio.h>
    #include <stdlib.h>

    #define NUM_ITEMS {num_items}
    #define CAPACITY {capacity}
    #define MAX_VALUE {max_value}
    """

    program += """
    // A structure to represent an item
    struct Item {
        unsigned int value;
        unsigned int weight;
    };

    // Function to sort items by value-to-weight ratio using integer comparison
    void sortItems(struct Item items[], int n) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                // Compare items[j].value * items[j + 1].weight with items[j + 1].value * items[j].weight
                if (items[j].value * items[j + 1].weight < items[j + 1].value * items[j].weight) {
                    struct Item temp = items[j];
                    items[j] = items[j + 1];
                    items[j + 1] = temp;
                }
            }
        }
    }

    // Function to implement the greedy knapsack algorithm
    void greedyKnapsack(struct Item items[], int n, unsigned int capacity, unsigned int *totalValue, unsigned int *totalWeight) {
        // Sort items by value-to-weight ratio
        sortItems(items, n);

        *totalValue = 0;
        *totalWeight = 0;

        for (int i = 0; i < n; i++) {
            if (*totalWeight + items[i].weight <= capacity) {
                *totalWeight += items[i].weight;
                *totalValue += items[i].value;
            } else {
                break;
            }
        }
    }

    """

    return program


class KnapsackProblem(Problem):
    def __init__(self, problem_config_path):
        super().__init__(problem_config_path)
        self.num_total_klee_inputs = 2 * self.problem_config["num_items"]
        self.all_klee_var_names = []
        for i in range(self.problem_config["num_items"]):
            self.all_klee_var_names.append(f"value_{i}")
            self.all_klee_var_names.append(f"weight_{i}")

    def get_values_and_weights_from_input(self, input_dict):
        values = []
        weights = []
        input_values = []
        # get the number of items from the input dictionary
        num_items = self.problem_config["num_items"]

        for i in range(num_items):
            values.append(input_dict[f"value_{i}"])
            weights.append(input_dict[f"weight_{i}"])
            input_values.append(input_dict[f"value_{i}"])
            input_values.append(input_dict[f"weight_{i}"])

        return values, weights, input_values

    def convert_input_dict_to_args(self, input_dict):
        values, weights, input_values = self.get_values_and_weights_from_input(
            input_dict
        )
        capacity = self.problem_config["capacity"]
        return {
            "values": values,
            "weights": weights,
            "capacity": capacity,
            "input_values": input_values,
            "input_dict": input_dict,
            "num_items": len(values),
        }
 # TODO: again what are these thresholds and what are they used for?
    def get_thresholds(self, relaxed_all_vars):
        thresholds = {key: (0, self.problem_config["capacity"]) for key in self.all_klee_var_names}
        extra_vars = ['lambda']
        for i in range(self.problem_config["num_items"]):
            extra_vars.append(f"aux_x_{i}")
        for key in extra_vars:
            if "aux" in key:
                thresholds[key] = (0, 1)
            elif "lambda" in key:
                thresholds[key] = (0, LAMBDA_MAX_VALUE)
        return thresholds

    def get_decision_to_input_map(self, all_vars):
        # Create a mapping of decision variables to their corresponding input variables
        decision_to_input_map = {}

        # Map aux_x variables to their corresponding value and weight variables
        for key in all_vars:
            if key.startswith('aux_x_'):
                # For aux_x_i, the corresponding inputs are value_i and weight_i
                i = key.split('_')[-1]
                input_vars = [f"value_{i}", f"weight_{i}"]
                decision_to_input_map[key] = input_vars

        return decision_to_input_map

    def compute_optimal_value(self, args_dict):
        self.num_compute_optimal_value_called += 1
        values = args_dict["values"]
        weights = args_dict["weights"]
        capacity = args_dict["capacity"]
        (
            all_vars,
            optimal_value,
            packed_items,
            packed_weights,
            total_weight,
        ) = optimal_knapsack(values, weights, capacity)
        gradient = get_knapsack_lagrangian_gradient(all_vars, len(values), capacity)
        return {
            "gradient": gradient,
            "all_vars": all_vars,
            "optimal_value": optimal_value,
            "packed_items": packed_items,
            "packed_weights": packed_weights,
            "total_weight": total_weight,
        }

    def compute_heuristic_value(self, args_dict):
        self.num_compute_heuristic_value_called += 1
        values = args_dict["values"]
        weights = args_dict["weights"]
        capacity = args_dict["capacity"]
        heuristic_value, code_path_num, packed_values, packed_weights, total_weight, packed_items = (
            greedy_knapsack(values, weights, capacity)
        )

        # Construct all_vars for heuristic solution
        all_vars = {}

        # Add value and weight variables
        for i in range(len(values)):
            all_vars[f"value_{i}"] = values[i]
            all_vars[f"weight_{i}"] = weights[i]

        # Add aux_x variables based on packed items
        for i in range(len(values)):
            if i in packed_items:
                all_vars[f"aux_x_{i}"] = 1
            else:
                all_vars[f"aux_x_{i}"] = 0

        # Add lambda variable (set to 0 for heuristic)
        all_vars["lambda"] = 0

        return {
            "heuristic_value": heuristic_value,
            "code_path_num": code_path_num,
            "packed_values": packed_values,
            "packed_weights": packed_weights,
            "total_weight": total_weight,
            "all_vars": all_vars,
        }

    def compute_lagrangian_gradient(self, args_dict):
        capacity = args_dict["capacity"]
        num_items = self.problem_config["num_items"]
        input_dict = args_dict["input_dict"]
        gradient = get_knapsack_lagrangian_gradient(input_dict, num_items, capacity)
        return gradient

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        capacity = args_dict["capacity"]
        num_items = self.problem_config["num_items"]
        input_dict = args_dict["input_dict"]
        constraints = {}
        lagrange = 0
        total_weight = 0
        for i in range(num_items):
            lagrange += input_dict[f"value_{i}"] * input_dict[f"aux_x_{i}"]
            total_weight += input_dict[f"weight_{i}"] * input_dict[f"aux_x_{i}"]
        lagrange += input_dict["lambda"] * (capacity - total_weight)
        constraints["lambda"] = capacity - total_weight
        return {
            "lagrange": lagrange,
            "constraints": constraints,
        }

    def compute_relaxed_optimal_value(self, args_dict):
        values = args_dict["values"]
        weights = args_dict["weights"]
        capacity = args_dict["capacity"]
        return optimal_relaxed_knapsack(values, weights, capacity)

    def get_common_header(self):
        num_items = self.problem_config["num_items"]
        capacity = self.problem_config["capacity"]
        max_value = self.problem_config["max_value"]

        return get_common_header(num_items, capacity, max_value)

    def generate_heuristic_program(
        self, program_type, list_of_input_paths_to_exclude=[], num_klee_inputs=None, path_to_assigned_fixed_points=None
    ):
        num_items = self.problem_config["num_items"]
        capacity = self.problem_config["capacity"]
        max_value = self.problem_config["max_value"]
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
            f"\033[91mSelected klee inputs: {selected_klee_inputs} from {self.num_total_klee_inputs}\033[0m"
        )

        klee_inputs_names = []
        fixed_points = {}
        header = self.get_common_header()

        klee_specific = """
        int main() {
            struct Item items[NUM_ITEMS];
            unsigned int totalValue;
            unsigned int totalWeight;
        """
        for i in range(num_items):
            if f"value_{i}" in selected_klee_inputs:
                klee_specific += f"""
                unsigned int value_{i};
                klee_make_symbolic(&value_{i}, sizeof(value_{i}), "value_{i}");
                klee_assume(value_{i} >= 1 & value_{i} <= MAX_VALUE);
                items[{i}].value = value_{i};
                """
                klee_inputs_names.append(f"value_{i}")
            else:
                value = random.randint(1, max_value)
                klee_specific += f"""
                items[{i}].value = {value};
                """
                fixed_points[f"value_{i}"] = value
            if f"weight_{i}" in selected_klee_inputs:
                klee_specific += f"""
                unsigned int weight_{i};
                klee_make_symbolic(&weight_{i}, sizeof(weight_{i}), "weight_{i}");
                klee_assume(weight_{i} >= 1 & weight_{i} <= CAPACITY);
                items[{i}].weight = weight_{i};
                """
                klee_inputs_names.append(f"weight_{i}")
            else:
                weight = random.randint(1, capacity)
                klee_specific += f"""
                items[{i}].weight = {weight};
                """
                fixed_points[f"weight_{i}"] = weight

        # Assume that sum of weights is more than capacity
        weight_sum_string = ""
        for i in range(num_items):
            weight_sum_string += f"items[{i}].weight + "
        weight_sum_string = weight_sum_string[:-2]
        klee_specific += f"""
        klee_assume({weight_sum_string} > CAPACITY);
        """

        for input_path in list_of_input_paths_to_exclude:
            # read the json file
            with open(input_path, "r") as f:
                test_cases = json.load(f)
            for _, test in test_cases.items():
                excluding_string = ""
                for key, value in test.items():
                    key = key.strip("'").strip()
                    if key in klee_inputs_names:
                        excluding_string += f"{key} != {value} | "
                excluding_string = excluding_string[:-3]

                klee_specific += f"""
                klee_assume({excluding_string});
                """

        klee_specific += """
            int n = sizeof(items) / sizeof(items[0]);
            greedyKnapsack(items, NUM_ITEMS, CAPACITY, &totalValue, &totalWeight);
            return 0;
        }
        """

        exec_specific = """
        int main(int argc, char *argv[]) {
            struct Item items[NUM_ITEMS];
            unsigned int totalValue;
            unsigned int totalWeight;
        """
        for i in range(num_items):
            exec_specific += f"""
            items[{i}].value = atoi(argv[{2 * i + 1}]);
            items[{i}].weight = atoi(argv[{2 * i + 2}]);
            """

        exec_specific += """
            int n = sizeof(items) / sizeof(items[0]);
            greedyKnapsack(items, NUM_ITEMS, CAPACITY, &totalValue, &totalWeight);
            printf("FINAL_OUTPUT: %u\\n", totalValue);
            printf("Total weight: %u\\n", totalWeight);
            return 0;
        }
        """

        if program_type == "klee":
            return {
                "program": header + klee_specific,
                "fixed_points": fixed_points,
            }
        elif program_type == "exec":
            return header + exec_specific
