import sys
import os
# Add parent directory to path for utils, common, and config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from utils import run_program, compile_program
from ortools.linear_solver import pywraplp
from .problem import Problem
from common import LAMBDA_MAX_VALUE

from ortools.linear_solver import pywraplp

def relaxed_optimal_mwm(edges, num_nodes):
    """
    Solve the relaxed maximum weight matching problem by allowing continuous variables.
    Extracts the lambda variables correctly as dual values of the node constraints.
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return None, None, None

    all_vars = {}
    # Create continuous variables for each edge
    x = {}
    for i, (u, v, weight) in enumerate(edges):
        x[i] = solver.NumVar(0.0, 1.0, f"aux_x_{u}_{v}")

    # Objective: Maximize the sum of selected edge weights
    objective = solver.Objective()
    for i, (u, v, weight) in enumerate(edges):
        objective.SetCoefficient(x[i], weight)
        all_vars[f"weight_{u}_{v}"] = weight
    objective.SetMaximization()

    # Constraint: Each node can be matched to at most one other node
    node_constraints = {}
    for node in range(num_nodes):
        constraint = solver.Constraint(0, 1, f"lambda_{node}")
        node_constraints[node] = constraint
        for i, (u, v, weight) in enumerate(edges):
            if u == node or v == node:
                constraint.SetCoefficient(x[i], 1)

    # Solve the problem
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        total_weight = 0
        matched_edges = []

        # Extract solution values for x variables
        for i, (u, v, weight) in enumerate(edges):
            solution_value = x[i].solution_value()
            all_vars[f"aux_x_{u}_{v}"] = solution_value
            if solution_value > 0:  # Include edges with fractional values
                matched_edges.append((u, v, weight, solution_value))
                total_weight += solution_value * weight

        # Extract lambda values (dual variables)
        for node in range(num_nodes):
            lambda_value = node_constraints[node].dual_value()
            all_vars[f"lambda_{node}"] = lambda_value

        return all_vars, matched_edges, total_weight
    else:
        print("No optimal solution found.")
        return None, None, None


def optimal_mwm(edges, num_nodes):
    """
    Make sure you won't add the lambda variables to the solution
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, None
    all_vars = {}
    # Create binary variables for each edge
    x = {}
    for i, (u, v, weight) in enumerate(edges):
        x[i] = solver.BoolVar(f"aux_x_{u}_{v}")

    # Objective: Maximize the sum of selected edge weights
    objective = solver.Objective()
    for i, (u, v, weight) in enumerate(edges):
        objective.SetCoefficient(x[i], weight)
        all_vars[f"weight_{u}_{v}"] = weight
    objective.SetMaximization()

    # Constraint: Each node can be matched to at most one other node
    node_constraints = {}
    for node in range(num_nodes):
        constraint = solver.Constraint(0, 1, f"lambda_{node}")
        node_constraints[node] = constraint
        for i, (u, v, weight) in enumerate(edges):
            if u == node or v == node:
                constraint.SetCoefficient(x[i], 1)

    # Solve the problem
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        total_weight = 0
        matched_edges = []
        for i, (u, v, weight) in enumerate(edges):
            solution_value = x[i].solution_value()
            all_vars[f"aux_x_{u}_{v}"] = solution_value
            # print(f'aux_x_{u}_{v}] = {solution_value}')
            if solution_value > 0.5:
                matched_edges.append((u, v, weight))
                total_weight += weight

        # Estimate dual values (lambda) since they're not available from SCIP solver
        # We'll solve the relaxed version with the same weights to get dual estimates
        relaxed_all_vars, _, _ = relaxed_optimal_mwm(edges, num_nodes)
        if relaxed_all_vars:
            for node in range(num_nodes):
                if f"lambda_{node}" in relaxed_all_vars:
                    all_vars[f"lambda_{node}"] = relaxed_all_vars[f"lambda_{node}"]
                else:
                    all_vars[f"lambda_{node}"] = 0.0
        else:
            # Fallback: estimate using complementary slackness conditions
            # For matched nodes, estimate lambda as half the weight of incident matched edge
            # For unmatched nodes, lambda = 0
            for node in range(num_nodes):
                lambda_estimate = 0.0
                # Check if this node is matched in the optimal solution
                for u, v, weight in matched_edges:
                    if u == node or v == node:
                        # For matched nodes, estimate lambda as a fraction of the edge weight
                        lambda_estimate = weight / 2.0
                        break
                all_vars[f"lambda_{node}"] = 0.0 #lambda_estimate
        
        return all_vars, matched_edges, total_weight
    else:
        print("No optimal solution found.")
        return None, None, None


# the greedy algorithm is not guaranteed to find the optimal solution
def greedy_mwm(edges, num_nodes):
    # Sort the edges by weight in descending order
    edges = sorted(edges, key=lambda x: x[2], reverse=True)

    # Initialize an empty matching
    matching = []
    total_weight = 0
    code_path_num = 0
    # Greedily add edges to the matching
    matched_nodes = set()
    for u, v, weight in edges:
        if u not in matched_nodes and v not in matched_nodes:
            matching.append((u, v, weight))
            total_weight += weight
            matched_nodes.add(u)
            matched_nodes.add(v)

    sorted_matching = sorted([(min(u, v), max(u, v)) for u, v, _ in matching])
    code_path_num = "" #.join([str(u) + str(v) for u, v in sorted_matching])
    return code_path_num, matching, total_weight


def get_mwm_lagrangian_gradient(num_nodes, edges, input_dict):
    """
    Compute the gradient of the Lagrangian function with respect to:
    - weight (w_i)
    - edges show the connection between nodes in the original graph
    - decision variables (aux_x_i_j), the same as p
    - lambdas (Lagrange multiplier)
    == > dual
    L(x, lambda) = sum_{i=1}^{n} sum_{j=1}^{n} aux_x_i * w_i - sum_{i=1}^{n}(lambda_i * (sum_{j=1}^{n} aux_x_i_j  - 1))

    Returns:
    - gradient: dictionary containing the gradients with respect to values, weights, x, and lambda.
    """
    gradient = {}
    for k, (i, j, weight) in enumerate(edges):
        gradient[f"aux_x_{i}_{j}"] = (
            input_dict[f"weight_{i}_{j}"]
            - input_dict[f"lambda_{i}"]
            - input_dict[f"lambda_{j}"]
        )
        gradient[f"weight_{i}_{j}"] = 1 if input_dict[f"aux_x_{i}_{j}"] > 0.5 else 0

    for l in range(num_nodes):
        gradient[f"lambda_{l}"] = 0.0
        for k, (i, j, weight) in enumerate(edges):
            if i == l:
                assert l < j
                gradient[f"lambda_{l}"] += (
                    1 if input_dict[f"aux_x_{l}_{j}"] > 0.5 else 0
                )
            if j == l:
                assert i < l
                gradient[f"lambda_{l}"] += (
                    1 if input_dict[f"aux_x_{i}_{l}"] > 0.5 else 0
                )
        gradient[f"lambda_{l}"] -= 1.0

    return gradient


class MWMProblem(Problem):
    def __init__(self, problem_config_path):
        super().__init__(problem_config_path)
        num_nodes = self.problem_config["num_nodes"]
        self.all_klee_var_names = []
        self.num_edges = sum(1 for key in self.problem_config.keys() if "edge_" in key)
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if f"edge_{i}_{j}" in self.problem_config.keys():
                    self.all_klee_var_names.append(f"weight_{i}_{j}")

    def get_thresholds(self, relaxed_all_vars):
        thresholds = {
            key: (0, self.problem_config["max_weight"])
            for key in self.all_klee_var_names
        }
        for key in relaxed_all_vars:
            if key.startswith("lambda_"):
                thresholds[key] = (0, LAMBDA_MAX_VALUE)
            elif key.startswith("aux_x_"):
                edge_key = key.split("_")[2:]
                thresholds[key] = (0, self.problem_config[f"edge_{edge_key[0]}_{edge_key[1]}"])
        return thresholds

    def generate_edges_and_weights_from_input_dict(self, input_dict):
        num_nodes = self.problem_config["num_nodes"]
        edges = []
        weights = []
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if f"edge_{i}_{j}" in self.problem_config.keys():
                    edges.append((i, j, input_dict[f"weight_{i}_{j}"]))
                    weights.append(input_dict[f"weight_{i}_{j}"])

        return edges, weights

    def convert_input_dict_to_args(self, input_dict):
        num_nodes = self.problem_config["num_nodes"]
        edges, weights = self.generate_edges_and_weights_from_input_dict(input_dict)
        return {
            "num_nodes": num_nodes,
            "edges": edges,
            "weights": weights,
            "input_dict": input_dict,
        }

    def compute_optimal_value(self, args_dict):
        self.num_compute_optimal_value_called += 1
        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        all_vars, matched_edges, total_weight = optimal_mwm(edges, num_nodes)
        gradient = self.compute_lagrangian_gradient(
            {"num_nodes": num_nodes, "edges": edges, "input_dict": all_vars}
        )
        return {
            "gradient": gradient,
            "all_vars": all_vars,
            "matched_edges": matched_edges,
            "optimal_value": total_weight,
        }

    def compute_heuristic_value(self, args_dict):
        self.num_compute_heuristic_value_called += 1
        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        code_path_num, matching, total_weight = greedy_mwm(edges, num_nodes)
        return {
            "code_path_num": code_path_num,
            "matching": matching,
            "heuristic_value": total_weight,
        }

    def compute_lagrangian_gradient(self, args_dict):
        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        input_dict = args_dict["input_dict"]
        return get_mwm_lagrangian_gradient(num_nodes, edges, input_dict)

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        num_nodes = args_dict["num_nodes"]
        input_dict = args_dict["input_dict"]
        edges = args_dict["edges"]
        lagrange = 0
        constraints = {}
        for i in range(num_nodes):
            for j in range(num_nodes):
                if f"edge_{i}_{j}" in self.problem_config.keys():
                    lagrange += (
                        input_dict[f"aux_x_{i}_{j}"] * input_dict[f"weight_{i}_{j}"]
                    )
        if not give_relaxed_gap:
            for i in range(num_nodes):
                constraint = 0
                for j in range(num_nodes):
                    if f"edge_{i}_{j}" in self.problem_config.keys():
                        constraint += input_dict[f"aux_x_{i}_{j}"]
                    if f"edge_{j}_{i}" in self.problem_config.keys():
                        constraint += input_dict[f"aux_x_{j}_{i}"]
                constraints[f"lambda_{i}"] = 1 - constraint
                lagrange += input_dict[f"lambda_{i}"] * (1 - constraint)
        return {
            "lagrange": lagrange,
            "constraints": constraints,
        }

    def compute_relaxed_optimal_value(self, args_dict):
        num_nodes = args_dict["num_nodes"]
        edges = args_dict["edges"]
        all_vars, matched_edges, relaxed_total_weight = relaxed_optimal_mwm(
            edges, num_nodes
        )
        return {
            "relaxed_optimal_value": relaxed_total_weight,
            "relaxed_all_vars": all_vars
        }

    def get_common_header(self, args_dict):
        max_weight = args_dict["max_weight"]
        program = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <klee/klee.h>

        #define MAX_Weight {max_weight}
        """

        program += """
        // Structure to represent an edge
        typedef struct {
            int u, v;
            unsigned int weight;
        } Edge;

        // Partition function for quicksort
        int partition(Edge edges[], int low, int high) {
            unsigned int pivot = edges[high].weight;
            int i = (low - 1);
            for (int j = low; j < high; j++) {
                if (edges[j].weight >= pivot) { // Sorting in descending order
                    i++;
                    Edge temp = edges[i];
                    edges[i] = edges[j];
                    edges[j] = temp;
                }
            }
            Edge temp = edges[i + 1];
            edges[i + 1] = edges[high];
            edges[high] = temp;
            return (i + 1);
        }

        // Quicksort function
        void quicksort(Edge edges[], int low, int high) {
            if (low < high) {
                int pi = partition(edges, low, high);
                quicksort(edges, low, pi - 1);
                quicksort(edges, pi + 1, high);
            }
        }

        // Function to perform the greedy maximum weighted matching
        void greedy_mwm(Edge edges[], int num_edges, int num_nodes, unsigned int* total_weight) {
            // Sort the edges by weight in descending order
            quicksort(edges, 0, num_edges - 1);

            // Initialize an empty matching and other necessary variables
            int* matched_nodes = (int*)calloc(num_nodes, sizeof(int));

            for (int i = 0; i < num_edges; i++) {
                int u = edges[i].u;
                int v = edges[i].v;
                unsigned int weight = edges[i].weight;

                // If neither of the nodes u or v are matched, add this edge to the matching
                if (!matched_nodes[u] && !matched_nodes[v]) {
                    *total_weight += weight;
                    matched_nodes[u] = 1;
                    matched_nodes[v] = 1;
                }
            }
            // Free allocated memory
            free(matched_nodes);
        }
        """
        return program

    def generate_heuristic_program(
        self, program_type, list_of_input_paths_to_exclude=[], path_to_assigned_fixed_points=None, num_klee_inputs=None
    ):
        num_nodes = self.problem_config["num_nodes"]
        num_edges = self.num_edges
        max_weight = self.problem_config["max_weight"]
        # num of keys that have edge_ in them
        num_edges_in_config = len(
            [key for key in self.problem_config.keys() if "edge_" in key]
        )
        assert num_edges == num_edges_in_config, f"Number of edges mismatched!"

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
                num_klee_inputs = min(num_klee_inputs, len(self.all_klee_var_names))
                selected_klee_inputs = random.sample(
                    self.all_klee_var_names, num_klee_inputs
                )
            else:
                selected_klee_inputs = self.all_klee_var_names
            print(
                f"Selected klee inputs: {selected_klee_inputs} from {len(self.all_klee_var_names)}"
            )

        count = 0
        fixed_points = {}
        header = self.get_common_header({"max_weight": max_weight})

        klee_specific = f"""
        int main() {{
            int num_nodes = {num_nodes};
            int num_edges = {num_edges};
            Edge edges[num_edges];
            unsigned int total_weight = 0;
        """

        # Generate symbolic input for edge weights
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if f"edge_{i}_{j}" in self.problem_config.keys():
                    weight_key = f"weight_{i}_{j}"
                    if weight_key in selected_klee_inputs:
                        klee_specific += f"""
                        edges[{count}] = (Edge){{ {i}, {j}, 0 }};
                        unsigned int weight_{i}_{j};
                        klee_make_symbolic(&weight_{i}_{j}, sizeof(weight_{i}_{j}), "weight_{i}_{j}");
                        klee_assume(weight_{i}_{j} >= 1  && weight_{i}_{j} <= MAX_Weight);
                        edges[{count}].weight = weight_{i}_{j};
                        """
                    else:
                        if file_fixed_points is not None:
                            weight = file_fixed_points[weight_key]
                        else:
                            weight = random.randint(1, max_weight)
                        fixed_points[weight_key] = weight
                        klee_specific += f"""
                        edges[{count}] = (Edge){{ {i}, {j}, {weight} }};
                        """
                    count += 1

        # Add hardness assumptions to ensure KLEE generates challenging inputs for greedy MWM
        # Strategy: Create scenarios where high-weight edges conflict with optimal matching
        if not selected_klee_inputs:  # Only add if we have symbolic inputs
            pass
        else:
            # Get all edges in the topology to identify potential conflicts
            all_edges = []
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if f"edge_{i}_{j}" in self.problem_config.keys():
                        all_edges.append((i, j))

            # Find edges that share nodes (conflicting edges)
            conflicting_edge_groups = []
            for i, edge1 in enumerate(all_edges):
                conflicts = []
                for j, edge2 in enumerate(all_edges):
                    if i != j and (edge1[0] in edge2 or edge1[1] in edge2):
                        conflicts.append(edge2)
                if len(conflicts) >= 2:  # Edge conflicts with at least 2 others
                    conflicting_edge_groups.append((edge1, conflicts))

            if conflicting_edge_groups:
                # Create hardness assumptions: ensure some high-weight edges conflict
                # This forces greedy to make suboptimal choices
                hardness_conditions = []

                # Strategy 1: Find any center edge with at least 1 conflicting edge (relaxed requirement)
                for center_edge, conflicts in conflicting_edge_groups[:3]:  # Limit to 3 groups
                    center_weight = f"weight_{center_edge[0]}_{center_edge[1]}"
                    if center_weight in selected_klee_inputs:
                        # Make the center edge have high weight
                        conflict_weights = []
                        for conf_edge in conflicts[:3]:  # Check more conflicts
                            conf_weight = f"weight_{conf_edge[0]}_{conf_edge[1]}"
                            if conf_weight in selected_klee_inputs:
                                conflict_weights.append(conf_weight)

                        if len(conflict_weights) >= 1:  # Relaxed from 2 to 1
                            # Condition: center edge is heavy AND at least one conflicting edge is also heavy
                            # This creates potential for greedy suboptimality
                            if len(conflict_weights) == 1:
                                condition = f"({center_weight} >= {max_weight - 1} && {conflict_weights[0]} >= {max_weight - 2})"
                            else:
                                condition = f"({center_weight} >= {max_weight - 2} && "
                                condition += f"({' + '.join(conflict_weights[:2])}) > {center_weight})"
                            hardness_conditions.append(condition)

                # Strategy 2: Ensure at least some edges have high weights to create competition
                weight_vars = [var for var in selected_klee_inputs if var.startswith("weight_")]
                if len(weight_vars) >= 2 and not hardness_conditions:
                    # Fallback: create weight competition among any available edges
                    high_weight_edges = weight_vars[:2]
                    condition = f"({high_weight_edges[0]} >= {max_weight - 1} && {high_weight_edges[1]} >= {max_weight - 1})"
                    hardness_conditions.append(condition)

                if hardness_conditions:
                    # Ensure at least one hardness condition is met
                    assumption_string = " || ".join(hardness_conditions)
                    klee_specific += f"""
                    // Hardness assumption: create greedy traps where high-weight edges conflict
                    klee_assume({assumption_string});
                    """

                # Additional assumption: ensure weight distribution has sufficient variance
                # to create meaningful optimization scenarios
                if len(selected_klee_inputs) >= 3:
                    weight_vars = [var for var in selected_klee_inputs if var.startswith("weight_")]
                    if len(weight_vars) >= 3:
                        # Ensure we have both high and medium weights
                        high_weight_conditions = []
                        medium_weight_conditions = []
                        for i, weight_var in enumerate(weight_vars[:6]):  # Limit to avoid overly complex assumptions
                            if i < 2:
                                high_weight_conditions.append(f"{weight_var} >= {max_weight - 1}")
                            else:
                                medium_weight_conditions.append(f"{weight_var} >= {max_weight // 2}")

                        if high_weight_conditions and medium_weight_conditions:
                            variance_assumption = f"({' || '.join(high_weight_conditions)}) && ({' || '.join(medium_weight_conditions)})"
                            klee_specific += f"""
                            // Weight variance assumption: ensure mix of high and medium weights
                            klee_assume({variance_assumption});
                            """

        for input_path in list_of_input_paths_to_exclude:
            # read the json file
            with open(input_path, "r") as f:
                test_cases = json.load(f)
            for _, test in test_cases.items():
                excluding_string = ""
                for key, value in test.items():
                    key = key.strip("'").strip()
                    if key in selected_klee_inputs:
                        excluding_string += f"{key} != {value} || "
                excluding_string = excluding_string[:-4]

                klee_specific += f"""
                klee_assume({excluding_string});
                """

        klee_specific += """
            greedy_mwm(edges, num_edges, num_nodes, &total_weight);
            return 0;
        }
        """
        if program_type == "klee":
            return {"program": header + klee_specific, "fixed_points": fixed_points}

