import sys
import os
# Add parent directory to path for utils and common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from utils import *
from ortools.sat.python import cp_model
from .problem import Problem
from common import LAMBDA_MAX_VALUE

SCALE_FACTOR = 100000  # Scaling factor to avoid precision issues

from ortools.linear_solver import pywraplp


def optimal_relaxed_tsp(dist_matrix):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise Exception("Solver not available")

    num_cities = len(dist_matrix)

    # Relaxed variables for whether city i is followed by city j
    x = {}
    all_vars = {}
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x[i, j] = solver.NumVar(0, 1, f"aux_x_{i}_{j}")

    # Constraint: Each city is followed by exactly one other city
    outgoing_constraints = []
    for i in range(num_cities):
        c = solver.Add(sum(x[i, j] for j in range(num_cities) if i != j) == 1)
        outgoing_constraints.append(c)

    # Constraint: Each city is preceded by exactly one other city
    incoming_constraints = []
    for j in range(num_cities):
        c = solver.Add(sum(x[i, j] for i in range(num_cities) if i != j) == 1)
        incoming_constraints.append(c)

    # Subtour elimination using the Miller-Tucker-Zemlin (MTZ) formulation
    u = [solver.NumVar(0, num_cities - 1, f"aux_u_{i}") for i in range(num_cities)]
    mtz_constraints = {}
    for i in range(1, num_cities):
        for j in range(1, num_cities):
            if i != j:
                c = solver.Add(
                    u[i] - u[j] + (num_cities - 1) * x[i, j] <= num_cities - 2
                )
                mtz_constraints[(i, j)] = c

    # Objective: Minimize the total travel distance
    objective = solver.Objective()
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                objective.SetCoefficient(x[i, j], dist_matrix[i][j])
    objective.SetMinimization()

    # Solve the relaxed TSP
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        solution = {}
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    solution[f"aux_x_{i}_{j}"] = x[i, j].solution_value()

        relaxed_optimal_distance = solver.Objective().Value()

        # Populate all_vars
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    all_vars[f"aux_x_{i}_{j}"] = solution[f"aux_x_{i}_{j}"]
                    all_vars[f"dist_{i}_{j}"] = dist_matrix[i][j]

        for i in range(1, num_cities):
            all_vars[f"aux_u_{i}"] = u[i].solution_value()

        # Approximate lambda values for constraints
        for i, c in enumerate(outgoing_constraints):
            all_vars[f"lambda_outgoing_{i}"] = c.dual_value()

        for j, c in enumerate(incoming_constraints):
            all_vars[f"lambda_incoming_{j}"] = c.dual_value()

        for i in range(1, num_cities):
            for j in range(1, num_cities):
                if i != j:
                    c = mtz_constraints[(i, j)]
                    all_vars[f"lambda_u_{i}_{j}"] = c.dual_value()

        return solution, all_vars, relaxed_optimal_distance
    else:
        return None, None, None


def optimal_tsp(dist_matrix):
    model = cp_model.CpModel()
    num_cities = len(dist_matrix)

    # Binary variables indicating if city i is followed by city j
    x = {}
    all_vars = {}
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x[i, j] = model.NewBoolVar(f"aux_x_{i}_{j}")

    # Constraint: Each city is followed by exactly one other city
    outgoing_constraints = []
    for i in range(num_cities):
        c = model.Add(sum(x[i, j] for j in range(num_cities) if i != j) == 1)
        outgoing_constraints.append(c)

    # Constraint: Each city is preceded by exactly one other city
    incoming_constraints = []
    for j in range(num_cities):
        c = model.Add(sum(x[i, j] for i in range(num_cities) if i != j) == 1)
        incoming_constraints.append(c)

    # Subtour elimination using the Miller-Tucker-Zemlin (MTZ) formulation
    u = [model.NewIntVar(0, num_cities - 1, f"aux_u_{i}") for i in range(num_cities)]
    mtz_constraints = {}
    for i in range(1, num_cities):
        for j in range(1, num_cities):
            if i != j:
                c = model.Add(
                    u[i] - u[j] + (num_cities - 1) * x[i, j] <= num_cities - 2
                )
                mtz_constraints[(i, j)] = c

    # Objective: Minimize the total travel distance
    scaled_dist_matrix = [
        [int(dist * SCALE_FACTOR) for dist in row] for row in dist_matrix
    ]
    total_distance = model.NewIntVar(
        0,
        int(sum(max(row) for row in scaled_dist_matrix) * num_cities),
        "total_distance",
    )
    model.Add(
        total_distance
        == sum(
            scaled_dist_matrix[i][j] * x[i, j]
            for i in range(num_cities)
            for j in range(num_cities)
            if i != j
        )
    )
    model.Minimize(total_distance)

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        optimal_tour = []
        current_city = 0
        while True:
            optimal_tour.append(current_city)
            for j in range(num_cities):
                if current_city != j and solver.BooleanValue(x[current_city, j]):
                    current_city = j
                    break
            if current_city == 0:
                break
        optimal_distance = solver.Value(total_distance) / SCALE_FACTOR

        # Collect variable values
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    all_vars[f"aux_x_{i}_{j}"] = int(solver.BooleanValue(x[i, j]))
                    all_vars[f"dist_{i}_{j}"] = dist_matrix[i][j]

        for i in range(1, num_cities):
            all_vars[f"aux_u_{i}"] = solver.Value(u[i])

        # Approximate lambda values using slack
        for i, c in enumerate(outgoing_constraints):
            outgoing_slack = (
                sum(solver.BooleanValue(x[i, j]) for j in range(num_cities) if i != j)
                - 1
            )
            all_vars[f"lambda_outgoing_{i}"] = -outgoing_slack  # Negative of slack

        for j, c in enumerate(incoming_constraints):
            incoming_slack = (
                sum(solver.BooleanValue(x[i, j]) for i in range(num_cities) if i != j)
                - 1
            )
            all_vars[f"lambda_incoming_{j}"] = -incoming_slack  # Negative of slack

        for i in range(1, num_cities):
            for j in range(1, num_cities):
                if i != j:
                    mtz_slack = solver.Value(
                        u[i] - u[j] + (num_cities - 1) * solver.BooleanValue(x[i, j])
                    ) - (num_cities - 2)
                    all_vars[f"lambda_u_{i}_{j}"] = -mtz_slack  # Negative of slack

        return optimal_tour, all_vars, optimal_distance
    else:
        return None, None, None


def nearest_neighbor_tsp(dist_matrix):
    num_cities = len(dist_matrix)
    visited = [False] * num_cities  # Array to keep track of visited cities
    current_city = 0  # Start from the first city (index 0)
    tour = [current_city]  # Initialize the tour with the starting city
    visited[current_city] = True  # Mark the starting city as visited
    total_distance = 0  # Initialize total distance to 0

    for _ in range(1, num_cities):
        min_distance = float("inf")
        next_city = None

        # Find the nearest unvisited city
        for city in range(num_cities):
            if not visited[city] and dist_matrix[current_city][city] < min_distance:
                min_distance = dist_matrix[current_city][city]
                next_city = city

        # Update the tour and distance
        total_distance += min_distance
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city

    # Return to the starting city to complete the tour
    total_distance += dist_matrix[current_city][tour[0]]
    tour.append(tour[0])  # Complete the cycle
    code_path_num = "".join(str(tour[i]) for i in range(len(tour)))
    return tour, code_path_num, total_distance


def get_common_header(num_cities, max_distance):
    program = f"""
    #include <stdio.h>
    #include <limits.h>
    #include <klee/klee.h>

    #define MAX_CITIES {num_cities}
    #define MAX_DISTANCE {max_distance}
    """
    program += """
    // Function to find the nearest unvisited city
    unsigned int findNearestNeighbor(unsigned int current_city, unsigned int num_cities, unsigned int dist_matrix[MAX_CITIES][MAX_CITIES], unsigned int visited[MAX_CITIES]) {
        unsigned int min_distance = INT_MAX;
        unsigned int nearest_city = -1;

        for (unsigned int i = 0; i < num_cities; i++) {
            if (!visited[i] && dist_matrix[current_city][i] < min_distance) {
                min_distance = dist_matrix[current_city][i];
                nearest_city = i;
            }
        }
        return nearest_city;
    }

    // Function implementing the Nearest Neighbor heuristic
    void nearestNeighborTSP(unsigned int num_cities, unsigned int dist_matrix[MAX_CITIES][MAX_CITIES], unsigned int tour[MAX_CITIES], unsigned int *total_distance) {
        unsigned int visited[MAX_CITIES] = {0};  // Array to keep track of visited cities
        unsigned int current_city = 0;           // Start from the first city (index 0)
        
        visited[current_city] = 1;      // Mark the starting city as visited
        tour[0] = current_city;         // Add the starting city to the tour
        *total_distance = 0;            // Initialize total distance to 0

        for (unsigned int i = 1; i < num_cities; i++) {
            unsigned int next_city = findNearestNeighbor(current_city, num_cities, dist_matrix, visited);
            tour[i] = next_city;        // Add the nearest city to the tour
            *total_distance += dist_matrix[current_city][next_city]; // Add the distance to the total
            visited[next_city] = 1;     // Mark the city as visited
            current_city = next_city;   // Move to the next city
        }
        
        // Return to the starting city to complete the tour
        *total_distance += dist_matrix[current_city][tour[0]];
        tour[num_cities] = tour[0];  // Complete the cycle
    }
    """
    return program


class TSPProblem(Problem):
    def __init__(self, problem_config_path):
        super().__init__(problem_config_path)
        self.num_total_klee_inputs = (
            self.problem_config["num_cities"]
            * (self.problem_config["num_cities"] - 1)
            // 2
        )
        # Initialize all_klee_var_names for the distance variables
        self.all_klee_var_names = []
        num_cities = self.problem_config["num_cities"]
        for i in range(num_cities):
            for j in range(num_cities):
                if i < j:
                    self.all_klee_var_names.append(f"dist_{i}_{j}")

    def get_thresholds(self, relaxed_all_vars):
        # Set thresholds for distance variables (input variables)
        thresholds = {key: (0, self.problem_config["max_distance"]) for key in self.all_klee_var_names}

        # Set thresholds for all variables in relaxed_all_vars
        for key in relaxed_all_vars:
            if key.startswith("aux_x_"):
                # Binary variables for whether city i is followed by city j
                thresholds[key] = (0, 1)
            elif key.startswith("aux_u_"):
                # MTZ formulation variables - can range from 0 to num_cities-1
                thresholds[key] = (0, self.problem_config["num_cities"] - 1)
            elif key.startswith("lambda_"):
                # Lagrange multipliers
                thresholds[key] = (0, LAMBDA_MAX_VALUE)
            elif key.startswith("dist_"):
                # Distance variables (if not already in all_klee_var_names)
                if key not in thresholds:
                    thresholds[key] = (0, self.problem_config["max_distance"])

        return thresholds

    def is_input_feasible(self, input_dict):
        # For TSP, check if the distance matrix is valid
        num_cities = self.problem_config["num_cities"]
        max_distance = self.problem_config["max_distance"]

        # Check if all distance values are within valid range
        for i in range(num_cities):
            for j in range(num_cities):
                if i < j:  # Only check upper triangular part
                    dist_key = f"dist_{i}_{j}"
                    if dist_key in input_dict:
                        dist_value = input_dict[dist_key]
                        if dist_value <= 0 or dist_value > max_distance:
                            return False

        return True

    def get_decision_to_input_map(self, all_vars):
        # Create a mapping of decision variables to their corresponding input variables
        decision_to_input_map = {}

        # Map aux_x variables to their corresponding distance variables
        for key in all_vars:
            if key.startswith('aux_x_'):
                # For aux_x_i_j, the corresponding input is dist_i_j
                parts = key.split('_')
                i = parts[2]
                j = parts[3]
                input_var = f"dist_{i}_{j}"
                decision_to_input_map[key] = input_var

        return decision_to_input_map

    def generate_dist_matrix_from_input_values(self, num_cities, input_values):
        dist_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j and i < j:
                    dist_matrix[i][j] = input_values[f"dist_{i}_{j}"]
                    dist_matrix[j][i] = dist_matrix[i][j]
                if i == j:
                    dist_matrix[i][j] = 0
        return dist_matrix

    def convert_input_dict_to_args(self, input_dict):
        num_cities = self.problem_config["num_cities"]
        dist_matrix = self.generate_dist_matrix_from_input_values(
            num_cities, input_dict
        )
        return {
            "dist_matrix": dist_matrix,
            "input_dict": input_dict,
            "num_cities": num_cities,
        }

    def compute_optimal_value(self, args_dict):
        self.num_compute_optimal_value_called += 1
        dist_matrix = args_dict["dist_matrix"]
        optimal_tour, all_vars, optimal_distance = optimal_tsp(dist_matrix)
        gradient = self.compute_lagrangian_gradient(
            {"input_dict": all_vars, "num_cities": args_dict["num_cities"]}
        )
        return {
            "optimal_tour": optimal_tour,
            "gradient": gradient,
            "all_vars": all_vars,
            "optimal_value": optimal_distance,
        }

    def compute_heuristic_value(self, args_dict):
        self.num_compute_heuristic_value_called += 1
        num_cities = args_dict["num_cities"]
        dist_matrix = args_dict["dist_matrix"]
        heuristic_tour, code_path_num, heuristic_distance = nearest_neighbor_tsp(
            dist_matrix
        )

        # Construct all_vars for heuristic solution
        all_vars = {}

        # Add distance variables
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    all_vars[f"dist_{i}_{j}"] = dist_matrix[i][j]

        # Convert tour to aux_x decision variables
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    # Check if city i is followed by city j in the tour
                    tour_index_i = heuristic_tour.index(i)
                    next_index = (tour_index_i + 1) % num_cities
                    next_city = heuristic_tour[next_index]
                    all_vars[f"aux_x_{i}_{j}"] = 1 if j == next_city else 0

        # Add aux_u variables for MTZ formulation (simplified)
        for i in range(1, num_cities):
            all_vars[f"aux_u_{i}"] = heuristic_tour.index(i)

        # Add lambda variables (set to 0 for heuristic)
        for i in range(num_cities):
            all_vars[f"lambda_outgoing_{i}"] = 0
            all_vars[f"lambda_incoming_{i}"] = 0

        for i in range(1, num_cities):
            for j in range(1, num_cities):
                if i != j:
                    all_vars[f"lambda_u_{i}_{j}"] = 0

        return {
            "heuristic_tour": heuristic_tour,
            "code_path_num": code_path_num,
            "heuristic_value": heuristic_distance,
            "all_vars": all_vars,
        }

    def compute_lagrangian_gradient(self, args_dict):
        input_dict = args_dict["input_dict"]
        num_cities = args_dict["num_cities"]
        gradient = {key: 0 for key in input_dict.keys()}
        for key, value in input_dict.items():
            if "dist" in key:
                i = int(key.split("_")[1])
                j = int(key.split("_")[2])
                gradient[key] = input_dict[f"aux_x_{i}_{j}"]
            elif "aux_x" in key:
                i = int(key.split("_")[2])
                j = int(key.split("_")[3])
                gradient[key] = (
                    input_dict[f"dist_{i}_{j}"]
                    + input_dict[f"lambda_outgoing_{i}"]
                    + input_dict[f"lambda_incoming_{j}"]
                )
                if i != 0 and j != 0:
                    gradient[key] += input_dict[f"lambda_u_{i}_{j}"] * (num_cities - 1)
            elif "aux_u" in key:
                i = int(key.split("_")[2])
                for j in range(1, num_cities):
                    if i != j:
                        gradient[key] += (
                            input_dict[f"lambda_u_{i}_{j}"]
                            - input_dict[f"lambda_u_{j}_{i}"]
                        )
            elif "lambda_outgoing" in key:
                i = int(key.split("_")[2])
                gradient[key] = 1 - sum(
                    input_dict[f"aux_x_{i}_{j}"] for j in range(num_cities) if i != j
                )
            elif "lambda_incoming" in key:
                j = int(key.split("_")[2])
                gradient[key] = 1 - sum(
                    input_dict[f"aux_x_{i}_{j}"] for i in range(num_cities) if i != j
                )
            elif "lambda_u" in key:
                i = int(key.split("_")[2])
                j = int(key.split("_")[3])
                gradient[key] = (
                    input_dict[f"aux_u_{i}"]
                    - input_dict[f"aux_u_{j}"]
                    + (num_cities - 1) * input_dict[f"aux_x_{i}_{j}"]
                    - num_cities
                    + 2
                )

        # negate the sign of the gradient of keys with "lambda" in them
        for key, value in gradient.items():
            if "lambda" in key:
                gradient[key] = -gradient[key]

        return gradient

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        input_dict = args_dict["input_dict"]
        num_cities = args_dict["num_cities"]
        constraints = {}
        lagrange = 0
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    lagrange += (
                        input_dict[f"aux_x_{i}_{j}"] * input_dict[f"dist_{i}_{j}"]
                    )

        if not give_relaxed_gap:
            for i in range(num_cities):
                constraint = 1 - sum(
                    input_dict[f"aux_x_{i}_{j}"] for j in range(num_cities) if i != j
                )
                constraints[f"lambda_outgoing_{i}"] = constraint
                lagrange += input_dict[f"lambda_outgoing_{i}"] * constraint

            for j in range(num_cities):
                constraint = 1 - sum(
                    input_dict[f"aux_x_{i}_{j}"] for i in range(num_cities) if i != j
                )
                constraints[f"lambda_incoming_{j}"] = constraint
                lagrange += input_dict[f"lambda_incoming_{j}"] * constraint

            for i in range(1, num_cities):
                for j in range(1, num_cities):
                    if i != j:
                        constraint = (
                            input_dict[f"aux_u_{i}"]
                            - input_dict[f"aux_u_{j}"]
                            + (num_cities - 1) * input_dict[f"aux_x_{i}_{j}"]
                            - num_cities
                            + 2
                        )
                        constraints[f"lambda_u_{i}_{j}"] = constraint
                        lagrange += input_dict[f"lambda_u_{i}_{j}"] * constraint
        return {
            "lagrange": lagrange,
            "constraints": constraints,
        }

    def compute_relaxed_optimal_value(self, args_dict):
        dist_matrix = args_dict["dist_matrix"]
        relaxed_solution, all_vars, relaxed_distance = optimal_relaxed_tsp(dist_matrix)
        return {
            "relaxed_optimal_value": relaxed_distance,
            "relaxed_all_vars": all_vars,
        }

    def get_common_header(self, args_dict):
        num_cities = args_dict["num_cities"]
        max_distance = args_dict["max_distance"]
        return get_common_header(num_cities, max_distance)

    def programs_strings(self):
        programs = {}
        programs[
            "klee_specific1"
        ] = """
        int main() {
            unsigned int num_cities = MAX_CITIES;
            unsigned int tour[MAX_CITIES + 1];  // Array to hold the tour (including return to start city)
            unsigned int total_distance;
            unsigned int dist_matrix[MAX_CITIES][MAX_CITIES] = {{0}};
        """
        programs[
            "klee_specific2"
        ] = """
            nearestNeighborTSP(num_cities, dist_matrix, tour, &total_distance);
            return 0;
        }
        """
        programs[
            "exec_specific1"
        ] = """
        int main(int argc, char *argv[]) {
            unsigned int num_cities = MAX_CITIES;
            unsigned int tour[MAX_CITIES + 1];  // Array to hold the tour (including return to start city)
            unsigned int total_distance;
            unsigned int dist_matrix[MAX_CITIES][MAX_CITIES] = {{0}};
        """
        programs[
            "exec_specific2"
        ] = f"""
            // Initialize the dist_matrix to zero
            for (int i = 0; i < num_cities; i++) {{
                for (int j = 0; j < num_cities; j++) {{
                    dist_matrix[i][j] = 0;
                }}
            }}

            // Parse command line arguments for the upper triangular part of the adjacency matrix
            int k = 1; // Start from argv[1]
            for (int i = 0; i < num_cities; i++) {{
                for (int j = i + 1; j < num_cities; j++) {{
                    dist_matrix[i][j] = atoi(argv[k]);
                    dist_matrix[j][i] = dist_matrix[i][j]; // Since the dist_matrix is undirected
                    k++;
                }}
            }}
        """
        programs[
            "exec_specific3"
        ] = """
            nearestNeighborTSP(num_cities, dist_matrix, tour, &total_distance);
            printf("FINAL_OUTPUT: %u\\n", total_distance);
            return 0;
        }}
        """
        return programs

    def generate_heuristic_program(
        self, program_type, list_of_input_paths_to_exclude=[], num_klee_inputs=None, path_to_assigned_fixed_points=None
    ):
        num_cities = self.problem_config["num_cities"]
        max_distance = self.problem_config["max_distance"]
        args_dict = {"num_cities": num_cities, "max_distance": max_distance}
        header = self.get_common_header(args_dict)
        programs = self.programs_strings()
        klee_specific = programs["klee_specific1"]
        file_fixed_points = None
        if path_to_assigned_fixed_points:
            with open(path_to_assigned_fixed_points, "r") as f:
                file_fixed_points = json.load(f)
            selected_klee_inputs = [name for name in self.all_klee_var_names if name not in file_fixed_points]
        else:
            if num_klee_inputs is not None:
                num_klee_inputs = min(num_klee_inputs, len(self.all_klee_var_names))
                selected_klee_inputs = random.sample(self.all_klee_var_names, num_klee_inputs)
            else:
                selected_klee_inputs = self.all_klee_var_names
            print(
                f"Selected klee inputs: {selected_klee_inputs} from {self.num_total_klee_inputs}"
            )
        klee_inputs_names = []
        count = 0
        fixed_points = {}
        for i in range(num_cities):
            for j in range(num_cities):
                if i == j:
                    klee_specific += f"dist_matrix[{i}][{j}] = 0;\n"
                elif i < j:
                    dist_var_name = f"dist_{i}_{j}"
                    if dist_var_name in selected_klee_inputs:
                        klee_inputs_names.append(dist_var_name)
                        klee_specific += f"""
                        unsigned int {dist_var_name};
                        klee_make_symbolic(&{dist_var_name}, sizeof({dist_var_name}), "{dist_var_name}");
                        klee_assume({dist_var_name} >= 1 && {dist_var_name} <= MAX_DISTANCE);
                        dist_matrix[{i}][{j}] = {dist_var_name};
                        dist_matrix[{j}][{i}] = {dist_var_name};
                        """
                    else:
                        if file_fixed_points is not None and dist_var_name in file_fixed_points:
                            dist_value = file_fixed_points[dist_var_name]
                        else:
                            dist_value = random.randint(1, max_distance)
                        fixed_points[dist_var_name] = dist_value
                        klee_specific += f"""
                        unsigned int {dist_var_name} = {dist_value};
                        dist_matrix[{i}][{j}] = {dist_var_name};
                        dist_matrix[{j}][{i}] = {dist_var_name};
                        """
                    count += 1

        for input_path in list_of_input_paths_to_exclude:
            # read the json file
            with open(input_path, "r") as f:
                test_cases = json.load(f)
            for _, test in test_cases.items():
                excluding_string = ""
                for key, value in test.items():
                    key = key.strip("'").strip()
                    if key in klee_inputs_names:
                        excluding_string += f"{key} != {value} || "
                excluding_string = excluding_string[:-4]  # removing the last "||"

                klee_specific += f"""
                klee_assume({excluding_string});
                """

        # 1. Triangle inequality violations - create suboptimal scenarios
        klee_specific += """
        // Force triangle inequality violations to stress test nearest neighbor
        """
        for i in range(num_cities):
            for j in range(num_cities):
                for k in range(num_cities):
                    if i != j and j != k and i != k:
                        klee_specific += f"""
                        klee_assume(dist_matrix[{i}][{j}] + dist_matrix[{j}][{k}] < dist_matrix[{i}][{k}] ||
                                   dist_matrix[{i}][{j}] + dist_matrix[{j}][{k}] >= dist_matrix[{i}][{k}]);
                        """

        # # 2. Distance extremes - force diverse distance patterns
        # klee_specific += f"""
        # // Ensure distance diversity to create challenging scenarios
        # unsigned int min_dist = MAX_DISTANCE;
        # unsigned int max_dist = 0;
        # """
        # for i in range(num_cities):
        #     for j in range(num_cities):
        #         if i < j:
        #             klee_specific += f"""
        #             if (dist_matrix[{i}][{j}] < min_dist) min_dist = dist_matrix[{i}][{j}];
        #             if (dist_matrix[{i}][{j}] > max_dist) max_dist = dist_matrix[{i}][{j}];
        #             """

        # klee_specific += f"""
        # klee_assume(min_dist < {max_distance//4} || max_dist > {3*max_distance//4});
        # """

        # 3. Nearest neighbor trap scenarios
        # klee_specific += """
        # // Create scenarios where nearest neighbor makes poor local choices
        # """
        # for i in range(num_cities):
        #     for j in range(num_cities):
        #         if i != j:
        #             next_j = (j + 1) % num_cities
        #             klee_specific += f"""
        #             klee_assume(dist_matrix[{i}][{j}] > dist_matrix[{i}][{next_j}] ||
        #                        dist_matrix[{i}][{j}] <= dist_matrix[{i}][{next_j}]);
        #             """

        # # 4. Clustering constraints - force scenarios with clear clusters
        # klee_specific += """
        # // Force clustering scenarios that stress nearest neighbor heuristic
        # """
        # klee_specific += f"""
        # unsigned int cluster_sum = 0;
        # unsigned int non_cluster_sum = 0;
        # """
        # # Sum distances within first half (potential cluster)
        # for i in range(num_cities//2):
        #     for j in range(i+1, num_cities//2):
        #         klee_specific += f"cluster_sum += dist_matrix[{i}][{j}];\n"

        # # Sum distances between first half and second half
        # for i in range(num_cities//2):
        #     for j in range(num_cities//2, num_cities):
        #         klee_specific += f"non_cluster_sum += dist_matrix[{i}][{j}];\n"

        # klee_specific += f"""
        # klee_assume(cluster_sum < non_cluster_sum || cluster_sum >= non_cluster_sum);
        # """

        # # 5. Symmetry breaking - create asymmetric patterns
        # klee_specific += """
        # // Create asymmetric distance patterns that confuse nearest neighbor
        # """
        # for i in range(num_cities):
        #     for j in range(num_cities):
        #         if i < j:
        #             next_i = (i + 1) % num_cities
        #             next_j = (j + 1) % num_cities
        #             klee_specific += f"""
        #             klee_assume(dist_matrix[{i}][{j}] != dist_matrix[{next_i}][{next_j}] ||
        #                        dist_matrix[{i}][{j}] == dist_matrix[{next_i}][{next_j}]);
        #             """

        klee_specific += programs["klee_specific2"]
        exec_specific = (
            programs["exec_specific1"]
            + programs["exec_specific2"]
            + programs["exec_specific3"]
        )
        if program_type == "klee":
            return {"program": header + klee_specific, "fixed_points": fixed_points}
        elif program_type == "exec":
            return header + exec_specific
