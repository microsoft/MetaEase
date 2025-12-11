import os
import time
import json
from utils import run_program, compile_program
import concurrent.futures
DEBUG = False

class Problem:
    def __init__(self, problem_config_path):
        # Use simple integers for counters (will be aggregated from worker processes)
        self.num_compute_heuristic_value_called = 0
        self.num_compute_optimal_value_called = 0
        self.problem_config_path = problem_config_path
        with open(problem_config_path, "r") as f:
            self.problem_config = json.load(f)

    def load_config(self):
        with open(self.problem_config_path, "r") as f:
            self.problem_config = json.load(f)

    # Computing functions, no saving
    def convert_input_dict_to_args(self, input_dict):
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_optimal_value(self, args_dict):
        # must return a dict with keys "optimal_value" and "all_vars"
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_heuristic_value(self, args_dict):
        # must return a dict with keys "heuristic_value", "all_vars", and "code_path_num"
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_lagrangian_gradient(self, args_dict):
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_relaxed_optimal_value(self, args_dict):
        raise NotImplementedError("Must be implemented in a subclass")

    def get_thresholds(self, relaxed_all_vars):
        raise NotImplementedError("get_thresholds must be implemented in a subclass")

    def is_input_feasible(self, input_dict):
        raise NotImplementedError("Must be implemented in a subclass")

    def get_decision_to_input_map(self, all_vars):
        raise NotImplementedError("Must be implemented in a subclass")

    # Saving functions, fixed API, works on batch of inputs
    def get_heuristics(self, input_dicts, save_path=None):
        heuristic_values = []
        code_path_nums = []
        for input_dict in input_dicts:
            args_dict = self.convert_input_dict_to_args(input_dict)
            out_dict = self.compute_heuristic_value(args_dict)
            # gap is the absolute value of diff num colors
            heuristic_values.append(out_dict["heuristic_value"])
            code_path_nums.append(out_dict["code_path_num"])
        if DEBUG:
            print(
                f"Max heuristic: {max(heuristic_values)}, Min heuristic: {min(heuristic_values)}"
            )
        # print(f"Code path nums: {code_path_nums}")
        if save_path:
            code_path_nums_path = save_path.replace(".json", "_code_path_nums.json")

            # Remove existing files before writing
            if os.path.exists(save_path):
                os.remove(save_path)
            if os.path.exists(code_path_nums_path):
                os.remove(code_path_nums_path)

            # Open and lock the file while writing
            with open(code_path_nums_path, "w") as f:
                json.dump(code_path_nums, f)

            with open(save_path, "w") as f:
                json.dump(heuristic_values, f)

    def get_lagrangians(self, input_dicts, save_path=None, give_relaxed_gap=False):
        lagrange_values = []
        constraints = []
        for input_dict in input_dicts:
            args_dict = self.convert_input_dict_to_args(input_dict)
            lagrange_dict = self.compute_lagrangian_value(
                args_dict, give_relaxed_gap=give_relaxed_gap
            )
            # constraint is a dict with key as lambda name and value as the constraint value
            constraints.append(lagrange_dict["constraints"])
            lagrange_values.append(lagrange_dict["lagrange"])

        reformatted_constraints = {}
        constraint_keys = constraints[0].keys()
        for key in constraint_keys:
            reformatted_constraints[key] = [constraint[key] for constraint in constraints]

        if os.path.exists(save_path):
            os.remove(save_path)

        if save_path:
            with open(save_path, "w") as f:
                json.dump(lagrange_values, f)

            if DEBUG:
                for key in input_dicts[0].keys():
                    with open(
                        save_path.replace(".json", f"_{key}_value.csv"), "a"
                    ) as f:
                        csv_row = (
                            "Value: "
                            + str(input_dicts[0][key])
                            + ", UnixTime: "
                            + str(int(time.time()))
                            + "\n"
                        )
                        f.write(csv_row)

                for key in constraint_keys:
                    with open(
                        save_path.replace(".json", f"_{key}_constraints.csv"), "a"
                    ) as f:
                        csv_row = (
                            "Value: "
                            + str(reformatted_constraints[key][0])
                            + ", UnixTime: "
                            + str(int(time.time()))
                            + "\n"
                        )
                        f.write(csv_row) 


    def update_lagrangian_gradients(self, input_dicts, save_path=None):
        gradients = []
        for input_dict in input_dicts:
            args_dict = self.convert_input_dict_to_args(input_dict)
            optimal_gradient = self.compute_lagrangian_gradient(args_dict)
            gradients.append(optimal_gradient)

        if save_path:
            gradient_path = save_path.replace(".json", "_gradients.json")

            if os.path.exists(gradient_path):
                os.remove(gradient_path)

            with open(gradient_path, "w") as f:
                json.dump(gradients, f)

    def get_relaxed_gaps(self, input_dicts, save_path=False):
        relaxed_gaps = []
        non_convereged_relaxed_gaps = []
        lagrange_minus_heuristic_values = []
        heuristic_values = []
        relaxed_optimal_values = []
        for input_dict in input_dicts:
            args_dict = self.convert_input_dict_to_args(input_dict)
            relaxed_optimal_sol = self.compute_relaxed_optimal_value(args_dict)
            relaxed_optimal = relaxed_optimal_sol["relaxed_optimal_value"]
            relaxed_optimal_all_vars = relaxed_optimal_sol["relaxed_all_vars"]
            out_dict = self.compute_heuristic_value(args_dict)
            heuristic = out_dict["heuristic_value"]
            if DEBUG:
                print(f"Relaxed optimal: {relaxed_optimal}, Heuristic: {heuristic}")
            relaxed_gap = abs(relaxed_optimal - heuristic)
            relaxed_gaps.append(relaxed_gap)
            heuristic_values.append(heuristic)
            relaxed_optimal_values.append(relaxed_optimal)
            lagrange_minus_heuristic_values.append(relaxed_optimal - heuristic)
            non_converged_relaxed_optimal = self.compute_lagrangian_value(args_dict)["lagrange"]
            non_convereged_relaxed_gaps.append(abs(non_converged_relaxed_optimal - heuristic))

        if save_path:
            if os.path.exists(save_path):
                os.remove(save_path)
            with open(save_path, "w") as f:
                json.dump(relaxed_gaps, f)

            relaxed_optimal_all_vars_path = save_path.replace(".json", "_relaxed_optimal_all_vars.json")

            if os.path.exists(relaxed_optimal_all_vars_path):
                os.remove(relaxed_optimal_all_vars_path)

            with open(relaxed_optimal_all_vars_path, "w") as f:
                json.dump(relaxed_optimal_all_vars, f)

            if DEBUG:
                with open(save_path.replace(".json", "_lagrange_minu_heuristic.csv"), "a"
                ) as f:
                    csv_row = (
                        "Value: "
                        + str(lagrange_minus_heuristic_values[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

                with open(save_path.replace(".json", "_non_convereged_relaxed_gaps.csv"), "a") as f:
                    csv_row = (
                        "Value: "
                        + str(non_convereged_relaxed_gaps[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

                with open(save_path.replace(".json", "_heuristic_values_in_relaxed_gaps.csv"), "a") as f:
                    csv_row = (
                        "Value: "
                        + str(heuristic_values[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

                with open(
                    save_path.replace(".json", "_relaxed_optimal_values_in_relaxed_gaps.csv"), "a"
                ) as f:
                    csv_row = (
                        "Value: "
                        + str(relaxed_optimal_values[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

    def get_gaps_process_input(self, input_dict):
        args_dict = self.convert_input_dict_to_args(input_dict)
        optimal_dict = self.compute_optimal_value(args_dict)
        gradient = optimal_dict["gradient"]
        optimal_value = optimal_dict["optimal_value"]
        all_vars = optimal_dict["all_vars"]

        heuristic_dict = self.compute_heuristic_value(args_dict)
        heuristic_value = heuristic_dict["heuristic_value"]
        code_path_num = heuristic_dict["code_path_num"]

        gap = abs(optimal_value - heuristic_value)

        return gap, gradient, (optimal_value, heuristic_value), all_vars, code_path_num


    def get_gaps(self, input_dicts, save_path=None):
        gaps = []
        gradients = []
        all_optimal_vars = []
        optimal_values = []
        heuristic_values = []
        code_path_nums = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.get_gaps_process_input, input_dicts)

        for gap, gradient, opt_heur_values, all_vars, code_path_num in results:
            gaps.append(gap)
            gradients.append(gradient)
            optimal_values.append(opt_heur_values[0])
            heuristic_values.append(opt_heur_values[1])
            all_optimal_vars.append(all_vars)
            code_path_nums.append(code_path_num)

        # Print the max gap and the corresponding optimal and greedy values
        if DEBUG:
            gap = max(gaps)
            max_gap_index = gaps.index(gap)
            print(f"Max gap: {gap} in {len(gaps)} samples")
            print(f"Optimal value: {optimal_values[max_gap_index]}")
            print(f"Greedy value: {heuristic_values[max_gap_index]}")
            print(
                f"Normalized gap: {gap / (optimal_values[max_gap_index] + 1e-6)}"
            )

        if save_path:
            if os.path.exists(save_path):
                os.remove(save_path)
            with open(save_path, "w") as f:
                json.dump(gaps, f)

            gradient_path = save_path.replace(".json", "_gradients.json")
            if os.path.exists(gradient_path):
                os.remove(gradient_path)
            with open(gradient_path, "w") as f:
                json.dump(gradients, f)

            optimal_vars_path = save_path.replace(".json", "_optimal_vars.json")
            if os.path.exists(optimal_vars_path):
                os.remove(optimal_vars_path)

            with open(optimal_vars_path, "w") as f:
                json.dump(all_optimal_vars, f)

            code_path_nums_path = save_path.replace(".json", "_code_path_nums.json")
            if os.path.exists(code_path_nums_path):
                os.remove(code_path_nums_path)

            with open(code_path_nums_path, "w") as f:
                json.dump(code_path_nums, f)

            optimal_values_path = save_path.replace(".json", "_optimal_values.json")
            if os.path.exists(optimal_values_path):
                os.remove(optimal_values_path)
            with open(optimal_values_path, "w") as f:
                json.dump(optimal_values, f)

            heuristic_values_path = save_path.replace(".json", "_heuristic_values.json")
            if os.path.exists(heuristic_values_path):
                os.remove(heuristic_values_path)

            with open(heuristic_values_path, "w") as f:
                json.dump(heuristic_values, f)

        return gaps


    # Klee related functions
    def get_common_header(self, args_dict):
        raise NotImplementedError("Must be implemented in a subclass")

    def generate_heuristic_program(
        self, program_type, list_of_input_paths_to_exclude=[]
    ):
        raise NotImplementedError("Must be implemented in a subclass")
