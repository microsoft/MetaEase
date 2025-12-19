import os
import time
import json
from utils import run_program, compile_program
import concurrent.futures

DEBUG = False


class Problem:
    """
    Base class for all optimization problems in MetaEase.

    This class defines the interface that all problem implementations (TE, VBP, Knapsack, etc.)
    must follow. It provides both core computation methods (called during optimization) and
    batch processing methods (used for analysis and KLEE integration).

    Subclasses must implement all abstract methods to define problem-specific behavior.
    """

    def __init__(self, problem_config_path):
        """
        Initialize problem instance from configuration file.

        Args:
            problem_config_path: Path to JSON file containing problem configuration
        """
        # Use simple integers for counters (will be aggregated from worker processes)
        self.num_compute_heuristic_value_called = 0
        self.num_compute_optimal_value_called = 0
        self.problem_config_path = problem_config_path
        with open(problem_config_path, "r") as f:
            self.problem_config = json.load(f)

    def load_config(self):
        """Reload problem configuration from file."""
        with open(self.problem_config_path, "r") as f:
            self.problem_config = json.load(f)

    # ============================================================================
    # Core Computing Functions (called during optimization)
    # These are used extensively in opt_utils.py and the main optimization loop
    # ============================================================================

    def convert_input_dict_to_args(self, input_dict):
        """
        Convert input dictionary to problem-specific argument format.

        Used in: opt_utils.py (get_heuristic_and_optimal_values, get_relaxed_optimal_values),
                 all compute_* methods, and batch processing methods.

        Args:
            input_dict: Dictionary mapping variable names to values (e.g., {"demand_0_1": 10.5})

        Returns:
            args_dict: Problem-specific format (e.g., {"demands": {...}, "num_nodes": 5, ...})
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_optimal_value(self, args_dict):
        """
        Compute the optimal (benchmark) solution value.

        Used in: opt_utils.py (get_heuristic_and_optimal_values), run_utils.py (filtering),
                 get_gaps_process_input, and throughout the optimization loop.

        Must return a dict with keys:
            - "optimal_value": float, the optimal objective value
            - "all_vars": dict, all solution variables (including dual variables, aux vars, etc.)
            - "gradient": dict, optional gradient for Lagrangian (used in gradient ascent)

        Returns:
            dict: {"optimal_value": float, "all_vars": dict, "gradient": dict}
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_heuristic_value(self, args_dict):
        """
        Compute the heuristic solution value and code path.

        Used in: opt_utils.py (get_heuristic_and_optimal_values, get_heuristic_values),
                 run_utils.py (filtering), get_gaps_process_input, and throughout optimization.

        Must return a dict with keys:
            - "heuristic_value": float, the heuristic objective value
            - "all_vars": dict, heuristic solution variables
            - "code_path_num": str/int, unique identifier for the code path taken

        The code_path_num is crucial for path-aware gradient ascent - it ensures we stay
        within the same code path to avoid instability at non-differentiable boundaries.

        Returns:
            dict: {"heuristic_value": float, "all_vars": dict, "code_path_num": str/int}
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        """
        Compute the Lagrangian value for gradient-based optimization.

        Used in: opt_utils.py (get_relaxed_optimal_gradient), get_relaxed_gaps,
                 and during gradient ascent to compute relaxed gaps.

        The Lagrangian is: L = objective + sum(lambda_i * constraint_i)
        This allows us to compute gradients without re-solving the optimization problem.

        Args:
            args_dict: Problem-specific arguments
            give_relaxed_gap: If True, return simplified Lagrangian (faster)

        Returns:
            dict: {"lagrange": float, "constraints": dict}
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_lagrangian_gradient(self, args_dict):
        """
        Compute the gradient of the Lagrangian with respect to input variables.

        Used in: opt_utils.py (get_relaxed_optimal_gradient), update_anchor_input_values,
                 and throughout gradient ascent to compute how to update inputs.

        This is the key method for gradient-based optimization. The gradient tells us
        which direction to move inputs to maximize the gap between optimal and heuristic.

        Returns:
            dict: Mapping from variable names to gradient values
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def compute_relaxed_optimal_value(self, args_dict):
        """
        Compute the relaxed (continuous) optimal solution.

        Used in: opt_utils.py (get_relaxed_optimal_values, update_anchor_input_values),
                 get_relaxed_gaps, and during gradient ascent for faster computation.

        The relaxed version allows fractional solutions, making it differentiable and
        enabling gradient computation via Lagrangian duality.

        Returns:
            dict: {"relaxed_optimal_value": float, "relaxed_all_vars": dict}
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def get_thresholds(self, relaxed_all_vars):
        """
        Get min/max bounds (thresholds) for all variables.

        Used in: opt_utils.py (generate_block_samples, update_anchor_input_values),
                 run_utils.py (handle_fixed_variables), and throughout optimization
                 to ensure variables stay within valid ranges.

        Args:
            relaxed_all_vars: Dictionary of relaxed solution variables

        Returns:
            dict: Mapping from variable names to (min, max) tuples
        """
        raise NotImplementedError("get_thresholds must be implemented in a subclass")

    def get_decision_to_input_map(self, all_vars):
        """
        Map decision variables to their corresponding input variables.

        Used in: run_utils.py (identify_gap_contributing_variables) to determine which
                 input variables affect which decisions, helping identify variables that
                 contribute to the gap between optimal and heuristic solutions.

        Args:
            all_vars: Dictionary of all solution variables (optimal or heuristic)

        Returns:
            dict: Mapping from decision variable names to lists of input variable names
        """
        raise NotImplementedError("Must be implemented in a subclass")

    # ============================================================================
    # Batch Processing Functions (used for analysis and KLEE integration)
    # These methods process multiple inputs and optionally save results to files
    # ============================================================================

    def get_heuristics(self, input_dicts, save_path=None):
        """
        Compute heuristic values for a batch of inputs.

        Used in: run_klee.py (when task="get_heuristic") for batch evaluation of
                 KLEE-generated inputs. This is called from the command line interface
                 to evaluate multiple inputs without running the full optimization loop.

        Args:
            input_dicts: List of input dictionaries to evaluate
            save_path: Optional path to save results (saves heuristic values and code paths)

        Saves:
            - {save_path}: JSON array of heuristic values
            - {save_path}_code_path_nums.json: JSON array of code path numbers
        """
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
        """
        Compute Lagrangian values for a batch of inputs.

        Used in: run_klee.py (when task="get_lagrangian") for batch evaluation.
                 This is useful for analyzing how Lagrangian values vary across inputs.

        Args:
            input_dicts: List of input dictionaries to evaluate
            save_path: Optional path to save results
            give_relaxed_gap: If True, compute simplified Lagrangian (faster)

        Saves:
            - {save_path}: JSON array of Lagrangian values
            - {save_path}_{variable}_value.csv: CSV files for each input variable (DEBUG mode)
            - {save_path}_{constraint}_constraints.csv: CSV files for each constraint (DEBUG mode)
        """
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
            reformatted_constraints[key] = [
                constraint[key] for constraint in constraints
            ]

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
        """
        Compute Lagrangian gradients for a batch of inputs.

        Used in: run_klee.py (when task="update_gradient") for batch gradient computation.
                 This is useful for analyzing gradients across multiple inputs.

        Args:
            input_dicts: List of input dictionaries to evaluate
            save_path: Optional path to save results

        Saves:
            - {save_path}_gradients.json: JSON array of gradient dictionaries
        """
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
        """
        Compute relaxed gaps (relaxed_optimal - heuristic) for a batch of inputs.

        Used in: run_klee.py (when task="get_relaxed_gap") for batch evaluation.
                 The relaxed gap is faster to compute than the true gap since it uses
                 the continuous relaxation instead of solving the integer program.

        Args:
            input_dicts: List of input dictionaries to evaluate
            save_path: Optional path to save results

        Saves:
            - {save_path}: JSON array of relaxed gap values
            - {save_path}_relaxed_optimal_all_vars.json: Relaxed solution variables
            - Various CSV files in DEBUG mode (lagrange_minus_heuristic, etc.)
        """
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
            non_converged_relaxed_optimal = self.compute_lagrangian_value(args_dict)[
                "lagrange"
            ]
            non_convereged_relaxed_gaps.append(
                abs(non_converged_relaxed_optimal - heuristic)
            )

        if save_path:
            if os.path.exists(save_path):
                os.remove(save_path)
            with open(save_path, "w") as f:
                json.dump(relaxed_gaps, f)

            relaxed_optimal_all_vars_path = save_path.replace(
                ".json", "_relaxed_optimal_all_vars.json"
            )

            if os.path.exists(relaxed_optimal_all_vars_path):
                os.remove(relaxed_optimal_all_vars_path)

            with open(relaxed_optimal_all_vars_path, "w") as f:
                json.dump(relaxed_optimal_all_vars, f)

            if DEBUG:
                with open(
                    save_path.replace(".json", "_lagrange_minu_heuristic.csv"), "a"
                ) as f:
                    csv_row = (
                        "Value: "
                        + str(lagrange_minus_heuristic_values[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

                with open(
                    save_path.replace(".json", "_non_convereged_relaxed_gaps.csv"), "a"
                ) as f:
                    csv_row = (
                        "Value: "
                        + str(non_convereged_relaxed_gaps[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

                with open(
                    save_path.replace(".json", "_heuristic_values_in_relaxed_gaps.csv"),
                    "a",
                ) as f:
                    csv_row = (
                        "Value: "
                        + str(heuristic_values[0])
                        + ", UnixTime: "
                        + str(int(time.time()))
                        + "\n"
                    )
                    f.write(csv_row)

                with open(
                    save_path.replace(
                        ".json", "_relaxed_optimal_values_in_relaxed_gaps.csv"
                    ),
                    "a",
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
        """
        Compute gap (optimal - heuristic) for a single input.

        Used in:
            - get_gaps() (internally, for parallel processing)
            - random_sampling.py (evaluating random samples)
            - hill_climbing.py (evaluating neighbors)
            - simulated_annealing.py (evaluating candidate solutions)
            - gap_sample_based.py (evaluating gradient-based samples)
            - metaease.py (final evaluation of best sample)

        This is the core evaluation function used by all baseline methods and the
        final evaluation in MetaEase. It computes both optimal and heuristic solutions
        and returns the gap along with all necessary metadata.

        Args:
            input_dict: Dictionary of input variable values

        Returns:
            tuple: (gap, gradient, (optimal_value, heuristic_value), all_vars, code_path_num)
                - gap: float, absolute difference between optimal and heuristic
                - gradient: dict, gradient for optimization
                - (optimal_value, heuristic_value): tuple of objective values
                - all_vars: dict, optimal solution variables
                - code_path_num: str/int, heuristic code path identifier
        """
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
        """
        Compute gaps (optimal - heuristic) for a batch of inputs in parallel.

        Used in: run_klee.py (when task="get_gap") for batch evaluation of KLEE inputs.
                 This processes inputs in parallel using ProcessPoolExecutor for efficiency.

        Args:
            input_dicts: List of input dictionaries to evaluate
            save_path: Optional path to save results

        Returns:
            list: List of gap values (floats)

        Saves:
            - {save_path}: JSON array of gap values
            - {save_path}_gradients.json: JSON array of gradient dictionaries
            - {save_path}_optimal_vars.json: JSON array of optimal solution variables
            - {save_path}_code_path_nums.json: JSON array of code path numbers
            - {save_path}_optimal_values.json: JSON array of optimal objective values
            - {save_path}_heuristic_values.json: JSON array of heuristic objective values
        """
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
            print(f"Normalized gap: {gap / (optimal_values[max_gap_index] + 1e-6)}")

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

    # ============================================================================
    # KLEE Integration Functions (used for symbolic execution)
    # These generate C programs that KLEE can analyze to find path-representative inputs
    # ============================================================================

    def get_common_header(self, args_dict):
        """
        Generate common C header code for KLEE programs.

        Used in: run_klee.py (generate_heuristic_C_program) to create the header portion
                 of C programs that KLEE will analyze. This includes includes, defines,
                 and common data structures/functions shared across all code paths.

        Args:
            args_dict: Problem-specific arguments needed to generate header

        Returns:
            str: C code string for the header section
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def generate_heuristic_C_program(
        self, program_type, list_of_input_paths_to_exclude=[]
    ):
        """
        Generate C program for KLEE symbolic execution.

        Used in: run_klee.py to create C programs that KLEE analyzes to find inputs
                 that explore different code paths in the heuristic implementation.

        The generated program:
        1. Makes certain variables symbolic (using klee_make_symbolic)
        2. Adds constraints (using klee_assume) to exclude already-explored paths
        3. Executes the heuristic algorithm
        4. KLEE explores all feasible paths and generates one input per path

        Args:
            program_type: "klee" (for KLEE analysis) or "exec" (for execution)
            list_of_input_paths_to_exclude: List of JSON file paths containing inputs
                                            to exclude (already explored paths)

        Returns:
            dict: {"program": str (C code), "fixed_points": dict (non-symbolic variable values)}
                  or str (C code) if program_type == "exec"
        """
        raise NotImplementedError("Must be implemented in a subclass")
