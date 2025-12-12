import os
import json
import time
import glob
import argparse
import subprocess
import tempfile
import textwrap
import numpy as np
from utils import *
from problems.programs_vbp import *
from problems.programs_TE import *
from problems.programs_knapsack import *
from problems.programs_max_weighted_matching import *
from problems.programs_arrow import *
import platform

DISABLE_FILE_REMOVAL = False
VBP_ITEM_USAGE = False #os.getenv("VBP_ITEM_USAGE", "True") == "True"
print(f"VBP_ITEM_USAGE: {VBP_ITEM_USAGE}")
class KleeRunner:
    def __init__(self, timeout_sec: int = 100):
        self.timeout_sec = timeout_sec

    def get_bitcode(self, c_filename: str) -> str:
        # Compile the program to LLVM bitcode
        bitcode = "test.bc"

        # Detect operating system and set appropriate compilation flags
        system = platform.system()

        if system == "Darwin":  # macOS
            # Automatically detect LLVM 16 clang path (compatible with KLEE)
            clang_path = "/opt/homebrew/Cellar/llvm@16/16.0.6_1/bin/clang"
            if not os.path.exists(clang_path):
                raise Exception("LLVM clang not found. Please ensure LLVM 16 is installed via Homebrew.")
            
            # Automatically detect KLEE include path
            klee_include_result = subprocess.run(["find", "/opt/homebrew", "-name", "klee.h", "-type", "f"], capture_output=True, text=True)
            if klee_include_result.returncode != 0 or not klee_include_result.stdout.strip():
                raise Exception("KLEE headers not found. Please ensure KLEE is installed.")
            klee_include_path = os.path.dirname(os.path.dirname(klee_include_result.stdout.strip().split('\n')[0]))
            
            compile_command = [
                clang_path,
                "-emit-llvm",
                "-g",
                "-c",
                "-O0",  # Add -O0 flag to avoid compilation crashes
                "-Xclang",
                "-disable-O0-optnone",
                f"-I{klee_include_path}",  # Add KLEE include path
                "-fno-stack-protector",  # Disable stack protector to avoid segmentation faults
                c_filename,
                "-o",
                bitcode,
            ]

            # Run the compilation
            compile_result = subprocess.run(compile_command, capture_output=True, text=True)
            if compile_result.returncode != 0:
                raise Exception(
                    f"Unable to compile generated program with error:\n{compile_result.stderr}"
                )

            return bitcode
        else:  # Linux
            compile_command = [
                "clang",
                "-emit-llvm",
                "-g",
                "-c",
                "-O0",  # Add -O0 flag to avoid compilation crashes
                "-Xclang",
                "-disable-O0-optnone",
                "--target=x86_64-unknown-linux-gnu",  # Set target triple explicitly
                "-I/home/linuxbrew/.linuxbrew/Cellar/klee/3.1_4/include",  # Add KLEE include path
                "-fno-stack-protector",  # Disable stack protector to avoid segmentation faults
                "-mno-red-zone",  # Disable red zone to avoid stack issues
                c_filename,
                "-o",
                bitcode,
            ]
            compile_result = subprocess.run(compile_command, capture_output=True, text=True)
            if compile_result.returncode != 0:
                raise Exception(
                    f"Unable to compile generated program with error:\n{compile_result.stderr}"
                )
            return bitcode

    def run_klee(self, c_filename: str) -> str:
        bitcode = None
        try:
            # Clean up any existing KLEE output directories
            if os.path.exists("klee-last"):
                subprocess.run(["rm", "-rf", "klee-last"])

            bitcode = self.get_bitcode(c_filename)
                        # Run KLEE on the bitcode file with deterministic search strategy
            # Automatically detect KLEE path
            klee_result = subprocess.run(["which", "klee"], capture_output=True, text=True)
            if klee_result.returncode != 0:
                raise Exception("KLEE not found in PATH. Please ensure KLEE is installed and available.")
            klee_path = klee_result.stdout.strip()

            klee_command = [
                klee_path,
                f"--max-time={self.timeout_sec}",
                "--watchdog",
                f"--max-solver-time={self.timeout_sec}",
                "--max-memory=10000",  # Add memory limit
                "--only-output-states-covering-new=false",  # Optimize state output
                "--rng-initial-seed=1",
                "--external-calls=all",
                "--search=random-path",  # Use random path to explore more diverse states
                "--use-batching-search=false",
                "--use-cex-cache=false",
                "--output-source=false",
                "--use-branch-cache=false",
                "--kdalloc",
                "--optimize=false",  # Disable optimizations
                "--max-tests=10000",  # Increase max number of tests
                "--max-instructions=1000000",  # Increase instruction limit
                bitcode,
            ]
            print(f"Running KLEE command: {' '.join(klee_command)}")
            klee_result = subprocess.run(klee_command, capture_output=True, text=True)

            if klee_result.returncode != 0:
                print("KLEE Error Output:")
                print(klee_result.stderr)
                raise Exception(f"KLEE execution failed with return code {klee_result.returncode}")

            print("KLEE Standard Output:")
            print(klee_result.stdout)

            # Check if any test cases were generated
            ktest_files = glob.glob("klee-last/*.ktest")
            print(f"Generated {len(ktest_files)} test cases")

            if not ktest_files:
                print("Warning: No test cases were generated!")
                print("KLEE Error Output:")
                print(klee_result.stderr)

            return klee_result.stdout
        finally:
            if not DISABLE_FILE_REMOVAL and bitcode is not None and os.path.exists(bitcode):
                os.remove(bitcode)

    def file_ready_check(
        self, docker_name, file_name, max_attempts=50, sleep_interval=5
    ):
        """Check if the file exists and is not empty."""
        for _ in range(max_attempts):
            check_command = ["docker", "exec", docker_name, "test", "-s", file_name]
            result = subprocess.run(check_command, capture_output=True)
            if result.returncode == 0:
                return True
            time.sleep(sleep_interval)
        return False

    def run_klee_from_string(self, program: str):
        """
        Runs KLEE given the program source code and returns the raw output
        from the KLEE tool as a string that contains the symbolic variable assignments.
        """
        # Write the program to a temporary file
        # program = "#include <klee/klee.h>\n" + program
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as temp:
            temp.write(program)
            temp_filename = temp.name

        self.run_klee(temp_filename)

    def run_klee_from_path(self, program_path: str):
        """
        Runs KLEE given the path to a C program file and returns the raw output
        from the KLEE tool as a string that contains the symbolic variable assignments.
        """
        self.run_klee(program_path)

    def extract_ktest(self, directory_path: str) -> str:
        """
        Extracts the symbolic variable assignments from the .ktest files
        generated by KLEE and returns them as a string.
        """
        try:
            # Glob pattern to match all .ktest files
            file_pattern = os.path.join(directory_path, "*.ktest")
            # Output file
            output_file = "output.txt"

            # Find all files in the directory matching the pattern
            ktest_files = glob.glob(file_pattern)

            script_content = textwrap.dedent(
                """
                import subprocess
                import os
                import glob

                directory_path = 'klee-last'
                file_pattern = os.path.join(directory_path, '*.ktest')
                output_file = 'output.txt'

                ktest_files = glob.glob(file_pattern)

                with open(output_file, 'w') as file:
                    batch = 50000
                    batch_files = []
                    for ktest_file in ktest_files:
                        batch_files.append(ktest_file)
                        batch -= 1
                        if batch != 0:
                            continue
                                                # Automatically detect ktest-tool path
                        ktest_tool_result = subprocess.run(["which", "ktest-tool"], capture_output=True, text=True)
                        if ktest_tool_result.returncode != 0:
                            raise Exception("ktest-tool not found in PATH. Please ensure KLEE is installed and available.")
                        ktest_tool_path = ktest_tool_result.stdout.strip()

                        command = [ktest_tool_path] + batch_files
                        try:
                            result = subprocess.run(command, capture_output=True, text=True, check=True)
                            file.write(f"{result.stdout}")
                        except subprocess.CalledProcessError as e:
                            file.write(f"Command failed for {ktest_file}: {e}")
                        batch = 50000
                        batch_files = []
                    if batch_files:
                        command = [ktest_tool_path] + batch_files
                        try:
                            result = subprocess.run(command, capture_output=True, text=True, check=True)
                            file.write(f"{result.stdout}")
                        except subprocess.CalledProcessError as e:
                            file.write(f"Command failed for {ktest_file}: {e}")
            """
            )

            # Write the script content to a temporary file
            script_path = "script.py"
            with open(script_path, "w") as script_file:
                script_file.write(script_content)

            # Run the script to process the .ktest files
            script_result = subprocess.run(
                ["python3", script_path], capture_output=True, text=True
            )
            if script_result.returncode != 0:
                raise Exception(
                    f"Failed to run the ktest-tool script with error:\n{script_result.stderr}"
                )

            # Read the output file
            with open(output_file, "r") as output:
                ktest_output = output.read()

        finally:
            if not DISABLE_FILE_REMOVAL and os.path.exists(script_path):
                os.remove(script_path)

        return ktest_output

    def run_klee_test_cases(self, program_string, ktest_dir, fixed_points=None):
        program_path, temp_filename = compile_program(program_string)
        self.extract_klee_test_cases(ktest_dir, fixed_points)
        # read args.save_name
        output_dict = {}
        with open(args.save_name, "r") as f:
            output_dict = json.load(f)

        for ktest_file, input_dict in output_dict.items():
            input_names = list(input_dict.keys())
            input_values = list(input_dict.values())
            # Run the compiled program with the extracted input value
            if program_path is not None:
                run_result = run_program(program_path, input_values)
            if not args.disable_print:
                print("=============================================")
                print(f"Extracted input names: {input_names}")
                print(f"Extracted input values: {input_values}")
                if program_path is not None:
                    # print(run_result.stdout)
                    # detect the FINAL_OUTPUT: from run_result.stdout
                    final_output = run_result.stdout.split("FINAL_OUTPUT: ")[1].strip()
                    # print(f"Final output: {final_output}")

        # Save the input values to a json file
        if os.path.exists(args.save_name):
            print(f"File {args.save_name} already exists. Overwriting the file.")
        with open(args.save_name, "w") as f:
            json.dump(output_dict, f)

        # Clean up
        if program_path is not None:
            if not DISABLE_FILE_REMOVAL and os.path.exists(temp_filename):
                os.remove(temp_filename)
            if not DISABLE_FILE_REMOVAL and os.path.exists(program_path):
                os.remove(program_path)

    def extract_klee_test_cases(self, ktest_dir, fixed_points=None, append_mode=False):
        # Find all .ktest files in the specified directory
        ktest_files = glob.glob(os.path.join(ktest_dir, "*.ktest"))
        err_files = glob.glob(os.path.join(ktest_dir, "*.err"))
        if fixed_points is not None:
            append_mode = True
        # Remove the test cases that have errors
        for err_file in err_files:
            ktest_files.remove(err_file.replace(".user.err", ".ktest"))
        if not args.disable_print:
            print(f"Found {len(ktest_files)} ktest files")

        output_dict = {}
        for ktest_file in ktest_files:
                        # Use ktest-tool to extract concrete inputs
            # Automatically detect ktest-tool path
            ktest_tool_result = subprocess.run(["which", "ktest-tool"], capture_output=True, text=True)
            if ktest_tool_result.returncode != 0:
                raise Exception("ktest-tool not found in PATH. Please ensure KLEE is installed and available.")
            ktest_tool_path = ktest_tool_result.stdout.strip()

            ktest_tool_cmd = [ktest_tool_path, ktest_file]
            result = subprocess.run(
                ktest_tool_cmd, capture_output=True, text=True, check=True
            )
            output = result.stdout

            # Parse the output to extract the input values
            lines = output.splitlines()
            input_values = []
            input_names = []
            for line in lines:
                if "object" in line and "name:" in line:
                    if "aux" in line:
                        continue
                    # Extract the input name
                    data_line = line.split(": ")[2]
                    input_name = data_line
                    if input_name[0] == "'":
                        input_name = input_name[1:-1]
                    input_names.append(input_name)
                if "object" in line and "int:" in line:
                    if len(input_names) <= len(input_values):
                        continue
                    # Extract the input value (assuming it's an integer)
                    data_line = line.split(": ")[2]
                    input_value = int(data_line)
                    # if data is read as bytes: below code can be used
                    # data_bytes = eval(data_line)
                    # input_value = int.from_bytes(data_bytes, 'little')
                    input_values.append(input_value)

            if not args.disable_print:
                print("=============================================")
                print(f"Extracted input names: {input_names}")
                print(f"Extracted input values: {input_values}")

            if fixed_points is not None:
                for key, value in fixed_points.items():
                    if key in input_names:
                        raise Exception(
                            f"Fixed point {key} already exists in the input names"
                        )
                    input_names.append(key)
                    input_values.append(value)
            test_dict = dict(zip(input_names, input_values))
            output_dict[str(ktest_file) + str(time.time())] = test_dict
        # save output_dict to a json file
        base_name, ext = os.path.splitext(args.save_name)
        timestamped_filename = f"{base_name}_{int(time.time())}{ext}"
        with open(timestamped_filename, "w") as f:
            json.dump(output_dict, f)

        if append_mode:
            if os.path.exists(args.save_name):
                with open(args.save_name, "r") as f:
                    existing_dict = json.load(f)
                    existing_dict.update(output_dict)
                    output_dict = existing_dict

        # Save the input values to a json file
        if os.path.exists(args.save_name):
            print(
                f"File {args.save_name} already exists. Overwriting the file with new extractions."
            )
        with open(args.save_name, "w") as f:
            json.dump(output_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KLEE on the heuristic program")
    parser.add_argument(
        "--save-name",
        type=str,
        help="Directory to save the end results",
        default="./klee_inputs.json",
    )
    parser.add_argument(
        "--disable-print", action="store_true", help="Disable printing the output"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="inputs: get the inputs, compare: compare two algorithms",
        default="inputs",
    )
    parser.add_argument("--algorithm1", type=str, help="Algorithm to run", default="FF")
    parser.add_argument(
        "--algorithm2", type=str, help="Algorithm to compare", default="FFDSum"
    )
    parser.add_argument(
        "--problem", type=str, help="Problem to run", default="TE"
    )
    parser.add_argument(
        "--problem-config-path",
        type=str,
        help="Problem configuration",
        default="config.json",
    )
    parser.add_argument(
        "--input-list-to-exclude",
        type=str,
        nargs="+",
        help="List of input paths to exclude",
        default=[],
    )
    parser.add_argument("--time-out", type=int, help="Time out for KLEE", default=100)
    parser.add_argument("--gap", type=int, help="Gap value to compare", default=1)
    parser.add_argument(
        "--input-path", type=str, help="Input to get the gap of", default="input.json"
    )
    parser.add_argument(
        "--max-num-scalable-klee-inputs",
        type=int,
        help="Maximum number of KLEE inputs to generate in scaling mode when task has inputs_scale",
        default=16,
    )
    parser.add_argument(
        "--max-num-scalable-rounds",
        type=int,
        help="Maximum number of round for KLEE generate in scaling mode when task == inputs_scale",
        default=4,
    )
    parser.add_argument(
        "--path-to-assigned-fixed-points",
        type=str,
        help="Path to the file that contains the fixed points",
        default=None,
    )
    parser.add_argument(
        "--use_num_bins",
        action="store_true",
        help="Use number of bins and loop over them",
        default=False,
    )
    args = parser.parse_args()
    INPUT_SEEKING_TASKS = [
        "get_gap",
        "get_heuristic",
        "update_gradient",
        "get_lagrangian",
        "get_relaxed_gap",
    ]
    # Initialize the KleeRunner
    runner = KleeRunner(args.time_out)
    if args.problem == "caching":
        runner.run_klee_from_string(
            generate_caching_program(
                args.problem_config_path,
                list_of_input_paths_to_exclude=args.input_list_to_exclude,
            )
        )
        runner.run_klee_test_cases(None, "klee-last")
    elif args.problem == "packet-scheduling":
        runner.run_klee_from_string(
            generate_packet_scheduling_program(
                args.problem_config_path,
                list_of_input_paths_to_exclude=args.input_list_to_exclude,
            )
        )
        runner.run_klee_test_cases(None, "klee-last")
    elif args.problem == "graph-coloring":
        runner.run_klee_from_string(
            generate_graph_coloring_program(
                args.problem_config_path,
                "klee",
                list_of_input_paths_to_exclude=args.input_list_to_exclude,
            )
        )
        if args.task == "inputs":
            runner.run_klee_test_cases(
                generate_graph_coloring_program(args.problem_config_path, "exec"),
                "klee-last",
            )
        elif args.task == "compare":
            runner.extract_klee_test_cases("klee-last")
            _, _ = compare_graph_coloring_programs(
                args.problem_config_path, args.save_name
            )
    elif args.problem in ["mwm", "knapsack", "TE", "vbp", "arrow" ]:
        if args.problem == "mwm":
            problem_class = MWMProblem(args.problem_config_path)
        elif args.problem == "knapsack":
            problem_class = KnapsackProblem(args.problem_config_path)
        elif args.problem == "TE":
            problem_class = TEProblem(args.problem_config_path)
        elif args.problem == "vbp":
            problem_class = VBPProblem(args.problem_config_path)
        elif args.problem == "arrow":
            problem_class = ArrowProblem(args.problem_config_path)

        if args.task == "inputs" and (args.problem != "vbp" or not VBP_ITEM_USAGE):
            output = problem_class.generate_heuristic_program(
                "klee",
                list_of_input_paths_to_exclude=args.input_list_to_exclude,
            )
            runner.run_klee_from_string(output["program"])
            runner.extract_klee_test_cases(
                "klee-last"
            )
        elif args.task == "inputs" and args.problem == "vbp" and VBP_ITEM_USAGE:
            if not DISABLE_FILE_REMOVAL and os.path.exists(args.save_name):
                os.remove(args.save_name)
            num_items = problem_class.num_items
            if args.use_num_bins:
                for num_bins in range(2, num_items - 2):
                    print(f"Num bins: {num_bins}")
                    output = problem_class.generate_heuristic_program(
                        "klee",
                        list_of_input_paths_to_exclude=args.input_list_to_exclude,
                        num_klee_inputs=args.max_num_scalable_klee_inputs,
                        num_optimal_bins=num_bins,
                    )
                    runner.run_klee_from_string(output["program"])
                    runner.extract_klee_test_cases(
                        "klee-last", fixed_points=output["fixed_points"], append_mode=True
                    )
            else:
                output = problem_class.generate_heuristic_program(
                    "klee",
                    list_of_input_paths_to_exclude=args.input_list_to_exclude,
                    num_klee_inputs=args.max_num_scalable_klee_inputs,
                )
                runner.run_klee_from_string(output["program"])
                runner.extract_klee_test_cases(
                    "klee-last", fixed_points=output["fixed_points"], append_mode=True
                )
        elif args.task == "inputs_scale" and (args.problem != "vbp" or not VBP_ITEM_USAGE):
            # remove args.save_name
            if not DISABLE_FILE_REMOVAL and os.path.exists(args.save_name):
                os.remove(args.save_name)
            num_rounds = min(max(
                int(
                    np.ceil(
                        problem_class.num_total_klee_inputs / args.max_num_scalable_klee_inputs
                    )
                ),
                1,
            ), args.max_num_scalable_rounds)

            for round_num in range(num_rounds):
                print(f"Round {(round_num + 1)}/{num_rounds}")
                output = problem_class.generate_heuristic_program(
                    "klee",
                    list_of_input_paths_to_exclude=args.input_list_to_exclude,
                    num_klee_inputs=args.max_num_scalable_klee_inputs,
                )
                runner.run_klee_from_string(output["program"])
                runner.extract_klee_test_cases(
                    "klee-last", fixed_points=output["fixed_points"], append_mode=True
                )
        elif args.task == "inputs_scale" and args.problem == "vbp" and VBP_ITEM_USAGE:
            # combining iteration over num_bins and num_klee_inputs
            num_items = problem_class.num_items
            if not DISABLE_FILE_REMOVAL and os.path.exists(args.save_name):
                os.remove(args.save_name)
            num_rounds = min(max(
                int(
                    np.ceil(
                        problem_class.num_total_klee_inputs / args.max_num_scalable_klee_inputs
                    )
                ),
                1,
            ), args.max_num_scalable_rounds)

            for round_num in range(num_rounds):
                print(f"Round {(round_num + 1)}/{num_rounds}")
                if args.use_num_bins:
                    for num_bins in range(2, num_items - 2):
                        print(f"Num bins: {num_bins}")
                        output = problem_class.generate_heuristic_program(
                            "klee",
                            list_of_input_paths_to_exclude=args.input_list_to_exclude,
                            num_klee_inputs=args.max_num_scalable_klee_inputs,
                            num_optimal_bins=num_bins,
                        )
                        runner.run_klee_from_string(output["program"])
                        runner.extract_klee_test_cases(
                            "klee-last", fixed_points=output["fixed_points"], append_mode=True
                        )
                else:
                    output = problem_class.generate_heuristic_program(
                        "klee",
                        list_of_input_paths_to_exclude=args.input_list_to_exclude,
                        num_klee_inputs=args.max_num_scalable_klee_inputs,
                    )
                    runner.run_klee_from_string(output["program"])
                    runner.extract_klee_test_cases(
                        "klee-last", fixed_points=output["fixed_points"], append_mode=True
                    )
        elif args.task == "inputs_scale_fixed_points" and (args.problem != "vbp" or not VBP_ITEM_USAGE):
            assert args.path_to_assigned_fixed_points is not None, "Path to the file that contains the fixed points is not provided"
            if not DISABLE_FILE_REMOVAL and os.path.exists(args.save_name):
                os.remove(args.save_name)
            output = problem_class.generate_heuristic_program(
                "klee",
                list_of_input_paths_to_exclude=args.input_list_to_exclude,
                num_klee_inputs=args.max_num_scalable_klee_inputs,
                path_to_assigned_fixed_points=args.path_to_assigned_fixed_points,
            )
            runner.run_klee_from_string(output["program"])
            runner.extract_klee_test_cases(
                "klee-last", fixed_points=output["fixed_points"], append_mode=True
            )
        elif args.task == "inputs_scale_fixed_points" and args.problem == "vbp" and VBP_ITEM_USAGE:
            assert args.path_to_assigned_fixed_points is not None, "Path to the file that contains the fixed points is not provided"
            if not DISABLE_FILE_REMOVAL and os.path.exists(args.save_name):
                os.remove(args.save_name)
            if args.use_num_bins:
                num_items = problem_class.num_items
                assigned_fixed_points = json.load(open(args.path_to_assigned_fixed_points, "r"))
                # find dimension with maximum sum of fixed points, items ending with _dim_k
                min_num_bins = 0
                sum_fixed_points = np.sum(list(assigned_fixed_points.values()))
                print(f"%%%%%% Sum of fixed points: {sum_fixed_points}")
                for key, value in assigned_fixed_points.items():
                    if key.endswith("_dim_0"):
                        min_num_bins += value / problem_class.bin_size
                min_num_bins = int(np.ceil(min_num_bins))
                num_max_bins = int(problem_class.num_items)
                max_bins = np.ceil(((num_max_bins - len(assigned_fixed_points)) * problem_class.bin_size + sum_fixed_points) / problem_class.bin_size)
                range_start = max(int(np.floor(problem_class.num_items / 2)), min_num_bins + 1)
                range_end = min(int(max_bins) + 1, num_max_bins - 1)
                print(f"XXXX num items {problem_class.num_items}, Min num bins: {range_start}, Max num bins: {range_end}")
                for num_bins in range(range_start, range_end):
                    print(f"Num bins: {num_bins}")
                    output = problem_class.generate_heuristic_program(
                        "klee",
                        list_of_input_paths_to_exclude=args.input_list_to_exclude,
                        num_klee_inputs=args.max_num_scalable_klee_inputs,
                        path_to_assigned_fixed_points=args.path_to_assigned_fixed_points,
                        num_optimal_bins=num_bins,
                    )
                    runner.run_klee_from_string(output["program"])
                    runner.extract_klee_test_cases(
                        "klee-last", fixed_points=output["fixed_points"], append_mode=True
                    )
            else:
                output = problem_class.generate_heuristic_program(
                    "klee",
                    list_of_input_paths_to_exclude=args.input_list_to_exclude,
                    num_klee_inputs=args.max_num_scalable_klee_inputs,
                    path_to_assigned_fixed_points=args.path_to_assigned_fixed_points,
                )
                runner.run_klee_from_string(output["program"])
                runner.extract_klee_test_cases(
                    "klee-last", fixed_points=output["fixed_points"], append_mode=True
                )
        elif args.task not in INPUT_SEEKING_TASKS:
            runner.run_klee_from_string(
                problem_class.generate_heuristic_program(
                    "klee", list_of_input_paths_to_exclude=args.input_list_to_exclude, num_klee_inputs=None
                )["program"]
            )

        elif args.task in INPUT_SEEKING_TASKS:
            input_dicts = {}
            try:
                with open(args.input_path, "r") as f:
                    input_dicts = json.load(f)
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"**************** Input path: {args.input_path}")
            if args.task == "get_gap":
                _ = problem_class.get_gaps(input_dicts, save_path=args.save_name)
            elif args.task == "get_heuristic":
                problem_class.get_heuristics(
                    input_dicts, save_path=args.save_name
                )
            elif args.task == "update_gradient":
                problem_class.update_lagrangian_gradients(
                    input_dicts, save_path=args.save_name
                )
            elif args.task == "get_lagrangian":
                problem_class.get_lagrangians(input_dicts, save_path=args.save_name)
            elif args.task == "get_relaxed_gap":
                problem_class.get_relaxed_gaps(
                    input_dicts, save_path=args.save_name
                )
    # Clean up, remove any of the klee-* directories
    if not DISABLE_FILE_REMOVAL and os.path.exists("klee-last"):
        os.system("rm -rf klee-*")
