import tempfile
import subprocess

def compile_program(program_string):
    temp_filename = None
    if program_string is not None:
        # write to a .c file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as temp:
            temp.write(program_string)
            temp_filename = temp.name
        # compile with gcc
        program_path = './test'
        compile_command = ['gcc', temp_filename, '-o', program_path]
        compile_result = subprocess.run(compile_command, capture_output=True, text=True)
        if compile_result.returncode != 0:
            raise Exception(f"Unable to compile generated program with error:\n{compile_result.stderr}")
    else:
        program_path = None
    return program_path, temp_filename

def run_program(program_path, input_values):
    run_cmd = [program_path]
    run_cmd.extend(str(input_value) for input_value in input_values)
    run_result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
    return run_result