from run import main
from common import get_problem_description

def metaease_main(args):
    problem_description = get_problem_description(args)
    main(problem_description, args.base_save_dir, args.klee_task)
