import argparse
from metaease import metaease_main
from random_sampling import random_sampling_main
from simulated_annealing import simulated_annealing_main
from hill_climbing import hill_climbing_main
from gap_sample_based import sample_based_gradient_main
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--base-save-dir", type=str, default="../logs_final")
    parser.add_argument("--method", type=str, default="MetaEase", choices=["MetaEase", "Random", "SimulatedAnnealing", "HillClimbing", "GradientSampleBased"])
    # MetaEase
    parser.add_argument("--klee-task", type=str, default="inputs_scale_fixed_points", choices=["inputs_scale_fixed_points", "inputs"])
    parser.add_argument("--baseline-max-time", type=float, default=3600, help="Maximum time in seconds for Random, Simulated Annealing and Hill Climbing")
    parser.add_argument("--baseline-repeat", type=int, default=10, help="Number of times to repeat the baseline method")
    # Random
    parser.add_argument("--num-random-samples", type=int, default=100, help="Number of random samples to generate for Random")
    # Simulated Annealing
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of iterations for Simulated Annealing and Hill Climbing")
    parser.add_argument("--SA-initial-temperature", type=float, default=1.0, help="Initial temperature for Simulated Annealing")
    parser.add_argument("--SA-cooling-rate", type=float, default=0.95, help="Cooling rate for Simulated Annealing")
    parser.add_argument("--SA-num-neighbors", type=int, default=1, help="Number of neighbors to evaluate per iteration for Simulated Annealing")
    # Hill Climbing
    parser.add_argument("--HC-num-neighbors", type=int, default=10, help="Number of neighbors to evaluate per iteration for Hill Climbing")
    parser.add_argument("--HC-step-size", type=float, default=0.1, help="Step size for Hill Climbing")
    args = parser.parse_args()

    if args.method == "MetaEase":
        if args.klee_task == "inputs_scale_fixed_points":
            print("Running MetaEase with partitioning. In each round, the partition demands are klee variables that get optimized, and the rest of the demands are frozen, i.e., in each round we optimize one partition at a time.")
            print("The first round of values are a random sample of preferred values if exist, otherwise a random sample of values.")
        elif args.klee_task == "inputs":
            print("Running MetaEase with klee on all variables.")
        else:
            sys.exit("Invalid klee task: " + args.klee_task)

        print("Running Full MetaEase Pipeline: This includes Partitioning, Klee, and Code-aware Gradient Ascent")
        metaease_main(args)
    elif args.method == "Random":
        print("Running Random Sampling")
        random_sampling_main(args, num_samples=args.num_random_samples, max_time=args.baseline_max_time)
    elif args.method == "SimulatedAnnealing":
        print("Running Simulated Annealing")
        for _ in range(args.baseline_repeat):
            simulated_annealing_main(args,
                                    num_iterations=args.num_iterations,
                                    initial_temperature=args.SA_initial_temperature,
                               cooling_rate=args.SA_cooling_rate,
                               num_neighbors=args.SA_num_neighbors,
                               max_time=args.baseline_max_time)
    elif args.method == "HillClimbing":
        print("Running Hill Climbing")
        for _ in range(args.baseline_repeat):
            hill_climbing_main(args,
                            num_iterations=args.num_iterations,
                            num_neighbors=args.HC_num_neighbors,
                            step_size=args.HC_step_size,
                            max_time=args.baseline_max_time)
    elif args.method == "GradientSampleBased":
        print("Running Gap Sample Based")
        for _ in range(1):
            sample_based_gradient_main(args,
                                    num_iterations=args.num_iterations,
                                    step_size=args.HC_step_size,
                                    max_time=args.baseline_max_time)
    else:
        sys.exit("Invalid method: " + args.method)