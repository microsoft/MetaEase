#!/bin/bash
# problem="tsp"
# baseline_max_time=3600
# for topology in "abilene" "b4-teavar" "swan"
# do
#     python paper.py --problem TE_DemandPinning_$topology --method HillClimbing --baseline-max-time $baseline_max_time
# done

python paper.py --problem TE_DemandPinning_abilene --method MetaEase
python paper.py --problem TE_DemandPinning_abilene --method HillClimbing --baseline-max-time 600 &
python paper.py --problem TE_DemandPinning_abilene --method SimulatedAnnealing --baseline-max-time 600 &
python paper.py --problem TE_DemandPinning_abilene --method Random --baseline-max-time 600 &
python paper.py --problem TE_DemandPinning_abilene --method GradientSampleBased --baseline-max-time 600 &
wait

# if [ "$problem" == "tsp" ]; then
#     for num_cities in 10 25 50
#     do
#         python paper.py --problem tsp_$num_cities
#     done

#     # run the following in parallel using background jobs
#     for method in Random SimulatedAnnealing HillClimbing
#     do
#         for num_cities in 10 25 50
#         do
#             python paper.py --problem tsp_$num_cities --method $method --baseline-max-time $baseline_max_time &
#         done
#     done

#     # wait for all background jobs to complete
#     wait
# fi

# elif [ "$problem" == "TE_DemandPinning" ]; then
#     for topology in "abilene" "b4-teavar" "swan"
#     do
#         python paper.py --problem TE_DemandPinning_$topology
#     done

#     # run the following in parallel using background jobs
#     for method in Random SimulatedAnnealing HillClimbing
#     do
#         for topology in "abilene" "b4-teavar" "swan"
#         do
#             python paper.py --problem TE_DemandPinning_$topology --method $method --baseline-max-time $baseline_max_time &
#         done
#     done

#     # wait for all background jobs to complete
#     wait
# fi
# fi