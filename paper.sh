# Function to run experiments for a problem across multiple settings
run_problem_on_settings() {
    local problem_base=$1  # e.g., "TE_DemandPinning"
    shift
    local settings=("$@")  # e.g., ("swan" "abilene")
    local base_save_dir=${BASE_SAVE_DIR:-"../logs_final_${problem_base}"}

    if [ ${#settings[@]} -eq 0 ]; then
        echo "Error: No settings provided for ${problem_base}"
        return 1
    fi

    echo "=========================================="
    echo "Running ${problem_base} on settings: ${settings[*]}"
    echo "=========================================="

    # Run experiments for each setting
    for setting in "${settings[@]}"; do
        local problem="${problem_base}_${setting}"
        echo ""
        echo "--- Processing ${problem} ---"
        echo "Setting: ${setting}"
        # Run MetaEase first and measure execution time
        echo "Running MetaEase to determine baseline time..."
        START_TIME=$(date +%s)
        python paper.py --problem ${problem} --method MetaEase --base-save-dir ${base_save_dir}
        END_TIME=$(date +%s)
        METAEASE_TIME=$((END_TIME - START_TIME))
        echo "MetaEase took ${METAEASE_TIME} seconds for ${problem}"

        # Run baseline methods with the same time limit
        echo "Running baseline methods with ${METAEASE_TIME} seconds time limit..."
        python paper.py --problem ${problem} --method HillClimbing --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir} &
        python paper.py --problem ${problem} --method SimulatedAnnealing --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir} &
        python paper.py --problem ${problem} --method Random --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir} &
        python paper.py --problem ${problem} --method GradientSampleBased --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir} &
        wait

        echo "Completed ${problem}"
    done

    # Generate experiment file and plots
    echo ""
    echo "--- Generating plots for ${problem_base} ---"
    cd ../scripts
    python generate_experiment_file.py --logs-dir ${base_save_dir} --output-name ${problem_base}
    python plot_methods.py --experiment_file ${problem_base}.txt --output_dir ../plots/${problem_base}
    cd ../src

    echo "=========================================="
    echo "Completed all experiments for ${problem_base}"
    echo "=========================================="
}

cd src
echo "Running experiments..."
# Traffic Engineering (TE) problem, small settings
# run_problem_on_settings "TE_DemandPinning" "swan" "abilene" "b4-teavar"
# run_problem_on_settings "TE_PoP" "swan" "abilene" "b4-teavar"
# run_problem_on_settings "TE_LLM" "swan" "abilene" "b4-teavar"
# run_problem_on_settings "TE_DOTE" "AbileneDOTE"

# Vector Bin Packing (vbp) problem
run_problem_on_settings "vbp_FFD" "8_1" # "10_1" "10_2" "15_1" "15_2" "20_1" "20_2"

# Knapsack problem
# run_problem_on_settings "knapsack" "20" "30" "40" "50"

# Maximum Weight Matching (mwm) problem
# run_problem_on_settings "mwm" "swan" "abilene" "b4-teavar"

# IP--Optical Network Optimization: Arrow Heuristic
# run_problem_on_settings "arrow" "IBM" "B4"


# larger settings
# run_problem_on_settings "TE_DemandPinning" "Cogentco" "Uninet2010"
# run_problem_on_settings "TE_PoP" "Cogentco" "Uninet2010"
# run_problem_on_settings "mwm" "Cogentco" "Uninet2010"