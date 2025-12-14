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
        python paper.py --problem ${problem} --method HillClimbing --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir}
        python paper.py --problem ${problem} --method SimulatedAnnealing --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir}
        python paper.py --problem ${problem} --method Random --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir}
        python paper.py --problem ${problem} --method GradientSampleBased --baseline-max-time ${METAEASE_TIME} --base-save-dir ${base_save_dir}

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

## Data for Figure 7 and Figure 8
run_problem_on_settings "TE_DemandPinning" "swan" "abilene"

## Data for Table 4
run_problem_on_settings "TE_DOTE" "AbileneDOTE"

## Data for Figure 9 and Table 5
run_problem_on_settings "TE_PoP" "swan"

## Data for Table 6
for topology in "swan" "abilene" "b4-teavar"; do
    python paper.py --problem TE_LLM_${topology} --method MetaEase --base-save-dir ../logs_final_TE_LLM
done


# Maximum Weight Matching (mwm) problem
## Data for Figure 12
topologies=("swan" "abilene" "b4-teavar")
run_problem_on_settings "mwm" "${topologies[@]}"