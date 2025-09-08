# tests to make sure that derive finds defenses in the expected number of
# attempts (overwhelming probability the entire derive process is intact)

# if trying having manually build the crate and not using the Dockerfile
#BINARY="./target/release/maybenot"
BINARY="maybenot"

# Function to run a test and verify the expected number of attempts
run_test() {
    local expected_attempts="$1"
    local config_file="$2"
    local seed="$3"
    
    echo "Running test expecting $expected_attempts attempts..."
    
    # Capture output
    local output
    output=$($BINARY derive -c "$config_file" -s "$seed" 2>&1)
    
    # Check if output contains the expected attempts
    if echo "$output" | grep -q "$expected_attempts attempts"; then
        echo "✓ Test passed: Found '$expected_attempts attempts' in output"
    else
        echo "✗ Test failed: Expected '$expected_attempts attempts' not found in output"
        echo "Actual output:"
        echo "$output"
        exit 1
    fi
}

printf "\nTUNED CIRCUIT FINGERPRINTING DEFENSES\n"
run_test 7 "experiments/circuit-fingerprinting/combo/combo-10.toml" "11431577435033622959-595-106"
run_test 42 "experiments/circuit-fingerprinting/combo/combo-10.toml" "11431577435033622959-195-21"

printf "\nEPHEMERAL PADDING-ONLY WF DEFENSES, INFINITE MODEL\n"
run_test 35 "experiments/wf-overview-table/eph-padding/infinite.toml" "11431577435033622959-562-2"
run_test 17 "experiments/wf-overview-table/eph-padding/infinite.toml" "11431577435033622959-808-1"

printf "\nEPHEMERAL PADDING-ONLY WF DEFENSES, BOTTLENECK MODEL\n"
run_test 47 "experiments/wf-overview-table/eph-padding/bottleneck.toml" "11431577435033622959-80-7"
run_test 15 "experiments/wf-overview-table/eph-padding/bottleneck.toml" "11431577435033622959-182-16"

printf "\nEPHEMERAL BLOCKING WF DEFENSES, INFINITE MODEL\n"
run_test 13 "experiments/wf-overview-table/eph-blocking/infinite.toml" "11431577435033622959-500-1210"
run_test 17 "experiments/wf-overview-table/eph-blocking/infinite.toml" "11431577435033622959-113-525"

printf "\nEPHEMERAL BLOCKING WF DEFENSES, BOTTLENECK MODEL\n"
run_test 51 "experiments/wf-overview-table/eph-blocking/bottleneck.toml" "11431577435033622959-322-138"
run_test 5 "experiments/wf-overview-table/eph-blocking/bottleneck.toml" "11431577435033622959-757-136"