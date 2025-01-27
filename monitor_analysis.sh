#!/bin/bash

# Path to the directory to monitor (update this path)
WATCH_DIR="/Users/michaelmoret/Library/CloudStorage/GoogleDrive-michael@externa.bio/.shortcut-targets-by-id/1BdUNsBjDh5Gee_76jCiKB1C_CwG0ercP/Pulling data"

# Path to store the last run state
STATE_FILE="$HOME/.krap_analysis_state"

# Function to get hash of all .txt files
get_files_hash() {
    find "$WATCH_DIR" -name "*.txt" -type f -exec stat -f "%m" {} \; | sort | md5
}

# Get current state of files
current_hash=$(get_files_hash)

# Check if state file exists and read last state
if [ -f "$STATE_FILE" ]; then
    last_hash=$(cat "$STATE_FILE")
else
    last_hash=""
fi

# Compare states
if [ "$current_hash" != "$last_hash" ]; then
    echo "New files detected, running analysis..."
    
    # Run your analysis script
    cd "$(dirname "$0")"  # Change to script directory
    ./run_plot_analysis.sh
    
    # Update state file
    echo "$current_hash" > "$STATE_FILE"
else
    echo "No new files detected. Skipping analysis."
fi