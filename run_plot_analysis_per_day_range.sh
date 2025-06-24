#!/bin/bash

source ~/.zshrc
conda activate krap

# Start date
start_date="2025-02-06"
# End date (today)
end_date=$(date +"%Y-%m-%d")

# Convert to seconds since epoch for comparison
start_seconds=$(date -j -f "%Y-%m-%d" "$start_date" "+%s")
end_seconds=$(date -j -f "%Y-%m-%d" "$end_date" "+%s")

# Loop through each day
current_seconds=$start_seconds
while [ $current_seconds -le $end_seconds ]; do
    # Format the date as YY-MM-DD (25-01-02 format)
    current_date=$(date -j -f "%s" "$current_seconds" "+%y-%m-%d")
    
    echo "Processing date: $current_date"
    
    # Run the Python script with the current date - fixed argument passing
    python plot_analysis.py --date "$current_date" \
                            --only_full_pptx_per_day
    
    # Move to the next day (add 86400 seconds = 1 day)
    current_seconds=$((current_seconds + 86400))
done

echo "All dates processed successfully!"