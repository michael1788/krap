#!/bin/bash

# Function to format date as YY-MM-DD
format_date() {
    date -j -f "%Y-%m-%d" "$1" "+%y-%m-%d"
}

# Function to add days to a date
add_days() {
    date -j -v+"$2"d -f "%Y-%m-%d" "$1" "+%Y-%m-%d"
}

# Start date
start_date="2024-12-03"
current_date="$start_date"

# Get today's date in YYYY-MM-DD format
end_date=$(date "+%Y-%m-%d")

while true; do
    # Format current date as YY-MM-DD for the Python script
    formatted_date=$(format_date "$current_date")
    
    echo "Processing date: $formatted_date"
    
    # Run the Python script with the current date
    python plot_analysis.py --date "$formatted_date" --summary_only
    
    # Break if we've reached today
    if [ "$current_date" = "$end_date" ]; then
        break
    fi
    
    # Move to next day
    current_date=$(add_days "$current_date" 1)
done

echo "Analysis completed for all dates"