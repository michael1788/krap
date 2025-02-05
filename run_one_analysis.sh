#!/bin/bash

# The specific folder name/date string you want to analyze
specific_date="24-12-13-14"

echo "Processing date: $specific_date"

# Run the Python script with the specific date
python plot_analysis.py --date "$specific_date" --summary_only --specific_savepath /Users/michaelmoret/Desktop/

echo "Analysis completed for: $specific_date"