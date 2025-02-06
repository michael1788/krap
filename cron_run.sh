#!/bin/bash

source ~/.zshrc
conda activate krap

YESTERDAY=$(date -v-1d +"%y-%m-%d")
TODAY=$(date +"%y-%m-%d")

# Run the Python script with today's date
python /Users/michaelmoret/tinyverse/krap/plot_analysis.py --date $TODAY

python /Users/michaelmoret/tinyverse/krap/plot_analysis.py --date $YESTERDAY