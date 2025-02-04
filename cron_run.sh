#!/bin/bash

source ~/.zshrc
conda activate krap

YESTERDAY=$(date -v-1d +"%d-%m-%y")
TODAY=$(date +"%d-%m-%y")

# Run the Python script with today's date
python /Users/michaelmoret/tinyverse/krap/plot_analysis.py --date $TODAY

python /Users/michaelmoret/tinyverse/krap/plot_analysis.py --date $YESTERDAY