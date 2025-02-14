#!/bin/bash

# Add logging
exec 1> /Users/michaelmoret/tinyverse/krap/cron_log/log.log 2>&1

# Check if Google Drive is mounted
echo "$(date): Checking Google Drive mount..."
ls -la "/Users/michaelmoret/Library/CloudStorage/GoogleDrive-michael@externa.bio/" 

# Source profile and activate conda
source ~/.zshrc
source ~/.bash_profile  # Add this line
conda activate krap

# Wait for Google Drive to be fully mounted
MAX_WAIT=300  # 5 minutes
WAIT_INTERVAL=10
TOTAL_WAIT=0

while [ $TOTAL_WAIT -lt $MAX_WAIT ]; do
    if [ -d "/Users/michaelmoret/Library/CloudStorage/GoogleDrive-michael@externa.bio/.shortcut-targets-by-id/1BdUNsBjDh5Gee_76jCiKB1C_CwG0ercP" ]; then
        echo "$(date): Google Drive mounted successfully"
        break
    fi
    echo "$(date): Waiting for Google Drive to mount... ($TOTAL_WAIT seconds elapsed)"
    sleep $WAIT_INTERVAL
    TOTAL_WAIT=$((TOTAL_WAIT + WAIT_INTERVAL))
done

if [ $TOTAL_WAIT -ge $MAX_WAIT ]; then
    echo "$(date): ERROR - Google Drive failed to mount within $MAX_WAIT seconds"
    exit 1
fi

# Get dates
YESTERDAY=$(date -v-1d +"%y-%m-%d")
TODAY=$(date +"%y-%m-%d")

# Run the Python script with dates
python /Users/michaelmoret/tinyverse/krap/plot_analysis.py --date $TODAY
python /Users/michaelmoret/tinyverse/krap/plot_analysis.py --date $YESTERDAY