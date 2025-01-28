#!/bin/zsh

# Initialize conda
source ~/.zshrc
conda activate krap

# Directory to monitor
WATCH_DIR="/Users/michaelmoret/Library/CloudStorage/GoogleDrive-michael@externa.bio/.shortcut-targets-by-id/1BdUNsBjDh5Gee_76jCiKB1C_CwG0ercP/Pulling data"

# File to store processed files
PROCESSED_FILES="$HOME/.krap_processed_files"

# Log file
LOG_FILE="$HOME/tinyverse/krap/analysis.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Create processed files list if it doesn't exist
touch "$PROCESSED_FILES"

log "Starting file check..."

# Find all .txt files
while IFS= read -r -d '' file; do
    # Get full path
    full_path=$(realpath "$file")
    
    # Check if file has been processed
    if ! grep -Fxq "$full_path" "$PROCESSED_FILES"; then
        log "New file found: $full_path"
        
        # Get the date directory (parent of the file's directory)
        date_dir=$(basename "$(dirname "$(dirname "$full_path")")")
        
        log "Running analysis for date: $date_dir"
        
        # Run the analysis script with the date
        cd "$(dirname "$0")"
        python plot_analysis.py --date "$date_dir"
        
        # Check if the analysis was successful
        if [ $? -eq 0 ]; then
            # Mark file as processed only if analysis was successful
            echo "$full_path" >> "$PROCESSED_FILES"
            log "Analysis completed successfully for date: $date_dir"
        else
            log "Error: Analysis failed for date: $date_dir"
        fi
    fi
done < <(find "$WATCH_DIR" -type f -name "*.txt" -print0)

log "File check completed"
echo "----------------------------------------" >> "$LOG_FILE"