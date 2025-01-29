#!/bin/zsh -l

# Log file for debugging
LOG_FILE="$HOME/tinyverse/krap/wrapper.log"

# File to store processed files
PROCESSED_FILES="$HOME/.krap_processed_files"

# Directory to monitor
WATCH_DIR="/Users/michaelmoret/Library/CloudStorage/GoogleDrive-michael@externa.bio/.shortcut-targets-by-id/1BdUNsBjDh5Gee_76jCiKB1C_CwG0ercP/Pulling data"

# Cut-off date
CUTOFF_DATE="25-01-28"

# Log start time
echo "[$(date)] Starting script" >> "$LOG_FILE"

# Create processed files list if it doesn't exist
touch "$PROCESSED_FILES"

# Change to the correct directory
cd $HOME/tinyverse/krap

/bin/zsh -l << ENDSCRIPT
  # Initialize environment
  source ~/.zshrc
  conda activate krap
  
  # Process files
  while IFS= read -r -d '' file; do
    # Get absolute path
    absolute_path=\$(realpath "\$file")
    
    # Extract date from path
    date_dir=\$(basename "\$(dirname "\$(dirname "\$absolute_path")")")
    
    # Check if date is in correct format and after cutoff
    if [[ \$date_dir =~ ^[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
      # Compare dates (convert to comparable format YYMMDD)
      file_date=\$(echo \$date_dir | tr -d '-')
      cutoff_date=\$(echo $CUTOFF_DATE | tr -d '-')
      
      if [[ \$file_date < \$cutoff_date ]]; then
        echo "[$(date)] Skipping file from before cutoff date: \$absolute_path" >> "$LOG_FILE"
        continue
      fi
      
      # Check if file has been processed before
      if ! grep -Fxq "\$absolute_path" "$PROCESSED_FILES"; then
        echo "[$(date)] Found new file: \$absolute_path" >> "$LOG_FILE"
        echo "[$(date)] Running analysis for date: \$date_dir" >> "$LOG_FILE"
        
        # Run analysis
        if python plot_analysis.py --date "\$date_dir"; then
          # Mark as processed only if successful
          echo "\$absolute_path" >> "$PROCESSED_FILES"
          echo "[$(date)] Successfully processed \$absolute_path" >> "$LOG_FILE"
        else
          echo "[$(date)] Failed to process \$absolute_path" >> "$LOG_FILE"
        fi
      else
        echo "[$(date)] File already processed: \$absolute_path" >> "$LOG_FILE"
      fi
    else
      echo "[$(date)] Skipping file in invalid date directory: \$date_dir" >> "$LOG_FILE"
    fi
  done < <(find "$WATCH_DIR" -type f -name "*.txt" -print0)
ENDSCRIPT