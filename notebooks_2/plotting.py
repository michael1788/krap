import matplotlib.pyplot as plt
import scipy.stats
from itertools import combinations
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np


def get_df_from_file(filepath, skip):
    # First, read all lines as strings
    raw_data = []
    with open(filepath, 'r') as f:
        # Skip the first skip lines
        for _ in range(skip):
            next(f)
        # Read remaining lines
        for line in f:
            fields = line.strip().split('\t')
            while len(fields) < 15:  # Pad with None if there are too few
                fields.append(None)
            raw_data.append(fields)

    # Convert to DataFrame with the same parameters you were using
    df = pd.DataFrame(raw_data)
    # Apply the same na_values treatment you had before
    df = df.replace(['', 'NA', 'null'], np.nan)

    return df

def replace_invalid_floats(df):
    """
    Replace '1.#IOe+000' and '-1.#IOe+000' values with numpy.nan in a DataFrame.
    Works on both string representations and actual float values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing invalid float values
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with invalid float values replaced by nan
    """
    # Create a copy to avoid modifying the original DataFrame
    df_cleaned = df.copy()
    
    # Replace string representations
    df_cleaned = df_cleaned.replace(['1.#IOe+000', '-1.#IOe+000'], np.nan)
    
    # For each numeric column, replace invalid float values
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # Check for invalid float values
        mask = df_cleaned[col].apply(lambda x: 
            isinstance(x, float) and (str(x) == '1.#IOe+000' or str(x) == '-1.#IOe+000'))
        # Replace with nan where mask is True
        df_cleaned.loc[mask, col] = np.nan
    
    return df_cleaned

def clean_single(df):
    # replace invalid floats
    df_clean = replace_invalid_floats(df)
    # set the header to the rows with RECORD
    df_clean.columns = df_clean.iloc[0]
    df_clean = df_clean[1:]
    df_clean.reset_index(drop=True, inplace=True)
    # removes row with the unit
    df_clean = df_clean.drop(index=0)
    df_clean.reset_index(drop=True, inplace=True)
    # remove empty rows at the end
    mask = pd.to_numeric(df_clean['RECORD'], errors='coerce').notnull()
    df_clean = df_clean[mask]
    df_clean.reset_index(drop=True, inplace=True)
    # save all the values as float 
    df_clean = df_clean.astype(float)

    return df_clean

def clean_triple(df):
    # replace invalid floats
    df_clean = replace_invalid_floats(df)
    # set the header to the rows with RECORD
    df_clean.columns = df_clean.iloc[0]
    df_clean = df_clean[1:]
    df_clean.reset_index(drop=True, inplace=True)
    # remove the first row
    df_clean = df_clean.drop(index=0)
    df_clean.reset_index(drop=True, inplace=True)
    # removes row with the unit
    df_clean = df_clean.drop(index=0)
    df_clean.reset_index(drop=True, inplace=True)
    # remove empty rows at the end
    mask = pd.to_numeric(df_clean['RECORD'], errors='coerce').notnull()
    df_clean = df_clean[mask]
    df_clean.reset_index(drop=True, inplace=True)
    # save all the values as float 
    df_clean = df_clean.astype(float)

    return df_clean

def format_number(number):
    """Format number: use scientific notation if more than 4 digits"""
    if abs(number) >= 10000:
        return f"{number:.2e}"
    return f"{number:.2f}"

def create_boxplot(df, metric_column, ymin, ymax, group_column='Name', figsize=(10, 6)):
    """Create a compact boxplot with statistical test, median values, spaced significance indicators,
    and individual data points (outliers removed)"""
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Function to remove outliers
    def remove_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    # Get unique base names and assign colors
    base_names = [name.split('*')[0] for name in df[group_column].unique()]
    unique_base_names = list(dict.fromkeys(base_names))  # Preserve order
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_base_names)))
    color_dict = dict(zip(unique_base_names, colors))
    
    # Create lists to store cleaned data for boxplot
    cleaned_data = []
    group_names = df[group_column].unique()
    
    for group in group_names:
        group_data = df[df[group_column] == group][metric_column]
        cleaned_group_data = remove_outliers(group_data)
        cleaned_data.append(cleaned_group_data)
    
    # Create boxplot with cleaned data
    bp = ax.boxplot(cleaned_data,
                    patch_artist=True,
                    medianprops={'color': 'red', 'linewidth': 1.5},
                    boxprops={'color': 'black'},
                    whiskerprops={'color': 'black'},
                    capprops={'color': 'black'},
                    showfliers=False)
    
    # Color the boxes
    for patch, group_name in zip(bp['boxes'], group_names):
        base_name = group_name.split('*')[0]
        patch.set_facecolor(color_dict[base_name])
        patch.set_alpha(0.7)
    
    # Add individual points with jitter (using cleaned data)
    for i, (group_name, group_data) in enumerate(zip(group_names, cleaned_data)):
        base_name = group_name.split('*')[0]
        
        # Create jitter
        x = np.random.normal(i + 1, 0.04, size=len(group_data))
        
        # Plot points with black edges
        ax.scatter(x, group_data, 
                  color=color_dict[base_name],
                  edgecolor='black',
                  linewidth=0.5,
                  alpha=0.5,
                  s=20,
                  zorder=2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add median values and set x-tick labels
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    
    # Dictionary to store significance markers and their colors for each group
    significance_dict = {group: [] for group in group_names}
    
    # List of base markers to use for different pairs
    base_markers = ['†', '‡', '§', '¶', '#', '@', '©', '®', '°', '∞', '∆', '∑', '√', '∫']
    marker_idx = 0
    
    # Define colors for different pairs (using a colorful qualitative colormap)
    pair_colors = plt.cm.Dark2(np.linspace(0, 1, len(list(combinations(group_names, 2)))))
    
    # Perform pairwise Mann-Whitney U tests
    pairs = list(combinations(group_names, 2))
    
    for pair_idx, (group1_name, group2_name) in enumerate(pairs):
        idx1 = list(group_names).index(group1_name)
        idx2 = list(group_names).index(group2_name)
        
        group1_data = cleaned_data[idx1]
        group2_data = cleaned_data[idx2]
        
        stat, p_value = scipy.stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        if p_value < 0.05:
            # Use next base marker in the list
            current_marker = base_markers[marker_idx % len(base_markers)]
            marker_idx += 1
            
            # Get color for this pair
            current_color = pair_colors[pair_idx]
            
            # Add appropriate number of markers based on p-value
            if p_value < 0.001:
                marker_with_significance = current_marker * 3
            elif p_value < 0.01:
                marker_with_significance = current_marker * 2
            else:
                marker_with_significance = current_marker
                
            # Add significance markers and colors to both groups
            significance_dict[group1_name].append((marker_with_significance, current_color))
            significance_dict[group2_name].append((marker_with_significance, current_color))
            
            print(f"Statistical significance between {group1_name} and {group2_name}: p = {p_value:.4f}")
    
    # Add median values and significance markers
    for i, group in enumerate(group_names):
        median = np.median(cleaned_data[i])
        y_pos = ax.get_ylim()[1]
        
        # Add median value with conditional scientific notation
        ax.text(i+1, y_pos*1.02, f'Median: {format_number(median)}',
                horizontalalignment='center', fontsize=8)
        
        # Add significance markers if they exist
        if significance_dict[group]:
            # Join all markers with spaces between them
            marker_text = '  '.join(marker for marker, _ in significance_dict[group])
            colors = [color for _, color in significance_dict[group]]
            
            # Calculate total width of the marker text
            total_width = len(marker_text) * 0.05  # Approximate width per character
            start_x = i + 1 - total_width/2
            
            # Place each marker with its color
            current_x = start_x
            for marker, color in significance_dict[group]:
                ax.text(current_x, y_pos*1.08, marker,
                       horizontalalignment='left', fontsize=19, color=color)
                current_x += len(marker) * 0.05 + 0.1  # Add space between markers
    
    plt.suptitle('')
    plt.title('')
    plt.xlabel('Experiment name', fontsize=10)
    plt.ylabel(metric_column.replace('_', ' ').title(), fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend for significance levels
    legend_text = 'Significance levels:\nOne symbol: p < 0.05\nTwo symbols: p < 0.01\nThree symbols: p < 0.001'
    plt.figtext(0.99, 0.01, legend_text, 
                horizontalalignment='right', 
                verticalalignment='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    return fig, ax