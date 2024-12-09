
import matplotlib.pyplot as plt
import scipy.stats
from itertools import combinations
import numpy as np

def create_boxplot(df, metric_column, ymin, ymax, group_column='Name', figsize=(10, 6)):
    """Create a compact boxplot with statistical test, median values, and spaced significance indicators"""
    
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
    
    # Create boxplot
    bp = ax.boxplot([df[df[group_column] == group][metric_column] for group in df[group_column].unique()],
                    patch_artist=True,
                    medianprops={'color': 'red', 'linewidth': 1.5},
                    boxprops={'color': 'black'},
                    whiskerprops={'color': 'black'},
                    capprops={'color': 'black'},
                    showfliers=False)
    
    # Color the boxes
    for patch, group_name in zip(bp['boxes'], df[group_column].unique()):
        base_name = group_name.split('*')[0]
        patch.set_facecolor(color_dict[base_name])
        patch.set_alpha(0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Get unique groups
    group_names = df[group_column].unique()
    
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
        group1_data = df[df[group_column] == group1_name][metric_column]
        group2_data = df[df[group_column] == group2_name][metric_column]
        
        group1_clean = remove_outliers(group1_data)
        group2_clean = remove_outliers(group2_data)
        
        stat, p_value = scipy.stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
        
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
        median = df[df[group_column] == group][metric_column].median()
        y_pos = ax.get_ylim()[1]
        
        # Add median value
        ax.text(i+1, y_pos*1.02, f'Median: {round(median, 2)}',
                horizontalalignment='center', fontsize=9)
        
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