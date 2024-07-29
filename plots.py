import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
#pip install pyperclip Pillow
from io import BytesIO
from PIL import Image
from AppKit import NSPasteboard, NSPasteboardTypePNG, NSPasteboardTypeTIFF
from Foundation import NSData
import pyperclip

def create_bar_chart(df, columns, labels, colors, date_labels, title='SX5E Parameters (1-year %ile)', filename='plot.png', add_percentage_sign=False):
    """
    Create a bar chart for 3 categories
    Dict colors should be the same as date_labels 
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.  Ex: pd.df(data, index=date_labels) ; data = {'spot': [15, 50],'vol': [5, 8]}
    columns (list): List of column names to plot. Ex: ['spot', 'vol', 'term_structure', 'skew', 'convex', 'correlation']
    labels (list): List of labels for the columns. Ex: ['Spot', 'Vol 3m ATMF', 'Term structure (3m/6m)', 'Skew', 'Convex', 'Correlation']
    date_labels (list): List of three date_labels.  Ex: ['before', 'peak', 'now']
    filename (str): The name of the file to save the plot to. Ex: 'plot.png'
    add_percentage_sign (bool): Whether to add a percentage sign to the x-axis labels. Ex: Boolean
    colors (dict): Dictionary of RGB colors for the bars. Ex: colors = {'before': colorpalette['green'],'peak': colorpalette['red'],'now': colorpalette['blue']}

    Additional:
    colorpalette = {'blue': (0.65, 0.65, 0.82),'red': (0.98, 0.65, 0.65),'green': (0.9, 0.76, 0.65)} - style.py
    """

    # Reverse the columns and labels to fit the param presentation order
    columns = columns [::-1]
    labels = labels [::-1]
    
    # Extract the data for the specified date_labels
    data_first = df.loc[date_labels[0], columns]
    data_second = df.loc[date_labels[1], columns]
    data_third = df.loc[date_labels[2], columns]
    
    # Define bar width and positions
    bar_width = 0.2
    indices = range(len(columns))
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # Increase DPI for higher quality
    
    # Plot the bars in the desired order: first, second, thrid
    bars_first = ax.barh([i + 2 * bar_width for i in indices], data_first, bar_width, label=date_labels[0], color=colors[date_labels[0]])
    bars_second = ax.barh([i + bar_width for i in indices], data_second, bar_width, label=date_labels[1], color=colors[date_labels[1]])
    bars_third = ax.barh([i for i in indices], data_third, bar_width, label=date_labels[2], color=colors[date_labels[2]])

    # Set y-axis labels
    ax.set_yticks([i + bar_width for i in indices])
    ax.set_yticklabels(labels, fontsize=12)
    
    # Add data labels
    for bars in [bars_first, bars_second, bars_third]:
        for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            ax.text(width + 2 if width >= 100 else width, label_y, f'{width:.0f}%', va='center', ha='left', fontsize=10, color='black')  # Add padding

    # Define font properties
    font_properties = {'family': 'sans-serif', 'size': 12}

    # Custom legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, color=colors[date_labels[0]], label=date_labels[0]),
        Rectangle((0, 0), 1, 1, color=colors[date_labels[1]], label=date_labels[1]),
        Rectangle((0, 0), 1, 1, color=colors[date_labels[2]], label=date_labels[2])
    ]
    
    # Format x-axis labels to include percentage sign if the parameter is set to True
    if add_percentage_sign:
        ax.set_xticks(ax.get_xticks())  # Ensure the ticks are set first
        ax.set_xticklabels([f'{int(tick)}%' for tick in ax.get_xticks()], fontdict=font_properties)
        
    # Increase the space between the legend entries and move it above the title
    # Adjust the legend position and spacing
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, handletextpad=2.0, columnspacing=4.0)

    # Set chart title
    ax.set_title(title, pad=40)
    
    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()

    # Set axis limits: Add space after the x-axis for % labels
    ax.set_xlim(0, 100)  # Increase x-axis limit to provide more space
    
    # Set axis colors to black and increase width
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.label.set_color('black')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', colors='black', width=1.5)
    ax.tick_params(axis='y', colors='black', width=1.5)
    
    # Reduce space between categories
    plt.subplots_adjust(hspace=0.1)
    
    # Save the plot to a file with more padding to the right
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2)  # Increase DPI and add padding
    
    print(f"Plot saved to {filename}")
    
    # Display the chart
    plt.show()

def create_stats_chart(categories, min_values, max_values, avg_values, last_values, percentile_20, percentile_80, title, colors, add_percentage_sign=False, filename='plot.png'):
    """
    Create a custom chart.
    
    Parameters:
    categories (list): List of category names.
    min_values (list): List of min values for each category.
    max_values (list): List of max values for each category.
    avg_values (list): List of average values for each category.
    last_values (list): List of last values for each category.
    percentile_20 (list): List of 20th percentile values for each category.
    percentile_80 (list): List of 80th percentile values for each category.
    title (str): The title of the chart.
    add_percentage_sign (bool): Whether to add a percentage sign to the y-axis labels.
    filename (str): The name of the file to save the plot to.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    icon_width = 0.3  # Increase width for all icons

    # Plot 20th-80th percentile range as bars
    for i, category in enumerate(categories):
        ax.bar(i, percentile_80[i] - percentile_20[i], bottom=percentile_20[i], color=colors['percentile_range'], alpha=0.5, edgecolor='none', width=icon_width, label='20th-80th %ile' if i == 0 else "")

    # Plot min/max as horizontal lines
    for i, category in enumerate(categories):
        ax.plot([i - icon_width / 2, i + icon_width / 2], [min_values[i], min_values[i]], color=colors['min_max'], linewidth=3, label='Min/Max' if i == 0 else "")
        ax.plot([i - icon_width / 2, i + icon_width / 2], [max_values[i], max_values[i]], color=colors['min_max'], linewidth=3)
    
    # Plot average as triangles
    for i, category in enumerate(categories):
        ax.plot(i, avg_values[i], marker='^', color=colors['avg'], markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Avg' if i == 0 else "")
    
    # Plot last values as diamonds
    for i, category in enumerate(categories):
        ax.plot(i, last_values[i], marker='D', color=colors['last'], markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Last' if i == 0 else "")

    # Customization
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_title(title, pad=40)
    ax.axhline(0, color='black', linewidth=2.0)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', bottom=False)  # Remove tick marks on the x-axis

    # Format y-axis labels to include percentage sign if the parameter is set to True
    if add_percentage_sign:
        ax.set_yticks(np.arange(-1, 6, 1))  # Adjust y-ticks as necessary
        ax.set_yticklabels([f'{int(tick)}%' for tick in ax.get_yticks()])

    # Define font properties
    font_properties = {'family': 'sans-serif', 'size': 12}

    # Custom legend
    legend_elements = [
        Patch(facecolor=colors['percentile_range'], edgecolor=colors['percentile_range'], alpha=0.5, label='20th-80th %ile'),
        plt.Line2D([0], [1], color=colors['min_max'], linewidth=3, label='Min/Max'),
        plt.Line2D([0], [0], marker='^', color=colors['avg'], markeredgecolor='black', markeredgewidth=1.5, label='Avg', markersize=12, linestyle='none'),
        plt.Line2D([0], [0], marker='D', color=colors['last'], markeredgecolor='black', markeredgewidth=1.5, label='Last', markersize=12, linestyle='none')
    ]

    # Adjust the legend position and spacing
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False, handletextpad=2.0, columnspacing=4.0, prop=font_properties)

    # Set axis limits: Add space after the x-axis for % labels
    ax.set_xlim(-0.5, len(categories) - 0.5)  # Adjust x-axis limit to provide more space
    
    # Set axis colors to black and increase width
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)  # Increase the width of the x-axis
    ax.xaxis.label.set_color('black')
    ax.yaxis.set_ticks_position('left')  # Only ticks on the left side
    ax.tick_params(axis='x', colors='black', width=1.5, direction='out')
    ax.tick_params(axis='y', colors='black', width=1.5, direction='out')
    
    # Reduce space between categories
    plt.subplots_adjust(hspace=0.1)
    
    # Save the plot to a file with more padding to the right
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2)  # Increase DPI and add padding
    
    print(f"Plot saved to {filename}")
    
    # Display the chart
    plt.tight_layout()
    plt.show()

def plot_term_structure(df, title='CAC term structure remains flat on the short end', filename='term_structure.png', show_grid=True, colors=None):
    """
    Plot the term structure graph.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    title (str): The title of the chart.
    filename (str): The name of the file to save the plot to.
    show_grid (bool): Whether to show the grid on the plot.
    colors (dict): Dictionary of RGB colors for the plot elements.
    """
    if colors is None:
        colors = {
            '10th_90th': (0.5, 0.5, 0.5),
            'current': (0.0, 0.5, 0.0),
            '1w_ago': (1.0, 0.65, 0.0),
            'peak_stress': (1.0, 0.0, 0.0)
        }

    tenors = df.columns

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot 10th/90th percentile range
    ax.fill_between(tenors, df.loc['10th'], df.loc['90th'], color=colors['10th_90th'], alpha=0.5, label='1Y 10th/90th %-ile')

    # Plot current term structure
    ax.plot(tenors, df.loc['current'], color=colors['current'], label='CAC Term Structure (Current)', linewidth=2)

    # Plot one week ago
    ax.plot(tenors, df.loc['1w_ago'], color=colors['1w_ago'], linestyle='--', label='1W ago', linewidth=2)

    # Plot peak stress
    ax.plot(tenors, df.loc['peak_stress'], color=colors['peak_stress'], label='Peak stress', linewidth=2)

    # Customization
    ax.set_xlabel('Tenors', fontsize=12, fontweight='bold')
    ax.set_ylabel('ATM Implied Volatility', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=70)
    if show_grid:
        ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust x and y axis limits
    ax.set_xlim(tenors[0], tenors[-1])
    ax.set_ylim(min(df.min()) - 1, max(df.max()) + 1)
    
    # Set y-axis labels as percentages
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(tick)}%' for tick in y_ticks])

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['10th_90th'], alpha=0.5, linewidth=10, label='1Y 10th/90th %-ile'),
        plt.Line2D([0], [0], color=colors['current'], linewidth=2, label='CAC Term Structure (Current)'),
        plt.Line2D([0], [0], color=colors['1w_ago'], linestyle='--', linewidth=2, label='1W ago'),
        plt.Line2D([0], [0], color=colors['peak_stress'], linewidth=2, label='Peak stress')
    ]
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=40)
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False, handletextpad=1.0, columnspacing=4.0)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    print(f"Plot saved to {filename}")
    
    # Display the chart
    plt.show()