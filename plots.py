import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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