"""

macro_regimes_analysis.py

This script performs the following:
1. Generates macro regime dates and raw data.
2. Calculates performance series (returns and shifts).
3. Prepares merged DataFrames for plotting.
4. Creates distribution plots and scatter plots per regime.

generate_macro_regime_dates
generate_raw_data
calc_performance
prepare_plot_data
plot_regime_distributions
plot_regime_scatter
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install --upgrade seaborn pandas
import warnings

warnings.filterwarnings(
    "ignore",
    message="use_inf_as_na option is deprecated and will be removed in a future version",
    category=FutureWarning
)
######################################################################################
############################################ INPUTS ##################################
######################################################################################
colorpalette = {
    'Recession':  (215/250, 48/250, 48/250),  # pure red
    'Slowdown':   (239/250, 123/250, 90/250),  # orange-ish
    'Recovery':   (0/250, 145/250, 90/250),  # darker green
    'Expansion':  (86/255, 180/255, 192/255),  # pure blue
}

######################################################################################
###################### Fake data - Macro regime and vols levels ######################
######################################################################################
def generate_macro_regime_dates(start_date='2020-01-01', end_date='2021-12-31'):
    """
    Generate a DataFrame named macro_regime_dates where each date
    is assigned a macro regime based on date intervals.
    """
    import pandas as pd

    # Create all dates between start and end (daily frequency)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize an empty DataFrame
    macro_regime_dates = pd.DataFrame(index=dates, columns=['regime'])

    # Assign regimes by date boundaries
    macro_regime_dates.loc['2020-01-01':'2020-03-31', 'regime'] = 'Recession'
    macro_regime_dates.loc['2020-04-01':'2020-09-30', 'regime'] = 'Slowdown'
    macro_regime_dates.loc['2020-10-01':'2021-03-31', 'regime'] = 'Recovery'
    macro_regime_dates.loc['2021-04-01':'2021-12-31', 'regime'] = 'Expansion'
    
    return macro_regime_dates

def generate_raw_data(start_date='2020-01-01', end_date='2021-12-31'):
    """
    Generate a synthetic DataFrame named raw_data.
    The DataFrame is indexed by dates, with columns:
      - spot
      - vol_1m
      - rv_1m
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # Synthetic time series (replace with real data loading in production)
    spot = 100 + np.cumsum(np.random.normal(loc=0, scale=1, size=n))
    vol_1m = np.random.normal(loc=20, scale=3, size=n).clip(min=0)    # vol can't be negative
    rv_1m = np.random.normal(loc=15, scale=2, size=n).clip(min=0)     # realized vol can't be negative
    
    raw_data = pd.DataFrame(
        index=dates,
        data={
            'spot': spot,
            'vol_1m': vol_1m,
            'rv_1m': rv_1m
        }
    )
    return raw_data

######################################################################################
###################### Compute perf - Returns and vols shifts ########################
######################################################################################
def calc_performance(df, col_name='spot', performance_type='return'):
    """
    Calculate performance from a given price or series column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least one column (e.g. 'spot') 
        to calculate returns or shifts.
    col_name : str
        The column in 'df' on which to perform the calculation.
    performance_type : str
        - 'return':  calculates percentage change 
                     (df[col_name].pct_change())
        - 'shift':   calculates absolute shift 
                     (df[col_name].diff())
    
    Returns
    -------
    pd.Series
        The performance series as either a percentage return or a shift.
    """
    if performance_type == 'return':
        # Percentage return
        return df[col_name].pct_change().fillna(0)
    elif performance_type == 'shift':
        # Absolute shift
        return df[col_name].diff().fillna(0)
    else:
        raise ValueError("performance_type must be one of ['return', 'shift']")

######################################################################################
############################################ Prepare data ############################
######################################################################################
def prepare_plot_data(raw_data, macro_regime_dates, performance_series, performance_label):
    """
    Merges raw data with macro regime dates and performance metrics.
    
    Parameters:
    - raw_data (pd.DataFrame): The main data containing 'date' and other metrics.
    - macro_regime_dates (pd.DataFrame): DataFrame with 'date' and 'regime' columns.
    - performance_series (pd.Series): Series containing performance metrics.
    - performance_label (str): The name for the performance metric column.
    
    Returns:
    - pd.DataFrame: Merged and cleaned DataFrame.
    """
    # Merge on 'date' column with suffixes to avoid overlapping
    merged_df = pd.merge(raw_data, macro_regime_dates, on='date', how='inner', suffixes=('', '_macro'))
    
    # Insert the performance series
    merged_df[performance_label] = performance_series

    # Handle missing regimes without using inplace=True
    merged_df["regime"] = merged_df["regime"].fillna("Unknown")

    # Replace ±inf with NaN and handle downcasting
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

    # Drop rows where performance_label or regime is NaN
    merged_df.dropna(subset=[performance_label, 'regime'], inplace=True)

    return merged_df


######################################################################################
############################################ Plot  ###################################
######################################################################################

def plot_regime_distributions(merged_df, performance_label, colorpalette):
    """
    Create a 2×2 grid of histograms for each unique regime in merged_df['regime'].
    Remove top/right spines, place axes at (0,0), and use our colorpalette dict.
    """
    unique_regimes = merged_df['regime'].unique()
    num_regimes = len(unique_regimes)
    
    n_cols = 2
    n_rows = math.ceil(num_regimes / n_cols) if num_regimes > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    
    if num_regimes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, regime_name in enumerate(unique_regimes):
        subset = merged_df[merged_df['regime'] == regime_name]
        this_color = colorpalette.get(regime_name, 'gray')
        
        sns.histplot(
            data=subset,
            x=performance_label,
            ax=axes[i],
            kde=True,
            color=this_color,
            alpha=0.7
        )
        axes[i].set_title(f"{regime_name} - {performance_label} Distribution")
        axes[i].set_xlabel(performance_label)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_regime_scatter(merged_df, performance_label="Performance"):
    """
    Create a scatter plot of performance_label vs 'vol_1m', 
    colored by 'regime' using our colorpalette dict.
    Remove top/right spines and place axes at (0,0).
    """
    plt.figure(figsize=(8, 6))
    scatter_ax = sns.scatterplot(
        data=merged_df,
        y=performance_label,
        x='SX5E spot (1m return)',
        hue='regime',
        palette=colorpalette,
        alpha=0.8,
        edgecolor='black'
    )
    scatter_ax.set_title(f"{performance_label} vs. vol_1m by regime", pad=10)
    scatter_ax.set_ylabel(performance_label)
    scatter_ax.set_xlabel("SX5E spot (1m return)")

    # Remove top/right spines
    scatter_ax.spines['top'].set_visible(False)
    scatter_ax.spines['right'].set_visible(False)

    # Move left/bottom spines to x=0, y=0
    scatter_ax.spines['left'].set_position(('data', 0))
    scatter_ax.spines['bottom'].set_position(('data', 0))

    # Draw lines at x=0 and y=0
    scatter_ax.axhline(0, color='black', linewidth=1)
    scatter_ax.axvline(0, color='black', linewidth=1)

    plt.legend(title='Regime', loc='best')
    plt.show()


### MY SCATTER
def plot_regime_scatter_individual(merged_df, performance_label="Spot_Return"):
    """
    Creates a 2×2 grid of scatter plots (one per regime), ensuring:
      - SAME x,y axis limits across subplots (for easy comparison),
      - Spines at (0,0), with crosshair lines at x=0 and y=0,
      - Tick labels positioned on the LEFT (y-axis) and BOTTOM (x-axis) only,
      - No axis labels or subplot titles (tick marks remain),
      - Top/right spines removed,
      - Color from your 'colorpalette'.
    """

    fig, ax = plt.subplots(figsize=(5.5, 2), dpi=200)

    # Ensure the correct columns are used
    if 'SX5E spot (1m return)' not in merged_df.columns:
        raise ValueError("The column 'SX5E spot (1m return)' is missing in the DataFrame.")
    if performance_label not in merged_df.columns:
        raise ValueError(f"The column '{performance_label}' is missing in the DataFrame.")

    # Plot the scatter plot
    sns.scatterplot(
        data=merged_df,
        x='SX5E spot (1m return)',
        y=performance_label,
        hue='regime',
        palette=colorpalette,
        alpha=0.8,
        edgecolor='black'
    )

    # Adjust spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))

    # Add crosshair lines
    ax.axvline(0, linewidth=0.9, color='black', linestyle='--')
    ax.axhline(0, linewidth=0.9, color='black', linestyle='--')

    # Adjust tick directions and positions
    ax.tick_params(axis='y', direction='in', width=1.2)
    ax.tick_params(axis='x', direction='out', width=1.2)

    # Set tick font size
    fontsize = 9
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()
    plt.show()

def plot_regime_scatter_combined(merged_df, x_label="SX5E spot (1m return)", y_label="vol_1m", colorpalette=None):
    """
    Create a single scatter plot with all regimes combined.
    Ensures:
      - Axis spines cross at the global minimum (x_min, y_min),
      - Each regime is color-coded,
      - A legend indicates the regimes, placed above the plot in a single line,
      - No forced zero-origin, allowing natural data range scaling.
    """
    # Compute global x and y limits across all regimes
    x_min, x_max = merged_df[x_label].min(), merged_df[x_label].max()
    y_min, y_max = merged_df[y_label].min(), merged_df[y_label].max()

    # Add margins
    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.05 * (y_max - y_min)
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    # Scatter plot for all regimes, with hue and legend
    sns.scatterplot(
        data=merged_df,
        x=x_label,
        y=y_label,
        hue='regime',
        palette=colorpalette,  # Use your predefined color palette
        alpha=0.8,
        edgecolor='black'
    )

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Adjust spines for crossing axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', x_min))
    ax.spines['bottom'].set_position(('data', y_min))

    # Add crosshair lines
    ax.axvline(x_min, linewidth=0.9, color='black', linestyle='--')
    ax.axhline(y_min, linewidth=0.9, color='black', linestyle='--')

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Custom Legend Placement Above Plot
    legend = ax.legend(
        title="regime",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),  # Adjust the legend position above the plot
        ncol=len(merged_df['regime'].unique()),  # Arrange legend items in one row
        frameon=False,  # Remove the box line
        fontsize=9
    )

    plt.tight_layout()
    plt.show()

def plot_regime_scatter_individual(merged_df, performance_label="Performance"):
    """
    Create a 2×2 grid of scatter plots (one subplot per regime).
    Let Matplotlib auto-scale the axes (no forced zero origin).
    Keeps the rest of styling the same (color palette, figure size, etc.).
    """
    regimes = merged_df['regime'].unique()

    # 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, regime_name in enumerate(regimes):
        ax = axes[i]
        subset = merged_df[merged_df['regime'] == regime_name]

        # Use your color palette
        this_color = colorpalette.get(regime_name, (0.3, 0.3, 0.3))

        # Scatter plot
        ax.scatter(
            subset['SX5E spot (1m return)'],
            subset[performance_label],
            color=this_color,
            alpha=0.8,
            edgecolor='black'
        )

        # Title & labels
        ax.set_title(f"{regime_name}: {performance_label} vs. vol_1m", pad=10)
        ax.set_xlabel('SX5E spot (1m return)')
        ax.set_ylabel(performance_label)

        # Remove top/right spines only
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # IMPORTANT: No lines placing spines at (0,0)
        # No ax.axhline(0), no ax.axvline(0), no set_position(('data', 0))

    # If fewer than 4 regimes, remove extra subplots
    if len(regimes) < 4:
        for j in range(len(regimes), 4):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_regime_pdf_overlay(merged_df, performance_label="Performance", colorpalette=None):
    """
    Plot a single figure with overlapping PDF (KDE) curves for each regime,
    and an overall PDF representing all regimes combined. This allows comparing
    the distribution shapes among regimes and against the overall distribution.

    Parameters:
    - merged_df (pd.DataFrame): DataFrame containing 'regime' and the performance metric.
    - performance_label (str): The name of the performance metric column to plot.
    """
    # Get unique regimes
    regimes = merged_df['regime'].unique()
    
    # Initialize the plot
    plt.figure(figsize=(10, 7))
    
    # Plot each regime's PDF (KDE) on the same axes
    for regime_name in regimes:
        subset = merged_df[merged_df['regime'] == regime_name]
        color = colorpalette.get(regime_name, (0.3, 0.3, 0.3))  # Default to gray if not found
        
        # Clean NaNs or ±inf if necessary
        valid_data = subset[performance_label].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Plot the KDE for the current regime
        sns.kdeplot(
            valid_data, 
            color=color, 
            label=regime_name, 
            fill=False,    # Set to True if you prefer filled density areas
            linewidth=2, 
            alpha=0.8
        )
    
    # Plot the overall PDF for all regimes combined
    all_valid_data = merged_df[performance_label].replace([np.inf, -np.inf], np.nan).dropna()
    sns.kdeplot(
        all_valid_data, 
        color='black', 
        label='All Regimes', 
        linestyle='--',  # Dashed line to differentiate from individual regimes
        linewidth=2, 
        alpha=0.9
    )
    
    # Set plot titles and labels
    plt.title(f"Overlapping PDF: {performance_label} by Regime", pad=15, fontsize=16)
    plt.xlabel(performance_label, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # Customize spines for a cleaner look
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Optional: Move left/bottom spines to the center for a "cross" look
    # Uncomment the following lines if desired
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.axhline(0, color='black', linewidth=1)
    # ax.axvline(0, color='black', linewidth=1)
    
    # Add legend with title
    plt.legend(title='Regime', loc='best', fontsize=12, title_fontsize=12)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def plot_regime_pdf_individual(merged_df, performance_label="Performance", colorpalette=None):
    """
    Creates a 2×2 grid of subplots (one per regime).
    In each subplot:
      - Plot the entire dataset's PDF (in black),
      - Plot that regime's subset PDF (in color).
    This makes it easy to see how each regime's distribution
    compares to the full distribution.

    Parameters:
    - merged_df (pd.DataFrame): DataFrame containing 'regime' and the performance metric.
    - performance_label (str): The column name for the performance metric.
    - colorpalette (dict): Dictionary mapping regimes to colors.
    """
    if colorpalette is None:
        colorpalette = {
            'Recession':  (215/255, 48/255, 48/255),      # Pure red
            'Slowdown':   (239/255, 123/255, 90/255),     # Orange-ish
            'Recovery':   (0/255, 145/255, 90/255),       # Darker green
            'Expansion':  (86/255, 180/255, 192/255),     # Pure blue
        }

    # Prepare the entire dataset's distribution (all regimes)
    all_data = merged_df[performance_label].replace([np.inf, -np.inf], np.nan).dropna()

    # Identify the unique regimes
    regimes = merged_df['regime'].unique()
    n_regimes = len(regimes)
    rows = 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = axes.flatten()

    for i, regime_name in enumerate(regimes):
        ax = axes[i]

        # Entire dataset PDF (in black)
        sns.kdeplot(
            all_data,
            color='black',
            label='All Data',
            fill=False,
            linewidth=2,
            alpha=0.8,
            ax=ax
        )

        # Subset for this regime
        subset = merged_df[merged_df['regime'] == regime_name]
        subset_data = subset[performance_label].replace([np.inf, -np.inf], np.nan).dropna()

        # Plot regime's distribution in color
        color = colorpalette.get(regime_name, (0.3, 0.3, 0.3))
        sns.kdeplot(
            subset_data,
            color=color,
            label=regime_name,
            fill=False,
            linewidth=2,
            alpha=0.8,
            ax=ax
        )

        ax.set_title(f"{regime_name}", pad=10)
        ax.set_xlabel(performance_label)
        ax.set_ylabel("Density")

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move left/bottom spines to x=0, y=0
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))

        # Draw lines at x=0 and y=0
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)

        ax.legend(loc='best')

    # Remove unused subplots if any
    if n_regimes < rows * cols:
        for j in range(n_regimes, rows * cols):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def plot_regime_scatter_2x2(merged_df, performance_label="spot_return", colorpalette=None):
    """
    Create a 2×2 grid of scatter plots (one subplot per regime),
    ensuring:
      - SAME x, y axis limits across all plots,
      - Axes cross at the minimum x and y values,
      - Proper 2×2 grid layout.
    """
    if performance_label not in merged_df.columns:
        raise ValueError(f"'{performance_label}' column not found in DataFrame. Available columns: {merged_df.columns.tolist()}")
    
    if colorpalette is None:
        colorpalette = {
            'Recession':  (215/255, 48/255, 48/255),      # Pure red
            'Slowdown':   (239/255, 123/255, 90/255),     # Orange-ish
            'Recovery':   (0/255, 145/255, 90/255),       # Darker green
            'Expansion':  (86/255, 180/255, 192/255),     # Pure blue
        }

    # Identify unique regimes
    regimes = merged_df['regime'].unique()

    # Compute global x and y limits across all regimes
    y_min, y_max = merged_df[performance_label].min(), merged_df[performance_label].max()
    x_min, x_max = merged_df[performance_label].min(), merged_df[performance_label].max()

    # Add margins to limits for better visualization
    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.05 * (y_max - y_min)

    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    # Define grid dimensions
    rows = 2
    cols = 2

    # Create a 2×2 grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(regimes):
            # Get the subset for the current regime
            regime_name = regimes[i]
            subset = merged_df[merged_df['regime'] == regime_name]
            
            # Pick color from your palette
            this_color = colorpalette.get(regime_name, (0.3, 0.3, 0.3))
            
            # Scatter plot
            ax.scatter(
                subset[performance_label],
                subset[performance_label],  # Adjust if another axis requires a different column
                color=this_color,
                alpha=0.8,
                edgecolor='black'
            )
            
            # Set title and axis labels
            ax.set_title(f"{regime_name}", fontsize=10, pad=5)
            ax.set_ylabel(performance_label)
            ax.set_xlabel(performance_label)
            
            # Set consistent axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Adjust spines to cross at the minimum values
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('data', x_min))
            ax.spines['bottom'].set_position(('data', y_min))
            
            # Add crosshair lines at the minimum x and y values
            ax.axvline(x_min, linewidth=0.9, color='black', linestyle='--')
            ax.axhline(y_min, linewidth=0.9, color='black', linestyle='--')
            
            # Adjust tick parameters
            ax.tick_params(axis='x', direction='out', width=1.2, labelsize=8)
            ax.tick_params(axis='y', direction='out', width=1.2, labelsize=8)
        else:
            # Remove any unused axes
            fig.delaxes(ax)

    # Adjust layout for proper alignment
    plt.tight_layout()
    plt.show()