import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Define the plotting function
def plot_term_structure(df1=None, df2=None, col1=None, col2=None, 
                        axis_labels={'x': 'SPX 2m vol', 'y': 'SPX 12m/2m 100% moneyness'},
                        rgb_color=(0, 128, 0), cross_date=None, cross_col=None, 
                        cross_value=None, decimals=2, regression=False):
    """
    Function to plot term structure and include customizable features.
    
    Parameters:
    df1 (pd.DataFrame): First dataframe containing the first set of data.
    df2 (pd.DataFrame): Second dataframe containing the second set of data.
    col1 (str): The column name from df1 for the x-axis.
    col2 (str): The column name from df2 for the y-axis.
    axis_labels (dict): A dictionary to customize x and y axis labels.
    rgb_color (tuple): A tuple containing the RGB values of the color for the plot points.
    cross_date (str): Date string to mark a red cross on the plot.
    cross_col (str): The column name where cross_date value is located (used to identify the point).
    cross_value (float): The y-axis value where to place the red cross.
    decimals (int): Number of decimals to show on the axes.
    regression (bool): Whether to plot a linear regression line.
    """
    
    # Prepare the data for plotting
    if df2 is None and df1 is not None:
        # Single dataframe case
        x = df1[col1]
        y = df1[col2]
    else:
        # Two dataframe case
        x = df1[col1]
        y = df2[col2]
    
    # Plot the scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, color=[rgb_color], label=f'{axis_labels["x"]} vs. {axis_labels["y"]}')
    
    # Check if cross_date is provided to place a red cross
    if cross_date is not None:
        cross_x = df1[df1[cross_col] == cross_date][col1].values[0]
        ax.scatter(cross_x, cross_value, color='red', marker='x', s=100, label=cross_date)
    
    # Check if regression line is required
    if regression:
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y)
        y_pred = model.predict(x.values.reshape(-1, 1))
        ax.plot(x, y_pred, color='blue', label='Regression Line')
    
    # Customizing the axes
    ax.set_xlabel(f'{axis_labels["x"]} ({decimals} decimals)')
    ax.set_ylabel(f'{axis_labels["y"]} ({decimals} decimals)')
    
    # Customize number of decimal places on the axes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.{decimals}f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.{decimals}f}'))
    
    plt.legend()
    plt.show()

# Create two fake dataframes for testing
np.random.seed(42)
df1 = pd.DataFrame({
    'SPX_2m_vol': np.random.uniform(10, 70, 200),
    'Date': pd.date_range(start='2024-01-01', periods=200, freq='D')
})

df2 = pd.DataFrame({
    'SPX_12m_2m_moneyness': np.random.uniform(0.6, 1.6, 200)
})

# Example call to the function
plot_term_structure(df1=df1, df2=df2, col1='SPX_2m_vol', col2='SPX_12m_2m_moneyness', 
                    axis_labels={'x': 'SPX 2m vol', 'y': 'SPX 12m/2m 100% moneyness'},
                    rgb_color=(0, 100, 0), cross_date='2024-04-24', 
                    cross_col='Date', cross_value=1.2, decimals=2, regression=True)