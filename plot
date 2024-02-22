# Here we define the code as if they were separate .py files but in a single execution environment.

# core/data_utils.py
data_utils_code = """
import pandas as pd

def load_data(file_path):
    # Placeholder for function to load data from a file
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Placeholder for data preprocessing steps
    return data.dropna()  # Example: dropping missing values
"""

# core/plotting_utils.py
plotting_utils_code = """
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def plot_ts(list_ts, dict_colors, y_axis_left, y_axis_right, time_frame, grid, start_date, end_date, recession_dates=None):
    fig, ax1 = plt.subplots()

    for ts, color in zip(list_ts, dict_colors.values()):
        ax1.plot(ts.index, ts.values, color=color)

    if recession_dates:
        for start, end in recession_dates:
            ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='grey', alpha=0.5)

    if time_frame == 'mm_yy':
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pd.to_datetime(x).strftime('%m_%y')))
    elif time_frame == 'yyyy':
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pd.to_datetime(x).strftime('%Y')))

    ax1.set_ylabel(y_axis_left)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax1.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    ax1.grid(grid)
    ax1.tick_params(axis='x', rotation=0)

    ax2 = ax1.twinx()
    secondary_data = np.random.rand(len(list_ts[0])) * 1000
    ax2.plot(list_ts[0].index, secondary_data, color='red')
    ax2.set_ylabel(y_axis_right, color='red')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:.2f}'))
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()
"""

# apps/graph_builder/builder.py
graph_builder_code = """
def build_graph(data, additional_data=None):
    # Placeholder for building complex graphs
    pass
"""

# global_module/base_app.py
base_app_code = """
class BaseApp:
    def __init__(self, data):
        self.state = self.initialize_state(data)

    def initialize_state(self, data):
        return {'data': data}
"""

# Simulate importing the defined modules by executing the code.
exec(data_utils_code)
exec(plotting_utils_code)
exec(graph_builder_code)
exec(base_app_code)

# Let's simulate creating fake data and plotting as you would in a Jupyter notebook.
date_range = pd.date_range(start='1995-01-01', end='2023-01-01', freq='M')
fake_series1 = pd.Series(np.random.rand(len(date_range)), index=date_range)
fake_series2 = pd.Series(np.random.rand(len(date_range)), index=date_range)
list_ts = [fake_series1, fake_series2]
dict_colors = {'fake_series1': 'blue', 'fake_series2': 'green'}
recession_dates = [('2001-03-01', '2001-11-01'), ('2007-12-01', '2009-06-01')]

# Now we plot using the sophisticated plotting function with proper formatting.
plot_ts(
    list_ts=list_ts,
    dict_colors=dict_colors,
    y_axis_left='GDP Growth Rate (%)',
    y_axis_right='Currency Value ($)',
    time_frame='yyyy',
    grid=True,
    start_date='1995-01-01',
    end_date='2023-01-01',
    recession_dates=recession_dates
)