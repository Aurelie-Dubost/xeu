import ipywidgets as widgets
from IPython.display import display, clear_output

# plot_functions.py

import matplotlib.pyplot as plt
import numpy as np

def plot_graph(ax, **params):
    print(f"plot_graph called with params: {params}")  # Debugging line
    param1 = params.get("param1", 1.0)
    param2 = params.get("param2", 1.0)
    color = params.get("color", "blue")
    
    x = np.linspace(0, 10, 100)
    y = param1 * np.sin(x) + param2 * np.cos(x)
    ax.plot(x, y, color=color)
    ax.set_title("Graph Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

def plot_histogram(ax, **params):
    """Plot a histogram using params for customization."""
    param1 = params.get("param1", 1.0)
    color = params.get("color", "blue")
    
    data = np.random.randn(1000) * param1
    ax.hist(data, bins=30, color=color, alpha=0.7)
    ax.set_title("Histogram Plot")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")


