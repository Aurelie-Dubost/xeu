import ipywidgets as widgets
from IPython.display import display

# Change the size of the existing output_scatter_indices to fixed dimensions
output_scatter_indices_plot.layout = widgets.Layout(width='640px', height='400px') # Adjusted height
output_stats_plot.layout = widgets.Layout(width='640px', height='400px') # Adjusted height

# Create the Layout
box = widgets.VBox([
    # Top row with udl_indices_widget, start_date_widget, and end_date_widget
    widgets.HBox([
        widgets.VBox([udl_indices_widget]), # Top Left
        
        # Top Right
        widgets.VBox([
            start_date_widget,
            end_date_widget,
        ], layout=widgets.Layout(margin='0 5px 0 0')), # Adjust the margin for spacing
    ], layout=widgets.Layout(justify_content='space-between')),

    # Call the existing output_scatter_indices
    widgets.HBox([
        widgets.VBox([output_scatter_indices_plot]), # Left
        widgets.VBox([]), # Center (remove extra space)
        widgets.VBox([output_stats_plot]), # Right
    ], layout=widgets.Layout(margin='0px')),

    # Additional widgets here with reduced space
    widgets.HBox([
        widgets.VBox([udl1_widget]), # Left
        widgets.VBox([udl2_widget]), # Right
    ], layout=widgets.Layout(margin='0px')), # Set margin to zero for less space

    # Output widgets
    widgets.HBox([
        widgets.VBox([output_udl1]),
        widgets.VBox([output_udl2]),
        widgets.VBox([output_spread]),
    ], layout=widgets.Layout(margin='0px')), # Adjusted margin to minimize space

], layout=widgets.Layout(margin='0px')) # Overall layout margin adjusted to zero

# Display the combined Layout
display(box)