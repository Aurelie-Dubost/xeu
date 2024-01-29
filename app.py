# Import necessary libraries
import ipywidgets as widgets
from IPython.display import display
from get_data import last_friday, extended_data
from data_output import create_download_link
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the list of extended parameters
extended_parameters = ['spot', 'atmfs', 'convexity', 'skew', 'down skew', 'up skew', 'rv', 'vrp', 'term structure']

# Function to create date pickers
def create_date_pickers():
    # Calculate the last trading day of the last year
    today = datetime.now().date()
    last_year = today.year - 1
    last_day_year = datetime(last_year, 12, 31)
    # Initialize widgets with last trading day of last year as the start date
    start_date_picker = widgets.DatePicker(description='Start Date', value=last_day_year)
    end_date_picker = widgets.DatePicker(description='End Date', value=today)

    return start_date_picker, end_date_picker

# Function to create underlying selectors
def create_underlying_selectors():
    # Initialize dropdown widgets for selecting underlyings
    underlying_selector_1 = widgets.Dropdown(options=['CAC', 'SPX'], description='Underlying 1', value='CAC')
    underlying_selector_2 = widgets.Dropdown(options=['None', 'CAC', 'SPX'], description='Underlying 2', value='None')

    return underlying_selector_1, underlying_selector_2

# Function to create plot output widget
def create_plot_output():
    return widgets.Output()

# Function to create data table output widget
def create_data_table_output():
    return widgets.Output()

# Function to create download button
def create_download_button():
    return widgets.Button(description="Download Data as Excel")

# Function to handle the download button click event
def on_download_button_clicked(b, underlying_selector_1):
    link = create_download_link(underlying_selector_1.value)
    display(link)

# Function to update plots
def update_plots(start_date, end_date, underlying1, underlying2, plot_output, extended_parameters):
    with plot_output:
        # Clear the previous chart output
        plot_output.clear_output(wait=True)

        if start_date is None or end_date is None:
            print("Please select both start and end dates.")
            return

        try:
            fig, axs = plt.subplots(3, 3, figsize=(15, 12))

            for i, param in enumerate(extended_parameters):
                ax = axs[i // 3, i % 3]

                # Attempt to fetch data
                data1 = extended_data[underlying1][param][start_date:end_date]
                data2 = extended_data[underlying2][param][start_date:end_date] if underlying2 != 'None' else None

                if data1 is not None and not data1.empty and (data2 is None or data2 is not None and not data2.empty):
                    spread_data = data1 - data2 if data2 is not None else data1
                    ax.plot(spread_data, color=colors[param])
                    ax.set_title(param)
                else:
                    print(f"No valid data available for parameter {param} in the selected date range.")

            plt.tight_layout()
            plt.show()
        except OverflowError as e:
            print("An OverflowError occurred. Please select a smaller date range.")
        except KeyError as e:
            print(f"An error occurred: {e}. Please check the selected dates and underlying instruments.")

# Function to update data view
def update_data_view(underlying, data_tab_output):
    with data_tab_output:
        data_tab_output.clear_output()
        display(extended_data[underlying])

# Event handler function
# Add a flag to track whether an error message is displayed
error_message_displayed = False

# Event handler function
def on_change(change):
    global error_message_displayed  # Declare the flag as global

    if change['type'] == 'change' and change['name'] == 'value':
        # Check if both start_date and end_date are not None
        if (start_date_picker.value is not None and end_date_picker.value is not None):
            # Check if the selected dates are in the correct format (yyyy-mm-dd)
            try:
                start_date = datetime.strptime(str(start_date_picker.value), '%Y-%m-%d').date()
                end_date = datetime.strptime(str(end_date_picker.value), '%Y-%m-%d').date()
            except ValueError:
                error_message_displayed = True  # Set the flag to indicate an error message is displayed
                print("Error: Please select dates in the format yyyy-mm-dd.")
                return
            
            # Check if the selected dates are within a reasonable range
            if (start_date <= datetime.now().date() and
                start_date >= datetime(2000, 1, 1).date() and
                end_date <= datetime.now().date() and
                end_date >= datetime(2000, 1, 1).date()):
                
                # Reset the error message flag
                error_message_displayed = False
                
                try:
                    update_plots(start_date, end_date, underlying_selector_1.value, 
                                 underlying_selector_2.value, plot_output, extended_parameters)
                    update_data_view(underlying_selector_1.value, data_tab_output)
                except KeyError as e:
                    # An error occurred during plot update, print the error message
                    print(f"Error: An error occurred while updating plots.")
                    print(f"Error Details: {e}")
            else:
                # Selected dates are out of range, print the error message
                print("Error: Please select valid dates within a reasonable range (e.g., between 2000-01-01 and today's date).")
        else:
            # Dates are not selected, print the error message
            print("Error: Please select both start and end dates.")

# Define colors in RGB format
colors = {
    'spot': (0, 0, 1),         # Blue
    'atmfs': (0, 0.5, 0),      # Green
    'convexity': (1, 0, 0),    # Red
    'skew': (0.5, 0, 0.5),     # Purple
    'down skew': (1, 0.549, 0),# Orange
    'up skew': (0, 1, 1),      # Cyan
    'rv': (1, 0, 1),           # Magenta
    'vrp': (1, 1, 0),          # Yellow
    'term structure': (0.647, 0.165, 0.165)  # Brown
}


# Initialize widgets
start_date_picker, end_date_picker = create_date_pickers()
underlying_selector_1, underlying_selector_2 = create_underlying_selectors()
plot_output = create_plot_output()
data_tab_output = create_data_table_output()
download_button = create_download_button()

# Attach event handler to widgets
start_date_picker.observe(on_change)
end_date_picker.observe(on_change)
underlying_selector_1.observe(on_change)
underlying_selector_2.observe(on_change)

# Check if both start_date and end_date are not None
if start_date_picker.value is not None and end_date_picker.value is not None:
    # Use the date values directly
    start_date = start_date_picker.value
    end_date = end_date_picker.value

    # Call the update_plots function with all required arguments
    update_plots(start_date, end_date, underlying_selector_1.value, underlying_selector_2.value, plot_output, extended_parameters)
else:
    print("Please select both start and end dates.")
    
    
# Create the app layout
app_layout = widgets.VBox([
    widgets.HBox([start_date_picker, end_date_picker]),
    widgets.HBox([underlying_selector_1, underlying_selector_2]),
    plot_output,
    data_tab_output,
    download_button
])