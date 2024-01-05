# Adjusting the bar chart for 0% series overlap

# Define the color in RGB
color = (0/255, 145/255, 90/255)  # RGB color for the bars

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Create the horizontal bar chart with the specific color
# Note: In Matplotlib, series overlap is not a concept like in Excel.
# Bars are plotted individually, thus there's inherently no overlap.
bars = ax.barh(categories, values, color=color, edgecolor='black')

# Adjusting the gap width by setting the bar height (182% of default height)
# Since there's no overlap, we don't adjust the bar width for overlap.
for bar in bars:
    bar.set_height(bar.get_height() * 1.82)

# Set tick marks to major and type to outside
ax.tick_params(axis='x', which='major', direction='out', length=6)
ax.tick_params(axis='y', which='major', direction='out', length=6)

# Set the axis labels
ax.set_xlabel('Values', fontsize=8)
ax.set_ylabel('Categories', fontsize=8)

# Invert the y-axis to have 'gte' at the top
ax.invert_yaxis()

# Remove the spines of the chart to match the image
for spine in ax.spines.values():
    spine.set_visible(False)

# Set the axis to automatically fit the data
ax.set_xlim(left=min(values)*0.98, right=max(values)*1.02)  # Slightly reduce/increase to fit the data

# Display the plot
plt.show()