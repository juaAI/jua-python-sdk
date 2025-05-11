import logging

import matplotlib.pyplot as plt
import numpy as np

from jua.client import JuaClient
from jua.weather.models import Model
from jua.weather.variables import Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient()
    model = client.weather.get_model(Model.EPT2)

    # Let' access the full, global dataset
    dataset = model.forecast.get_latest_forecast_as_dataset()
    print(dataset.to_xarray())

    # Fenerate a plot for air temperature and wind speed for 0, 12, and 24 hours
    rows = 2
    cols = 3
    fig, axs = plt.subplots(
        rows, cols, figsize=(18, 10), sharex=True, sharey=True
    )  # Share axes for maps

    variable_display_names = {
        Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M: "Temperature (K)",
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M: "Wind Speed (m/s)",
    }

    variables_to_plot = [
        Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
    ]

    lead_times_hours = [0, 12, 24]
    colormaps = ["viridis", "plasma"]  # One cmap per variable/row

    # Loop through each variable (row)
    for r_idx, variable_key in enumerate(variables_to_plot):
        data_for_row = []
        # First, collect all data arrays for the current variable to find global
        # min/max for the row
        for lead_time_h in lead_times_hours:
            data_array = dataset[variable_key].jua.sel(prediction_timedelta=lead_time_h)
            data_for_row.append(data_array.data)  # Append numpy array

        # Determine vmin and vmax for the current row using all its data
        # Using np.nanmin and np.nanmax to be robust to NaNs if any
        current_vmin = np.nanmin([arr.min() for arr in data_for_row if arr.size > 0])
        current_vmax = np.nanmax([arr.max() for arr in data_for_row if arr.size > 0])

        print(
            f"Variable: {variable_display_names[variable_key]}, "
            f"vmin: {current_vmin:.2f}, vmax: {current_vmax:.2f}"
        )

        last_plot_in_row = None  # To store the mappable for the colorbar

        # Second, plot each heatmap in the row using the determined vmin/vmax
        for c_idx, lead_time_h in enumerate(lead_times_hours):
            ax = axs[r_idx, c_idx]
            data_array_to_plot = dataset[variable_key].jua.sel(
                prediction_timedelta=lead_time_h
            )

            im = data_array_to_plot.plot(
                ax=ax,
                add_colorbar=False,  # We add a shared colorbar manually
                cmap=colormaps[r_idx],
                vmin=current_vmin,
                vmax=current_vmax,
            )
            last_plot_in_row = im  # Store the QuadMesh object (or similar)

            ax.set_title(f"T+{lead_time_h}h")

            # Xarray's plot usually sets good axis labels (Longitude, Latitude)
            # If you want to override or ensure:
            if r_idx == rows - 1:  # Only for the last row
                ax.set_xlabel("Longitude")
            else:
                ax.set_xlabel("")

            if c_idx == 0:  # Only for the first column
                ax.set_ylabel("Latitude")
            else:
                ax.set_ylabel("")

        # Add a shared colorbar for the current row
        if last_plot_in_row:
            # Position the colorbar to the right of the row of subplots
            # [left, bottom, width, height] in figure coordinates
            # Adjust these values based on your fig_size and subplot layout
            cbar_left = (
                axs[r_idx, -1].get_position().x1 + 0.015
            )  # Right of last subplot + padding
            cbar_bottom = axs[r_idx, -1].get_position().y0  # Align bottom with subplots
            cbar_width = 0.015  # Width of colorbar
            cbar_height = (
                axs[r_idx, -1].get_position().height
            )  # Align height with subplots

            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            cb = fig.colorbar(last_plot_in_row, cax=cbar_ax)
            cb.set_label(variable_display_names[variable_key])

    # Add prominent row labels on the far left using fig.text (optional, if desired)
    # These are in addition to axis labels or colorbar labels
    y_positions_row_labels = [0.75, 0.28]  # Adjusted based on typical subplot heights
    for r_idx, variable_key in enumerate(variables_to_plot):
        # Using the actual variable names from the dictionary for these labels
        label_text = variable_display_names[variable_key].split(" (")[
            0
        ]  # e.g., "Temperature"
        fig.text(
            0.01,  # x-position (very left)
            y_positions_row_labels[r_idx],  # y-position (centered for each row)
            label_text,
            rotation=90,
            va="center",  # Vertical alignment
            ha="left",  # Horizontal alignment
            fontsize=14,
            fontweight="bold",
        )

    # Adjust layout to prevent overlap and make space for colorbars and fig.text labels
    fig.subplots_adjust(
        left=0.08, right=0.90, bottom=0.08, top=0.92, hspace=0.3, wspace=0.15
    )

    # Add a title for the entire figure
    fig.suptitle("Global Weather Forecast", fontsize=18, y=0.98, fontweight="bold")

    plt.show()


if __name__ == "__main__":
    main()
