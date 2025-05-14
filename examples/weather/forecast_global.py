import logging

import matplotlib.pyplot as plt

from jua import JuaClient
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient()
    model = client.weather.get_model(Models.EPT1_5)

    # Let' access the full, global dataset
    lead_times_hours = [0, 12, 24]
    variables = [
        Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
    ]
    init_time = model.forecast.get_available_init_times()[1]
    dataset = model.forecast.get_forecast(
        init_time=init_time,
        prediction_timedelta=lead_times_hours,
        variables=variables,
    )

    # Fenerate a plot for air temperature and wind speed for 0, 12, and 24 hours
    rows = 2
    cols = 3
    fig, axs = plt.subplots(
        rows, cols, figsize=(18, 10), sharex=True, sharey=True
    )  # Share axes for maps

    # Loop through each variable (row)
    for r_idx, variable in enumerate(variables):
        print(f"Plotting data for {variable} at lead times {lead_times_hours}")
        data = dataset[variable]

        # Determine vmin and vmax for the current row using all its data
        # Using np.nanmin and np.nanmax to be robust to NaNs if any
        current_vmin = data.min()
        current_vmax = data.max()

        print(
            f"Variable: {variable.display_name}, "
            f"vmin: {current_vmin:.2f}, vmax: {current_vmax:.2f}"
        )

        last_plot_in_row = None  # To store the mappable for the colorbar

        # Second, plot each heatmap in the row using the determined vmin/vmax
        for c_idx, lead_time_h in enumerate(lead_times_hours):
            ax = axs[r_idx, c_idx]
            data_array_to_plot = data.sel(prediction_timedelta=lead_time_h)

            im = data_array_to_plot.plot(
                ax=ax,
                add_colorbar=False,  # We add a shared colorbar manually
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
            cb.set_label(variable.display_name_with_unit)

    # Add prominent row labels on the far left using fig.text (optional, if desired)
    # These are in addition to axis labels or colorbar labels
    y_positions_row_labels = [0.75, 0.28]  # Adjusted based on typical subplot heights
    for r_idx, variable in enumerate(variables):
        # Using the actual variable names from the dictionary for these labels
        label_text = variable.display_name  # e.g., "Temperature"
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
