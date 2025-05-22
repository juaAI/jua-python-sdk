from pathlib import Path
from typing import Any

import xarray as xr
from pydantic import validate_call

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua._utils.spinner import Spinner
from jua.logging import get_logger
from jua.settings.jua_settings import JuaSettings
from jua.weather._xarray_patches import (
    TypedDataArray,
    TypedDataset,
    as_typed_dataarray,
    as_typed_dataset,
)
from jua.weather.conversions import bytes_to_gb
from jua.weather.models import Models

logger = get_logger(__name__)


class JuaDataset:
    """Weather dataset containing forecast or hindcast data from a Jua model.

    JuaDataset is the primary container for weather data returned by forecast and
    hindcast queries.

    Use `to_xarray` to get the data as an xarray dataset.

    You can access individual variables using dictionary-like syntax with either
    variable names or Variables enum members.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.weather import Models, Variables
        >>> client = JuaClient()
        >>> model = client.weather.get_model(Models.EPT2)
        >>> jua_ds = model.forecast.get_forecast()
        >>> # Access temperature data
        >>> temp = jua_ds[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
        >>> # Plot the data
        >>> temp.to_celcius().plot()
        >>> # Use xarray (equivalent to above)
        >>> ds = jua_ds.to_xarray()
        >>> temp = ds[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
        >>> # Plot the data
        >>> temp.to_celcius().plot()
    """

    _DOWLOAD_SIZE_WARNING_THRESHOLD_GB = 20

    def __init__(
        self,
        settings: JuaSettings,
        dataset_name: str,
        raw_data: xr.Dataset,
        model: Models,
    ):
        """Initialize a JuaDataset.

        Args:
            settings: Client settings.
            dataset_name: Name identifier for the dataset.
            raw_data: The underlying xarray Dataset.
            model: The model that produced this data.
        """
        self._settings = settings
        self._dataset_name = dataset_name
        self._raw_data = raw_data
        self._model = model

    @property
    def nbytes(self) -> int:
        """Get the memory size of the dataset in bytes.

        Returns:
            Memory size in bytes.
        """
        return self._raw_data.nbytes

    @property
    def nbytes_gb(self) -> float:
        """Get the memory size of the dataset in gigabytes.

        Returns:
            Memory size in gigabytes.
        """
        return bytes_to_gb(self.nbytes)

    def _get_default_output_path(self) -> Path:
        return Path.home() / ".jua" / "datasets" / self._model.value

    def to_xarray(self) -> TypedDataset:
        """Convert to a TypedDataset for advanced xarray operations.

        Returns:
            A TypedDataset that extends xarray.Dataset with additional Jua-specific
            functionality.
        """
        return as_typed_dataset(self._raw_data)

    def __getitem__(self, key: Any) -> TypedDataArray:
        """Access a specific variable from the dataset.

        You can use either string variable names or Variables enum members
        as keys.

        Args:
            key: Variable identifier (string name or Variables enum member).

        Returns:
            TypedDataArray for the requested variable.
        """
        return as_typed_dataarray(self._raw_data[str(key)])

    @validate_call(config={"arbitrary_types_allowed": True})
    def save(
        self,
        output_path: Path | None = None,
        show_progress: bool | None = None,
        overwrite: bool = False,
        ignore_size_warning: bool = False,
    ) -> None:
        """Save the dataset to disk in Zarr format.

        The dataset is saved to disk in Zarr format, which is an efficient format for
        large multidimensional arrays. By default, datasets are saved to
        ~/.jua/datasets/<model_name>/<dataset_name>.zarr.

        Args:
            output_path: Path to save the dataset to.
                If None, uses the default location.
            show_progress: Whether to display a progress bar during saving.
                If None, uses the client's default setting.
            overwrite: Whether to overwrite an existing dataset at the same location.
            ignore_size_warning: Whether to skip the confirmation prompt for large
                datasets.

        Raises:
            ValueError: If the dataset is too large and user declines to continue.
        """
        if output_path is None:
            output_path = self._get_default_output_path()

        output_name = self._dataset_name
        if output_path.suffix != ".zarr":
            output_path = output_path / f"{output_name}.zarr"

        if output_path.exists() and not overwrite:
            logger.warning(
                f"Dataset {self._dataset_name} already exists at {output_path}. "
                "Skipping download."
            )
            return

        data_to_save = self._raw_data
        data_size = bytes_to_gb(data_to_save.nbytes)
        if (
            not ignore_size_warning
            and data_size > self._DOWLOAD_SIZE_WARNING_THRESHOLD_GB
        ):
            logger.warning(
                f"Dataset {self._dataset_name} is large ({data_size:.2f}GB). "
                "This may take a while to save."
            )
            yn = input("Do you want to continue? (y/N) ")
            if yn.lower() != "y":
                logger.info("Skipping save.")
                return

        logger.info(
            f"Saving a {data_size:.2f}GB dataset "
            f"{self._dataset_name} to {output_path}..."
        )

        with Spinner(
            "Preparing save. This might take a while...",
            enabled=self._settings.should_print_progress(show_progress),
        ):
            delayed = data_to_save.to_zarr(output_path, mode="w", compute=False)

        with OptionalProgressBar(self._settings, show_progress):
            logger.info("Saving dataset...")
            delayed.compute()
        logger.info(f"Dataset {self._dataset_name} saved to {output_path}.")
