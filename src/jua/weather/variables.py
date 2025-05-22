from enum import Enum

import xarray as xr


class Variable:
    """Internal representation of a weather variable with associated metadata.

    This class represents a weather variable with standardized naming and metadata
    across different weather models. It stores the canonical variable name,
    its unit of measurement, and model-specific name mappings.

    Attributes:
        name: The standardized name of the weather variable.
        unit: The unit of measurement for this variable.
        emcwf_code: Optional. The name/code for this variable in ECMWF data.
        name_ept2: Optional. The name for this variable in EPT2 data.
    """

    def __init__(
        self,
        name: str,
        unit: str,
        emcwf_code: str | None = None,
        name_ept2: str | None = None,
    ):
        """Initialize a weather Variable with its metadata.

        Args:
            name: Standardized variable name.
            unit: Unit of measurement (e.g., "K" for Kelvin, "m s⁻¹" for wind speed).
            emcwf_code: Optional. The code used for this variable in ECMWF data.
            name_ept2: Optional. The name used for this variable in EPT2 data.
        """
        self.name = name
        self.unit = unit
        self.emcwf_code = emcwf_code
        self.name_ept2 = name_ept2

    @property
    def display_name(self) -> str:
        """Return the display name of the variable.

        Returns:
            The display name of the variable.
        """
        return " ".join(word.capitalize() for word in self.name.split("_"))

    @property
    def display_name_with_unit(self) -> str:
        """Return the display name of the variable with its unit.

        Returns:
            The display name of the variable with its unit.
        """
        return f"{self.display_name} ({self.unit})"

    def __eq__(self, other):
        """Check if two Variable objects are equal.

        Variables are considered equal if all their attributes match.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, Variable):
            return NotImplemented
        return (
            self.name == other.name
            and self.unit == other.unit
            and self.emcwf_code == other.emcwf_code
            and self.name_ept2 == other.name_ept2
        )

    def __str__(self):
        """Return the standardized name of the variable.

        Returns:
            The name attribute as a string.
        """
        return self.name

    def __repr__(self):
        """Return a string representation of the Variable for debugging.

        Returns:
            A string showing all attributes of the Variable.
        """
        return (
            f"Variable(name={self.name}, unit={self.unit}, "
            f"name_ept1_5={self.emcwf_code}, name_ept2={self.name_ept2})"
        )

    def __hash__(self):
        return hash(self.name)


class Variables(Enum):
    """Standard weather variables available through the Jua API.

    This enum defines the set of weather variables that can be requested
    in weather forecasts and hindcasts. Each variable has a standardized name
    and unit of measurement.

    Examples:
        >>> from jua.weather import Variables
        >>> # Request air temperature in a forecast
        >>> ds[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M].plot()
    """

    AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M = Variable(
        "air_temperature_at_height_level_2m", "K", "2t", "air_temperature_2m"
    )
    AIR_PRESSURE_AT_MEAN_SEA_LEVEL = Variable(
        "air_pressure_at_mean_sea_level", "Pa", "msl", "air_pressure_at_mean_sea_level"
    )
    WIND_SPEED_AT_HEIGHT_LEVEL_10M = Variable(
        "wind_speed_at_height_level_10m", "m s⁻¹", "10si", "wind_speed_10m"
    )
    WIND_DIRECTION_AT_HEIGHT_LEVEL_10M = Variable(
        "wind_direction_at_height_level_10m", "°", "10wdir", "wind_direction_10m"
    )
    WIND_SPEED_AT_HEIGHT_LEVEL_100M = Variable(
        "wind_speed_at_height_level_100m", "m s⁻¹", "100si", "wind_speed_100m"
    )
    WIND_DIRECTION_AT_HEIGHT_LEVEL_100M = Variable(
        "wind_direction_at_height_level_100m", "°", "100wdir", "wind_direction_100m"
    )
    GEOPOTENTIAL_AT_PRESSURE_LEVEL_50000PA = Variable(
        "geopotential_at_pressure_level_50000Pa",
        "m² s⁻²",
        "z_500",
        "geopotential_500hpa",
    )

    SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H = Variable(
        "surface_downwelling_shortwave_flux_sum_1h", "J m⁻²", "ssrd", None
    )

    SURFACE_DIRECT_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H = Variable(
        "surface_direct_downwelling_shortwave_flux_sum_1h", "J m⁻²", "fdir", None
    )

    SURFACE_NET_DOWNWARD_SHORTWAVE_FLUX_SUM_1H = Variable(
        "surface_net_downward_shortwave_flux_sum_1h", "J m⁻²", "ssr", None
    )

    SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_6H = Variable(
        "surface_downwelling_shortwave_flux_sum_6h", "J m⁻²", "ssrd", None
    )

    # Additional variables from EPT2 that don't have a direct EPT1_5 equivalent
    EASTWARD_WIND_AT_HEIGHT_LEVEL_10M = Variable(
        "eastward_wind_at_height_level_10m", "m s⁻¹", None, "eastward_wind_10m"
    )
    NORTHWARD_WIND_AT_HEIGHT_LEVEL_10M = Variable(
        "northward_wind_at_height_level_10m", "m s⁻¹", None, "northward_wind_10m"
    )
    EASTWARD_WIND_AT_HEIGHT_LEVEL_100M = Variable(
        "eastward_wind_at_height_level_100m", "m s⁻¹", None, "eastward_wind_100m"
    )
    NORTHWARD_WIND_AT_HEIGHT_LEVEL_100M = Variable(
        "northward_wind_at_height_level_100m", "m s⁻¹", None, "northward_wind_100m"
    )

    HIGH_TYPE_CLOUD_AREA_FRACTION = Variable(
        "high_type_cloud_area_fraction", "[0,1]", "hcc", None
    )

    MEDIUM_TYPE_CLOUD_AREA_FRACTION = Variable(
        "medium_type_cloud_area_fraction", "[0,1]", "mcc", None
    )

    LOW_TYPE_CLOUD_AREA_FRACTION = Variable(
        "low_type_cloud_area_fraction", "[0,1]", "lcc", None
    )

    CLOUD_AREA_FRACTION_AT_ENTIRE_ATMOSPHERE = Variable(
        "cloud_area_fraction_at_entire_atmosphere", "[0,1]", "tcc", None
    )

    @property
    def display_name(self) -> str:
        """Return the display name of the variable.

        Returns:
            The display name of the variable.
        """
        return self.value.display_name

    @property
    def name(self) -> str:
        """Return the name of the variable.

        Returns:
            The name of the variable.
        """
        return self.value.name

    @property
    def emcwf_code(self) -> str | None:
        """Return the EMCWF code of the variable.

        Returns:
            The EMCWF code of the variable.
        """
        return self.value.emcwf_code

    @property
    def unit(self) -> str:
        """Return the unit of the variable.

        Returns:
            The unit of the variable.
        """
        return self.value.unit

    @property
    def display_name_with_unit(self) -> str:
        """Return the display name of the variable with its unit.

        Returns:
            The display name of the variable with its unit.
        """
        return self.value.display_name_with_unit

    def __str__(self) -> str:
        """Return the variable's standard name as a string.

        Returns:
            The standardized name of the weather variable.
        """
        return self.value.name

    def __repr__(self) -> str:
        return self.value.__repr__()

    def __hash__(self) -> int:
        # Ensures that we can use the enum variable as a dictionary key
        # and that is is the same as the variable name as string
        # i.e. data[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M] is the same as
        # data["air_temperature_at_height_level_2m"]
        return hash(self.value.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.value.name == other
        if isinstance(other, Variable):
            return self.value.name == other.name
        if isinstance(other, Variables):
            return self.value.name == other.value.name
        return NotImplemented


_RENAMING_DICT = {
    **{
        v.value.emcwf_code: v.value.name
        for v in Variables
        if v.value.emcwf_code is not None
    },
    **{
        v.value.name_ept2: v.value.name
        for v in Variables
        if v.value.name_ept2 is not None
    },
}

_RENAME_TO_EPT2_DICT = {
    **{
        v.value.emcwf_code: v.value.name_ept2
        for v in Variables
        if v.value.emcwf_code is not None
    },
    **{
        v.value.name: v.value.name_ept2
        for v in Variables
        if v.value.name_ept2 is not None
    },
}


def rename_variable(variable: str) -> str:
    """Convert variable names from model-specific formats to standardized names.

    Args:
        variable: The source variable name to convert.

    Returns:
        The standardized variable name if recognized, otherwise the original name.
    """
    return _RENAMING_DICT.get(variable, variable)


def rename_to_ept2(variable: str | Variables) -> str:
    """Convert variable names from model-specific formats to EPT2 names.

    Args:
        variable: The source variable name to convert.

    Returns:
        The EPT2 variable name if recognized, otherwise the original name.
    """
    return _RENAME_TO_EPT2_DICT.get(str(variable), str(variable))  # type: ignore


def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    output_variable_names = {k: rename_variable(k) for k in ds.variables}
    return ds.rename(output_variable_names)
