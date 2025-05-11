from enum import Enum


class Variable:
    def __init__(
        self,
        name: str,
        unit: str,
        name_ept1_5: str | None = None,
        name_ept2: str | None = None,
    ):
        self.name = name
        self.unit = unit
        self.name_ept1_5 = name_ept1_5
        self.name_ept2 = name_ept2

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return (
            self.name == other.name
            and self.unit == other.unit
            and self.name_ept1_5 == other.name_ept1_5
            and self.name_ept2 == other.name_ept2
        )

    def __str__(self):
        return self.name

    def __repr__(self):
        return (
            f"Variable(name={self.name}, unit={self.unit}, "
            f"name_ept1_5={self.name_ept1_5}, name_ept2={self.name_ept2})"
        )

    def __hash__(self):
        return hash((self.name, self.unit, self.name_ept1_5, self.name_ept2))


class Variables(Enum):
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
    SURFACE_DOWNWELLING_SHORTWAVE_FLUX = Variable(
        "surface_downwelling_shortwave_flux", "J m⁻²", "ssrd", None
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

    def __str__(self) -> str:
        return self.value.name

    def __repr__(self) -> str:
        return self.value.__repr__()

    def __hash__(self) -> int:
        return hash(self.value.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.value.name == other
        if isinstance(other, Variable):
            return self.value.name == other.value.name
        return NotImplemented
