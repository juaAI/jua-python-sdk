from dataclasses import dataclass
from enum import Enum, StrEnum

from jua.weather.variables import Variables


class Weighting(StrEnum):
    """Weighting type for an AggregateVariable"""

    POPULATION = "population"
    SOLAR_CAPACITY = "solar_capacity"
    WIND_CAPACITY = "wind_capacity"


@dataclass(frozen=True)
class AggregateVariable:
    """Internal representation of a variable to be used for market aggregates.

    This class represents a market aggregate variable with standardized naming and
    metadata, including the appropriate weighting type that should be used when
    querying data for this variable.

    Attributes:
        variable: The market aggregate variable.
        weighting_type: The weighting method to use for this variable.
    """

    variable: Variables
    weighting: Weighting

    @property
    def name(self) -> str:
        """Return the name of the variable.

        Returns:
            The name of the variable.
        """
        return self.variable.name

    @property
    def unit(self) -> str:
        """Return the unit of the variable.

        Returns:
            The unit of the variable.
        """
        return self.variable.unit

    @property
    def display_name(self) -> str:
        """Return the display name of the variable.

        Returns:
            The display name of the variable.
        """
        return self.variable.display_name

    @property
    def display_name_with_unit(self) -> str:
        """Return the display name of the variable with its unit.

        Returns:
            The display name of the variable with its unit.
        """
        return self.variable.display_name_with_unit

    def __eq__(self, other):
        """Check if two metadata objects are equal.

        Variables are considered equal if all their attributes match.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, AggregateVariable):
            return NotImplemented
        return self.variable == other.variable and self.weighting == other.weighting

    def __hash__(self):
        return hash(self.display_name_with_unit)


class AggregateVariables(Enum):
    """Market aggregate variables available through the Jua API.

    This enum defines the set of market aggregate variables that can be requested
    in forecasts.
    """

    AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M = AggregateVariable(
        Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M, Weighting.POPULATION
    )
    SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H = AggregateVariable(
        Variables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H, Weighting.SOLAR_CAPACITY
    )
    WIND_SPEED_AT_HEIGHT_LEVEL_10M = AggregateVariable(
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M, Weighting.WIND_CAPACITY
    )
    WIND_SPEED_AT_HEIGHT_LEVEL_100M = AggregateVariable(
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_100M, Weighting.WIND_CAPACITY
    )

    @property
    def variable(self) -> Variables:
        """Return the variable for which to compute aggregates

        Returns:
            The variable for which to compute aggregates.
        """
        return self.value.variable

    @property
    def weighting(self) -> Weighting:
        """Return the weighting type for the variable.

        Returns:
            The weighting type for the variable.
        """
        return self.value.weighting

    def __hash__(self) -> int:
        # Ensures that we can use the enum variable as a dictionary key
        # and that it is the same as the variable name as string
        return hash(self.value.variable.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, AggregateVariable):
            return self.value.variable.name == other.variable.name
        if isinstance(other, AggregateVariables):
            return self.value.variable.name == other.value.variable.name
        return NotImplemented
