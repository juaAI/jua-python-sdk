# statistics.py

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Statistic:
    """Internal representation of a statistic with associated metadata.

    This class represents a statistic with standardized naming and metadata.

    Attributes:
        key: The key for the statistic in the data returned by the Jua API.
        name: The standardized name of the statistic.
    """

    key: str
    name: str

    @property
    def display_name(self) -> str:
        """Return the display name of the statistic.

        Returns:
            The display name of the statistic.
        """
        return " ".join(word.capitalize() for word in self.name.split("_"))

    def __eq__(self, other):
        """Check if two Statistic objects are equal.

        Statistics are considered equal if all their attributes match.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, Statistic):
            return NotImplemented
        return self.name == other.name

    def __str__(self):
        """Return the standardized name of the statistic.

        Returns:
            The name attribute as a string.
        """
        return self.name

    def __hash__(self):
        return hash(self.name)


class Statistics(Enum):
    """Statistics available through the Jua API.

    This enum defines the set of statistics that can be requested
    when fetching forecasts or hindcasts. Use these constants when
    specifying which statistic to use with weather data functions.
    """

    MEAN = Statistic(key="mean", name="mean")
    STD = Statistic(key="std", name="standard deviation")
    QUANTILE_5 = Statistic(key="q5", name="5th_quantile")
    QUANTILE_25 = Statistic(key="q25", name="25th_quantile")
    QUANTILE_75 = Statistic(key="q75", name="75th_quantile")
    QUANTILE_95 = Statistic(key="q95", name="95th_quantile")

    @property
    def display_name(self) -> str:
        """Return the display name of the statistic.

        Returns:
            The display name of the statistic.
        """
        return self.value.display_name

    @property
    def key(self) -> str:
        """Return the key of the statistic.

        Returns:
            The key of the statistic.
        """
        return self.value.key

    @property
    def name(self) -> str:
        """Return the name of the statistic.

        Returns:
            The name of the statistic.
        """
        return self.value.name
