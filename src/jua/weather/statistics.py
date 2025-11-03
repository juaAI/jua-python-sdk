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
    aggregation: str

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

    MEAN = Statistic(key="mean", name="mean", aggregation="avg")
    STD = Statistic(key="std", name="standard deviation", aggregation="std")
    QUANTILE_5 = Statistic(key="q5", name="5th_quantile", aggregation="quantile_(0.05)")
    QUANTILE_25 = Statistic(
        key="q25", name="25th_quantile", aggregation="quantile_(0.25)"
    )
    QUANTILE_75 = Statistic(
        key="q75", name="75th_quantile", aggregation="quantile_(0.75)"
    )
    QUANTILE_95 = Statistic(
        key="q95", name="95th_quantile", aggregation="quantile_(0.95)"
    )

    @property
    def display_name(self) -> str:
        """Return the display name of the statistic.

        Returns:
            The display name of the statistic.
        """
        return self.value.display_name

    @property
    def agg(self) -> str:
        """Return the name of the aggregation for the statistic.

        Returns:
            The name of the aggregation for the statistic.
        """
        return self.value.aggregation

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

    @classmethod
    def from_key(cls, key: str) -> "Statistics":
        """Get a Statistics enum member from its key.

        Args:
            key: The key of the statistic (e.g., "mean", "std", "q5").

        Returns:
            The corresponding Statistics enum member.

        Raises:
            ValueError: If the key does not correspond to any Statistics member.
        """
        for stat in cls:
            if stat.key == key:
                return stat
        raise ValueError(
            f"No statistic found with key '{key}'. "
            f"Available keys: {', '.join(s.key for s in cls)}"
        )
