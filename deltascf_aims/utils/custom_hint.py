from dataclasses import dataclass


@dataclass
class BetweenInclusive:
    """
    Represent a range of values.
    """

    min: float
    max: float


@dataclass
class GreaterThan:
    """
    Represent a value greater than a given value.
    """

    min: float
