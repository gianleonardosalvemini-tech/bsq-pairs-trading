"""Domain-specific exceptions for expected failure states."""

from __future__ import annotations


class PairsError(Exception):
    """Base class for every expected failure in this project."""


class ConfigError(PairsError):
    """A configuration value is missing or inconsistent."""


class DataFetchError(PairsError):
    """A remote data source (Yahoo Finance, Nasdaq) failed or returned nothing usable."""


class StaleDataError(PairsError):
    """The data does not cover the period that the analysis requires."""


class ShortInterestUnavailableError(PairsError):
    """The Nasdaq short-interest endpoint has no data for a ticker (e.g. NYSE-listed)."""
