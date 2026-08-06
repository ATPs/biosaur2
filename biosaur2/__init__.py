"""Biosaur2 public package metadata."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("biosaur2")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = ["__version__"]
