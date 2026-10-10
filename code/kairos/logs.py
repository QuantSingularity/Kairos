"""Logging setup shared by the command line entry points."""

import logging

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure(level: str = "INFO") -> None:
    """Send log records to stderr at the requested level."""
    logging.basicConfig(level=level.upper(), format=FORMAT, datefmt="%H:%M:%S")
