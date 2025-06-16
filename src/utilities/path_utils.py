"""Utility functions for path operations."""

import logging
import os

logger = logging.getLogger(__name__)


def validate_paths_exist(paths: list[str]) -> bool:
    """Check if all paths exist."""
    for path in paths:
        if not os.path.exists(path):
            logger.error("Path does not exist: %s", path)
            return False
    return True
