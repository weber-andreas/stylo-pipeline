from abc import ABC, abstractmethod
from typing import Any


class BaseBlock(ABC):
    """Abstract base class for pipeline blocks."""

    @abstractmethod
    def unload_model(self, *args, **kwargs):
        """Unload the model if it exists."""
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load the model"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass
