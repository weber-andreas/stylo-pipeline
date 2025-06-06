from abc import abstractmethod


class BaseBlock:
    """
    Abstract base class for pipeline blocks.
    Each block should implement the `run` method.
    """

    def unload_model(self):
        """
        Unload the model if it exists.
        This method should be overridden by subclasses if they have a model to unload.
        """
        pass

    def load_model(self):
        """
        Load the model from the specified path.
        This method should be overridden by subclasses if they have a model to load.
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
