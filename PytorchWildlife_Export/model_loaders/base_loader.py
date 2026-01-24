from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModelLoader(ABC):
    """
    An abstract base class for model loaders.
    """
    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Loads the model and returns the underlying PyTorch nn.Module.
        """
        pass
