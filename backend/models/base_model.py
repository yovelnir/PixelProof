from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all deepfake detection models."""
    
    def __init__(self, model_name):
        """Initialize a model.
        
        Args:
            model_name (str): Name identifier for this model
        """
        self.model_name = model_name
        self.model = None
    
    @abstractmethod
    def load(self, model_path):
        """Load the model from a file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def analyze(self, image_path):
        """Analyze an image for deepfake detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results
        """
        pass 