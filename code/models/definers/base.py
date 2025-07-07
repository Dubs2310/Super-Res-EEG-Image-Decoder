from torch import nn

class BaseModelDefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.register_model()
    
    def register_model(self):
        """Register a model. Users should override this to return the model they want to use."""
        raise NotImplementedError("Subclasses must implement register_model method")
    
    def compute_loss(self, predictions, targets):
        """Compute loss function. Users should override this to define how the loss function works."""
        if self.model is None:
            raise ValueError("Model not registered. register_model must return a valid model.")
        raise NotImplementedError("Subclasses must implement compute_loss method")
    
    def forward(self, x):
        if self.model is None:
            raise ValueError("Model not registered. register_model must return a valid model.")
        return self.model(x)
    
    def predict(self, x):
        if self.model is None:
            raise ValueError("Model not registered. register_model must return a valid model.")
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            return self.model(x)