import pickle
class Layer:
    """
        Base class for all neural layers in `snnframe`
    """
    def __init__(self, size: tuple):
        self.size = size
        self.training = True
    
    def setPreviousLayer(self, layer: Layer):
        self.prev_layer = layer

    def setTraining(self, training):
        self.training = training

    def initializeLayer(self):
        """
            This should create any parameters needed for computation
            Any allocation of resources, and also use `self.prev_layer` 
            to compute the actual parameter sizes needed 
        """
        pass

    def compute(self, input):
        """
            Computes the output spikes from the `input` passed
            This should be overriden by the sub classes
            This should use `self.training` to update parameters accordingly
        """
        pass
    
    def dump(self):
        """
            Must **implement** for checkpointing. 
            Return a Pickle object
        """
        pass

    @staticmethod
    def restore(pkl):
        """
            Must **implement** for checkpointing
            Return a `Layer` or `Layer`-like object using the pkl object
        """
        pass