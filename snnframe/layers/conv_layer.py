from .layer import Layer

class ConvLayer(Layer):
    """
        Convolutional Layer for SNNs

        TODO: Update Description while implementing 
    """
    def __init__(self, neuron_model, window_size, stride):
        self.neuron_model = neuron_model
        self.window_size = window_size
        self.stride = stride
    
    def initializeLayer(self):
        