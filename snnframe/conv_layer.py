from .layer import Layer

class ConvLayer(Layer):
    """
        Convolutional Layer for SNNs

        TODO: Update Description while implementing 
    """
    def __init__(self, window_size, stride, firing_threshold):
        self.window_size = window_size
        self.stride = stride
        self.firing_threshold = firing_threshold