from .layer import Layer

class PoolingLayer(Layer):
    """
        Pooling Layer for SNNs

        TODO: Update Description while implementing 
    """
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

