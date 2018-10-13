from .layer import Layer

class PoolingLayer(Layer):
    """
        Pooling Layer for SNNs

        TODO: Update Description while implementing 
    """
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride
    
    def compute(self, input):
        raise NotImplementedError

class GlobalPoolingLayer(PoolingLayer):
    """
        Global Pooling Layer for SNNs

        TODO: Update Description while implementing
    """
    def __init__(self):
        self.stride = 1

    def setPreviousLayer(self, layer: Layer):
        super().setPreviousLayer(layer)
        self.window_size = self.prev_layer.getOutSize()[:-1]