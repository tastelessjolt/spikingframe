import numpy as np

from .layer import Layer

class SNNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer: Layer): 
        prev_layer = None
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
        
        layer.prev_layer(prev_layer)
        self.layers.append(layer)

    def initialize(self):
        for layer in self.layers:
            layer.initialize()

    def training(self):
        curr_output = None
        for layer in self.layers:
            curr_output = layer.compute(curr_output)