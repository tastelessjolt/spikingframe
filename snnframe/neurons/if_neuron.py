from .neuron import Neuron
import numpy as np

class IFNeuron(Neuron):
    firing_threshold = 0
    current_potential = 0
    def __init__(self, firing_threshold):
        self.firing_threshold = firing_threshold

    def reset(self, inputShape):
        """
            Resets the neuron to resting state
            with the given size of `input`
        """
        self.current_potential = np.zeros(inputShape)

    def compute(self, input):
        """
            Integrate and Fire if `current_potential` is 
            more than `firing_threshold`

            TODO
        """
        pass