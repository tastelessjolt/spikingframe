from snnframe import SNNetwork
from snnframe.layers import ConvLayer, PoolingLayer, InputLayer
from snnframe.neurons import IFNeuron

if __name__ == '__main__':
    snn = SNNetwork()
    snn.add_layer(InputLayer())
    snn.add_layer(ConvLayer(IFNeuron(10), (5, 5), 1))
    snn.add_layer(PoolingLayer((2, 2), 2))

