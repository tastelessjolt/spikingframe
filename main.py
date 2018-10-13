from snnframe import SNNetwork, ConvLayer, PoolingLayer, InputLayer


if __name__ == '__main__':
    snn = SNNetwork()
    snn.add_layer(InputLayer())
    snn.add_layer(ConvLayer())
    snn.add_layer(PoolingLayer())

