import mnist_loader

import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(30, 10, 3.0, training_data, test_data=test_data)
