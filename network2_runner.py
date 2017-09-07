import mnist_loader

import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10])

net.SGD(60, 10, .1, training_data, 5.0, test_data = test_data)

