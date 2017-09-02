# import system

import numpy as np

import random

class Network(object):
    def __init__(self, layers):
        """ Initalizes the weights and biases with a random array
         according to a normal distribuation, with a mean of 0,
         and a variance of 1. The biases need only be taken from sizes[1:]
         since there is no bias present in the first layer of the network. """
        self.num_layers = len(layers)
        self.sizes = layers
        for y in layers[1:]:
            self.biases = [np.random.randn(y,1)]          # Creates an array representing all of the biases of each layer, 
                                                          # The input layer is not included when creating the bias array,
                                                          # since the bias is used to help calculate the output of a layer. 

        for x , y in zip(layers[:-1], layers[1:]):
            self.weights = [np.random.randn(y , x)]          # Creates an array representing all of the weights between the nodes of each respective layer,
                                                             # There are no weights protruding out of the output layer.


    def forward_pass(self,a):
        """ Performs a forward pass through the neural network with input,
        a, applying the sigmoid function at each node """
        for w , b in zip(self.weights, self.bias):
            a = sigmoid(np.dot(w , a) + b)
        return a

    def SGD(self, batches, mini_batch_size, learning_rate, training_data, test_data = None):
        """ Stochastic gradient descent takes small batches of the training
        data, which will then be passed through the backpropagation algorithm.
        New weights and biases will be generated, and the network will be updated
        to reflect this. """
        for i in range(batches):
            random.shuffle(training_data)
            for j in range(0, len(training_data), mini_batch_size):
                mini_batches = [training_data[i:i+mini_batch_size]]
                for mini_batch in mini_batches:
                    self.update_network(mini_batch, learning_rate)
            if test_data:
                print ("Batch {0}  {1} / {2}".format(
                    i, self.evaluate_network(test_data), len(training_data)))
            else:
                print ("Batch {0} complete".format(i))


    def update_network(self, mini_batch, learning_rate):
        """ Updates the weights and biases of the network according to the new 
        values calculated by the backpropogation algorithm. """
        
        # Creates an array of zeros in the same shape of weights and biases which
        # will later be filled in with the proper gradient values.
        for w in self.weights:
            grad_wrt_w = [np.zeros(w.shape)]
        for b in self.biases:
            grad_wrt_b = [np.zeros(b.shape)]
        
        for x , y in mini_batch:
            delta_grad_wrt_b, delta_grad_wrt_w = self.backpropogation( x , y )
            for dgrad_b, grad_b in zip(delta_grad_wrt_b, grad_wrt_b):
                grad_wrt_b = [dgrad_b + grad_b]
            for dgrad_w, grad_w in zip(delta_grad_wrt_w, grad_wrt_w):
                grad_wrt_w = [dgrad_w + grad_w]
        
        for b , grad_wrt_b in zip(self.biases, grad_wrt_b):
            self.biases = [ b - ( learning_rate / len(mini_batch)) * grad_wrt_b]
        for w , grad_wrt_w in zip(self.weights, grad_wrt_w):
            self.weights = [ w - ( learning_rate / len(mini_batch)) * grad_wrt_w]


        



    def backpropogation(self, x, y):
        """ Takes in a tuple ( x , y ) and returns the gradients with respect to 
        the biases and weights of the networks. These gradients will be later used
        to update the network's biases and weights in preparation for the next
        mini_batch of data. """
        for w in self.weights:
            grad_wrt_w = [np.zeros(w.shape)]
        for b in self.biases:
            grad_wrt_b = [np.zeros(b.shape)]

        activation = x
        activations = [x]                                # activations refers to the weighted inputs of each respective level, after being passed through the sigmoid activation function
        zs = []                                          # outputs refers the outputs of each respective level, excluding the sigmoid activation function
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b                # z can be visualized as a weighted input into a layer in the network
            np.append(z, zs)
            activation = sigmoid(z)
            activations.append(activation)
        ## The following variables account for when the activation is at an output layer.
        delta = self.cost_derivative(activations[-1], y) * deriv_sigmoid(zs[-1])

        grad_wrt_b[-1] = delta
        grad_wrt_w[-1] = np.dot(activations[-1], delta)
        ## The following variables account for when the activation is at a hidden layer. """
        for i in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * deriv_sigmoid(z[-l])
            
            grad_wrt_b[-l] = delta
            grad_wrt_w[-l] = np.dot(activations[-l-1], delta)
        return(grad_wrt_b, grad_wrt_w)



    def evaluate_network(self, x, y):
        """ Takes in a tuple ( x , y ) and passes the input, x,
        throught the network. Then compares the final output from the
        forward pass with the correct label, y. Returns the amount of times 
        the network is correct in its evaluation of the image. """
        for x , y in test_data:
            test_results = [(np.argmax(forward_pass(x)), y)]
        for x , y in test_results:  
            return sum(int(x == y))



    def cost_derivative(self, out_activation, y):
        """ Derivative of the cost function, which in this case is
        is a quadratic loss function. """
        return out_activation - y


###########################################################################################################################################

def sigmoid(x):
        """ Sigmoid function, used as activation for neurons """
        return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
        """ Derivative of sigmoid function, used in backpropagation """
        return (sigmoid(x))(1.0 - sigmoid(x))
        
