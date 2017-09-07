import random

import numpy as np

class Network(object):
    
    def __init__(self, layers):
        """ Initalizes the weights and biases with a random array
         according to a normal distribuation, with a mean of 0,
         and a variance of 1. The biases need only be taken from sizes[1:]
         since there is no bias present in the first layer of the network. 
         Similarly, the weights are only taken from layers after the first layer,
         since weights serve as a connection between each layer of neurons. """
         ## Weight initialization using /np.sqrt(x) in order to prevent learning,
         ## slowdown. When the values of the weights and biases are very spread
         ## out, it is likely to create a learning slowdown due to the nature
         ## of the sigmoid function. By dividing by the square root of the 
         ## "length" of the distribution, the distribuation becomes much 
         ## tighter, which prevents the learning slowdowns that occur when
         ## z approaches very large positive or negative numbers. 
        self.num_layers = len(layers)
        self.sizes = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]         
        self.weights = [np.random.randn(y, x)/np.sqrt(x)             
                        for x, y in zip(layers[:-1], layers[1:])]         

    def forward_pass(self, a):
        """ Performs a forward pass through the neural network with input,
        a, applying the activation function to each weighted input. In 
        this case, we will be using the sigmoid function. """
        for w , b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w , a) + b)
        return a

    def SGD(self, batches, mini_batch_size, learning_rate, training_data, lmbda, test_data = None):
        """ Stochastic gradient descent randomizes and passes through batches of
        the training data, which will be used to train the network's weights and
        biases using the backpropagation algorithm. The function also passes a learning
        rate, which can be determined by the user. """
        len_train = len(training_data)
        if test_data:
            length_test = len(list(test_data))
        for i in range(batches):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, len_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_network(mini_batch, learning_rate, lmbda, training_data)
            if test_data:
                print ("Batch {0}  {1} / {2}".format(
                    i, self.evaluate_network(test_data), length_test))
            else:
                print ("Batch {0} complete".format(i))


    def update_network(self, mini_batch, learning_rate, lmbda, training_data):
        """ Updates the weights and biases of the network according to the new 
        values calculated by the backpropogation algorithm. """
        
        # Creates an array of zeros in the same shape of weights and biases which
        # will later be filled in with the proper gradient values.
        grad_wrt_w = [np.zeros(w.shape) for w in self.weights]
        grad_wrt_b = [np.zeros(b.shape) for b in self.biases]

        for x , y in mini_batch:
            delta_grad_wrt_b, delta_grad_wrt_w = self.backpropagation( x , y )
            grad_wrt_b = [dgrad_b + grad_b for dgrad_b, grad_b in zip(delta_grad_wrt_b, grad_wrt_b)]
            grad_wrt_w = [dgrad_w + grad_w for dgrad_w, grad_w in zip(delta_grad_wrt_w, grad_wrt_w)]
        
        self.biases = [ b - ( learning_rate / len(mini_batch)) * grad_b
                        for b, grad_b in zip(self.biases, grad_wrt_b)]
        # Using L2 Regularization, also known as weight decay
        self.weights = [ ((1 - (( learning_rate * lmbda) / len(training_data))) * \
                        w ) - ((learning_rate / len(mini_batch))* grad_w) \
                        for w , grad_w in zip(self.weights, grad_wrt_w)] 


    def backpropagation(self, x, y):
        """ Takes in a tuple ( x , y ) and returns the gradients with respect to 
        the biases and weights of the network. These gradients will be later used
        to update the network's biases and weights in preparation for the next
        mini_batch of data. The function calculates these values for output layers
        and hidden layers separately. The term backpropagation refers to how the
        gradients are taken going backwards through the network. """
        grad_wrt_w = [np.zeros(w.shape) for w in self.weights]
        grad_wrt_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]                                
        zs = []                                          # zs refers to the weighted inputs for each layer of neurons
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b                # This loop stores the activations and weighted inputs at each layer,
            zs.append(z)                                 # which will later be used to calculate the weight and bias gradients
            activation = sigmoid(z)                      # at each layer. 
            activations.append(activation)
        ## The following variables account for when the activation is at an output layer.
        ## The sigmoid prime term is dropped since this is utilizing a cross-entropy cost function. 
        delta = self.cost_derivative(activations[-1], y) 
        
        grad_wrt_b[-1] = delta
        grad_wrt_w[-1] = np.dot(delta, activations[-2].transpose())
        ## The following variables account for when the activation is at a hidden layer. 
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * deriv_sigmoid(zs[-l])
            
            grad_wrt_b[-l] = delta 
            grad_wrt_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return(grad_wrt_b, grad_wrt_w)
    
    
    def evaluate_network(self, test_data):
        """ Takes in a tuple ( x , y ) from the test_data and passes the input, x,
        throught the network. Then compares the final output from the
        forward pass with the correct label, y. Returns the amount of times 
        the network is correct in its evaluation of the image. """
        test_results = [(np.argmax(self.forward_pass(x)), y) for (x , y) in test_data]          # checks if the vector ouputed by the network matches
        return sum(int(x == y) for (x , y) in test_results)                                     # the vector representing the correct label. 

    
    def cost_derivative(self, out_activation, y):
                """ Derivative of the cost function, which in this case is
                is a quadratic loss function. """
                return (out_activation - y)

###########################################################################################################################################

def sigmoid(e):
        """ Sigmoid function, used as activation for neurons """
        return 1.0 / (1.0 + np.exp(-e))

def deriv_sigmoid(e):
        """ Derivative of sigmoid function, used in backpropagation """
        return (sigmoid(e)) * (1.0 - sigmoid(e))
        
