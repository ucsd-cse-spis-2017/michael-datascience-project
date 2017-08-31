1. Using Neural Network to Analzye MNIST 

2. MNIST Dataset

Algorithm:

class Network(object):
  """ Defines basic features of the network, such as the number of layers, neurons per layer, weights, and biases """
  
  def forward_pass(self, input):
  """ Takes an input and passes it through the network to return an ouput """
      y = sigmoid( dotproduct( Weights, input) + bias )
  
  def Stochastic Gradient Descent(self, training_data, size of each batch, number of batches, learning rate)
   """ Performs a Stochastic Gradient Descent on the training data. Randomly shuffles the training_data, takes the data and splits                                                  it into batches. Passes each batch into a function that will update the network, using the backpropagation algorithm. """
   
  def update network(batches of data, learning rate):
  """ Updates the weight and bias values of the network using the backpropagation algorithm """
  
  def backpropagation(input data, correct input label)
  """ Calculates the gradient and error values from each layer of the network. Adjusts the weights and biases of the network accordingly """
  
  *** REQUIRES MORE RESEARCH ON SPECIFICS OF ALGORITHM ***
  
  def evaluate
  """ Evaluates how many images the network labels correctly and """
  
  End of Minimum Viable Product ( including poster )
################################################################################################################################

Possible additions ( if possible )

  Visualization of network
  Visualization of changing weights and biases
  Tests between different learning rates or even activiation functions
  
   
      

