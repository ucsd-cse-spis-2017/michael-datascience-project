The code is pretty nice already. I think given your current stage, once you get the sklearn's nn to work well for mnist, you can do the following
1. test learning rate and different actuation function (other than sigmoid) to see how they affect accuracy. When you calcluate accuracy, do 10-fold cross validation which means you randomly pick 90% of the data as training and 10% as testing. Record the testing accuracy. Do this 10 times and calcualte the average accuracy. This will be the accuracy that you report
2. Apply NN to other datasets. You can get other datasets such as malicious URL detection, and other datasets.
3. Debug your own NN and see what goes wrong. This may take a while though.

```

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
  
  
  
```
   
      

