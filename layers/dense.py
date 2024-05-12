# this file contains the code for a dense hidden layer of a neural network
# michael rizig
# 5/11/24
import numpy as np
class layer_dense:
    def __init__(self,numNeurons,numInputs):
        self.weights = 0.1*np.random.randn(numNeurons,numInputs)
        self.biases = np.zeros((1,numInputs))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases
