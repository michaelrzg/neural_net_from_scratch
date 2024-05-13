# this file contains the code for a dense hidden layer of a neural network
# michael rizig
# 5/11/24
import numpy as np
class layer_dense:
    def __init__(self,numNeurons,numInputs):
        self.weights = 0.1*np.random.randn(numNeurons,numInputs)
        self.biases = np.zeros((1,numInputs))
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights)+self.biases
    def backward(self, derivative_values):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, derivative_values)
        self.dbiases = np.sum(derivative_values, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(derivative_values, self.weights.T)
