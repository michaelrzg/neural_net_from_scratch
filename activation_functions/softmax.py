# this file contains the code for softmax activation function for neural network nodes
# for use with output layer nodes.
# michael rizig
# 5/12/24
import numpy as np
class softmax:
    def forward(self,input):
        self.inputs = input
        expvalues = np.exp(input-np.max(input, axis=1, keepdims=True))
        probs = expvalues / np.sum(expvalues, axis =1, keepdims=True)
        self.output = probs
    # Backward pass
def backward(self, derivative_values, labels):
    # Create uninitialized array
    self.dinputs = np.empty_like(derivative_values)
    # Enumerate outputs and gradients
    for index, (single_output, single_dvalues) in \
        enumerate(zip(self.output, derivative_values)):
        # Flatten output array
        single_output = single_output.reshape(-1, 1)
        # Calculate Jacobian matrix of the output and
        jacobian_matrix = np.diagflat(single_output) - \
        np.dot(single_output, single_output.T)
        # Calculate sample-wise gradient
        # and add it to the array of sample gradients
        self.dinputs[index] = np.dot(jacobian_matrix,
        single_dvalues)

