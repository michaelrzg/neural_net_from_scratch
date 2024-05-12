# this file contains the code for softmax activation function for neural network nodes
# for use with output layer nodes.
# michael rizig
# 5/12/24
import numpy as np
class softmax:
    def forward(self,input):
        expvalues = np.exp(input-np.max(input, axis=1, keepdims=True))
        probs = expvalues / np.sum(expvalues, axis =1, keepdims=True)
        self.output = probs