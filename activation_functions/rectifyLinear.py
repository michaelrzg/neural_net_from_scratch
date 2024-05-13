# this file contains the code for rectify linear activaton function for neural network nodes.
# for use with hidden layer nodes
# michael rizig
# 5/12/24
import numpy as np
class rectefiedLinear:
    def forward(self,input):
        self.inputs = input
        self.output= np.maximum(0,input)      
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0