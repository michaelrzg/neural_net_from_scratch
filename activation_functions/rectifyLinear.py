# this file contains the code for rectify linear activaton function for neural network nodes.
# for use with hidden layer nodes.
# michael rizig
# 5/12/24
import numpy as np
class Act_rectefiedLinear:
    def forward(self,input):
        self.output= np.maximum(0,input)      