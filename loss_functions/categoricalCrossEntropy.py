# this file contains the code for categorical cross entropy loss calculation
# michael rizig
# 5/12/24
import numpy as np
class loss:
    def calculateLoss(self,input,labels):
        loss = self.forward(input,labels)
        mse = np.mean(loss)
        return mse
#first loss type
class categoricalCrossEntropy(loss):
    def forward(self,input,labels):  
        # negative log of error of all predictions output from forward propogation  
        n = len(input)
        clippedLossVector = np.clip(input,1e-9,1-1e-9) # this line ensures we dont get a infinite loss when finding mse of a 0
        if len(labels.shape) == 1: #case for if we have a scalar
            confidences = clippedLossVector[range(n),labels]      
        else :  #else we have a one-hot encoded vector (0,0,0,...,1,0,0,)
            confidences = np.sum(clippedLossVector*labels, axis = 1)
        final_likelyhood = -np.log(confidences) 
        return final_likelyhood    