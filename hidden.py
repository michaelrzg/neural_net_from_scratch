# this file contains the hidden layers source code 
# michael rizig
# 5/12/24

import numpy as np
from generateData import generateData
#good activation function for inner layers, not good for learning
class Act_rectefiedLinear:
    def forward(self,input):
        self.output= np.maximum(0,input)      
#not good for classification, good for training/learning
class Act_softmax:
    def forward(self,input):
        expvalues = np.exp(input-np.max(input, axis=1, keepdims=True))
        probs = expvalues / np.sum(expvalues, axis =1, keepdims=True)
        self.output = probs
#define a internal hidden layer
class loss:
    def calculateLoss(self,input,labels):
        loss = self.forward(input,labels)
        mse = np.mean(loss)
        return mse
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
class layer_dense:
    def __init__(self,numNeurons,numInputs):
        self.weights = 0.1*np.random.randn(numNeurons,numInputs)
        self.biases = np.zeros((1,numInputs))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases


#main:

#eulers number (e^x) used for softmax
E = 2.71828182846

#generate data
np.random.seed(123)
X,y = generateData(100,3)
# define first layer & activation
dense1 = layer_dense(2,3)
activation1 = Act_rectefiedLinear()
#define second layer and activation
dense2 = layer_dense(3,3)
activation2 = Act_softmax()

#forward propogation
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
#print results of forward propogation
firstpass= activation2.output
#categorical cross entropy to calculate loss
lossfunc = categoricalCrossEntropy()
passloss = lossfunc.calculateLoss(firstpass,y)
#get accuracy
predictions = np.argmax(firstpass,axis=1)
accuracy = np.mean(predictions==y)
print(passloss)
print(accuracy)