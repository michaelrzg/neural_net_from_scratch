# this file contains main file for the neural network
# michael rizig
# 5/12/24

import numpy as np #for math
from layers import dense
from data_generation.generateSpiralData import generateData #from generateData file
from activation_functions import rectifyLinear,softmax # from activation_functions folder
from loss_functions.categoricalCrossEntropy import categoricalCrossEntropy #from loss functions folder

#_main:

#eulers number (e^x) used for softmax
E = 2.71828182846

#generate data
np.random.seed(123)
X,y = generateData(100,3)

# define first layer & activation
dense1 =dense.layer_dense(2,3)
activation1 = rectifyLinear.rectefiedLinear()

#define second layer and activation
dense2 = dense.layer_dense(3,3)
activation2 = softmax.softmax()

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