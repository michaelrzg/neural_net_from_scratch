# this file contains main file for the neural network
# michael rizig
# 5/12/24

import numpy as np #for math
from layers import dense
from data_generation.generateSpiralData import generateData #from generateData file
from activation_functions import rectifyLinear # from activation_functions folder
from activation_functions.activation_softmax_loss_categorical import Activation_Softmax_Loss_CategoricalCrossentropy
from loss_functions.categoricalCrossEntropy import categoricalCrossEntropy #from loss functions folder
from Optimisers.Optimizer_SGD import Optimizer_SGD
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
activation2 = Activation_Softmax_Loss_CategoricalCrossentropy()

#define optimizer to calibrate weights and biases
optimizer = Optimizer_SGD()

for i in range(10):
    #forward propogation
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    #results of forward propogation
    los= activation2.forward(dense2.output,y)


    #get accuracy
    predictions = np.argmax(los)
    accuracy = np.mean(predictions==y)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    #backward propogation
    activation2.backward(activation2.output,y)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    print("Accuracy: ",accuracy)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

# Print gradients
print("1",dense1.dweights)
print("2",dense1.dbiases)
print("3",dense2.dweights)
print("4",dense2.dbiases)

