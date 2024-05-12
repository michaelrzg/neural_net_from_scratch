# this file contains the code generation functions for testing 
# michael rizig
# 5/12/24
# modified from:
#https://cs231n.github.io/neural-networks-case-study/
import numpy as np
#define the  number of features:
D=2

def generateData(points, classes):
    X = np.zeros((points*classes,D)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for j in range(classes):
        ix = range(points*j,points*(j+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y