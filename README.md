init

BASIC IDEA:

import numpy as np

layer1Output = []
inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-.8]]
weights = [[0.2,0.8,-.5,1],[0.5,-0.91,0.26,-.5],[-.26,-.27,.17,.87]]
bias = [2,3,0.5]

#alternative method to avoid numpy:
'''for neuronBias,neuronWeights in zip(bias,weights):
neuronOut = 0
for weight_n,input_n in zip(neuronWeights,inputs):
neuronOut += weight_n \* input_n
neuronOut += neuronBias  
 layerOutput.append(neuronOut)
print (layerOutput)'''

layer1Output = np.dot(inputs,np.array(weights).T) + bias
print (layer1Output)

layer2Output = []
weights2 = [[0.1,-.14,.5],[-0.5,0.12,-0.33],[-.44,.73,-.13]]
bias2 = [-1,2,-.5]

layer2Output = np.dot(layer1Output,np.array(weights2).T) + bias2

print (layer2Output)
