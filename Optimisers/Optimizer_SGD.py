#this file contians the code for the optimizer utilizing stochastic gradient descent to update the layers biases and weights
# michael rizig
# 5/13/24

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
