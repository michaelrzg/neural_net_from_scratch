import numpy as np 
class dense: 
 
    def __init__(self, numInputs, numNeurons): 
        self.weights = 0.01 * np.random.randn(numInputs, numNeurons) 
        self.biases = np.zeros((1, numNeurons)) 
 
  
    def propogate(self, data): 
        self.inputs = data 
        self.output = np.dot(data, self.weights) + self.biases 
 
    def backwards_propogate(self, dvalues): 
        self.derivative_weights = np.dot(self.inputs.T, dvalues) 
        self.derivative_bias = np.sum(dvalues, axis=0, keepdims=True) 
        self.dinputs = np.dot(dvalues, self.weights.T) 
class RectifiedLinearActivation: 
 
    def forward(self, inputs): 
        self.inputs = inputs 
        self.output = np.maximum(0, inputs) 
 
    def backward(self, dvalues): 
        self.dinputs = dvalues.copy() 
 
        self.dinputs[self.inputs <= 0] = 0
class softmaxActivation: 
 
    def forward(self, inputs): 
        self.inputs = inputs 
 
        values_e = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        p = values_e / np.sum(values_e, axis=1, keepdims=True) 
 
        self.output = p 
    def backward(self, dvalues): 
 
        self.dinputs = np.empty_like(dvalues) 
 
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): 
            single_output = single_output.reshape(-1, 1) 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) 
            self.dinputs[index] = np.dot(jacobian_matrix,  single_dvalues) 
class Stochastic_Gradient_Descent: 

    def __init__(self, LR=1., decay=0., momentum=0.): 
        self.LR = LR 
        self.current_LR = LR 
        self.decay = decay 
        self.index = 0 
        self.momentum = momentum
    
    def pre_updateLayerValues(self): 
        if self.decay: 
            self.current_LR = self.LR * (1. / (1. + self.decay * self.index)) 
 
    def updateLayerValues(self, layer): 
 
        if self.momentum: 
 
            if not hasattr(layer, 'weight_momentums'): 
                layer.weight_momentums = np.zeros_like(layer.weights) 
                layer.bias_momentums = np.zeros_like(layer.biases) 
            weight_updates = self.momentum * layer.weight_momentums - self.current_LR * layer.derivative_weights 
            layer.weight_momentums = weight_updates 
 
            bias_updates = self.momentum * layer.bias_momentums - self.current_LR * layer.derivative_bias 
            layer.bias_momentums = bias_updates 
 
        else: 
            weight_updates = -self.current_LR * layer.derivative_weights 
            bias_updates = -self.current_LR * layer.derivative_bias 
        layer.weights += weight_updates 
        layer.biases += bias_updates            
   
    def post(self): 
        self.index += 1 
 
 
class ADAM: 
 
    def __init__(self, LR=0.001, decay=0., epsilon=1e-7, 
                 beta_1=0.9, beta_2=0.999): 
        self.LR = LR 
        self.current_LR = LR 
        self.decay = decay 
        self.index = 0 
        self.epsilon = epsilon 
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
 
    def pre_updateLayerValues(self): 
        if self.decay: 
            self.current_LR = self.LR * (1. / (1. + self.decay * self.index)) 
 
    def updateLayerValues(self, layer): 
 
        if not hasattr(layer, 'weight_cache'): 
            layer.weight_momentums = np.zeros_like(layer.weights) 
            layer.weight_cache = np.zeros_like(layer.weights) 
            layer.bias_momentums = np.zeros_like(layer.biases) 
            layer.bias_cache = np.zeros_like(layer.biases) 
 
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.derivative_weights 
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.derivative_bias 
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.index + 1)) 
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.index + 1)) 
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.derivative_weights**2 
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.derivative_bias**2 
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.index + 1)) 
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.index + 1)) 
 
        layer.weights += -self.current_LR * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon) 
        layer.biases += -self.current_LR * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon) 
 
    def post(self): 
        self.index += 1
class lossParentClass: 

    def calc(self, output, y): 
        sample_losses = self.forward(output, y) 
        data_loss = np.mean(sample_losses) 
        return data_loss 
 
 
class CatCrossEntropy(lossParentClass): 
 
    def forward(self, output, labels): 
 
        passes = len(output) 
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7) 
        if len(labels.shape) == 1: 
            conf = output_clipped[ 
                range(passes), 
                labels 
            ] 
 
        elif len(labels.shape) == 2: 
            conf = np.sum( 
                output_clipped * labels, 
                axis=1 
            ) 
 
        minusLog = -np.log(conf) 
        return minusLog
    def backward(self, dvalues, labels): 
 
        passes = len(dvalues) 
        labels = len(dvalues[0]) 
        if len(labels.shape) == 1: 
            labels = np.eye(labels)[labels] 
 
        self.dinputs = -labels / dvalues 
        self.dinputs = self.dinputs / passes

        
class softmaxActivation_CatCrossEntropy(): 
 
    def __init__(self): 
        self.activation_layer = softmaxActivation() 
        self.loss = CatCrossEntropy() 
 
    def forward(self, inputs, labels): 
        self.activation_layer.forward(inputs) 
        self.output = self.activation_layer.output 
        return self.loss.calc(self.output, labels) 

    def backward(self, dvalues, labels): 
  
        passes = len(dvalues) 
        if len(labels.shape) == 2: 
            labels = np.argmax(labels, axis=1) 
 
        self.dinputs = dvalues.copy() 
        self.dinputs[range(passes), labels] -= 1  
        self.dinputs = self.dinputs / passes
         
def generateData(points, classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes, dtype='uint8')
    for j in range(classes):
        ix = range(points*j,points*(j+1))
        r = np.linspace(0.0,1,points) 
        t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y 
X, y = generateData(points=500, classes=3) 
 
first_hidden = dense(2, 64) 
 
activation_layer = RectifiedLinearActivation() 
 
second_hidden = dense(64, 3) 
loss_activation_layer = softmaxActivation_CatCrossEntropy() 
 
layer_optimizer = ADAM(LR=0.05, decay=5e-7) 
 
for i in range(10001): 
 
    first_hidden.propogate(X) 
 
    activation_layer.forward(first_hidden.output) 
 
    second_hidden.propogate(activation_layer.output) 
 
    current_pass = loss_activation_layer.forward(second_hidden.output, y) 
 
    predictions = np.argmax(loss_activation_layer.output, axis=1) 
    if len(y.shape) == 2: 
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y) 
 
    if not i % 100: 
        print(f'itteration: {i}, ' + 
              f'accuracy: {accuracy:.3f}, ' + 
              f'loss: {current_pass:.3f}, ' + 
              f'learning rate: {layer_optimizer.current_LR}') 
 
    loss_activation_layer.backward(loss_activation_layer.output, y) 
    second_hidden.backwards_propogate(loss_activation_layer.dinputs) 
    activation_layer.backward(second_hidden.dinputs) 
    first_hidden.backwards_propogate(activation_layer.dinputs) 

    layer_optimizer.pre_updateLayerValues() 
    layer_optimizer.updateLayerValues(first_hidden) 
    layer_optimizer.updateLayerValues(second_hidden) 
    layer_optimizer.post() 