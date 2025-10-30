import numpy as np

class Activation :
    def forward(self , inputs):
        raise NotImplementedError("forward() must be implemented in subclass of Activation")
    def backward(self , dvalues):
        raise NotImplementedError("backward() must be implemented in subclass of Activation")
    
class Sigmoid(Activation):
    def forward(self , inputs):
        self.outputs = 1/(1 + np.exp(-inputs))
        return self.outputs
     
    def backward(self , dvalues ):
        self.dinputs = dvalues* self.outputs * (1 - self.outputs)

class ReLu(Activation):
    def forward(self , inputs):
        self.inputs = inputs.copy() 
        self.outputs = np.maximum(0 , self.inputs)
        return self.outputs
     
    def backward(self , dvalues ):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class LeakyReLu(Activation):
    def __init__(self , alpha=0.01):
        self.alpha = alpha  
    
    def forward(self , inputs):
        self.inputs = inputs.copy()
        self.outputs = np.where( inputs > 0 , inputs , self.alpha*inputs)
        return self.outputs
     
    def backward(self , dvalues ):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha
        return self.dinputs


class Tanh(Activation):
    def forward(self, inputs):
        self.inputs = inputs.copy()
        np.clip(self.inputs, -500, 500, out=self.inputs)
        ex = np.exp(self.inputs) 
        exi = np.exp(-self.inputs) 
        self.outputs = (ex - exi) / (ex + exi) 
        return self.outputs 
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.outputs ** 2)
        return self.dinputs
    

class Softmax(Activation):
    def forward(self , inputs ):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs