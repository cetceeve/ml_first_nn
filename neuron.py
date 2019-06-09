import numpy as np

class Neuron:
    def __init__(self):
        self.activation = 0
        self.firstDerivative = 0

    def feedForward(self, inputs, weights):
        self.activation = np.tanh(np.dot(inputs, weights))
        return self.activation